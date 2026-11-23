# Search for supporting evidence (NRC-5): full-text and vector search for every statement in one
# query each, fused per statement.
#
# Search only ever adds evidence. Rules are selected by their conditions alone and exceptions are
# reached through the graph, so neither is searched: a missed search result can never drop a
# rule, and a spurious one can never add or excuse one.

from collections.abc import Iterable, Mapping, Sequence
from typing import Final, Literal
from uuid import UUID

from pgvector.sqlalchemy import Vector
from sqlalchemy import Text, and_, bindparam, cast, func, literal, select, true
from sqlalchemy.dialects.postgresql import ARRAY, REGCONFIG
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.sql.selectable import TableValuedAlias

from app.db.session import bound_tenant
from app.models import PolicyNode, PolicyNodeEmbedding
from app.models.enums import PolicyLanguage
from app.models.policy_graph import SEARCHABLE_NODE_TYPES

# Text-search configuration per policy language (the same mapping the generated search_vector
# column uses).
TEXT_SEARCH_CONFIG: Final[Mapping[PolicyLanguage, str]] = {
    PolicyLanguage.EN: "english",
    PolicyLanguage.ES: "spanish",
}

Source = Literal["lexical", "semantic"]


def _statements(values: Sequence[str]) -> TableValuedAlias:
    # One row per statement, numbered from 1: (value, position). Every statement is searched in
    # one query (a LATERAL top-k per row), not one query per statement.
    return (
        func.unnest(bindparam("statements", list(values), type_=ARRAY(Text)))
        .table_valued("value", with_ordinality="position")
        .render_derived(name="statements")
    )


def _per_statement(count: int, rows: Iterable[tuple[int, UUID]]) -> list[list[UUID]]:
    ranked: list[list[UUID]] = [[] for _ in range(count)]
    for position, node_id in rows:
        ranked[position - 1].append(node_id)
    return ranked


async def lexical(
    db: AsyncSession,
    bundle_id: UUID,
    texts: Sequence[str],
    language: PolicyLanguage,
    limit: int,
) -> list[list[UUID]]:
    """Best full-text matches for each statement, best first (ties by node key), in one query.

    websearch_to_tsquery parses the text safely but ANDs every word, so a whole sentence would
    rarely match anything; its terms are ORed instead. ts_rank (not ts_rank_cd) orders them: with
    an OR query, cover density scores one repeated word as high as two different ones, while
    ts_rank ranks a node that shares more distinct terms higher.
    """
    if not texts:
        return []
    tenant_id = bound_tenant(db)
    statements = _statements(texts)
    config = cast(literal(TEXT_SEARCH_CONFIG[language]), REGCONFIG)
    terms = cast(func.websearch_to_tsquery(config, statements.c.value), Text)
    # Negations ("-word") are dropped: ORed, `!word` would match every node without that word.
    ored = func.regexp_replace(func.replace(terms, " & ", " | "), "!", "", "g")
    query = func.to_tsquery(config, ored)
    score = func.ts_rank(PolicyNode.search_vector, query)
    best = (
        select(PolicyNode.id, PolicyNode.node_key, score.label("score"))
        .where(
            PolicyNode.tenant_id == tenant_id,
            PolicyNode.policy_bundle_id == bundle_id,
            PolicyNode.approved.is_(True),
            PolicyNode.node_type.in_(SEARCHABLE_NODE_TYPES),
            # The statement's language only: other languages' stems would match by accident;
            # finding text across languages is the vector search's job.
            PolicyNode.language == language,
            PolicyNode.search_vector.bool_op("@@")(query),
        )
        .order_by(score.desc(), PolicyNode.node_key)
        .limit(limit)
        .lateral("best")
    )
    rows = await db.execute(
        select(statements.c.position, best.c.id)
        .select_from(statements.join(best, true()))
        .order_by(statements.c.position, best.c.score.desc(), best.c.node_key)
    )
    return _per_statement(len(texts), rows)


async def semantic(
    db: AsyncSession,
    bundle_id: UUID,
    model_id: str,
    vectors: Sequence[Sequence[float]],
    limit: int,
) -> list[list[UUID]]:
    """Nearest nodes by cosine distance for each statement's vector, best first, in one query.

    Exact search over one bundle's vectors (no ANN index: a bundle is a few hundred nodes, and a
    filtered approximate scan could return fewer than `limit` rows).
    """
    if not vectors:
        return []
    tenant_id = bound_tenant(db)
    # pgvector's text form, "[x,y,...]"; cast back to vector per row.
    statements = _statements(["[" + ",".join(map(str, vector)) + "]" for vector in vectors])
    distance = PolicyNodeEmbedding.embedding.cosine_distance(cast(statements.c.value, Vector))
    best = (
        select(PolicyNode.id, PolicyNode.node_key, distance.label("distance"))
        .join(
            PolicyNodeEmbedding,
            and_(
                PolicyNodeEmbedding.tenant_id == PolicyNode.tenant_id,
                PolicyNodeEmbedding.policy_bundle_id == PolicyNode.policy_bundle_id,
                PolicyNodeEmbedding.node_id == PolicyNode.id,
            ),
        )
        .where(
            PolicyNode.tenant_id == tenant_id,
            PolicyNode.policy_bundle_id == bundle_id,
            PolicyNode.approved.is_(True),
            PolicyNode.node_type.in_(SEARCHABLE_NODE_TYPES),
            PolicyNodeEmbedding.model_id == model_id,
        )
        .order_by(distance, PolicyNode.node_key)
        .limit(limit)
        .lateral("best")
    )
    rows = await db.execute(
        select(statements.c.position, best.c.id)
        .select_from(statements.join(best, true()))
        .order_by(statements.c.position, best.c.distance, best.c.node_key)
    )
    return _per_statement(len(vectors), rows)


def fuse(
    ranked: Mapping[Source, Sequence[UUID]],
    names: Mapping[UUID, str],
    limit: int,
    rrf_k: int,
) -> list[tuple[UUID, tuple[Source, ...]]]:
    """Reciprocal rank fusion of the per-search lists: the `limit` best nodes, best first.

    Each list contributes 1 / (rrf_k + rank) for the nodes it found; ties go to the node key so
    the order is deterministic. Returns each node with the searches that found it.
    """
    scores: dict[UUID, float] = {}
    found: dict[UUID, list[Source]] = {}
    for source in sorted(ranked):
        for rank, node_id in enumerate(ranked[source], start=1):
            scores[node_id] = scores.get(node_id, 0.0) + 1.0 / (rrf_k + rank)
            found.setdefault(node_id, []).append(source)
    best = sorted(scores, key=lambda node_id: (-scores[node_id], names[node_id]))[:limit]
    return [(node_id, tuple(found[node_id])) for node_id in best]
