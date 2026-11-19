# Search for supporting evidence (NRC-5): full-text and vector search per statement, fused.
#
# Search only ever adds evidence. Rules are selected by their conditions alone and exceptions are
# reached through the graph, so neither is searched: a missed search result can never drop a
# rule, and a spurious one can never add or excuse one.

from collections.abc import Mapping, Sequence
from typing import Final, Literal
from uuid import UUID

from sqlalchemy import Text, and_, cast, func, literal, select
from sqlalchemy.dialects.postgresql import REGCONFIG
from sqlalchemy.ext.asyncio import AsyncSession

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


async def lexical(
    db: AsyncSession,
    tenant_id: UUID,
    bundle_id: UUID,
    text: str,
    language: PolicyLanguage,
    limit: int,
) -> list[UUID]:
    """Best full-text matches for one statement, best first (ties by node key).

    websearch_to_tsquery parses the text safely but ANDs every word, so a whole sentence would
    rarely match anything; its terms are ORed instead. ts_rank (not ts_rank_cd) orders them: with
    an OR query, cover density scores one repeated word as high as two different ones, while
    ts_rank ranks a node that shares more distinct terms higher.
    """
    config = cast(literal(TEXT_SEARCH_CONFIG[language]), REGCONFIG)
    terms = cast(func.websearch_to_tsquery(config, text), Text)
    # Negations ("-word") are dropped: ORed, `!word` would match every node without that word.
    ored = func.regexp_replace(func.replace(terms, " & ", " | "), "!", "", "g")
    query = func.to_tsquery(config, ored)
    rows = await db.execute(
        select(PolicyNode.id)
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
        .order_by(func.ts_rank(PolicyNode.search_vector, query).desc(), PolicyNode.node_key)
        .limit(limit)
    )
    return list(rows.scalars())


async def semantic(
    db: AsyncSession,
    tenant_id: UUID,
    bundle_id: UUID,
    model_id: str,
    vector: Sequence[float],
    limit: int,
) -> list[UUID]:
    """Nearest nodes by cosine distance for one statement's vector, best first.

    Exact search over one bundle's vectors (no ANN index: a bundle is a few hundred nodes, and a
    filtered approximate scan could return fewer than `limit` rows).
    """
    rows = await db.execute(
        select(PolicyNode.id)
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
        .order_by(PolicyNodeEmbedding.embedding.cosine_distance(list(vector)), PolicyNode.node_key)
        .limit(limit)
    )
    return list(rows.scalars())


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
