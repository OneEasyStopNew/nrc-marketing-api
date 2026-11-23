"""Import a policy graph (source, clauses, nodes, rules, edges) into one policy bundle.

Usage (runs as the database owner, MIGRATION_DATABASE_URL):
    uv run python -m scripts.import_policy_graph --tenant-id <uuid> --bundle-id <uuid> \
        [--file path/to/graph.json]

Without --file the synthetic rulebook is imported. A bundle gets its graph once: a bundle that
already has one is never changed (rerunning with the same file does nothing; a different file is
rejected). This is also why importing into an ACTIVE bundle is allowed: it backfills legacy
bundles that have no graph yet, and an existing graph is never rewritten. Prints ids and counts.
"""

import argparse
import asyncio
import hashlib
import json
from collections import Counter
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Annotated, Final
from uuid import UUID, uuid4

from pydantic import AwareDatetime, BaseModel, ConfigDict, Field, StringConstraints
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.pool import NullPool

from app.core.config import OwnerDatabaseSettings
from app.db.session import TenantSession, bind_tenant
from app.models import (
    PolicyBundle,
    PolicyEdge,
    PolicyNode,
    PolicyRuleRecord,
    PolicySource,
    PolicySourceClause,
)
from app.models.enums import (
    CheckType,
    Gate,
    PolicyEdgeType,
    PolicyLanguage,
    PolicyNodeType,
    Severity,
)
from app.models.policy_graph import RULE_ID_PATTERN
from app.schemas.tenancy import StudyFacts

SYNTHETIC_GRAPH_PATH: Final = (
    Path(__file__).resolve().parent / "data" / "synthetic_policy_graph.json"
)
# Recorded as created_by on imported sources.
IMPORT_ACTOR: Final = "import_policy_graph"

_N = PolicyNodeType
# Which node types an edge type may join: (from, to). Mirrors the table in the design note.
_EDGE_ENDS: Final[dict[PolicyEdgeType, tuple[frozenset[_N], frozenset[_N]]]] = {
    PolicyEdgeType.SUPPORTED_BY: (
        frozenset({_N.RULE, _N.EXCEPTION}),
        frozenset({_N.CLAUSE, _N.INTERPRETATION, _N.EXAMPLE}),
    ),
    PolicyEdgeType.REQUIRES_DISCLOSURE: (frozenset({_N.RULE}), frozenset({_N.DISCLOSURE})),
    PolicyEdgeType.EXCEPT_WHEN: (frozenset({_N.RULE, _N.EXCEPTION}), frozenset({_N.EXCEPTION})),
    PolicyEdgeType.SUPERSEDES: (frozenset({_N.RULE}), frozenset({_N.RULE})),
    PolicyEdgeType.APPLIES_TO: (frozenset({_N.INTERPRETATION}), frozenset({_N.RULE})),
}

_Tag = Annotated[str, StringConstraints(min_length=1, max_length=64)]


class PolicyGraphImportError(ValueError):
    """The graph file is inconsistent; nothing was written."""


class _ImportModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class ClauseImport(_ImportModel):
    clause_ref: str = Field(min_length=1, max_length=100)
    section: str | None = Field(default=None, max_length=200)
    page: int | None = Field(default=None, ge=1)
    text: str = Field(min_length=1)
    language: PolicyLanguage = PolicyLanguage.EN


class SourceImport(_ImportModel):
    title: str = Field(min_length=1, max_length=300)
    source_ref: str = Field(min_length=1, max_length=300)
    language: PolicyLanguage = PolicyLanguage.EN
    is_synthetic: bool
    clauses: list[ClauseImport] = Field(min_length=1)


class NodeImport(_ImportModel):
    node_key: str = Field(min_length=1, max_length=80)
    node_type: PolicyNodeType
    text: str = Field(min_length=1)
    language: PolicyLanguage = PolicyLanguage.EN
    # Source clause the node quotes (its clause_ref in the file's source).
    clause_ref: str | None = None
    # A node proposed but not approved never takes part in retrieval.
    approved: bool = False
    input_kinds: list[_Tag] = Field(default_factory=list)
    channels: list[_Tag] = Field(default_factory=list)
    post_formats: list[_Tag] = Field(default_factory=list)
    languages: list[_Tag] = Field(default_factory=list)
    audiences: list[_Tag] = Field(default_factory=list)
    jurisdictions: list[_Tag] = Field(default_factory=list)
    study_ids: list[UUID] = Field(default_factory=list)
    # With an offset: a date without one would land in timestamptz in whatever timezone the
    # importing session assumes, shifting a rule's window by hours.
    effective_from: AwareDatetime | None = None
    effective_to: AwareDatetime | None = None
    required_study_fact: str | None = Field(default=None, max_length=64)


class RuleImport(_ImportModel):
    node_key: str
    rule_id: str = Field(pattern=RULE_ID_PATTERN)
    rule_version: int = Field(ge=1)
    title: str = Field(min_length=1, max_length=200)
    requirement: str = Field(min_length=1, max_length=1000)
    gate: Gate
    severity: Severity
    mandatory: bool
    check_type: CheckType
    exact_wording: str | None = None
    # Plain str: a blank phrase must reach validate_graph and raise the typed error.
    prohibited_phrases: list[str] = Field(default_factory=list)


class EdgeImport(_ImportModel):
    edge_type: PolicyEdgeType
    from_node_key: str
    to_node_key: str
    approved: bool = False
    provenance_clause_ref: str | None = None
    provenance_note: str | None = None


class PolicyGraphImport(_ImportModel):
    source: SourceImport
    nodes: list[NodeImport]
    rules: list[RuleImport]
    edges: list[EdgeImport]


@dataclass(frozen=True, slots=True)
class ImportResult:
    skipped: bool
    sources_created: int = 0
    sources_reused: int = 0
    clauses_created: int = 0
    nodes: int = 0
    rules: int = 0
    edges: int = 0


def load_graph(path: Path) -> PolicyGraphImport:
    return PolicyGraphImport.model_validate_json(path.read_text(encoding="utf-8"))


def load_synthetic_graph() -> PolicyGraphImport:
    return load_graph(SYNTHETIC_GRAPH_PATH)


def clean_phrases(rule: RuleImport) -> list[str]:
    phrases = [phrase.strip() for phrase in rule.prohibited_phrases]
    if not all(phrases):
        raise PolicyGraphImportError(f"Rule {rule.rule_id} has a blank prohibited phrase.")
    return phrases


def _reject_duplicates(what: str, values: Iterable[object]) -> None:
    repeated = sorted(str(v) for v, n in Counter(values).items() if n > 1)
    if repeated:
        raise PolicyGraphImportError(f"Duplicate {what}: {', '.join(repeated)}.")


def validate_graph(graph: PolicyGraphImport) -> None:
    # Everything that can be checked without the database; runs before any write.
    clauses = {clause.clause_ref: clause for clause in graph.source.clauses}
    clause_refs = set(clauses)
    _reject_duplicates("clause_ref", (c.clause_ref for c in graph.source.clauses))
    _reject_duplicates("node_key", (n.node_key for n in graph.nodes))
    node_types = {node.node_key: node.node_type for node in graph.nodes}
    for node in graph.nodes:
        if node.clause_ref is not None:
            _validate_node_clause(node, clauses.get(node.clause_ref))
        if node.required_study_fact and node.required_study_fact not in StudyFacts.model_fields:
            raise PolicyGraphImportError(
                f"Node {node.node_key} requires study fact {node.required_study_fact}, "
                "which is not a StudyFacts field."
            )
    _reject_duplicates("rule node_key", (r.node_key for r in graph.rules))
    _reject_duplicates("rule_id", (r.rule_id for r in graph.rules))
    for rule in graph.rules:
        node_type = node_types.get(rule.node_key)
        if node_type is None:
            raise PolicyGraphImportError(f"Rule {rule.rule_id} names unknown node {rule.node_key}.")
        if node_type is not PolicyNodeType.RULE:
            raise PolicyGraphImportError(
                f"Rule {rule.rule_id} is on node {rule.node_key}, which is a {node_type} node."
            )
        clean_phrases(rule)
    with_rule = {rule.node_key for rule in graph.rules}
    bare = sorted(
        key
        for key, kind in node_types.items()
        if kind is PolicyNodeType.RULE and key not in with_rule
    )
    if bare:
        raise PolicyGraphImportError(f"RULE node without a rule row: {', '.join(bare)}.")
    _reject_duplicates("edge", ((e.edge_type, e.from_node_key, e.to_node_key) for e in graph.edges))
    for edge in graph.edges:
        _validate_edge(edge, node_types, clause_refs)


def _validate_node_clause(node: NodeImport, clause: ClauseImport | None) -> None:
    if clause is None:
        raise PolicyGraphImportError(
            f"Node {node.node_key} cites clause_ref {node.clause_ref}, not in the source."
        )
    # The search column uses the node's language; a mismatch indexes text with the wrong stemmer.
    if node.language is not clause.language:
        raise PolicyGraphImportError(
            f"Node {node.node_key} is {node.language} but cites a {clause.language} clause."
        )
    # Evidence quotes the node; it must say exactly what the cited clause says.
    if node.node_type is PolicyNodeType.CLAUSE and node.text != clause.text:
        raise PolicyGraphImportError(
            f"CLAUSE node {node.node_key} text differs from clause {clause.clause_ref}."
        )


def _validate_edge(
    edge: EdgeImport, node_types: dict[str, PolicyNodeType], clause_refs: set[str]
) -> None:
    label = f"{edge.edge_type} edge {edge.from_node_key} -> {edge.to_node_key}"
    from_type = node_types.get(edge.from_node_key)
    to_type = node_types.get(edge.to_node_key)
    if from_type is None or to_type is None:
        raise PolicyGraphImportError(f"The {label} has an endpoint that is not a node.")
    if edge.from_node_key == edge.to_node_key:
        raise PolicyGraphImportError(f"The {label} links a node to itself.")
    from_types, to_types = _EDGE_ENDS[edge.edge_type]
    if from_type not in from_types or to_type not in to_types:
        raise PolicyGraphImportError(
            f"The {label} joins {from_type} to {to_type}, which this edge type does not allow."
        )
    ref = edge.provenance_clause_ref
    if ref is not None and ref not in clause_refs:
        raise PolicyGraphImportError(f"The {label} cites clause_ref {ref}, not in the source.")
    if edge.approved and ref is None and not (edge.provenance_note or "").strip():
        raise PolicyGraphImportError(f"The approved {label} has no provenance.")


# What decides whether a stored graph matches a file: every node's key, type, language and text,
# and every rule's id, check type and phrases.
_Fingerprint = tuple[
    frozenset[tuple[str, str, str, str]], frozenset[tuple[str, str, tuple[str, ...]]]
]


def _fingerprint(graph: PolicyGraphImport) -> _Fingerprint:
    nodes = frozenset((n.node_key, n.node_type, n.language, n.text) for n in graph.nodes)
    rules = frozenset((r.rule_id, r.check_type, tuple(clean_phrases(r))) for r in graph.rules)
    return nodes, rules


async def _stored_fingerprint(db: AsyncSession, tenant_id: UUID, bundle_id: UUID) -> _Fingerprint:
    scope = (PolicyNode.tenant_id == tenant_id, PolicyNode.policy_bundle_id == bundle_id)
    node_rows = await db.execute(
        select(
            PolicyNode.node_key, PolicyNode.node_type, PolicyNode.language, PolicyNode.text
        ).where(*scope)
    )
    rule_rows = await db.execute(
        select(
            PolicyRuleRecord.rule_id,
            PolicyRuleRecord.check_type,
            PolicyRuleRecord.prohibited_phrases,
        ).where(
            PolicyRuleRecord.tenant_id == tenant_id, PolicyRuleRecord.policy_bundle_id == bundle_id
        )
    )
    nodes = frozenset((key, kind, lang, text) for key, kind, lang, text in node_rows)
    rules = frozenset((rid, check, tuple(phrases)) for rid, check, phrases in rule_rows)
    return nodes, rules


def source_checksum(source: SourceImport) -> str:
    # Canonical JSON, so key order and whitespace in the file never change the checksum.
    canonical = json.dumps(
        source.model_dump(mode="json"), sort_keys=True, separators=(",", ":"), ensure_ascii=False
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


async def import_policy_graph(
    db: AsyncSession, tenant_id: UUID, bundle_id: UUID, graph: PolicyGraphImport
) -> ImportResult:
    # Runs in the caller's transaction, which has already bound the tenant. Writes nothing if the
    # graph is inconsistent or the bundle already has a graph.
    validate_graph(graph)
    row = (
        await db.execute(
            select(PolicyBundle.rules).where(
                PolicyBundle.tenant_id == tenant_id, PolicyBundle.id == bundle_id
            )
        )
    ).first()
    if row is None:
        raise PolicyGraphImportError(f"Policy bundle {bundle_id} not found for this tenant.")
    bundle_rule_ids = {str(rule["rule_id"]) for rule in row[0]}
    graph_rule_ids = {rule.rule_id for rule in graph.rules}
    # The bundle's JSONB rules are what validation uses today; evidence must cite the same ones.
    if bundle_rule_ids and bundle_rule_ids != graph_rule_ids:
        raise PolicyGraphImportError(
            "Graph rule ids differ from the bundle's: only in graph "
            f"{sorted(graph_rule_ids - bundle_rule_ids)}, only in bundle "
            f"{sorted(bundle_rule_ids - graph_rule_ids)}."
        )
    existing = await db.scalar(
        select(func.count())
        .select_from(PolicyNode)
        .where(PolicyNode.tenant_id == tenant_id, PolicyNode.policy_bundle_id == bundle_id)
    )
    if existing:
        # A bundle's graph is written once; a different file must not look like a no-op.
        if await _stored_fingerprint(db, tenant_id, bundle_id) != _fingerprint(graph):
            raise PolicyGraphImportError(
                f"Policy bundle {bundle_id} already has a different graph; it is never rewritten."
            )
        return ImportResult(skipped=True)

    clause_ids, created = await _ensure_source(db, tenant_id, graph.source)
    node_ids = {node.node_key: uuid4() for node in graph.nodes}
    db.add_all(
        PolicyNode(
            id=node_ids[node.node_key],
            tenant_id=tenant_id,
            policy_bundle_id=bundle_id,
            source_clause_id=clause_ids[node.clause_ref] if node.clause_ref else None,
            **node.model_dump(exclude={"clause_ref"}),
        )
        for node in graph.nodes
    )
    await db.flush()
    db.add_all(
        PolicyRuleRecord(
            tenant_id=tenant_id,
            policy_bundle_id=bundle_id,
            node_id=node_ids[rule.node_key],
            prohibited_phrases=clean_phrases(rule),
            **rule.model_dump(exclude={"node_key", "prohibited_phrases"}),
        )
        for rule in graph.rules
    )
    db.add_all(
        PolicyEdge(
            tenant_id=tenant_id,
            policy_bundle_id=bundle_id,
            edge_type=edge.edge_type,
            from_node_id=node_ids[edge.from_node_key],
            to_node_id=node_ids[edge.to_node_key],
            approved=edge.approved,
            provenance_clause_id=(
                clause_ids[edge.provenance_clause_ref] if edge.provenance_clause_ref else None
            ),
            provenance_note=edge.provenance_note,
        )
        for edge in graph.edges
    )
    await db.flush()
    return ImportResult(
        skipped=False,
        sources_created=int(created),
        sources_reused=int(not created),
        clauses_created=len(graph.source.clauses) if created else 0,
        nodes=len(graph.nodes),
        rules=len(graph.rules),
        edges=len(graph.edges),
    )


async def _ensure_source(
    db: AsyncSession, tenant_id: UUID, data: SourceImport
) -> tuple[dict[str, UUID], bool]:
    # A source is shared by every bundle that imports the same content (same checksum). Returns
    # the clause ids by clause_ref, and whether the source was created now.
    checksum = source_checksum(data)
    source_id = await db.scalar(
        select(PolicySource.id).where(
            PolicySource.tenant_id == tenant_id, PolicySource.checksum_sha256 == checksum
        )
    )
    created = source_id is None
    if source_id is None:
        source_id = uuid4()
        db.add(
            PolicySource(
                id=source_id,
                tenant_id=tenant_id,
                title=data.title,
                source_ref=data.source_ref,
                checksum_sha256=checksum,
                language=data.language,
                is_synthetic=data.is_synthetic,
                created_by=IMPORT_ACTOR,
            )
        )
        await db.flush()
        db.add_all(
            PolicySourceClause(
                id=uuid4(),
                tenant_id=tenant_id,
                source_id=source_id,
                checksum_sha256=_sha256(clause.text),
                **clause.model_dump(),
            )
            for clause in data.clauses
        )
        await db.flush()
    rows = await db.execute(
        select(PolicySourceClause.clause_ref, PolicySourceClause.id).where(
            PolicySourceClause.tenant_id == tenant_id, PolicySourceClause.source_id == source_id
        )
    )
    return {ref: clause_id for ref, clause_id in rows}, created


async def _run(tenant_id: UUID, bundle_id: UUID, graph: PolicyGraphImport) -> ImportResult:
    engine = create_async_engine(
        OwnerDatabaseSettings().migration_database_url.get_secret_value(), poolclass=NullPool
    )
    try:
        async with AsyncSession(engine, sync_session_class=TenantSession) as db:
            await bind_tenant(db, tenant_id)
            result = await import_policy_graph(db, tenant_id, bundle_id, graph)
            await db.commit()
    finally:
        await engine.dispose()
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0] if __doc__ else None)
    parser.add_argument("--tenant-id", type=UUID, required=True)
    parser.add_argument("--bundle-id", type=UUID, required=True)
    parser.add_argument("--file", type=Path, help="graph JSON; default is the synthetic rulebook")
    args = parser.parse_args()
    graph = load_graph(args.file) if args.file else load_synthetic_graph()
    result = asyncio.run(_run(args.tenant_id, args.bundle_id, graph))
    summary = {
        "tenant_id": str(args.tenant_id),
        "policy_bundle_id": str(args.bundle_id),
        **asdict(result),
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
