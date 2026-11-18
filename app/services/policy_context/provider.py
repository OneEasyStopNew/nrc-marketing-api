"""Policy context retrieval (NRC-5): select the rules a check must use and gather their evidence.

Steps for one request (design note §3.1):
1. Load the pinned bundle and study snapshot; refuse a missing or unusable bundle, or one with no
   graph (PolicyContextUnavailableError; callers record it and never report PASS).
2. Select rules by their stored conditions only. A rule whose conditions name a dimension the
   request doesn't know is kept and flagged, never dropped.
3. Resolve supersedes between selected rules; anything ambiguous is a CONFLICT and both stay.
4. Walk approved edges from each selected rule (bounded hops and nodes) for clauses,
   disclosures, interpretations and exceptions. Exceptions are listed with whether they apply;
   they never remove a rule.
5. Store the manifest once per distinct set of inputs (the cache key) and return it.

Search (full-text and vector) is added on top of this in a later NRC-5 change.
"""

import hashlib
import json
import logging
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Final, Protocol
from uuid import UUID, uuid4

from sqlalchemy import and_, select
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import Settings
from app.core.errors import PolicyContextUnavailableError
from app.models import (
    ContextManifestRecord,
    PolicyBundle,
    PolicyEdge,
    PolicyIndexBuild,
    PolicyNode,
    PolicyRuleRecord,
    PolicySourceClause,
    StudySnapshot,
)
from app.models.enums import (
    Applicability,
    BundleLifecycleState,
    CheckType,
    Gate,
    IndexBuildState,
    MissingContextCode,
    PolicyEdgeType,
    PolicyLanguage,
    PolicyNodeType,
    RetrievalMode,
    Severity,
)
from app.models.policy_graph import SEARCHABLE_NODE_TYPES
from app.schemas.tenancy import StudyFacts
from app.services.policy_context import search
from app.services.policy_context.applicability import (
    Conditions,
    Verdict,
    evaluate,
    window_reasons,
)
from app.services.policy_context.graph import adjacency, walk
from app.services.policy_context.manifest import (
    EXCERPT_MAX_CHARS,
    RETRIEVAL_CONFIG_VERSION,
    BundleRef,
    ContextManifest,
    ContextRequest,
    ContextScope,
    EvidenceItem,
    ExceptionItem,
    ManifestRule,
    MissingContext,
    RetrievalConfig,
    SearchMatch,
    SnapshotRef,
    Statement,
    StatementRef,
    SupersededRule,
)
from app.services.policy_index import index_content_checksum
from app.services.providers.embeddings import (
    EmbeddingProvider,
    EmbeddingPurpose,
    EmbeddingResponseError,
)

logger = logging.getLogger(__name__)

# ACTIVE for new checks; RETIRED so an earlier decision can be rebuilt from its pinned bundle.
USABLE_BUNDLE_STATES: Final = frozenset({BundleLifecycleState.ACTIVE, BundleLifecycleState.RETIRED})
# A rule has provenance if its node quotes a clause or it has one of these approved edges (an
# approved edge always cites a clause or a note, by a database check).
_PROVENANCE_EDGES: Final = frozenset(
    {PolicyEdgeType.SUPPORTED_BY, PolicyEdgeType.REQUIRES_DISCLOSURE}
)

Edge = tuple[PolicyEdgeType, UUID, UUID]


@dataclass(frozen=True, slots=True)
class StoredManifest:
    id: UUID
    manifest: ContextManifest
    # True when an identical request had already stored this manifest.
    reused: bool


class PolicyContextProvider(Protocol):
    """Builds the policy context for one check. Callers depend on this, not on Postgres."""

    async def build(
        self, db: AsyncSession, tenant_id: UUID, request: ContextRequest, *, actor: str
    ) -> StoredManifest: ...


@dataclass(frozen=True, slots=True)
class GraphNode:
    id: UUID
    key: str
    node_type: PolicyNodeType
    language: PolicyLanguage
    text: str
    clause_ref: str | None
    conditions: Conditions


@dataclass(frozen=True, slots=True)
class GraphRule:
    node_id: UUID
    rule_id: str
    rule_version: int
    mandatory: bool
    check_type: CheckType
    gate: Gate
    severity: Severity


@dataclass(frozen=True, slots=True)
class SearchPlan:
    mode: RetrievalMode
    # The statements searched (at most max_statements) and their manifest references.
    statements: tuple[Statement, ...]
    refs: tuple[StatementRef, ...]
    # More statements were given than were searched.
    truncated: bool
    embedding_model_id: str | None


@dataclass(frozen=True, slots=True)
class SearchResult:
    plan: SearchPlan
    # Search hits per node (only approved, searchable nodes of the bundle).
    matches: Mapping[UUID, tuple[SearchMatch, ...]]


@dataclass(frozen=True, slots=True)
class LoadedContext:
    # Everything one manifest is built from, loaded in a fixed number of queries.
    bundle: BundleRef
    snapshot: SnapshotRef
    study_id: UUID
    facts: StudyFacts
    nodes: Mapping[UUID, GraphNode]
    rules: Sequence[GraphRule]
    edges: Sequence[Edge]


class AuroraPolicyContextProvider:
    """Reads the approved graph from Postgres (Aurora in AWS) under the caller's tenant.

    With an embedder, statements are searched by full-text and vector search; without one (the
    API process, which never calls providers) by full-text only, and the manifest says so.
    """

    def __init__(self, settings: Settings, embedder: EmbeddingProvider | None = None) -> None:
        self._config = RetrievalConfig(
            version=RETRIEVAL_CONFIG_VERSION,
            max_hops=settings.policy_retrieval_max_hops,
            max_nodes=settings.policy_retrieval_max_nodes,
            top_k=settings.policy_retrieval_top_k,
            max_statements=settings.policy_retrieval_max_statements,
        )
        self._embedder = embedder

    async def build(
        self, db: AsyncSession, tenant_id: UUID, request: ContextRequest, *, actor: str
    ) -> StoredManifest:
        """Return the stored manifest for this request, building and storing it if new.

        Runs in the caller's transaction, which has bound the tenant; the caller commits. A
        stored manifest is found before any search, so a repeat costs no embedding call.

        Raises:
            PolicyContextUnavailableError: the bundle is missing, not usable or has no graph,
                the study snapshot is missing, or (with an embedder) the bundle has no READY
                index for the embedder's model or it doesn't match the graph
                (details.reason says which).
            ProviderError: the embedder failed (only with an embedder).
        """
        loaded = await load_context(db, tenant_id, request)
        plan = self._plan(request)
        if plan.mode is RetrievalMode.FULL and plan.embedding_model_id is not None:
            await _require_index(db, tenant_id, loaded, plan.embedding_model_id)
        key = cache_key(tenant_id, loaded, request.scope, plan, self._config)
        existing = await _find(db, tenant_id, key)
        if existing is not None:
            return existing
        matches = await self._search(db, tenant_id, loaded, request.scope.language, plan)
        manifest = assemble(loaded, request.scope, self._config, SearchResult(plan, matches))
        stored = await _store(db, tenant_id, key, loaded, manifest, actor)
        logger.info(
            "policy context built manifest=%s tenant=%s bundle=%s reused=%s",
            stored.id,
            tenant_id,
            loaded.bundle.id,
            stored.reused,
        )
        return stored

    def _plan(self, request: ContextRequest) -> SearchPlan:
        statements = request.statements[: self._config.max_statements]
        if not statements:
            mode = RetrievalMode.GRAPH_ONLY
        elif self._embedder is None:
            mode = RetrievalMode.LEXICAL_ONLY
        else:
            mode = RetrievalMode.FULL
        return SearchPlan(
            mode=mode,
            statements=statements,
            refs=tuple(StatementRef(sha256=_sha256(s.text), span=s.span) for s in statements),
            truncated=len(request.statements) > len(statements),
            embedding_model_id=(
                self._embedder.model_id
                if mode is RetrievalMode.FULL and self._embedder is not None
                else None
            ),
        )

    async def _search(
        self,
        db: AsyncSession,
        tenant_id: UUID,
        loaded: LoadedContext,
        language: PolicyLanguage,
        plan: SearchPlan,
    ) -> dict[UUID, tuple[SearchMatch, ...]]:
        # Per statement: full-text (and vector) top_k, fused into one ranked list.
        if not plan.statements:
            return {}
        texts = [statement.text for statement in plan.statements]
        vectors: list[list[float]] | None = None
        if plan.mode is RetrievalMode.FULL and self._embedder is not None:
            vectors = await self._embedder.embed(texts, EmbeddingPurpose.QUERY)
            if len(vectors) != len(texts):
                raise EmbeddingResponseError(
                    f"{self._embedder.model_id} returned {len(vectors)} vectors for {len(texts)}"
                    " statements"
                )
        names = {node_id: node.key for node_id, node in loaded.nodes.items()}
        top_k, bundle_id = self._config.top_k, loaded.bundle.id
        matches: dict[UUID, list[SearchMatch]] = {}
        for index, text in enumerate(texts):
            ranked: dict[search.Source, Sequence[UUID]] = {
                "lexical": await search.lexical(db, tenant_id, bundle_id, text, language, top_k)
            }
            if vectors is not None and plan.embedding_model_id is not None:
                ranked["semantic"] = await search.semantic(
                    db, tenant_id, bundle_id, plan.embedding_model_id, vectors[index], top_k
                )
            fused = search.fuse(ranked, names, top_k, self._config.rrf_k)
            for rank, (node_id, found_by) in enumerate(fused, start=1):
                if node_id in names:
                    matches.setdefault(node_id, []).append(
                        SearchMatch(statement_index=index, found_by=found_by, rank=rank)
                    )
        return {node_id: tuple(found) for node_id, found in matches.items()}


async def load_context(db: AsyncSession, tenant_id: UUID, request: ContextRequest) -> LoadedContext:
    bundle_row = (
        await db.execute(
            select(
                PolicyBundle.id,
                PolicyBundle.version,
                PolicyBundle.is_synthetic,
                PolicyBundle.lifecycle_state,
            ).where(
                PolicyBundle.tenant_id == tenant_id, PolicyBundle.id == request.policy_bundle_id
            )
        )
    ).one_or_none()
    if bundle_row is None:
        raise _unavailable("BUNDLE_NOT_FOUND")
    if bundle_row.lifecycle_state not in USABLE_BUNDLE_STATES:
        raise _unavailable("BUNDLE_NOT_USABLE", lifecycle_state=bundle_row.lifecycle_state)
    snapshot_row = (
        await db.execute(
            select(
                StudySnapshot.id, StudySnapshot.version, StudySnapshot.study_id, StudySnapshot.facts
            ).where(
                StudySnapshot.tenant_id == tenant_id,
                StudySnapshot.id == request.study_snapshot_id,
            )
        )
    ).one_or_none()
    if snapshot_row is None:
        raise _unavailable("SNAPSHOT_NOT_FOUND")
    nodes = await _load_nodes(db, tenant_id, bundle_row.id)
    rules = [
        rule for rule in await _load_rules(db, tenant_id, bundle_row.id) if rule.node_id in nodes
    ]
    if not rules:
        # A legacy bundle without a graph (or with none approved): nothing can be selected.
        raise _unavailable("GRAPH_MISSING")
    return LoadedContext(
        bundle=BundleRef(
            id=bundle_row.id, version=bundle_row.version, is_synthetic=bundle_row.is_synthetic
        ),
        snapshot=SnapshotRef(id=snapshot_row.id, version=snapshot_row.version),
        study_id=snapshot_row.study_id,
        facts=StudyFacts.model_validate(snapshot_row.facts),
        nodes=nodes,
        rules=rules,
        edges=await _load_edges(db, tenant_id, bundle_row.id),
    )


def cache_key(
    tenant_id: UUID,
    loaded: LoadedContext,
    scope: ContextScope,
    plan: SearchPlan,
    config: RetrievalConfig,
) -> str:
    """Hash of every input the manifest depends on; any change gives a new key.

    The graph's content is hashed as well rather than trusted to be immutable, so a changed
    node, rule or edge never reuses an old manifest. evaluated_at itself is left out (it changes
    on every request); what it decides, which dated nodes are in effect, is in the key instead.
    The statements, the search mode and the embedding model are in it too; the index build for
    that model is checked against the graph before searching, so its vectors are covered.
    """
    windows = sorted(
        (node.key, not window_reasons(node.conditions, scope.evaluated_at))
        for node in loaded.nodes.values()
        if node.conditions.effective_from or node.conditions.effective_to
    )
    payload = {
        "tenant": str(tenant_id),
        "bundle": [str(loaded.bundle.id), loaded.bundle.version],
        "snapshot": [str(loaded.snapshot.id), loaded.snapshot.version],
        "scope": scope.model_dump(mode="json", exclude={"evaluated_at"}),
        "windows": windows,
        "graph": graph_fingerprint(loaded),
        "search": {
            "mode": plan.mode,
            "embedding_model_id": plan.embedding_model_id,
            "statements": [ref.model_dump(mode="json") for ref in plan.refs],
            "truncated": plan.truncated,
        },
        "config": config.model_dump(mode="json"),
    }
    return _sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")))


def graph_fingerprint(loaded: LoadedContext) -> str:
    # Every approved node, rule and edge that selection and the walk read, in a fixed order.
    names = {node_id: node.key for node_id, node in loaded.nodes.items()}
    nodes = sorted(
        [
            node.key,
            node.node_type,
            node.language,
            _sha256(node.text),
            node.clause_ref or "",
            repr(node.conditions),
        ]
        for node in loaded.nodes.values()
    )
    rules = sorted(
        [
            names[r.node_id],
            r.rule_id,
            str(r.rule_version),
            str(r.mandatory),
            r.check_type,
            r.gate,
            r.severity,
        ]
        for r in loaded.rules
    )
    edges = sorted(
        [kind, names.get(source, str(source)), names.get(target, str(target))]
        for kind, source, target in loaded.edges
    )
    return _sha256(json.dumps([nodes, rules, edges], separators=(",", ":")))


def _sha256(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def assemble(
    loaded: LoadedContext,
    scope: ContextScope,
    config: RetrievalConfig,
    search: SearchResult | None = None,
) -> ContextManifest:
    """Build the manifest from loaded data and search results.

    Pure and deterministic: same input, same output. Search results only add evidence.
    """
    if search is None:
        search = SearchResult(SearchPlan(RetrievalMode.GRAPH_ONLY, (), (), False, None), {})
    nodes = loaded.nodes
    verdicts = {rule.node_id: _verdict(nodes[rule.node_id], scope, loaded) for rule in loaded.rules}
    applicable = {
        rule.node_id: rule
        for rule in loaded.rules
        if verdicts[rule.node_id].applies is not Applicability.FALSE
    }
    superseded, conflicts = _resolve_supersedes(applicable, loaded.edges)
    selected = sorted(
        (rule for node_id, rule in applicable.items() if node_id not in superseded),
        key=lambda rule: rule.rule_id,
    )
    missing = [*conflicts]
    for rule in selected:
        unresolved = verdicts[rule.node_id].unresolved
        if unresolved:
            missing.append(
                MissingContext(
                    code=MissingContextCode.APPLICABILITY_UNRESOLVED,
                    rule_id=rule.rule_id,
                    detail=", ".join(unresolved),
                )
            )
    missing.extend(_missing_sources(selected, nodes, loaded.edges))
    paths, exceptions, truncated = _walk_rules(selected, loaded, scope, config)
    evidence, dropped = _evidence(paths, search.matches, len(exceptions), nodes, config)
    truncated = truncated or dropped
    plan = search.plan
    if plan.mode is RetrievalMode.LEXICAL_ONLY:
        missing.append(MissingContext(code=MissingContextCode.SEMANTIC_SEARCH_SKIPPED))
    if plan.truncated:
        missing.append(
            MissingContext(
                code=MissingContextCode.STATEMENTS_TRUNCATED,
                detail=f"searched the first {config.max_statements}",
            )
        )
    if truncated:
        missing.append(
            MissingContext(
                code=MissingContextCode.TRAVERSAL_TRUNCATED,
                detail=f"max_hops={config.max_hops}, max_nodes={config.max_nodes}",
            )
        )
    rules = [_manifest_rule(rule, nodes, verdicts[rule.node_id]) for rule in selected]
    return ContextManifest(
        policy_bundle=loaded.bundle,
        study_snapshot=loaded.snapshot,
        scope=scope,
        retrieval_mode=plan.mode,
        embedding_model_id=plan.embedding_model_id,
        statements=plan.refs,
        required_rules=tuple(rule for rule in rules if rule.mandatory),
        optional_rules=tuple(rule for rule in rules if not rule.mandatory),
        evidence=evidence,
        exceptions=tuple(sorted(exceptions, key=lambda item: (item.for_rule_id, item.node_key))),
        superseded=tuple(
            sorted(
                (
                    SupersededRule(
                        rule_id=applicable[old].rule_id, superseded_by=applicable[new].rule_id
                    )
                    for old, new in superseded.items()
                ),
                key=lambda item: item.rule_id,
            )
        ),
        missing_context=tuple(sorted(missing, key=lambda m: (m.code, m.rule_id or ""))),
        retrieval_config=config,
    )


def _resolve_supersedes(
    applicable: Mapping[UUID, GraphRule], edges: Iterable[Edge]
) -> tuple[dict[UUID, UUID], list[MissingContext]]:
    # Returns {superseded rule node: superseding rule node} and the conflicts. Only rules that
    # both apply are compared; a superseding rule that doesn't apply leaves the old one in force.
    pairs = {
        (source, target)
        for edge_type, source, target in edges
        if edge_type is PolicyEdgeType.SUPERSEDES and source in applicable and target in applicable
    }
    superseded: dict[UUID, UUID] = {}
    conflicts: list[MissingContext] = []
    for new, old in sorted(
        pairs, key=lambda pair: (applicable[pair[1]].rule_id, applicable[pair[0]].rule_id)
    ):
        newer, older = applicable[new], applicable[old]
        if (old, new) in pairs:
            reason = f"{newer.rule_id} and {older.rule_id} supersede each other"
        elif older.mandatory and not newer.mandatory:
            # Never trade a mandatory rule for an optional one.
            reason = f"optional {newer.rule_id} supersedes mandatory {older.rule_id}"
        elif old in superseded:
            reason = f"{older.rule_id} is superseded by more than one rule"
        else:
            superseded[old] = new
            continue
        superseded.pop(old, None)
        conflicts.append(
            MissingContext(code=MissingContextCode.CONFLICT, rule_id=older.rule_id, detail=reason)
        )
    return superseded, conflicts


def _missing_sources(
    selected: Sequence[GraphRule], nodes: Mapping[UUID, GraphNode], edges: Iterable[Edge]
) -> list[MissingContext]:
    with_provenance = {
        source
        for edge_type, source, target in edges
        if edge_type in _PROVENANCE_EDGES and target in nodes
    }
    return [
        MissingContext(code=MissingContextCode.SOURCE_MISSING, rule_id=rule.rule_id)
        for rule in selected
        if nodes[rule.node_id].clause_ref is None and rule.node_id not in with_provenance
    ]


def _walk_rules(
    selected: Sequence[GraphRule],
    loaded: LoadedContext,
    scope: ContextScope,
    config: RetrievalConfig,
) -> tuple[dict[UUID, list[tuple[str, ...]]], list[ExceptionItem], bool]:
    # One walk per rule, so evidence or an exception shared by two rules is tied to each of them.
    # max_nodes bounds the manifest: distinct evidence nodes plus exception entries. Returns the
    # graph paths per evidence node (in walk order), the exceptions and whether a bound was hit.
    nodes = loaded.nodes
    names = {node_id: node.key for node_id, node in nodes.items()}
    graph = adjacency(loaded.edges, names)
    paths: dict[UUID, list[tuple[str, ...]]] = {}
    exceptions: list[ExceptionItem] = []
    truncated = False
    for rule in selected:
        result = walk(rule.node_id, graph, names, config.max_hops)
        truncated = truncated or result.truncated
        for step in result.reached:
            if step.node_id in paths:
                # Already evidence for an earlier rule: record this rule's path too.
                paths[step.node_id].append(step.path)
                continue
            if len(paths) + len(exceptions) >= config.max_nodes:
                truncated = True
                continue
            node = nodes[step.node_id]
            if node.node_type is not PolicyNodeType.EXCEPTION:
                paths[step.node_id] = [step.path]
                continue
            verdict = _verdict(node, scope, loaded)
            exceptions.append(
                ExceptionItem(
                    node_key=node.key,
                    for_rule_id=rule.rule_id,
                    applicable=verdict.applies,
                    reasons=verdict.reasons or verdict.unresolved,
                    excerpt=node.text[:EXCERPT_MAX_CHARS],
                    path=step.path,
                )
            )
    return paths, exceptions, truncated


def _evidence(
    paths: Mapping[UUID, Sequence[tuple[str, ...]]],
    matches: Mapping[UUID, Sequence[SearchMatch]],
    exception_count: int,
    nodes: Mapping[UUID, GraphNode],
    config: RetrievalConfig,
) -> tuple[tuple[EvidenceItem, ...], bool]:
    # Graph evidence first (walk order), then nodes only search found, best match first. Search
    # hits share the max_nodes bound; returns whether any had to be left out.
    search_only = sorted(
        (node_id for node_id in matches if node_id not in paths),
        key=lambda node_id: (
            min(m.rank for m in matches[node_id]),
            min(m.statement_index for m in matches[node_id]),
            nodes[node_id].key,
        ),
    )
    room = max(config.max_nodes - len(paths) - exception_count, 0)
    ordered = [*paths, *search_only[:room]]
    items = tuple(
        EvidenceItem(
            node_key=nodes[node_id].key,
            node_type=nodes[node_id].node_type,
            language=nodes[node_id].language,
            excerpt=nodes[node_id].text[:EXCERPT_MAX_CHARS],
            clause_ref=nodes[node_id].clause_ref,
            paths=tuple(paths.get(node_id, ())),
            matches=tuple(matches.get(node_id, ())),
        )
        for node_id in ordered
    )
    return items, len(search_only) > room


def _verdict(node: GraphNode, scope: ContextScope, loaded: LoadedContext) -> Verdict:
    return evaluate(node.conditions, scope, loaded.study_id, loaded.facts)


def _manifest_rule(
    rule: GraphRule, nodes: Mapping[UUID, GraphNode], verdict: Verdict
) -> ManifestRule:
    return ManifestRule(
        rule_id=rule.rule_id,
        rule_version=rule.rule_version,
        node_key=nodes[rule.node_id].key,
        mandatory=rule.mandatory,
        check_type=rule.check_type,
        gate=rule.gate,
        severity=rule.severity,
        unresolved=verdict.unresolved,
    )


async def _require_index(
    db: AsyncSession, tenant_id: UUID, loaded: LoadedContext, model_id: str
) -> None:
    # Vector search is only trusted on a READY index that covers exactly the approved graph.
    row = (
        await db.execute(
            select(PolicyIndexBuild.node_count, PolicyIndexBuild.content_checksum).where(
                PolicyIndexBuild.tenant_id == tenant_id,
                PolicyIndexBuild.policy_bundle_id == loaded.bundle.id,
                PolicyIndexBuild.model_id == model_id,
                PolicyIndexBuild.state == IndexBuildState.READY,
            )
        )
    ).one_or_none()
    if row is None:
        raise _unavailable("INDEX_NOT_READY", embedding_model_id=model_id)
    # The index covers the approved searchable nodes, the same set the build embedded.
    indexed = [node for node in loaded.nodes.values() if node.node_type in SEARCHABLE_NODE_TYPES]
    checksum = index_content_checksum((node.id, node.text) for node in indexed)
    if row.node_count != len(indexed) or row.content_checksum != checksum:
        raise _unavailable("INDEX_STALE", embedding_model_id=model_id)


def _unavailable(reason: str, **details: object) -> PolicyContextUnavailableError:
    return PolicyContextUnavailableError(details={"reason": reason, **details})


async def _load_nodes(db: AsyncSession, tenant_id: UUID, bundle_id: UUID) -> dict[UUID, GraphNode]:
    # Approved nodes only: unreviewed proposals never take part in retrieval.
    rows = await db.execute(
        select(
            PolicyNode.id,
            PolicyNode.node_key,
            PolicyNode.node_type,
            PolicyNode.language,
            PolicyNode.text,
            PolicySourceClause.clause_ref,
            PolicyNode.input_kinds,
            PolicyNode.channels,
            PolicyNode.languages,
            PolicyNode.audiences,
            PolicyNode.jurisdictions,
            PolicyNode.study_ids,
            PolicyNode.effective_from,
            PolicyNode.effective_to,
            PolicyNode.required_study_fact,
        )
        .outerjoin(
            PolicySourceClause,
            and_(
                PolicySourceClause.tenant_id == PolicyNode.tenant_id,
                PolicySourceClause.id == PolicyNode.source_clause_id,
            ),
        )
        .where(
            PolicyNode.tenant_id == tenant_id,
            PolicyNode.policy_bundle_id == bundle_id,
            PolicyNode.approved.is_(True),
        )
    )
    return {
        row.id: GraphNode(
            id=row.id,
            key=row.node_key,
            node_type=row.node_type,
            language=row.language,
            text=row.text,
            clause_ref=row.clause_ref,
            conditions=Conditions(
                input_kinds=tuple(row.input_kinds),
                channels=tuple(row.channels),
                languages=tuple(row.languages),
                audiences=tuple(row.audiences),
                jurisdictions=tuple(row.jurisdictions),
                study_ids=tuple(row.study_ids),
                effective_from=row.effective_from,
                effective_to=row.effective_to,
                required_study_fact=row.required_study_fact,
            ),
        )
        for row in rows
    }


async def _load_rules(db: AsyncSession, tenant_id: UUID, bundle_id: UUID) -> list[GraphRule]:
    rows = await db.execute(
        select(
            PolicyRuleRecord.node_id,
            PolicyRuleRecord.rule_id,
            PolicyRuleRecord.rule_version,
            PolicyRuleRecord.mandatory,
            PolicyRuleRecord.check_type,
            PolicyRuleRecord.gate,
            PolicyRuleRecord.severity,
        ).where(
            PolicyRuleRecord.tenant_id == tenant_id, PolicyRuleRecord.policy_bundle_id == bundle_id
        )
    )
    return [GraphRule(*row) for row in rows]


async def _load_edges(db: AsyncSession, tenant_id: UUID, bundle_id: UUID) -> list[Edge]:
    # Approved edges only; edges to unapproved nodes are dropped when the walk is built.
    rows = await db.execute(
        select(PolicyEdge.edge_type, PolicyEdge.from_node_id, PolicyEdge.to_node_id).where(
            PolicyEdge.tenant_id == tenant_id,
            PolicyEdge.policy_bundle_id == bundle_id,
            PolicyEdge.approved.is_(True),
        )
    )
    return [(row.edge_type, row.from_node_id, row.to_node_id) for row in rows]


async def _find(db: AsyncSession, tenant_id: UUID, key: str) -> StoredManifest | None:
    row = (
        await db.execute(
            select(ContextManifestRecord.id, ContextManifestRecord.manifest).where(
                ContextManifestRecord.tenant_id == tenant_id,
                ContextManifestRecord.cache_key == key,
            )
        )
    ).one_or_none()
    if row is None:
        return None
    return StoredManifest(row.id, ContextManifest.model_validate(row.manifest), reused=True)


async def _store(
    db: AsyncSession,
    tenant_id: UUID,
    key: str,
    loaded: LoadedContext,
    manifest: ContextManifest,
    actor: str,
) -> StoredManifest:
    # Two identical requests at once: one insert wins, the other reads the winner's row.
    manifest_id = await db.scalar(
        insert(ContextManifestRecord)
        .values(
            id=uuid4(),
            tenant_id=tenant_id,
            cache_key=key,
            policy_bundle_id=loaded.bundle.id,
            policy_bundle_version=loaded.bundle.version,
            study_snapshot_id=loaded.snapshot.id,
            study_snapshot_version=loaded.snapshot.version,
            retrieval_config_version=manifest.retrieval_config.version,
            manifest=manifest.model_dump(mode="json"),
            created_by=actor,
        )
        .on_conflict_do_nothing(index_elements=["tenant_id", "cache_key"])
        .returning(ContextManifestRecord.id)
    )
    if manifest_id is not None:
        return StoredManifest(manifest_id, manifest, reused=False)
    existing = await _find(db, tenant_id, key)
    if existing is None:
        # The conflicting row must exist and be visible to this tenant; anything else is a bug.
        raise _unavailable("MANIFEST_STORE_FAILED")
    return existing
