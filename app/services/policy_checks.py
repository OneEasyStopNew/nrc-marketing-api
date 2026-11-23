"""Checks that decide whether a policy bundle may be approved or activated (NRC-4).

Each check is a pure function over a snapshot of one bundle's graph and returns issues; any issue
blocks approval and activation (AC2). Rows marked REJECTED are inert and ignored. To add a check,
write a function and list it in CHECKS.
"""

import re
from collections import Counter, defaultdict
from collections.abc import Callable, Collection, Iterable, Iterator, Mapping
from dataclasses import dataclass
from enum import StrEnum
from typing import Final
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models import (
    PolicyEdge,
    PolicyNode,
    PolicyRuleRecord,
    PolicyRuleWording,
    PolicySourceClause,
)
from app.models.enums import (
    PolicyEdgeType,
    PolicyLanguage,
    PolicyNodeType,
    ReviewState,
    WordingKind,
)


class CheckCode(StrEnum):
    EMPTY_BUNDLE = "EMPTY_BUNDLE"
    UNREVIEWED_ITEM = "UNREVIEWED_ITEM"
    MISSING_PROVENANCE = "MISSING_PROVENANCE"
    EDGE_TO_UNAPPROVED_NODE = "EDGE_TO_UNAPPROVED_NODE"
    SUPERSEDES_CYCLE = "SUPERSEDES_CYCLE"
    SUPERSEDES_CONFLICT = "SUPERSEDES_CONFLICT"
    CONTRADICTORY_WORDING = "CONTRADICTORY_WORDING"
    UNRESOLVED_EXCEPTION = "UNRESOLVED_EXCEPTION"
    UNSUPPORTED_LANGUAGE = "UNSUPPORTED_LANGUAGE"
    DUPLICATE_OBLIGATION = "DUPLICATE_OBLIGATION"
    EXACT_WORDING_NOT_IN_SOURCE = "EXACT_WORDING_NOT_IN_SOURCE"


@dataclass(frozen=True, slots=True)
class Issue:
    code: CheckCode
    # What the issue is about, e.g. "rule:REC-01", "node:CL-FDA-1", "edge:supported_by:A->B".
    subject: str
    message: str


@dataclass(frozen=True, slots=True)
class NodeView:
    id: UUID
    key: str
    type: PolicyNodeType
    review_state: ReviewState
    language: PolicyLanguage
    source_clause_id: UUID | None


@dataclass(frozen=True, slots=True)
class EdgeView:
    type: PolicyEdgeType
    from_id: UUID
    to_id: UUID
    review_state: ReviewState


@dataclass(frozen=True, slots=True)
class RuleView:
    id: UUID
    node_id: UUID
    rule_id: str
    requirement: str
    mandatory: bool


@dataclass(frozen=True, slots=True)
class WordingView:
    id: UUID
    rule_record_id: UUID
    language: PolicyLanguage
    kind: WordingKind
    text: str
    review_state: ReviewState
    source_clause_id: UUID | None


@dataclass(frozen=True, slots=True)
class BundleSnapshot:
    nodes: tuple[NodeView, ...]
    edges: tuple[EdgeView, ...]
    rules: tuple[RuleView, ...]
    wordings: tuple[WordingView, ...]
    # Text of every source clause the bundle's rows point at, by clause id.
    clause_texts: Mapping[UUID, str]


Check = Callable[[BundleSnapshot, Collection[PolicyLanguage]], Iterable[Issue]]

_APPROVED: Final = ReviewState.APPROVED


def _normalised(text: str) -> str:
    # Wording comparisons ignore line breaks and runs of spaces, never the words themselves.
    return " ".join(text.split())


class _Graph:
    # Lookups shared by the checks, over approved rows only.

    def __init__(self, snapshot: BundleSnapshot) -> None:
        self.nodes = {node.id: node for node in snapshot.nodes}
        self.approved_nodes = {
            node.id: node for node in snapshot.nodes if node.review_state is _APPROVED
        }
        self.approved_edges = [edge for edge in snapshot.edges if edge.review_state is _APPROVED]
        # Rules whose RULE node is approved: the obligations that would become active.
        self.rules = [rule for rule in snapshot.rules if rule.node_id in self.approved_nodes]
        self.rules_by_node = {rule.node_id: rule for rule in self.rules}
        self.rule_ids_by_record = {rule.id: rule.rule_id for rule in snapshot.rules}
        self.approved_wordings = [w for w in snapshot.wordings if w.review_state is _APPROVED]

    def key(self, node_id: UUID) -> str:
        node = self.nodes.get(node_id)
        return node.key if node else str(node_id)

    def edge_subject(self, edge: EdgeView) -> str:
        return f"edge:{edge.type.value}:{self.key(edge.from_id)}->{self.key(edge.to_id)}"

    def targets(self, node_id: UUID, edge_type: PolicyEdgeType) -> Iterator[NodeView]:
        for edge in self.approved_edges:
            if edge.type is edge_type and edge.from_id == node_id:
                target = self.approved_nodes.get(edge.to_id)
                if target is not None:
                    yield target


def check_empty(snapshot: BundleSnapshot, languages: Collection[PolicyLanguage]) -> list[Issue]:
    if _Graph(snapshot).rules:
        return []
    return [Issue(CheckCode.EMPTY_BUNDLE, "bundle", "The bundle has no approved rule.")]


def check_unreviewed(
    snapshot: BundleSnapshot, languages: Collection[PolicyLanguage]
) -> list[Issue]:
    graph = _Graph(snapshot)
    issues = [
        Issue(CheckCode.UNREVIEWED_ITEM, f"node:{node.key}", "Node has not been reviewed.")
        for node in snapshot.nodes
        if node.review_state is ReviewState.PROPOSED
    ]
    issues += [
        Issue(CheckCode.UNREVIEWED_ITEM, graph.edge_subject(edge), "Edge has not been reviewed.")
        for edge in snapshot.edges
        if edge.review_state is ReviewState.PROPOSED
    ]
    issues += [
        Issue(
            CheckCode.UNREVIEWED_ITEM,
            f"wording:{graph.rule_ids_by_record.get(w.rule_record_id, '?')}:{w.id}",
            "Wording has not been reviewed.",
        )
        for w in snapshot.wordings
        if w.review_state is ReviewState.PROPOSED
    ]
    return issues


def check_provenance(
    snapshot: BundleSnapshot, languages: Collection[PolicyLanguage]
) -> list[Issue]:
    # AC1: an active rule rests on an approved source clause or an approved interpretation.
    graph = _Graph(snapshot)
    issues = [
        Issue(
            CheckCode.MISSING_PROVENANCE,
            f"node:{node.key}",
            "Clause node does not point at a source clause.",
        )
        for node in graph.approved_nodes.values()
        if node.type is PolicyNodeType.CLAUSE and node.source_clause_id is None
    ]
    for rule in graph.rules:
        support = list(graph.targets(rule.node_id, PolicyEdgeType.SUPPORTED_BY))
        if not any(
            (target.type is PolicyNodeType.CLAUSE and target.source_clause_id is not None)
            or target.type is PolicyNodeType.INTERPRETATION
            for target in support
        ):
            issues.append(
                Issue(
                    CheckCode.MISSING_PROVENANCE,
                    f"rule:{rule.rule_id}",
                    "Rule is not supported by an approved source clause or interpretation.",
                )
            )
    return issues


def check_edge_endpoints(
    snapshot: BundleSnapshot, languages: Collection[PolicyLanguage]
) -> list[Issue]:
    graph = _Graph(snapshot)
    return [
        Issue(
            CheckCode.EDGE_TO_UNAPPROVED_NODE,
            graph.edge_subject(edge),
            "Approved edge joins a node that is not approved.",
        )
        for edge in graph.approved_edges
        if edge.from_id not in graph.approved_nodes or edge.to_id not in graph.approved_nodes
    ]


def check_supersedes(
    snapshot: BundleSnapshot, languages: Collection[PolicyLanguage]
) -> list[Issue]:
    # The same conflicts retrieval would report at request time (policy_context.provider), caught
    # before activation instead: a cycle, an optional rule replacing a mandatory one, or a rule
    # replaced by more than one rule.
    graph = _Graph(snapshot)
    replaces: defaultdict[UUID, set[UUID]] = defaultdict(set)
    replaced_by: Counter[UUID] = Counter()
    issues: list[Issue] = []
    for edge in graph.approved_edges:
        if edge.type is not PolicyEdgeType.SUPERSEDES:
            continue
        newer, older = graph.rules_by_node.get(edge.from_id), graph.rules_by_node.get(edge.to_id)
        if newer is None or older is None:
            issues.append(
                Issue(
                    CheckCode.SUPERSEDES_CONFLICT,
                    graph.edge_subject(edge),
                    "Supersedes must join two approved rules.",
                )
            )
            continue
        replaces[edge.from_id].add(edge.to_id)
        replaced_by[edge.to_id] += 1
        if older.mandatory and not newer.mandatory:
            issues.append(
                Issue(
                    CheckCode.SUPERSEDES_CONFLICT,
                    graph.edge_subject(edge),
                    f"Optional {newer.rule_id} supersedes mandatory {older.rule_id}.",
                )
            )
    issues += [
        Issue(
            CheckCode.SUPERSEDES_CONFLICT,
            f"rule:{graph.rules_by_node[node_id].rule_id}",
            "Rule is superseded by more than one rule.",
        )
        for node_id, count in replaced_by.items()
        if count > 1
    ]
    issues += [
        Issue(
            CheckCode.SUPERSEDES_CYCLE,
            f"rule:{graph.rules_by_node[node_id].rule_id}",
            "Rule supersedes itself through a chain of supersedes edges.",
        )
        for node_id in sorted(_on_cycles(replaces), key=lambda n: graph.rules_by_node[n].rule_id)
    ]
    return issues


def _on_cycles(edges: Mapping[UUID, Collection[UUID]]) -> set[UUID]:
    # Nodes that can reach themselves. Bundles hold tens of rules, so a walk per node is enough.
    def reaches(start: UUID) -> bool:
        seen: set[UUID] = set()
        stack = list(edges.get(start, ()))
        while stack:
            node = stack.pop()
            if node == start:
                return True
            if node not in seen:
                seen.add(node)
                stack.extend(edges.get(node, ()))
        return False

    return {node for node in edges if reaches(node)}


def check_contradictions(
    snapshot: BundleSnapshot, languages: Collection[PolicyLanguage]
) -> list[Issue]:
    # One rule requiring wording that another rule prohibits, in the same language.
    graph = _Graph(snapshot)
    live = {rule.id for rule in graph.rules}
    wordings = [w for w in graph.approved_wordings if w.rule_record_id in live]
    return [
        Issue(
            CheckCode.CONTRADICTORY_WORDING,
            f"rule:{graph.rule_ids_by_record[exact.rule_record_id]}",
            "Required wording contains a phrase prohibited by "
            f"{graph.rule_ids_by_record[phrase.rule_record_id]}.",
        )
        for exact in wordings
        if exact.kind is WordingKind.EXACT_WORDING
        for phrase in wordings
        if phrase.kind is WordingKind.PROHIBITED_PHRASE
        and phrase.language is exact.language
        and _contains_phrase(exact.text, phrase.text)
    ]


def _contains_phrase(text: str, phrase: str) -> bool:
    # Whole words only, so a ban on "cure" doesn't match "secure".
    pattern = rf"(?<!\w){re.escape(_normalised(phrase))}(?!\w)"
    return re.search(pattern, _normalised(text), re.IGNORECASE) is not None


def check_exceptions(
    snapshot: BundleSnapshot, languages: Collection[PolicyLanguage]
) -> list[Issue]:
    # An exception only means something when a rule leads to it through except_when, directly or
    # through another exception (retrieval walks except_when from any node it reaches).
    graph = _Graph(snapshot)
    issues = []
    children: defaultdict[UUID, list[UUID]] = defaultdict(list)
    for edge in graph.approved_edges:
        if edge.type is not PolicyEdgeType.EXCEPT_WHEN:
            continue
        source = graph.approved_nodes.get(edge.from_id)
        target = graph.approved_nodes.get(edge.to_id)
        if (
            source is None
            or source.type not in (PolicyNodeType.RULE, PolicyNodeType.EXCEPTION)
            or target is None
            or target.type is not PolicyNodeType.EXCEPTION
        ):
            issues.append(
                Issue(
                    CheckCode.UNRESOLVED_EXCEPTION,
                    graph.edge_subject(edge),
                    "except_when must lead from an approved rule or exception to an approved "
                    "exception.",
                )
            )
        else:
            children[source.id].append(target.id)
    attached: set[UUID] = set()
    stack = [child for rule_node in graph.rules_by_node for child in children[rule_node]]
    while stack:
        node_id = stack.pop()
        if node_id not in attached:
            attached.add(node_id)
            stack.extend(children[node_id])
    issues += [
        Issue(
            CheckCode.UNRESOLVED_EXCEPTION,
            f"node:{node.key}",
            "Exception is not attached to any rule.",
        )
        for node in graph.approved_nodes.values()
        if node.type is PolicyNodeType.EXCEPTION and node.id not in attached
    ]
    return issues


def check_languages(snapshot: BundleSnapshot, languages: Collection[PolicyLanguage]) -> list[Issue]:
    graph = _Graph(snapshot)
    allowed = ", ".join(sorted(language.value for language in languages))
    issues = [
        Issue(
            CheckCode.UNSUPPORTED_LANGUAGE,
            f"node:{node.key}",
            f"Language {node.language.value} is not supported ({allowed}).",
        )
        for node in graph.approved_nodes.values()
        if node.language not in languages
    ]
    issues += [
        Issue(
            CheckCode.UNSUPPORTED_LANGUAGE,
            f"wording:{graph.rule_ids_by_record.get(w.rule_record_id, '?')}:{w.id}",
            f"Language {w.language.value} is not supported ({allowed}).",
        )
        for w in graph.approved_wordings
        if w.language not in languages
    ]
    return issues


def check_duplicates(
    snapshot: BundleSnapshot, languages: Collection[PolicyLanguage]
) -> list[Issue]:
    # AC3: the same obligation imported twice must not become two active rules.
    graph = _Graph(snapshot)
    by_requirement: defaultdict[str, list[str]] = defaultdict(list)
    for rule in graph.rules:
        by_requirement[_normalised(rule.requirement).casefold()].append(rule.rule_id)
    return [
        Issue(
            CheckCode.DUPLICATE_OBLIGATION,
            f"rule:{rule_id}",
            f"Same requirement as {', '.join(sorted(set(rule_ids) - {rule_id}))}.",
        )
        for rule_ids in by_requirement.values()
        if len(rule_ids) > 1
        for rule_id in sorted(rule_ids)
    ]


def check_exact_wording(
    snapshot: BundleSnapshot, languages: Collection[PolicyLanguage]
) -> list[Issue]:
    # AC7: required wording is the source's own words, never a paraphrase. It must appear in the
    # clause it cites or, without one, in a source clause that supports its rule.
    graph = _Graph(snapshot)
    rules = {rule.id: rule for rule in graph.rules}
    issues = []
    for wording in graph.approved_wordings:
        rule = rules.get(wording.rule_record_id)
        if wording.kind is not WordingKind.EXACT_WORDING or rule is None:
            continue
        if wording.source_clause_id is not None:
            candidates = [wording.source_clause_id]
        else:
            candidates = [
                target.source_clause_id
                for target in graph.targets(rule.node_id, PolicyEdgeType.SUPPORTED_BY)
                if target.source_clause_id is not None and target.language is wording.language
            ]
        needle = _normalised(wording.text)
        if not any(
            needle in _normalised(snapshot.clause_texts.get(clause_id, ""))
            for clause_id in candidates
        ):
            issues.append(
                Issue(
                    CheckCode.EXACT_WORDING_NOT_IN_SOURCE,
                    f"rule:{rule.rule_id}",
                    f"Exact wording ({wording.language.value}) does not appear verbatim in its "
                    "source clause.",
                )
            )
    return issues


CHECKS: Final[tuple[Check, ...]] = (
    check_empty,
    check_unreviewed,
    check_provenance,
    check_edge_endpoints,
    check_supersedes,
    check_contradictions,
    check_exceptions,
    check_languages,
    check_duplicates,
    check_exact_wording,
)


def run_checks(
    snapshot: BundleSnapshot, supported_languages: Collection[PolicyLanguage]
) -> list[Issue]:
    return [issue for check in CHECKS for issue in check(snapshot, supported_languages)]


async def load_snapshot(db: AsyncSession, tenant_id: UUID, bundle_id: UUID) -> BundleSnapshot:
    # Reads in the caller's transaction (tenant already bound). Callers that act on the result
    # lock the bundle row first, so the graph cannot change between this read and their write.
    nodes = (
        await db.scalars(
            select(PolicyNode).where(
                PolicyNode.tenant_id == tenant_id, PolicyNode.policy_bundle_id == bundle_id
            )
        )
    ).all()
    edges = (
        await db.scalars(
            select(PolicyEdge).where(
                PolicyEdge.tenant_id == tenant_id, PolicyEdge.policy_bundle_id == bundle_id
            )
        )
    ).all()
    rules = (
        await db.scalars(
            select(PolicyRuleRecord).where(
                PolicyRuleRecord.tenant_id == tenant_id,
                PolicyRuleRecord.policy_bundle_id == bundle_id,
            )
        )
    ).all()
    wordings = (
        await db.scalars(
            select(PolicyRuleWording).where(
                PolicyRuleWording.tenant_id == tenant_id,
                PolicyRuleWording.policy_bundle_id == bundle_id,
            )
        )
    ).all()
    clause_ids = {
        clause_id
        for clause_id in [
            *(node.source_clause_id for node in nodes),
            *(wording.source_clause_id for wording in wordings),
        ]
        if clause_id is not None
    }
    clauses = (
        (
            await db.execute(
                select(PolicySourceClause.id, PolicySourceClause.text).where(
                    PolicySourceClause.tenant_id == tenant_id,
                    PolicySourceClause.id.in_(clause_ids),
                )
            )
        ).all()
        if clause_ids
        else []
    )
    return BundleSnapshot(
        nodes=tuple(
            NodeView(
                node.id,
                node.node_key,
                node.node_type,
                node.review_state,
                node.language,
                node.source_clause_id,
            )
            for node in nodes
        ),
        edges=tuple(
            EdgeView(edge.edge_type, edge.from_node_id, edge.to_node_id, edge.review_state)
            for edge in edges
        ),
        rules=tuple(
            RuleView(rule.id, rule.node_id, rule.rule_id, rule.requirement, rule.mandatory)
            for rule in rules
        ),
        wordings=tuple(
            WordingView(
                w.id,
                w.policy_rule_id,
                w.language,
                w.kind,
                w.text,
                w.review_state,
                w.source_clause_id,
            )
            for w in wordings
        ),
        clause_texts={clause_id: text for clause_id, text in clauses},
    )
