"""Checks that decide whether a policy bundle may be approved or activated (NRC-4)."""

from dataclasses import replace
from uuid import UUID, uuid4

import pytest
from sqlalchemy.ext.asyncio import AsyncConnection

from app.models.enums import (
    CheckCode,
    PolicyEdgeType,
    PolicyLanguage,
    PolicyNodeType,
    ReviewState,
    StudyWordingSource,
    WordingKind,
)
from app.services.policy_checks import (
    CHECKS,
    BundleSnapshot,
    EdgeView,
    NodeView,
    RuleView,
    WordingView,
    load_snapshot,
    run_checks,
)
from scripts.import_policy_graph import import_policy_graph, load_synthetic_graph
from tests.conftest import World, owner_session

EN = (PolicyLanguage.EN,)
APPROVED = ReviewState.APPROVED
CLAUSE_TEXT = "[SYNTHETIC] Say:  This is a research study.\nTaking part is voluntary."


class Graph:
    # A clean one-rule bundle (rule -> clause node -> source clause, one exact wording) that each
    # test then breaks in exactly one way.

    def __init__(self) -> None:
        self.clause_id = uuid4()
        self.nodes: list[NodeView] = []
        self.edges: list[EdgeView] = []
        self.rules: list[RuleView] = []
        self.wordings: list[WordingView] = []
        self.clause_texts: dict[UUID, str] = {self.clause_id: CLAUSE_TEXT}
        self.rule_node = self.rule("REC-01", "[SYNTHETIC] Say it is a research study.")
        self.clause_node = self.node("CL-1", PolicyNodeType.CLAUSE, clause_id=self.clause_id)
        self.edge(PolicyEdgeType.SUPPORTED_BY, self.rule_node, self.clause_node)
        self.wording(self.rules[0], WordingKind.EXACT_WORDING, "This is a research study.")

    def node(
        self,
        key: str,
        node_type: PolicyNodeType,
        *,
        clause_id: UUID | None = None,
        state: ReviewState = APPROVED,
        language: PolicyLanguage = PolicyLanguage.EN,
    ) -> NodeView:
        node = NodeView(uuid4(), key, node_type, state, language, clause_id)
        self.nodes.append(node)
        return node

    def rule(self, rule_id: str, requirement: str, *, mandatory: bool = True) -> NodeView:
        node = self.node(rule_id, PolicyNodeType.RULE)
        self.rules.append(RuleView(uuid4(), node.id, rule_id, requirement, mandatory))
        return node

    def supported_rule(self, rule_id: str, requirement: str, *, mandatory: bool = True) -> NodeView:
        node = self.rule(rule_id, requirement, mandatory=mandatory)
        self.edge(PolicyEdgeType.SUPPORTED_BY, node, self.clause_node)
        return node

    def edge(
        self,
        edge_type: PolicyEdgeType,
        source: NodeView,
        target: NodeView,
        state: ReviewState = APPROVED,
    ) -> None:
        self.edges.append(EdgeView(edge_type, source.id, target.id, state))

    def wording(
        self,
        rule: RuleView,
        kind: WordingKind,
        text: str,
        *,
        state: ReviewState = APPROVED,
        language: PolicyLanguage = PolicyLanguage.EN,
        clause_id: UUID | None = None,
    ) -> WordingView:
        wording = WordingView(uuid4(), rule.id, language, kind, text, state, clause_id)
        self.wordings.append(wording)
        return wording

    def snapshot(self) -> BundleSnapshot:
        return BundleSnapshot(
            tuple(self.nodes),
            tuple(self.edges),
            tuple(self.rules),
            tuple(self.wordings),
            self.clause_texts,
        )

    def issues(self, languages: tuple[PolicyLanguage, ...] = EN) -> set[tuple[CheckCode, str]]:
        return {(i.code, i.subject) for i in run_checks(self.snapshot(), languages)}


def test_the_clean_graph_has_no_issues() -> None:
    assert Graph().issues() == set()


def test_every_check_code_has_a_check() -> None:
    # check_supersedes reports two codes; INDEX_STALE comes from activation, not from a check.
    assert len(CHECKS) == len(CheckCode) - 2


def test_a_bundle_without_rules_is_empty() -> None:
    snapshot = BundleSnapshot((), (), (), (), {})

    assert {i.code for i in run_checks(snapshot, EN)} == {CheckCode.EMPTY_BUNDLE}


def test_a_bundle_whose_only_rule_is_rejected_is_empty() -> None:
    graph = Graph()
    graph.nodes[0] = replace(graph.rule_node, review_state=ReviewState.REJECTED)

    assert (CheckCode.EMPTY_BUNDLE, "bundle") in graph.issues()


@pytest.mark.parametrize("item", ["node", "edge", "wording"])
def test_unreviewed_items_are_reported(item: str) -> None:
    graph = Graph()
    if item == "node":
        graph.node("INT-1", PolicyNodeType.INTERPRETATION, state=ReviewState.PROPOSED)
        subject = "node:INT-1"
    elif item == "edge":
        graph.edge(
            PolicyEdgeType.REQUIRES_DISCLOSURE,
            graph.rule_node,
            graph.node("D-1", PolicyNodeType.DISCLOSURE),
            ReviewState.PROPOSED,
        )
        subject = "edge:requires_disclosure:REC-01->D-1"
    else:
        wording = graph.wording(
            graph.rules[0], WordingKind.PROHIBITED_PHRASE, "cure", state=ReviewState.PROPOSED
        )
        subject = f"wording:REC-01:{wording.id}"

    assert graph.issues() == {(CheckCode.UNREVIEWED_ITEM, subject)}


def test_rejected_items_are_ignored() -> None:
    graph = Graph()
    graph.node("INT-1", PolicyNodeType.INTERPRETATION, state=ReviewState.REJECTED)
    graph.wording(
        graph.rules[0], WordingKind.EXACT_WORDING, "Not in the source.", state=ReviewState.REJECTED
    )

    assert graph.issues() == set()


def test_a_rule_without_support_has_no_provenance() -> None:
    graph = Graph()
    graph.rule("REC-02", "[SYNTHETIC] Do not promise payment.")

    assert graph.issues() == {(CheckCode.MISSING_PROVENANCE, "rule:REC-02")}


def test_a_rule_supported_only_by_an_unapproved_edge_has_no_provenance() -> None:
    graph = Graph()
    graph.edges[0] = replace(graph.edges[0], review_state=ReviewState.REJECTED)

    assert (CheckCode.MISSING_PROVENANCE, "rule:REC-01") in graph.issues()


def test_an_approved_interpretation_is_enough_provenance() -> None:
    graph = Graph()
    rule = graph.rule("REC-02", "[SYNTHETIC] Do not promise payment.")
    graph.edge(
        PolicyEdgeType.SUPPORTED_BY, rule, graph.node("INT-1", PolicyNodeType.INTERPRETATION)
    )

    assert graph.issues() == set()


def test_a_clause_node_without_a_source_clause_has_no_provenance() -> None:
    graph = Graph()
    rule = graph.rule("REC-02", "[SYNTHETIC] Do not promise payment.")
    graph.edge(PolicyEdgeType.SUPPORTED_BY, rule, graph.node("CL-2", PolicyNodeType.CLAUSE))

    assert graph.issues() == {
        (CheckCode.MISSING_PROVENANCE, "node:CL-2"),
        (CheckCode.MISSING_PROVENANCE, "rule:REC-02"),
    }


def test_an_approved_edge_to_a_rejected_node_is_reported() -> None:
    graph = Graph()
    disclosure = graph.node("D-1", PolicyNodeType.DISCLOSURE, state=ReviewState.REJECTED)
    graph.edge(PolicyEdgeType.REQUIRES_DISCLOSURE, graph.rule_node, disclosure)

    assert graph.issues() == {
        (CheckCode.EDGE_TO_UNAPPROVED_NODE, "edge:requires_disclosure:REC-01->D-1")
    }


def test_a_supersedes_cycle_is_reported() -> None:
    graph = Graph()
    two = graph.supported_rule("REC-02", "[SYNTHETIC] Two.")
    three = graph.supported_rule("REC-03", "[SYNTHETIC] Three.")
    graph.edge(PolicyEdgeType.SUPERSEDES, graph.rule_node, two)
    graph.edge(PolicyEdgeType.SUPERSEDES, two, three)
    graph.edge(PolicyEdgeType.SUPERSEDES, three, graph.rule_node)

    assert graph.issues() == {
        (CheckCode.SUPERSEDES_CYCLE, "rule:REC-01"),
        (CheckCode.SUPERSEDES_CYCLE, "rule:REC-02"),
        (CheckCode.SUPERSEDES_CYCLE, "rule:REC-03"),
    }


def test_a_plain_supersedes_chain_is_fine() -> None:
    graph = Graph()
    two = graph.supported_rule("REC-02", "[SYNTHETIC] Two.")
    graph.edge(PolicyEdgeType.SUPERSEDES, two, graph.rule_node)

    assert graph.issues() == set()


def test_an_optional_rule_cannot_supersede_a_mandatory_one() -> None:
    graph = Graph()
    optional = graph.supported_rule("REC-02", "[SYNTHETIC] Two.", mandatory=False)
    graph.edge(PolicyEdgeType.SUPERSEDES, optional, graph.rule_node)

    assert graph.issues() == {
        (CheckCode.SUPERSEDES_CONFLICT, "edge:supersedes:REC-02->REC-01"),
    }


def test_a_rule_superseded_twice_is_a_conflict() -> None:
    graph = Graph()
    for rule_id in ("REC-02", "REC-03"):
        newer = graph.supported_rule(rule_id, f"[SYNTHETIC] {rule_id}.")
        graph.edge(PolicyEdgeType.SUPERSEDES, newer, graph.rule_node)

    assert graph.issues() == {(CheckCode.SUPERSEDES_CONFLICT, "rule:REC-01")}


def test_supersedes_must_join_two_rules() -> None:
    graph = Graph()
    graph.edge(PolicyEdgeType.SUPERSEDES, graph.rule_node, graph.clause_node)

    assert graph.issues() == {(CheckCode.SUPERSEDES_CONFLICT, "edge:supersedes:REC-01->CL-1")}


def test_required_wording_with_a_prohibited_phrase_is_a_contradiction() -> None:
    graph = Graph()
    graph.supported_rule("REC-02", "[SYNTHETIC] No research-study talk.")
    graph.wording(graph.rules[1], WordingKind.PROHIBITED_PHRASE, "Research  Study")

    assert graph.issues() == {(CheckCode.CONTRADICTORY_WORDING, "rule:REC-01")}


def test_a_prohibited_phrase_inside_another_word_is_not_a_contradiction() -> None:
    graph = Graph()
    graph.supported_rule("REC-02", "[SYNTHETIC] No cure claims.")
    graph.wording(graph.rules[1], WordingKind.PROHIBITED_PHRASE, "search")

    assert graph.issues() == set()


def test_a_contradiction_needs_the_same_language() -> None:
    graph = Graph()
    graph.supported_rule("REC-02", "[SYNTHETIC] No research-study talk.")
    graph.wording(
        graph.rules[1], WordingKind.PROHIBITED_PHRASE, "research study", language=PolicyLanguage.ES
    )

    assert graph.issues(languages=(PolicyLanguage.EN, PolicyLanguage.ES)) == set()


def test_an_exception_not_attached_to_a_rule_is_unresolved() -> None:
    graph = Graph()
    graph.node("EX-1", PolicyNodeType.EXCEPTION)

    assert graph.issues() == {(CheckCode.UNRESOLVED_EXCEPTION, "node:EX-1")}


def test_an_attached_exception_is_fine() -> None:
    graph = Graph()
    graph.edge(
        PolicyEdgeType.EXCEPT_WHEN, graph.rule_node, graph.node("EX-1", PolicyNodeType.EXCEPTION)
    )

    assert graph.issues() == set()


def test_an_exception_of_an_attached_exception_is_fine() -> None:
    graph = Graph()
    first = graph.node("EX-1", PolicyNodeType.EXCEPTION)
    graph.edge(PolicyEdgeType.EXCEPT_WHEN, graph.rule_node, first)
    graph.edge(PolicyEdgeType.EXCEPT_WHEN, first, graph.node("EX-2", PolicyNodeType.EXCEPTION))

    assert graph.issues() == set()


def test_an_exception_chain_not_reaching_a_rule_is_unresolved() -> None:
    graph = Graph()
    first = graph.node("EX-1", PolicyNodeType.EXCEPTION)
    graph.edge(PolicyEdgeType.EXCEPT_WHEN, first, graph.node("EX-2", PolicyNodeType.EXCEPTION))

    assert graph.issues() == {
        (CheckCode.UNRESOLVED_EXCEPTION, "node:EX-1"),
        (CheckCode.UNRESOLVED_EXCEPTION, "node:EX-2"),
    }


def test_except_when_must_lead_to_an_exception() -> None:
    graph = Graph()
    graph.edge(
        PolicyEdgeType.EXCEPT_WHEN, graph.rule_node, graph.node("D-1", PolicyNodeType.DISCLOSURE)
    )

    assert graph.issues() == {(CheckCode.UNRESOLVED_EXCEPTION, "edge:except_when:REC-01->D-1")}


def test_unsupported_languages_are_reported() -> None:
    graph = Graph()
    spanish = graph.node(
        "CL-ES", PolicyNodeType.CLAUSE, clause_id=graph.clause_id, language=PolicyLanguage.ES
    )
    graph.edge(PolicyEdgeType.SUPPORTED_BY, graph.rule_node, spanish)
    wording = graph.wording(
        graph.rules[0], WordingKind.PROHIBITED_PHRASE, "cura", language=PolicyLanguage.ES
    )

    assert graph.issues() == {
        (CheckCode.UNSUPPORTED_LANGUAGE, "node:CL-ES"),
        (CheckCode.UNSUPPORTED_LANGUAGE, f"wording:REC-01:{wording.id}"),
    }
    assert graph.issues(languages=(PolicyLanguage.EN, PolicyLanguage.ES)) == set()


def test_the_same_requirement_twice_is_a_duplicate_obligation() -> None:
    graph = Graph()
    graph.supported_rule("REC-02", "[SYNTHETIC]  say it is a RESEARCH study.")

    assert graph.issues() == {
        (CheckCode.DUPLICATE_OBLIGATION, "rule:REC-01"),
        (CheckCode.DUPLICATE_OBLIGATION, "rule:REC-02"),
    }


def test_exact_wording_not_in_the_source_is_reported() -> None:
    graph = Graph()
    graph.wordings[0] = replace(graph.wordings[0], text="This is a clinical research study.")

    assert graph.issues() == {(CheckCode.EXACT_WORDING_NOT_IN_SOURCE, "rule:REC-01")}


def test_study_approved_wording_needs_no_source_clause() -> None:
    # v0.3: a study disclosure is checked against its own approval record, not a source clause.
    graph = Graph()
    graph.wordings[0] = replace(
        graph.wordings[0],
        kind=WordingKind.STUDY_APPROVED_WORDING,
        text=StudyWordingSource.REQUIRED_DISCLOSURES.value,
    )

    assert graph.issues() == set()


def test_a_regulatory_quotation_still_needs_its_source_clause() -> None:
    # Next to a study wording, the rule's quotation is still checked word for word.
    graph = Graph()
    graph.wording(
        graph.rules[0],
        WordingKind.STUDY_APPROVED_WORDING,
        StudyWordingSource.APPROVED_PHRASES.value,
    )
    graph.wordings[0] = replace(graph.wordings[0], text="Taking part is always paid.")

    assert graph.issues() == {(CheckCode.EXACT_WORDING_NOT_IN_SOURCE, "rule:REC-01")}


def test_exact_wording_keeps_its_words_but_not_its_line_breaks() -> None:
    graph = Graph()
    graph.wordings[0] = replace(graph.wordings[0], text="research study.\nTaking part")

    assert graph.issues() == set()


def test_exact_wording_is_case_sensitive() -> None:
    graph = Graph()
    graph.wordings[0] = replace(graph.wordings[0], text="this is a research study.")

    assert graph.issues() == {(CheckCode.EXACT_WORDING_NOT_IN_SOURCE, "rule:REC-01")}


def test_exact_wording_is_checked_against_the_clause_it_cites() -> None:
    graph = Graph()
    other = uuid4()
    graph.clause_texts[other] = "[SYNTHETIC] Something else entirely."
    graph.wordings[0] = replace(graph.wordings[0], source_clause_id=other)

    assert graph.issues() == {(CheckCode.EXACT_WORDING_NOT_IN_SOURCE, "rule:REC-01")}


def test_exact_wording_may_only_cite_a_clause_that_supports_its_rule() -> None:
    # The other clause holds the very words, but it is not the rule's source.
    graph = Graph()
    other = uuid4()
    graph.clause_texts[other] = CLAUSE_TEXT
    graph.wordings[0] = replace(graph.wordings[0], source_clause_id=other)

    assert graph.issues() == {(CheckCode.EXACT_WORDING_NOT_IN_SOURCE, "rule:REC-01")}


def test_exact_wording_citing_its_supporting_clause_is_clean() -> None:
    graph = Graph()
    graph.wordings[0] = replace(graph.wordings[0], source_clause_id=graph.clause_id)

    assert graph.issues() == set()


def test_proposals_hanging_off_a_rejected_rule_do_not_block() -> None:
    graph = Graph()
    rejected = graph.rule("REC-02", "[SYNTHETIC] Something else.")
    graph.nodes[-1] = rejected = replace(rejected, review_state=ReviewState.REJECTED)
    graph.wording(
        graph.rules[-1], WordingKind.PROHIBITED_PHRASE, "cure", state=ReviewState.PROPOSED
    )
    graph.edge(PolicyEdgeType.SUPPORTED_BY, rejected, graph.clause_node, ReviewState.PROPOSED)

    assert graph.issues() == set()


async def test_the_imported_synthetic_rulebook_reports_only_its_known_gap(
    connection: AsyncConnection, world: World
) -> None:
    # The shared synthetic graph (NRC-5) has Spanish rows and an exception of an exception, both
    # fine; REC-07 rests only on disclosures with no English source clause, which AC1 must catch.
    tenant_id = world.a.tenant_id
    assert world.a.policy_bundle_id is not None
    async with owner_session(connection, tenant_id) as db:
        await import_policy_graph(db, tenant_id, world.a.policy_bundle_id, load_synthetic_graph())
        snapshot = await load_snapshot(db, tenant_id, world.a.policy_bundle_id)

    assert len(snapshot.rules) == len(load_synthetic_graph().rules)
    assert snapshot.clause_texts
    issues = run_checks(snapshot, (PolicyLanguage.EN, PolicyLanguage.ES))
    assert [(issue.code, issue.subject) for issue in issues] == [
        (CheckCode.MISSING_PROVENANCE, "rule:REC-07")
    ]


async def test_a_snapshot_never_reads_another_tenants_bundle(
    connection: AsyncConnection, world: World
) -> None:
    assert world.a.policy_bundle_id is not None
    async with owner_session(connection, world.a.tenant_id) as db:
        await import_policy_graph(
            db, world.a.tenant_id, world.a.policy_bundle_id, load_synthetic_graph()
        )
    async with owner_session(connection, world.b.tenant_id) as db:
        snapshot = await load_snapshot(db, world.b.tenant_id, world.a.policy_bundle_id)

    assert snapshot == BundleSnapshot((), (), (), (), {})
