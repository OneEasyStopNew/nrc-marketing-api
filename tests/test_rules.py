# Deterministic brief and caption checks (app.services.rules): no database, no network.

from datetime import UTC, datetime
from uuid import uuid4

import pytest

from app.models.enums import (
    CheckType,
    Gate,
    JudgeFailure,
    ModelCheckFailure,
    Outcome,
    Severity,
    ValidationInputKind,
)
from app.schemas.policies import PolicyBundleRead, PolicyRule
from app.schemas.tenancy import StudyFacts
from app.schemas.validations import RuleResultValue, SourceReference
from app.services import rules
from app.services.providers.text_judge import JudgeFinding, JudgeRuleAnswer
from scripts.seed_demo import DEMO_POLICY_NAME, DEMO_POLICY_RULES, SYNTHETIC_FACTS

BRIEF = ValidationInputKind.BRIEF
TEXT = ValidationInputKind.TEXT
DISCLOSURE = "This is a research study. Taking part is voluntary."
# The two captions rehearsed for the demo (steps 4 and 6 of the run sheet).
FAILING_CAPTION = (
    "Join our study of a new treatment for high blood pressure. It is proven to work, and you "
    "get free treatment plus $150."
)
PASSING_CAPTION = (
    "Adults aged 18 to 75 with high blood pressure may be able to join a research study of an "
    "investigational treatment. Taking part is voluntary, and there is no cost to take part."
)


def _policy(extra: list[dict[str, object]] | None = None) -> PolicyBundleRead:
    return PolicyBundleRead(
        policy_bundle_id=uuid4(),
        version=1,
        name=DEMO_POLICY_NAME,
        source_ref="synthetic:test",
        is_synthetic=True,
        approved_by="synthetic-fixture",
        approved_at=datetime(2026, 11, 1, tzinfo=UTC),
        rules=[PolicyRule.model_validate(rule) for rule in [*DEMO_POLICY_RULES, *(extra or [])]],
    )


FACTS = StudyFacts.model_validate(SYNTHETIC_FACTS)


def _excerpts(evaluation: rules.Evaluation) -> dict[str, list[str | None]]:
    found: dict[str, list[str | None]] = {}
    for finding in evaluation.findings:
        found.setdefault(finding.rule_id, []).append(finding.offending_excerpt)
    return found


def test_risky_brief_fails_with_exact_spans() -> None:
    text = "Guaranteed results! Try this new drug for free treatment."

    evaluation = rules.evaluate_text(BRIEF, text, _policy(), FACTS)

    assert evaluation.outcome is Outcome.FAIL
    assert _excerpts(evaluation) == {
        "REC-03": ["new drug"],
        "REC-04": ["Guaranteed"],
        "REC-05": ["free treatment"],
    }
    for finding in evaluation.findings:
        assert finding.text_span is not None
        assert text[finding.text_span.start : finding.text_span.end] == finding.offending_excerpt


def test_clean_brief_passes_every_applicable_rule() -> None:
    text = "Invite adults with high blood pressure to learn about our research study."

    evaluation = rules.evaluate_text(BRIEF, text, _policy(), FACTS)

    assert evaluation.outcome is Outcome.PASS
    assert evaluation.findings == []
    assert {r.result for r in evaluation.rule_results} == {RuleResultValue.PASS}
    # Briefs are not checked for the disclosure (REC-07 applies to captions only).
    assert "REC-07" not in {r.rule_id for r in evaluation.rule_results}
    assert evaluation.coverage_gaps == [rules.PHRASE_AND_FACT_GAP]


def test_rehearsed_fail_caption() -> None:
    evaluation = rules.evaluate_text(TEXT, FAILING_CAPTION, _policy(), FACTS)

    assert evaluation.outcome is Outcome.FAIL
    assert set(_excerpts(evaluation)) == {
        "REC-01",
        "REC-03",
        "REC-05",
        "REC-06",
        "REC-07",
        "REC-08",
    }
    corrections = {f.rule_id: f.suggested_correction for f in evaluation.findings}
    assert corrections["REC-03"] == "investigational treatment"
    assert corrections["REC-07"] == DISCLOSURE


def test_rehearsed_pass_caption() -> None:
    evaluation = rules.evaluate_text(TEXT, PASSING_CAPTION, _policy(), FACTS)

    assert evaluation.outcome is Outcome.PASS
    assert evaluation.findings == []


@pytest.mark.parametrize("stated", ["aged 21 to 65", "ages 21-65", "aged 21\u201365"])
def test_age_range_outside_the_approved_facts_fails(stated: str) -> None:
    text = f"Adults {stated} may join this research study. Taking part is voluntary."

    evaluation = rules.evaluate_text(TEXT, text, _policy(), FACTS)

    assert evaluation.outcome is Outcome.FAIL
    (finding,) = evaluation.findings
    assert finding.rule_id == "REC-08"
    assert finding.offending_excerpt == stated
    assert finding.suggested_correction == "aged 18 to 75"


def test_amount_without_approved_compensation_fails_the_facts_rule() -> None:
    text = f"Earn $150. {DISCLOSURE}"

    evaluation = rules.evaluate_text(TEXT, text, _policy(), FACTS)

    assert evaluation.outcome is Outcome.FAIL
    assert _excerpts(evaluation) == {"REC-06": ["$150"], "REC-08": ["$150"]}


def test_amount_is_allowed_by_the_facts_rule_when_compensation_is_approved() -> None:
    facts = FACTS.model_copy(update={"compensation_statement": "Up to $150 for your time."})

    evaluation = rules.evaluate_text(
        TEXT, f"Up to $150 for your time. {DISCLOSURE}", _policy(), facts
    )

    # Only REC-06 (payment emphasis) remains, and it is optional: a human decides.
    assert _excerpts(evaluation) == {"REC-06": ["$150"]}
    assert evaluation.outcome is Outcome.NEEDS_REVIEW
    assert evaluation.findings[0].needs_human_decision is True


def test_offsets_are_code_points() -> None:
    text = f"\N{SLIGHTLY SMILING FACE} Guaranteed. {DISCLOSURE}"

    evaluation = rules.evaluate_text(TEXT, text, _policy(), FACTS)

    span = evaluation.findings[0].text_span
    assert span is not None
    assert (span.start, span.end) == (2, 12)


def test_unknown_rule_is_unable_to_assess_and_needs_review() -> None:
    extra = {
        **DEMO_POLICY_RULES[0],
        "rule_id": "REC-99",
        "title": "A rule with no automated check",
        "applies_to": ["brief"],
    }
    text = "Invite adults with high blood pressure to learn about our research study."

    evaluation = rules.evaluate_text(BRIEF, text, _policy([extra]), FACTS)

    result = next(r for r in evaluation.rule_results if r.rule_id == "REC-99")
    assert result.result is RuleResultValue.UNABLE_TO_ASSESS
    assert result.reason == rules.NOT_CHECKED
    assert any(gap.rule_id == "REC-99" for gap in evaluation.coverage_gaps)
    assert evaluation.outcome is Outcome.NEEDS_REVIEW


def test_gates_follow_their_rules() -> None:
    evaluation = rules.evaluate_text(TEXT, f"Adults aged 30 to 40. {DISCLOSURE}", _policy(), FACTS)

    gates = {gate.gate.value: gate.outcome for gate in evaluation.gates}
    assert gates == {
        "completeness": Outcome.PASS,
        "study_facts": Outcome.FAIL,
        "policy": Outcome.PASS,
    }


# Graph rules (evaluate_graph_rules): the rules a context manifest selected, checked from their
# own data.


def _graph_rule(
    rule_id: str = "R-1",
    *,
    check_type: CheckType = CheckType.SEMANTIC,
    mandatory: bool = True,
    gate: Gate = Gate.POLICY,
    phrases: tuple[str, ...] = (),
    wording: str | None = None,
    loaded: bool = True,
) -> rules.GraphRule:
    content = rules.RuleContent(
        title="Test rule",
        requirement="Test requirement.",
        exact_wording=wording,
        prohibited_phrases=phrases,
    )
    return rules.GraphRule(
        rule_id=rule_id,
        rule_version=1,
        node_key=rule_id,
        mandatory=mandatory,
        gate=gate,
        severity=Severity.MAJOR,
        source=SourceReference(source_id="synthetic:test", clause=rule_id),
        check_type=check_type,
        content=content if loaded else None,
    )


def _result_of(evaluation: rules.Evaluation, rule_id: str = "R-1") -> RuleResultValue:
    return next(r.result for r in evaluation.rule_results if r.rule_id == rule_id)


def test_no_selected_rules_needs_review_instead_of_passing() -> None:
    # Nothing checked is not the same as nothing broken.
    evaluation = rules.evaluate_graph_rules("Join our study.", [], FACTS)

    assert evaluation.outcome is Outcome.NEEDS_REVIEW
    assert all(gate.outcome is not Outcome.PASS for gate in evaluation.gates)
    assert rules.NO_RULES_SELECTED_GAP in evaluation.coverage_gaps


def test_graph_phrase_hit_fails_with_code_point_span() -> None:
    text = "🙂 Café: results GUARANTEED for you."
    evaluation = rules.evaluate_graph_rules(text, [_graph_rule(phrases=("guaranteed",))], FACTS)

    (finding,) = evaluation.findings
    assert finding.text_span is not None
    assert text[finding.text_span.start : finding.text_span.end] == "GUARANTEED"
    assert finding.text_span.start == 16  # the emoji and é are one code point each
    assert evaluation.outcome is Outcome.FAIL


@pytest.mark.parametrize(
    ("text", "hit"),
    [
        ("It is proven.", True),
        ("It is unproven.", False),
        ("A proven-ish claim.", True),  # a hyphen is not a word character
        ("Provená", False),  # a Unicode letter continues the word
        ("safe   and\neffective", True),  # whitespace runs inside a phrase
    ],
)
def test_graph_phrases_match_whole_words_only(text: str, hit: bool) -> None:
    rule = _graph_rule(phrases=("proven", "safe and effective"))

    evaluation = rules.evaluate_graph_rules(text, [rule], FACTS)

    assert bool(evaluation.findings) is hit


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("This is a research study. Taking part is voluntary.", RuleResultValue.PASS),
        ("THIS IS A RESEARCH   STUDY.\nTaking part is voluntary.", RuleResultValue.PASS),
        ("This is a study. Taking part is voluntary.", RuleResultValue.FAIL),
    ],
)
def test_graph_exact_wording_must_appear(text: str, expected: RuleResultValue) -> None:
    rule = _graph_rule(
        check_type=CheckType.DETERMINISTIC, gate=Gate.COMPLETENESS, wording=DISCLOSURE
    )

    evaluation = rules.evaluate_graph_rules(text, [rule], FACTS)

    assert _result_of(evaluation) is expected
    if expected is RuleResultValue.FAIL:
        (finding,) = evaluation.findings
        assert finding.suggested_correction == DISCLOSURE
        assert finding.text_span is None


def test_graph_semantic_rule_without_a_hit_is_unable_to_assess() -> None:
    evaluation = rules.evaluate_graph_rules(
        "A clean text.", [_graph_rule(phrases=("cure",))], FACTS
    )

    assert _result_of(evaluation) is RuleResultValue.UNABLE_TO_ASSESS
    assert evaluation.outcome is Outcome.NEEDS_REVIEW
    (gap,) = evaluation.coverage_gaps
    assert (gap.rule_id, gap.reason) == ("R-1", rules.NEEDS_MEANING)


def test_graph_deterministic_rule_with_nothing_to_check_is_unable_to_assess() -> None:
    rule = _graph_rule(check_type=CheckType.DETERMINISTIC)

    evaluation = rules.evaluate_graph_rules("A clean text.", [rule], FACTS)

    assert _result_of(evaluation) is RuleResultValue.UNABLE_TO_ASSESS
    assert evaluation.rule_results[0].reason == rules.NOT_CHECKED


def test_graph_rule_without_its_row_is_unable_to_assess_even_if_deterministic() -> None:
    rule = _graph_rule(check_type=CheckType.DETERMINISTIC, loaded=False)

    evaluation = rules.evaluate_graph_rules("A clean text.", [rule], FACTS)

    assert _result_of(evaluation) is RuleResultValue.UNABLE_TO_ASSESS
    assert evaluation.rule_results[0].reason == rules.RULE_NOT_LOADED


def test_graph_study_fact_rules_use_the_registry() -> None:
    disclosure = _graph_rule("REC-07", check_type=CheckType.DETERMINISTIC, gate=Gate.COMPLETENESS)
    facts = _graph_rule("REC-08", check_type=CheckType.DETERMINISTIC, gate=Gate.STUDY_FACTS)

    clean = rules.evaluate_graph_rules(PASSING_CAPTION, [disclosure, facts], FACTS)
    wrong = rules.evaluate_graph_rules("Adults aged 30 to 40.", [disclosure, facts], FACTS)

    assert clean.outcome is Outcome.PASS
    assert {r.rule_id: r.result for r in wrong.rule_results} == {
        "REC-07": RuleResultValue.FAIL,
        "REC-08": RuleResultValue.FAIL,
    }


@pytest.mark.parametrize(
    ("mandatory", "hit", "expected"),
    [
        (True, True, Outcome.FAIL),
        (False, True, Outcome.NEEDS_REVIEW),
        (False, False, Outcome.PASS),  # an unassessed optional rule doesn't block
        (True, False, Outcome.NEEDS_REVIEW),
    ],
)
def test_graph_outcome_table(mandatory: bool, hit: bool, expected: Outcome) -> None:
    rule = _graph_rule(mandatory=mandatory, phrases=("cure",))

    evaluation = rules.evaluate_graph_rules("a cure" if hit else "clean", [rule], FACTS)

    assert evaluation.outcome is expected


# Merging the model check (ModelReview) into the graph rules.


def _review(
    *answers: JudgeRuleAnswer, failure: JudgeFailure | ModelCheckFailure | None = None
) -> rules.ModelReview:
    return rules.ModelReview(
        answers={a.rule_id: a for a in answers}, failure=failure, is_simulated=False
    )


def _said(result: RuleResultValue, **extra: object) -> JudgeRuleAnswer:
    return JudgeRuleAnswer.model_validate(
        {"rule_id": "R-1", "result": result, "explanation": "The model's reason.", **extra}
    )


def test_model_fail_becomes_findings_with_its_spans() -> None:
    text = "🙂 This will fix you."
    finding = JudgeFinding(start=2, end=20, excerpt="This will fix you.", suggested_correction="x")

    evaluation = rules.evaluate_graph_rules(
        text, [_graph_rule()], FACTS, _review(_said(RuleResultValue.FAIL, findings=[finding]))
    )

    (found,) = evaluation.findings
    assert found.text_span is not None
    assert text[found.text_span.start : found.text_span.end] == "This will fix you."
    assert (found.explanation, found.suggested_correction) == ("The model's reason.", "x")
    assert evaluation.outcome is Outcome.FAIL


def test_model_fail_for_something_missing_has_one_finding_without_a_span() -> None:
    evaluation = rules.evaluate_graph_rules(
        "text", [_graph_rule()], FACTS, _review(_said(RuleResultValue.FAIL))
    )

    (found,) = evaluation.findings
    assert found.text_span is None


def test_model_pass_on_every_open_rule_lets_the_text_pass() -> None:
    evaluation = rules.evaluate_graph_rules(
        "text", [_graph_rule()], FACTS, _review(_said(RuleResultValue.PASS))
    )

    assert evaluation.outcome is Outcome.PASS
    assert evaluation.coverage_gaps == []


def test_model_not_applicable_keeps_its_reason_and_exception() -> None:
    answer = _said(RuleResultValue.NOT_APPLICABLE, exception_node_key="EX-1")

    evaluation = rules.evaluate_graph_rules("text", [_graph_rule()], FACTS, _review(answer))

    (result,) = evaluation.rule_results
    assert result.result is RuleResultValue.NOT_APPLICABLE
    assert result.reason == "The model's reason. (exception EX-1)"
    assert evaluation.outcome is Outcome.PASS


def test_model_unable_to_assess_needs_review_with_its_explanation() -> None:
    evaluation = rules.evaluate_graph_rules(
        "text", [_graph_rule()], FACTS, _review(_said(RuleResultValue.UNABLE_TO_ASSESS))
    )

    assert evaluation.outcome is Outcome.NEEDS_REVIEW
    assert evaluation.rule_results[0].reason == "The model's reason."


def test_rejected_model_answer_leaves_the_rule_unassessed_with_the_failure() -> None:
    evaluation = rules.evaluate_graph_rules(
        "text", [_graph_rule()], FACTS, _review(failure=JudgeFailure.MALFORMED)
    )

    assert _result_of(evaluation) is RuleResultValue.UNABLE_TO_ASSESS
    assert evaluation.rule_results[0].reason == rules.MODEL_CHECK_FAILED.format(
        failure=JudgeFailure.MALFORMED
    )
    assert evaluation.outcome is Outcome.NEEDS_REVIEW


def test_a_code_hit_wins_over_a_model_pass() -> None:
    rule = _graph_rule(phrases=("cure",))

    evaluation = rules.evaluate_graph_rules(
        "a cure", [rule], FACTS, _review(_said(RuleResultValue.PASS))
    )

    assert evaluation.outcome is Outcome.FAIL


def test_a_model_answer_never_changes_a_deterministic_rule() -> None:
    rule = _graph_rule(check_type=CheckType.DETERMINISTIC)

    evaluation = rules.evaluate_graph_rules(
        "text", [rule], FACTS, _review(_said(RuleResultValue.PASS))
    )

    assert evaluation.rule_results[0].reason == rules.NOT_CHECKED


def test_a_simulated_review_is_labelled_in_the_coverage_gaps() -> None:
    review = rules.ModelReview(
        answers={"R-1": _said(RuleResultValue.PASS)}, failure=None, is_simulated=True
    )

    evaluation = rules.evaluate_graph_rules("text", [_graph_rule()], FACTS, review)

    assert rules.SIMULATED_REVIEW_GAP in evaluation.coverage_gaps


def test_only_open_semantic_rules_go_to_the_model() -> None:
    open_rule = _graph_rule("R-OPEN", phrases=("cure",))
    hit_rule = _graph_rule("R-HIT", phrases=("guaranteed",))
    missing = _graph_rule("R-MISSING", loaded=False)
    code_rule = _graph_rule("R-CODE", check_type=CheckType.DETERMINISTIC, phrases=("free",))

    sent = rules.rules_for_model(
        "Guaranteed help.", [open_rule, hit_rule, missing, code_rule], FACTS
    )

    assert [rule.rule_id for rule in sent] == ["R-OPEN"]


def test_results_from_the_policy_context_only_ever_get_stricter() -> None:
    # A conflict (NEEDS_REVIEW) can't soften a rule already unassessed, and FAIL never changes:
    # a reviewer may resolve NEEDS_REVIEW, never UNABLE_TO_ASSESS or FAIL.
    policy = _policy()
    checked = rules.applicable_rules(policy, BRIEF)
    evaluation = rules.evaluate_text(BRIEF, "Guaranteed results!", policy, FACTS)
    unassessed = rules.with_results(
        evaluation, checked, {"REC-01": (RuleResultValue.UNABLE_TO_ASSESS, "gap")}, []
    )

    changed = rules.with_results(
        unassessed,
        checked,
        {
            "REC-01": (RuleResultValue.NEEDS_REVIEW, "conflict"),
            "REC-04": (RuleResultValue.UNABLE_TO_ASSESS, "gap"),
        },
        [],
    )

    results = {result.rule_id: result.result for result in changed.rule_results}
    assert results["REC-01"] is RuleResultValue.UNABLE_TO_ASSESS
    assert results["REC-04"] is RuleResultValue.FAIL
    assert changed.outcome is Outcome.FAIL
