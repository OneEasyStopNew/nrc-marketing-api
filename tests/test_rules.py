# Deterministic brief and caption checks (app.services.rules): no database, no network.

from datetime import UTC, datetime
from uuid import uuid4

import pytest

from app.models.enums import Outcome, ValidationInputKind
from app.schemas.policies import PolicyBundleRead, PolicyRule
from app.schemas.tenancy import StudyFacts
from app.schemas.validations import RuleResultValue
from app.services import rules
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
