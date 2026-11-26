"""Deterministic checks of a brief or caption against the active policy and the study facts.

Pure functions: no database, no network, no model. Each check covers one policy rule by ID
(REC-01 to REC-08 in the seeded policy). A rule in the policy with no check here is reported
as UNABLE_TO_ASSESS with a coverage gap, so an unknown rule can never pass silently.

The checks are phrase and fact matches, not a reading of meaning, and every result says so in
its coverage gaps. Text offsets are Python string indices, which are Unicode code points: the
contract's unit (see TextSpan).
"""

import re
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, replace
from typing import Final

from app.models.enums import MissingContextCode, Outcome, ValidationInputKind
from app.schemas.policies import PolicyBundleRead, PolicyRule
from app.schemas.tenancy import StudyFacts
from app.schemas.validations import (
    CoverageGap,
    Finding,
    Gate,
    GateResult,
    ImageRegion,
    RuleResult,
    RuleResultValue,
    TextSpan,
)
from app.services.policy_context.manifest import ContextManifest

PHRASE_AND_FACT_GAP: Final = CoverageGap(
    reason="Phrase and study-fact checks only; a review of meaning (AI or human) has not run."
)
NOT_CHECKED: Final = "No automated check exists for this rule yet."
# A gap in the rule's policy context, saved with the rule result. {code} is a MissingContextCode.
CONTEXT_GAP: Final = "The policy context for this rule is incomplete ({code}{detail})."
# The rule result each blocking context gap gives a mandatory rule (policy bundle contract).
_CONTEXT_RESULTS: Final[Mapping[MissingContextCode, RuleResultValue]] = {
    MissingContextCode.APPLICABILITY_UNRESOLVED: RuleResultValue.UNABLE_TO_ASSESS,
    MissingContextCode.SOURCE_MISSING: RuleResultValue.UNABLE_TO_ASSESS,
    MissingContextCode.TRAVERSAL_TRUNCATED: RuleResultValue.UNABLE_TO_ASSESS,
    MissingContextCode.CONFLICT: RuleResultValue.NEEDS_REVIEW,
}
# Mandatory results that keep a run from passing without failing it.
_UNSETTLED: Final = frozenset({RuleResultValue.UNABLE_TO_ASSESS, RuleResultValue.NEEDS_REVIEW})
# A result from outside the checks only replaces a less strict one. UNABLE_TO_ASSESS outranks
# NEEDS_REVIEW because a reviewer may resolve only the latter.
_STRICTNESS: Final[Mapping[RuleResultValue, int]] = {
    RuleResultValue.PASS: 0,
    RuleResultValue.NOT_APPLICABLE: 0,
    RuleResultValue.NEEDS_REVIEW: 1,
    RuleResultValue.UNABLE_TO_ASSESS: 2,
    RuleResultValue.FAIL: 3,
}


@dataclass(frozen=True, slots=True)
class Evaluation:
    outcome: Outcome
    gates: list[GateResult]
    rule_results: list[RuleResult]
    findings: list[Finding]
    coverage_gaps: list[CoverageGap]


@dataclass(frozen=True, slots=True)
class Hit:
    explanation: str
    span: TextSpan | None = None
    excerpt: str | None = None
    correction: str | None = None


@dataclass(frozen=True, slots=True)
class _PhraseCheck:
    rule_id: str
    pattern: re.Pattern[str]
    explanation: str
    suggested_correction: str | None


# Every pattern is linear (no nested quantifiers) and inputs are capped at 4000 characters.
# re.ASCII keeps \b and [0-9] to ASCII, so look-alike digits never count as an amount or age.
_FLAGS: Final = re.IGNORECASE | re.ASCII
PHRASE_CHECKS: Final[tuple[_PhraseCheck, ...]] = (
    _PhraseCheck(
        "REC-01",
        re.compile(r"\b(?:safe and effective|proven|cures?)\b", _FLAGS),
        "States or implies that the investigational treatment is safe or works.",
        None,
    ),
    _PhraseCheck(
        "REC-02",
        re.compile(r"\b(?:better than|superior to|best treatment)\b", _FLAGS),
        "Compares the investigational treatment favourably with other treatments.",
        None,
    ),
    _PhraseCheck(
        "REC-03",
        re.compile(r"\bnew (?:treatment|drug|medication|medicine)\b", _FLAGS),
        "Calls the treatment new without saying it is investigational.",
        "investigational treatment",
    ),
    _PhraseCheck(
        "REC-04",
        re.compile(r"\bguarantee[ds]?\b", _FLAGS),
        "Promises a certain favourable outcome.",
        None,
    ),
    _PhraseCheck(
        "REC-05",
        re.compile(r"\bfree (?:treatment|medication|medicine|care)\b", _FLAGS),
        "Promises free treatment.",
        "no cost to take part",
    ),
    _PhraseCheck(
        "REC-06",
        re.compile(r"\$\s?[0-9][0-9,]*|\b(?:get|earn) paid\b", _FLAGS),
        "Emphasises payment for taking part.",
        None,
    ),
)
PAYMENT_CHECK: Final = PHRASE_CHECKS[5]
_DISCLOSURE_RULE: Final = "REC-07"
_DISCLOSURE_PATTERNS: Final = (
    re.compile(r"\bresearch study\b", _FLAGS),
    re.compile(r"\bvoluntary\b", _FLAGS),
)
_FACTS_RULE: Final = "REC-08"
_AGE_RANGE: Final = re.compile(
    # "aged 18 to 75", "ages 18-75", and both with an en dash.
    r"\bage[sd]?\s+([0-9]{1,3})\s*(?:to|and|[-\u2013])\s*([0-9]{1,3})\b",
    _FLAGS,
)
_AMOUNT: Final = re.compile(r"\$\s?[0-9][0-9,]*", _FLAGS)

_Check = Callable[[str, StudyFacts], list[Hit]]


def _phrase_check(check: _PhraseCheck) -> _Check:
    def run(text: str, facts: StudyFacts) -> list[Hit]:
        return [
            Hit(
                check.explanation,
                span=TextSpan(start=match.start(), end=match.end()),
                excerpt=match.group(),
                correction=check.suggested_correction,
            )
            for match in check.pattern.finditer(text)
        ]

    return run


def _disclosure(text: str, facts: StudyFacts) -> list[Hit]:
    if all(pattern.search(text) for pattern in _DISCLOSURE_PATTERNS):
        return []
    # The study's own required wording is the correction; the seed has exactly one.
    correction = " ".join(facts.required_disclosures) or None
    return [
        Hit(
            "Does not say that this is a research study and that taking part is voluntary.",
            correction=correction,
        )
    ]


def _study_facts(text: str, facts: StudyFacts) -> list[Hit]:
    hits = [
        _age_hit(match, facts)
        for match in _AGE_RANGE.finditer(text)
        if (int(match.group(1)), int(match.group(2))) != (facts.min_age, facts.max_age)
    ]
    if facts.compensation_statement is None:
        hits.extend(
            Hit(
                "Mentions a payment, but the approved study facts include no compensation.",
                span=TextSpan(start=match.start(), end=match.end()),
                excerpt=match.group(),
            )
            for match in _AMOUNT.finditer(text)
        )
    return hits


def _age_hit(match: re.Match[str], facts: StudyFacts) -> Hit:
    if facts.min_age is None or facts.max_age is None:
        explanation = "States an age range, but the approved study facts give none."
        correction = None
    else:
        explanation = (
            f"States ages {match.group(1)} to {match.group(2)}, but the approved range is "
            f"{facts.min_age} to {facts.max_age}."
        )
        correction = f"aged {facts.min_age} to {facts.max_age}"
    return Hit(
        explanation,
        span=TextSpan(start=match.start(), end=match.end()),
        excerpt=match.group(),
        correction=correction,
    )


TEXT_CHECKS: Final[Mapping[str, _Check]] = {
    **{check.rule_id: _phrase_check(check) for check in PHRASE_CHECKS},
    _DISCLOSURE_RULE: _disclosure,
    _FACTS_RULE: _study_facts,
}


def evaluate_text(
    kind: ValidationInputKind, text: str, policy: PolicyBundleRead, facts: StudyFacts
) -> Evaluation:
    """Check a brief (kind BRIEF) or caption (kind TEXT) against the rules that apply to it."""
    rules = applicable_rules(policy, kind)
    hits = {
        rule_id: TEXT_CHECKS[rule_id](text, facts) for rule_id in rules if rule_id in TEXT_CHECKS
    }
    return summarise(rules, hits, extra_gaps=[PHRASE_AND_FACT_GAP])


def applicable_rules(policy: PolicyBundleRead, kind: ValidationInputKind) -> dict[str, PolicyRule]:
    return {rule.rule_id: rule for rule in policy.rules if kind in rule.applies_to}


def summarise(
    rules: Mapping[str, PolicyRule],
    hits: Mapping[str, list[Hit]],
    *,
    extra_gaps: list[CoverageGap],
    regions: Mapping[str, ImageRegion] | None = None,
) -> Evaluation:
    """Turn per-rule hits into rule results, gates, findings and an outcome.

    A rule with hits FAILs, a checked rule without hits PASSes, and a rule with no check is
    UNABLE_TO_ASSESS (reported as a coverage gap).
    """
    findings = [
        _finding(rules[rule_id], hit, None if regions is None else regions.get(rule_id))
        for rule_id, rule_hits in hits.items()
        for hit in rule_hits
    ]
    results = {rule_id: _result(rule_id, hits) for rule_id in rules}
    rule_results = [
        RuleResult(
            rule_id=rule.rule_id,
            rule_version=rule.rule_version,
            mandatory=rule.mandatory,
            result=results[rule.rule_id],
            reason=NOT_CHECKED
            if results[rule.rule_id] is RuleResultValue.UNABLE_TO_ASSESS
            else None,
        )
        for rule in rules.values()
    ]
    gaps = [
        CoverageGap(rule_id=rule_id, reason=NOT_CHECKED)
        for rule_id, result in results.items()
        if result is RuleResultValue.UNABLE_TO_ASSESS
    ]
    gates = [
        GateResult(
            gate=gate,
            outcome=outcome_of(
                (rule, results[rule.rule_id]) for rule in rules.values() if rule.gate is gate
            ),
        )
        for gate in Gate
        if any(rule.gate is gate for rule in rules.values())
    ]
    return Evaluation(
        outcome=outcome_of((rule, results[rule.rule_id]) for rule in rules.values()),
        gates=gates,
        rule_results=rule_results,
        findings=findings,
        coverage_gaps=[*gaps, *extra_gaps],
    )


def outcome_of(results: Iterable[tuple[PolicyRule, RuleResultValue]]) -> Outcome:
    """A failed mandatory rule fails; a failed optional rule, or a mandatory rule that is
    unchecked or waiting for a person, needs a human decision; otherwise the input passes."""
    outcome = Outcome.PASS
    for rule, result in results:
        if result is RuleResultValue.FAIL and rule.mandatory:
            return Outcome.FAIL
        if result is RuleResultValue.FAIL or (result in _UNSETTLED and rule.mandatory):
            outcome = Outcome.NEEDS_REVIEW
    return outcome


def apply_policy_context(
    evaluation: Evaluation, rules: Mapping[str, PolicyRule], manifest: ContextManifest
) -> Evaluation:
    """The evaluation with what the policy context couldn't settle made part of the saved result.

    Contract (policy bundle, "Retrieval gaps"): a mandatory rule with unresolved applicability, a
    missing source or an incomplete graph walk is UNABLE_TO_ASSESS; one in a conflict is
    NEEDS_REVIEW; FAIL stays FAIL. A gap on an optional rule is a warning (a coverage gap only).
    A walk cut short names no rule, so it affects every mandatory rule. Search-only codes
    (full-text only, statements cut) are left in the manifest: search never selects a rule.
    """
    mandatory = {rule.rule_id for rule in manifest.required_rules} | {
        rule_id for rule_id, rule in rules.items() if rule.mandatory
    }
    changes: dict[str, tuple[RuleResultValue, str]] = {}
    gaps: list[CoverageGap] = []
    for item in manifest.missing_context:
        value = _CONTEXT_RESULTS.get(item.code)
        if value is None:
            continue
        reason = CONTEXT_GAP.format(
            code=item.code, detail=f": {item.detail}" if item.detail else ""
        )
        for rule_id in [item.rule_id] if item.rule_id else sorted(mandatory):
            if rule_id not in mandatory:
                gaps.append(CoverageGap(rule_id=rule_id, reason=f"{reason} (optional rule)"))
                continue
            gaps.append(CoverageGap(rule_id=rule_id, reason=reason))
            if rule_id not in changes or _STRICTNESS[value] > _STRICTNESS[changes[rule_id][0]]:
                changes[rule_id] = (value, reason)
    result = with_results(evaluation, rules, changes, gaps)
    # The graph may call a rule mandatory that the checks didn't run, or ran as optional (the
    # import matches rule ids, not flags). outcome_of can't see those, but they still block: the
    # run and the rule's gate can't pass.
    unseen = {
        rule_id for rule_id in changes if rule_id not in rules or not rules[rule_id].mandatory
    }
    if unseen:
        gate_of = {rule.rule_id: rule.gate for rule in manifest.required_rules}
        blocked = {gate_of.get(rule_id) for rule_id in unseen}
        result = replace(
            result,
            outcome=_settled_at_most(result.outcome),
            gates=[
                gate.model_copy(update={"outcome": _settled_at_most(gate.outcome)})
                if gate.gate in blocked or None in blocked
                else gate
                for gate in result.gates
            ],
        )
    return result


def apply_missing_policy_context(
    evaluation: Evaluation, rules: Mapping[str, PolicyRule], reason: str
) -> Evaluation:
    """No policy context at all: every mandatory rule is UNABLE_TO_ASSESS (FAIL stays FAIL), and
    the run can't pass even when it has no mandatory rule."""
    changes = {
        rule_id: (RuleResultValue.UNABLE_TO_ASSESS, reason)
        for rule_id, rule in rules.items()
        if rule.mandatory
    }
    result = with_results(evaluation, rules, changes, [])
    return replace(
        result,
        outcome=_settled_at_most(result.outcome),
        gates=[
            gate.model_copy(update={"outcome": _settled_at_most(gate.outcome)})
            for gate in result.gates
        ],
    )


def with_results(
    evaluation: Evaluation,
    rules: Mapping[str, PolicyRule],
    changes: Mapping[str, tuple[RuleResultValue, str]],
    gaps: Iterable[CoverageGap],
) -> Evaluation:
    """The evaluation with rule results changed from outside the checks, gates and outcome
    recomputed. A change only ever makes a result stricter: FAIL stays FAIL."""
    rule_results = [
        result.model_copy(
            update={"result": changes[result.rule_id][0], "reason": changes[result.rule_id][1]}
        )
        if result.rule_id in changes
        and _STRICTNESS[changes[result.rule_id][0]] > _STRICTNESS[result.result]
        else result
        for result in evaluation.rule_results
    ]
    values = {result.rule_id: result.result for result in rule_results}
    return replace(
        evaluation,
        outcome=outcome_of((rule, values[rule_id]) for rule_id, rule in rules.items()),
        gates=[
            gate.model_copy(
                update={
                    "outcome": outcome_of(
                        (rule, values[rule_id])
                        for rule_id, rule in rules.items()
                        if rule.gate is gate.gate
                    )
                }
            )
            for gate in evaluation.gates
        ],
        rule_results=rule_results,
        coverage_gaps=[*evaluation.coverage_gaps, *gaps],
    )


def _settled_at_most(outcome: Outcome) -> Outcome:
    # PASS becomes NEEDS_REVIEW; FAIL and NEEDS_REVIEW stay.
    return Outcome.NEEDS_REVIEW if outcome is Outcome.PASS else outcome


def _result(rule_id: str, hits: Mapping[str, list[Hit]]) -> RuleResultValue:
    if rule_id not in hits:
        return RuleResultValue.UNABLE_TO_ASSESS
    return RuleResultValue.FAIL if hits[rule_id] else RuleResultValue.PASS


def _finding(rule: PolicyRule, hit: Hit, region: ImageRegion | None) -> Finding:
    return Finding(
        rule_id=rule.rule_id,
        rule_version=rule.rule_version,
        severity=rule.severity,
        explanation=hit.explanation,
        source=rule.source,
        text_span=hit.span,
        image_region=region,
        offending_excerpt=hit.excerpt,
        suggested_correction=hit.correction,
        needs_human_decision=not rule.mandatory,
    )
