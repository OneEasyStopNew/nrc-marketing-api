"""Deterministic checks of a brief or caption against policy rules and the study facts.

Pure functions: no database, no network, no model. There are two paths:

- Graph rules (evaluate_graph_rules, NRC-7): the rules the context manifest selected, each
  checked from its own data (prohibited phrases, exact wording) plus a small registry of
  study-fact checks keyed by rule id. A semantic rule needs a review of meaning, which is not
  built yet, so without a phrase hit it is UNABLE_TO_ASSESS with a coverage gap.
- Legacy (evaluate_text): the hard-coded PHRASE_CHECKS over the bundle's JSONB rules
  (REC-01 to REC-08). Used only when a run has no context manifest. Each check covers one rule
  by id, and a rule with no check is UNABLE_TO_ASSESS with a coverage gap.

Either way an unassessed rule can never pass silently. The checks are phrase and fact matches,
not a reading of meaning. Text offsets are Python string indices, which are Unicode code
points: the contract's unit (see TextSpan).
"""

import re
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Final, Protocol

from app.models.enums import (
    CheckType,
    JudgeFailure,
    ModelCheckFailure,
    Outcome,
    ValidationInputKind,
)
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
    Severity,
    SourceReference,
    TextSpan,
)
from app.services.providers.text_judge import JudgeRuleAnswer

PHRASE_AND_FACT_GAP: Final = CoverageGap(
    reason="Phrase and study-fact checks only; a review of meaning (AI or human) has not run."
)
NOT_CHECKED: Final = "No automated check exists for this rule yet."
# Recorded on each run. Change it whenever the checks or the merge with the model's answers
# change, so stored results can be traced and are never reused across a change.
EVALUATION_VERSION: Final = "checks-v1"
NEEDS_MEANING: Final = "Needs a review of meaning; the model check has not run."
RULE_NOT_LOADED: Final = "The rule's text is not in the policy graph, so it could not be checked."
MODEL_CHECK_FAILED: Final = "The model check failed ({failure}); a person must review this rule."
SIMULATED_REVIEW_GAP: Final = CoverageGap(
    reason="Semantic rules were judged by the SIMULATED model check; this is not evidence."
)
NO_RULES_SELECTED_GAP: Final = CoverageGap(
    reason="The policy context selected no rules for this text, so nothing was checked; a person "
    "must review it."
)


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


class RuleInfo(Protocol):
    # What summarise needs of a rule: the legacy PolicyRule and GraphRule both fit.
    @property
    def rule_id(self) -> str: ...
    @property
    def rule_version(self) -> int: ...
    @property
    def mandatory(self) -> bool: ...
    @property
    def gate(self) -> Gate: ...
    @property
    def severity(self) -> Severity: ...
    @property
    def source(self) -> SourceReference: ...


@dataclass(frozen=True, slots=True)
class RuleContent:
    # What the policy graph stores about a rule that the checks read.
    title: str
    requirement: str
    exact_wording: str | None
    prohibited_phrases: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class GraphRule:
    # A rule the manifest selected. content is None when its policy_rules row was not found.
    rule_id: str
    rule_version: int
    node_key: str
    mandatory: bool
    gate: Gate
    severity: Severity
    source: SourceReference
    check_type: CheckType
    content: RuleContent | None


@dataclass(frozen=True, slots=True)
class ModelReview:
    # The model check of semantic rules, already verified by app.services.judging. answers is
    # empty when the check ran but its answer was rejected; failure then says why.
    answers: Mapping[str, JudgeRuleAnswer]
    failure: JudgeFailure | ModelCheckFailure | None
    is_simulated: bool


@dataclass(frozen=True, slots=True)
class _NotApplicable:
    reason: str


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


# shortcut: study-fact checks are keyed by rule id; move to a fact-check field on
# policy_rules when NRC-4's schema lands.
FACT_CHECKS: Final[Mapping[str, _Check]] = {
    _DISCLOSURE_RULE: _disclosure,
    _FACTS_RULE: _study_facts,
}
# Legacy path only: the graph path reads its phrases from the rules themselves.
TEXT_CHECKS: Final[Mapping[str, _Check]] = {
    **{check.rule_id: _phrase_check(check) for check in PHRASE_CHECKS},
    **FACT_CHECKS,
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


def evaluate_graph_rules(
    text: str,
    rules: Sequence[GraphRule],
    facts: StudyFacts,
    review: ModelReview | None = None,
) -> Evaluation:
    """Check a text against the rules a context manifest selected, from their stored data.

    A prohibited phrase or missing exact wording FAILs the rule, whatever the model says. A
    deterministic rule that had something to check otherwise PASSes. A semantic rule takes the
    model's verified answer from `review`; without one it is UNABLE_TO_ASSESS, as is a rule with
    nothing to check or no stored row.
    """
    hits: dict[str, list[Hit]] = {}
    unassessed: dict[str, str] = {}
    not_applicable: dict[str, str] = {}
    for rule in rules:
        outcome = _check_graph_rule(rule, text, facts, review)
        if isinstance(outcome, str):
            unassessed[rule.rule_id] = outcome
        elif isinstance(outcome, _NotApplicable):
            not_applicable[rule.rule_id] = outcome.reason
        else:
            hits[rule.rule_id] = outcome
    return summarise(
        {rule.rule_id: rule for rule in rules},
        hits,
        extra_gaps=[SIMULATED_REVIEW_GAP] if review is not None and review.is_simulated else [],
        unassessed=unassessed,
        not_applicable=not_applicable,
    )


def rules_for_model(text: str, rules: Sequence[GraphRule], facts: StudyFacts) -> list[GraphRule]:
    # The semantic rules the code checks left open: no phrase or wording hit, row loaded.
    return [rule for rule in rules if _check_graph_rule(rule, text, facts, None) == NEEDS_MEANING]


def _check_graph_rule(
    rule: GraphRule, text: str, facts: StudyFacts, review: ModelReview | None
) -> list[Hit] | str | _NotApplicable:
    # The rule's hits (empty means checked and clean), the reason it can't be assessed, or why
    # it does not apply.
    content = rule.content
    if content is None:
        return RULE_NOT_LOADED
    hits = _phrase_hits(content, text)
    fact_check = FACT_CHECKS.get(rule.rule_id)
    if fact_check is not None:
        hits.extend(fact_check(text, facts))
    if content.exact_wording and _normalise(content.exact_wording) not in _normalise(text):
        hits.append(
            Hit(
                f'Does not include the wording required by the rule "{content.title}".',
                correction=content.exact_wording,
            )
        )
    if hits:
        return hits
    if rule.check_type is CheckType.SEMANTIC:
        return _model_result(rule, review)
    checked = bool(content.prohibited_phrases or content.exact_wording or fact_check)
    return [] if checked else NOT_CHECKED


def _model_result(rule: GraphRule, review: ModelReview | None) -> list[Hit] | str | _NotApplicable:
    if review is None:
        return NEEDS_MEANING
    answer = review.answers.get(rule.rule_id)
    if answer is None:
        return (
            MODEL_CHECK_FAILED.format(failure=review.failure) if review.failure else NEEDS_MEANING
        )
    match answer.result:
        case RuleResultValue.FAIL:
            # Something missing has no span: one finding with the explanation alone.
            return [
                Hit(
                    answer.explanation,
                    span=TextSpan(start=finding.start, end=finding.end),
                    excerpt=finding.excerpt,
                    correction=finding.suggested_correction,
                )
                for finding in answer.findings
            ] or [Hit(answer.explanation)]
        case RuleResultValue.PASS:
            return []
        case RuleResultValue.NOT_APPLICABLE:
            return _NotApplicable(f"{answer.explanation} (exception {answer.exception_node_key})")
        case RuleResultValue.UNABLE_TO_ASSESS:
            return answer.explanation


def _normalise(value: str) -> str:
    # Case-insensitive, whitespace runs collapsed.
    return " ".join(value.split()).casefold()


def _phrase_hits(content: RuleContent, text: str) -> list[Hit]:
    if not content.prohibited_phrases:
        return []
    # Whole words by lookaround rather than \b: Unicode-aware (phrases may be Spanish) and right
    # for a phrase that starts or ends with a symbol. A space in a phrase matches any whitespace
    # run. Linear: the alternatives are escaped literals.
    alternatives = "|".join(
        r"\s+".join(re.escape(word) for word in phrase.split())
        for phrase in content.prohibited_phrases
    )
    pattern = re.compile(rf"(?<!\w)(?:{alternatives})(?!\w)", re.IGNORECASE)
    return [
        Hit(
            f'Uses a phrase the rule "{content.title}" prohibits.',
            span=TextSpan(start=match.start(), end=match.end()),
            excerpt=match.group(),
        )
        for match in pattern.finditer(text)
    ]


def applicable_rules(policy: PolicyBundleRead, kind: ValidationInputKind) -> dict[str, PolicyRule]:
    return {rule.rule_id: rule for rule in policy.rules if kind in rule.applies_to}


def summarise(
    rules: Mapping[str, RuleInfo],
    hits: Mapping[str, list[Hit]],
    *,
    extra_gaps: list[CoverageGap],
    regions: Mapping[str, ImageRegion] | None = None,
    unassessed: Mapping[str, str] | None = None,
    not_applicable: Mapping[str, str] | None = None,
) -> Evaluation:
    """Turn per-rule hits into rule results, gates, findings and an outcome.

    A rule with hits FAILs, a checked rule without hits PASSes, and a rule with no check is
    UNABLE_TO_ASSESS (reported as a coverage gap). `unassessed` gives the reason for such a rule
    (default NOT_CHECKED); a rule in `not_applicable` is NOT_APPLICABLE with that reason.
    """
    reasons = unassessed or {}
    exempt = not_applicable or {}
    findings = [
        _finding(rules[rule_id], hit, None if regions is None else regions.get(rule_id))
        for rule_id, rule_hits in hits.items()
        for hit in rule_hits
    ]
    results = {
        rule_id: RuleResultValue.NOT_APPLICABLE if rule_id in exempt else _result(rule_id, hits)
        for rule_id in rules
    }
    rule_results = [
        RuleResult(
            rule_id=rule.rule_id,
            rule_version=rule.rule_version,
            mandatory=rule.mandatory,
            result=results[rule.rule_id],
            reason=_reason(results[rule.rule_id], rule.rule_id, reasons, exempt),
        )
        for rule in rules.values()
    ]
    gaps = [
        CoverageGap(rule_id=rule_id, reason=reasons.get(rule_id, NOT_CHECKED))
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


def outcome_of(results: Iterable[tuple[RuleInfo, RuleResultValue]]) -> Outcome:
    """A failed mandatory rule fails; a failed optional rule or an unchecked mandatory rule
    needs a human decision; otherwise the input passes."""
    outcome = Outcome.PASS
    for rule, result in results:
        if result is RuleResultValue.FAIL and rule.mandatory:
            return Outcome.FAIL
        if result is RuleResultValue.FAIL or (
            result is RuleResultValue.UNABLE_TO_ASSESS and rule.mandatory
        ):
            outcome = Outcome.NEEDS_REVIEW
    return outcome


def _reason(
    result: RuleResultValue, rule_id: str, unassessed: Mapping[str, str], exempt: Mapping[str, str]
) -> str | None:
    if result is RuleResultValue.UNABLE_TO_ASSESS:
        return unassessed.get(rule_id, NOT_CHECKED)
    if result is RuleResultValue.NOT_APPLICABLE:
        return exempt[rule_id]
    return None


def _result(rule_id: str, hits: Mapping[str, list[Hit]]) -> RuleResultValue:
    if rule_id not in hits:
        return RuleResultValue.UNABLE_TO_ASSESS
    return RuleResultValue.FAIL if hits[rule_id] else RuleResultValue.PASS


def _finding(rule: RuleInfo, hit: Hit, region: ImageRegion | None) -> Finding:
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
