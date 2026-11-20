# Whether a rule or exception applies to one request: a pure, three-way decision.

from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import datetime
from uuid import UUID

from app.models.enums import Applicability
from app.schemas.tenancy import StudyFacts
from app.services.policy_context.manifest import ContextScope


@dataclass(frozen=True, slots=True)
class Conditions:
    # A node's applicability columns. An empty tuple means "no restriction".
    input_kinds: tuple[str, ...] = ()
    channels: tuple[str, ...] = ()
    post_formats: tuple[str, ...] = ()
    languages: tuple[str, ...] = ()
    audiences: tuple[str, ...] = ()
    jurisdictions: tuple[str, ...] = ()
    study_ids: tuple[UUID, ...] = ()
    effective_from: datetime | None = None
    effective_to: datetime | None = None
    required_study_fact: str | None = None


@dataclass(frozen=True, slots=True)
class Verdict:
    applies: Applicability
    # Dimensions restricted by the conditions but unknown in the request.
    unresolved: tuple[str, ...] = ()
    # Why it does not apply; empty unless applies is FALSE.
    reasons: tuple[str, ...] = field(default=())


def evaluate(
    conditions: Conditions, scope: ContextScope, study_id: UUID, facts: StudyFacts
) -> Verdict:
    """Return TRUE, FALSE or UNRESOLVED for one node's conditions.

    Any condition that clearly fails makes it FALSE. Otherwise a restricted dimension the
    request doesn't know (audience, jurisdiction, an unknown study fact) makes it UNRESOLVED:
    callers keep unresolved rules and flag them, so missing input never drops a mandatory rule.
    """
    unresolved: list[str] = []
    reasons: list[str] = []
    dimensions: Sequence[tuple[str, tuple[str, ...], str | None]] = (
        ("input_kind", conditions.input_kinds, scope.input_kind.value),
        ("channel", conditions.channels, scope.channel.value),
        (
            "post_format",
            conditions.post_formats,
            scope.post_format.value if scope.post_format else None,
        ),
        ("language", conditions.languages, scope.language.value),
        ("audience", conditions.audiences, scope.audience),
        ("jurisdiction", conditions.jurisdictions, scope.jurisdiction),
        ("study", tuple(str(s) for s in conditions.study_ids), str(study_id)),
    )
    for name, allowed, value in dimensions:
        if not allowed:
            continue
        if value is None:
            unresolved.append(name)
        elif value not in allowed:
            reasons.append(f"{name.upper()}_NOT_MATCHED")
    reasons.extend(window_reasons(conditions, scope.evaluated_at))
    fact = conditions.required_study_fact
    if fact is not None:
        if fact not in StudyFacts.model_fields:
            # The importer rejects unknown names; if one slips through, don't guess.
            unresolved.append(f"study_fact:{fact}")
        elif not _present(getattr(facts, fact)):
            reasons.append(f"STUDY_FACT_MISSING:{fact}")
    if reasons:
        return Verdict(Applicability.FALSE, tuple(unresolved), tuple(reasons))
    if unresolved:
        return Verdict(Applicability.UNRESOLVED, tuple(unresolved))
    return Verdict(Applicability.TRUE)


def window_reasons(conditions: Conditions, at: datetime) -> tuple[str, ...]:
    # The one place that decides the effective window: from is inclusive, to is exclusive.
    reasons: list[str] = []
    if conditions.effective_from is not None and at < conditions.effective_from:
        reasons.append("NOT_YET_EFFECTIVE")
    if conditions.effective_to is not None and at >= conditions.effective_to:
        reasons.append("NO_LONGER_EFFECTIVE")
    return tuple(reasons)


def _present(value: object) -> bool:
    # 0 is a real value (min_age, max_age); only absent, blank or empty counts as missing.
    return value is not None and value != "" and value != []
