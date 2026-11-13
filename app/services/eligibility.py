"""Generation eligibility: may image generation start for this creative version?

Two entry points share one validation rule:

- check_generation_eligibility answers the UI, so the Generate button can say why it is disabled.
- assert_generation_allowed is the gate job admission runs (app.services.generation_checks); it
  raises ValidationNotPassedError instead of answering.

The latest stored brief validation of the version decides. A run checked against an older policy
version no longer counts, so a new policy makes every brief validate again.
"""

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Final
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.auth.access import TenantContext
from app.core.errors import NotFoundError, ValidationNotPassedError
from app.models import Creative, StudySnapshot
from app.models.enums import RecruitmentStatus
from app.schemas.policies import EligibilityReason, GenerationEligibility
from app.schemas.validations import Outcome
from app.services import policies, validations
from app.services.creatives import visible_to

REASON_MESSAGES: Final[Mapping[EligibilityReason, str]] = MappingProxyType(
    {
        EligibilityReason.ELIGIBLE: "The brief passed validation against the active policy.",
        EligibilityReason.STALE_VERSION: "A newer version of this creative exists. Reload it.",
        EligibilityReason.STUDY_CLOSED: "The study is closed to recruitment.",
        EligibilityReason.NO_ACTIVE_POLICY: "This tenant has no active policy to validate against.",
        EligibilityReason.NOT_VALIDATED: "Validate the brief for this version before generating.",
        EligibilityReason.VALIDATION_FAILED: "The brief failed validation. Fix and revalidate.",
        EligibilityReason.NEEDS_REVIEW: "The brief needs a human decision before generating.",
    }
)
_REASON_BY_OUTCOME: Final[Mapping[Outcome, EligibilityReason]] = MappingProxyType(
    {
        Outcome.PASS: EligibilityReason.ELIGIBLE,
        Outcome.FAIL: EligibilityReason.VALIDATION_FAILED,
        Outcome.NEEDS_REVIEW: EligibilityReason.NEEDS_REVIEW,
    }
)


@dataclass(frozen=True, slots=True)
class _ValidationStatus:
    # reason is ELIGIBLE only for a PASS against the active policy version.
    reason: EligibilityReason
    policy_bundle_version: int | None
    # The latest finished brief run of the version, whatever its outcome; None when there is none.
    latest_run_id: UUID | None


async def check_generation_eligibility(
    db: AsyncSession,
    ctx: TenantContext,
    creative_id: UUID,
    creative_version: int | None = None,
) -> GenerationEligibility:
    """Decide whether generation may start for one creative version (default: the current one).

    Raises:
        NotFoundError: no such creative visible to the caller, or no such version yet.
    """
    creative = (
        await db.execute(
            select(Creative.current_version, Creative.study_id).where(
                *visible_to(ctx), Creative.id == creative_id
            )
        )
    ).one_or_none()
    if creative is None:
        raise NotFoundError("Creative not found.")

    version = creative.current_version if creative_version is None else creative_version
    if version > creative.current_version:
        raise NotFoundError("Creative version not found.")
    status = await _validation_status(db, ctx.tenant_id, creative_id, version)
    reason = await _blocking_reason(db, ctx, creative.study_id, version, creative.current_version)
    if reason is None:
        reason = status.reason
    eligible = reason is EligibilityReason.ELIGIBLE

    return GenerationEligibility(
        creative_id=creative_id,
        creative_version=version,
        eligible=eligible,
        reason=reason,
        message=REASON_MESSAGES[reason],
        policy_bundle_version=status.policy_bundle_version,
        # Reported only when it is what makes the version eligible.
        validation_run_id=status.latest_run_id if eligible else None,
    )


async def assert_generation_allowed(
    session: AsyncSession, tenant_id: UUID, creative_id: UUID, creative_version: int
) -> None:
    """The validation-PASS gate for generating this exact creative version.

    The caller (app.services.generation_checks) has already checked that the creative is visible,
    current and in a recruiting study.

    Raises:
        ValidationNotPassedError: details.validation_run_id is the latest brief run of the version
            (None when there is none) and details.status the EligibilityReason: NOT_VALIDATED,
            VALIDATION_FAILED, NEEDS_REVIEW or NO_ACTIVE_POLICY.
    """
    status = await _validation_status(session, tenant_id, creative_id, creative_version)
    if status.reason is EligibilityReason.ELIGIBLE:
        return
    run_id = None if status.latest_run_id is None else str(status.latest_run_id)
    raise ValidationNotPassedError(
        REASON_MESSAGES[status.reason],
        details={"validation_run_id": run_id, "status": status.reason.value},
    )


async def _blocking_reason(
    db: AsyncSession, ctx: TenantContext, study_id: UUID, version: int, current_version: int
) -> EligibilityReason | None:
    if version != current_version:
        return EligibilityReason.STALE_VERSION
    status = await db.scalar(
        select(StudySnapshot.recruitment_status)
        .where(StudySnapshot.tenant_id == ctx.tenant_id, StudySnapshot.study_id == study_id)
        .order_by(StudySnapshot.version.desc())
        .limit(1)
    )
    if status is not RecruitmentStatus.RECRUITING:
        return EligibilityReason.STUDY_CLOSED
    return None


async def _validation_status(
    db: AsyncSession, tenant_id: UUID, creative_id: UUID, version: int
) -> _ValidationStatus:
    policy = await policies.get_active_policy_summary(db, tenant_id)
    latest = await validations.latest_brief_outcome(db, tenant_id, creative_id, version)
    latest_run_id = None if latest is None else latest.validation_run_id
    if policy is None:
        return _ValidationStatus(EligibilityReason.NO_ACTIVE_POLICY, None, latest_run_id)
    if latest is None or latest.policy_bundle_version != policy.version:
        return _ValidationStatus(EligibilityReason.NOT_VALIDATED, policy.version, latest_run_id)
    return _ValidationStatus(_REASON_BY_OUTCOME[latest.outcome], policy.version, latest_run_id)
