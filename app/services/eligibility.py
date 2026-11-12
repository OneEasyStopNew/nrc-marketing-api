"""Generation eligibility: may image generation start for this creative version?

Two entry points share the same validation rule:

- check_generation_eligibility answers the UI, so the Generate button can say why it is disabled.
- assert_generation_eligible is the gate job admission (app.services.jobs) runs before a job is
  queued; it raises instead of answering.

The latest stored brief validation of the version decides. A run checked against an older policy
version no longer counts, so a new policy makes every brief validate again.
"""

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Final, NoReturn
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.auth.access import TenantContext
from app.core.errors import GenerationNotEligibleError, NotFoundError, VersionConflictError
from app.models import Creative, StudySnapshot
from app.models.enums import FORMAT_SPECS, PostFormat, RecruitmentStatus
from app.schemas.policies import EligibilityReason, GenerationEligibility, PolicySummary
from app.schemas.validations import Outcome
from app.services import creatives, policies, validations
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
class EligibleCreative:
    # The study snapshot the content was written against, recorded on the job.

    study_snapshot_id: UUID
    study_snapshot_version: int
    post_format: PostFormat


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
    policy = await policies.get_active_policy_summary(db, ctx)
    reason = await _blocking_reason(db, ctx, creative.study_id, version, creative.current_version)
    run_id = None
    if reason is None:
        if policy is None:
            reason = EligibilityReason.NO_ACTIVE_POLICY
        else:
            reason, run_id = await _validation_reason(db, ctx, creative_id, version, policy)

    return GenerationEligibility(
        creative_id=creative_id,
        creative_version=version,
        eligible=reason is EligibilityReason.ELIGIBLE,
        reason=reason,
        message=REASON_MESSAGES[reason],
        policy_bundle_version=None if policy is None else policy.version,
        validation_run_id=run_id,
    )


async def assert_generation_eligible(
    db: AsyncSession, ctx: TenantContext, creative_id: UUID, creative_version: int
) -> EligibleCreative:
    """Check every precondition for generating an image of this exact creative version.

    Raises:
        NotFoundError: no such creative visible to the caller (tenant and study access).
        VersionConflictError: creative_version is not the current one (details.current_version).
        GenerationNotEligibleError: details.reason names the failed check: STUDY_NOT_RECRUITING,
            LANGUAGE_NOT_ALLOWED, FORMAT_NOT_ALLOWED, or one of the EligibilityReason values
            NO_ACTIVE_POLICY, NOT_VALIDATED, VALIDATION_FAILED and NEEDS_REVIEW.
    """
    creative = await creatives.get_creative(db, ctx, creative_id)
    if creative.version != creative_version:
        raise VersionConflictError(details={"current_version": creative.version})

    # The latest snapshot decides: a study closed after the draft was written is not eligible.
    latest = (
        await db.execute(
            select(StudySnapshot.recruitment_status, StudySnapshot.language)
            .where(
                StudySnapshot.tenant_id == ctx.tenant_id,
                StudySnapshot.study_id == creative.study_id,
            )
            .order_by(StudySnapshot.version.desc())
            .limit(1)
        )
    ).one()
    if latest.recruitment_status is not RecruitmentStatus.RECRUITING:
        _not_eligible("STUDY_NOT_RECRUITING")
    if creative.language != latest.language:
        _not_eligible("LANGUAGE_NOT_ALLOWED")
    spec = FORMAT_SPECS.get(creative.post_format)
    if spec is None or spec.channel != creative.channel:
        _not_eligible("FORMAT_NOT_ALLOWED")

    policy = await policies.get_active_policy_summary(db, ctx)
    if policy is None:
        _not_eligible(EligibilityReason.NO_ACTIVE_POLICY)
    reason, _ = await _validation_reason(db, ctx, creative_id, creative_version, policy)
    if reason is not EligibilityReason.ELIGIBLE:
        _not_eligible(reason)
    return EligibleCreative(
        creative.study_snapshot_id, creative.study_snapshot_version, creative.post_format
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


async def _validation_reason(
    db: AsyncSession, ctx: TenantContext, creative_id: UUID, version: int, policy: PolicySummary
) -> tuple[EligibilityReason, UUID | None]:
    # The run ID is reported only when it is what makes the version eligible.
    latest = await validations.latest_brief_outcome(db, ctx, creative_id, version)
    if latest is None or latest.policy_bundle_version != policy.version:
        return EligibilityReason.NOT_VALIDATED, None
    reason = _REASON_BY_OUTCOME[latest.outcome]
    return reason, latest.validation_run_id if reason is EligibilityReason.ELIGIBLE else None


def _not_eligible(reason: str) -> NoReturn:
    raise GenerationNotEligibleError(details={"reason": str(reason)})
