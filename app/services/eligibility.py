"""Generation eligibility: may image generation start for this creative version?

The generation job (NRC-6/NRC-10) calls check_generation_eligibility before it submits anything
to a provider, and raises GenerationNotEligibleError when the result is not eligible. The same
result is exposed to the UI so the Generate button can explain why it is disabled.

The latest finished brief validation of the version decides the rest. Until D2 stores real
runs, a BriefValidationLookup supplies them (the demo stubs); without one, the best possible
answer is NOT_VALIDATED.
"""

from collections.abc import Mapping
from types import MappingProxyType
from typing import Final, Protocol
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.auth.access import TenantContext
from app.core.errors import NotFoundError
from app.models import Creative, StudySnapshot
from app.models.enums import RecruitmentStatus
from app.schemas.policies import EligibilityReason, GenerationEligibility
from app.schemas.validations import Outcome
from app.services import policies
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


class BriefValidationLookup(Protocol):
    def latest_brief_outcome(
        self, tenant_id: UUID, creative_id: UUID, version: int
    ) -> tuple[UUID, Outcome] | None:
        """Latest finished brief validation run of the version and its outcome, if any."""
        ...


async def check_generation_eligibility(
    db: AsyncSession,
    ctx: TenantContext,
    creative_id: UUID,
    creative_version: int | None = None,
    *,
    brief_runs: BriefValidationLookup | None = None,
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
    if reason is None and policy is None:
        reason = EligibilityReason.NO_ACTIVE_POLICY
    run_id = None
    if reason is None:
        reason, run_id = _validation_reason(brief_runs, ctx.tenant_id, creative_id, version)

    return GenerationEligibility(
        creative_id=creative_id,
        creative_version=version,
        eligible=reason is EligibilityReason.ELIGIBLE,
        reason=reason,
        message=REASON_MESSAGES[reason],
        policy_bundle_version=None if policy is None else policy.version,
        validation_run_id=run_id,
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


def _validation_reason(
    brief_runs: BriefValidationLookup | None, tenant_id: UUID, creative_id: UUID, version: int
) -> tuple[EligibilityReason, UUID | None]:
    # The run ID is reported only when it is what makes the version eligible.
    latest = (
        None
        if brief_runs is None
        else brief_runs.latest_brief_outcome(tenant_id, creative_id, version)
    )
    if latest is None:
        return EligibilityReason.NOT_VALIDATED, None
    run_id, outcome = latest
    reason = _REASON_BY_OUTCOME[outcome]
    return reason, run_id if reason is EligibilityReason.ELIGIBLE else None
