# Whether a creative version may be sent for image generation.

from dataclasses import dataclass
from typing import Final, NoReturn
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.auth.access import TenantContext
from app.core.errors import GenerationNotEligibleError, VersionConflictError
from app.models import StudySnapshot
from app.models.enums import FORMAT_SPECS, PostFormat, RecruitmentStatus
from app.services import creatives

# Seam owned by Dinesh (NRC-7). While True, GENERATION_MODE=live refuses to start, so no paid
# call can run on a brief that has not passed validation.
VALIDATION_GATE_IS_STUB: Final = True


@dataclass(frozen=True, slots=True)
class EligibleCreative:
    # The study snapshot the content was written against, recorded on the job.

    study_snapshot_id: UUID
    study_snapshot_version: int
    post_format: PostFormat


async def assert_generation_eligible(
    db: AsyncSession, ctx: TenantContext, creative_id: UUID, creative_version: int
) -> EligibleCreative:
    """Check every precondition for generating an image of this exact creative version.

    Raises:
        NotFoundError: no such creative visible to the caller (tenant and study access).
        VersionConflictError: creative_version is not the current one (details.current_version).
        GenerationNotEligibleError: details.reason names the failed check.
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
    if not await _validation_passed(db, ctx, creative_id, creative_version):
        _not_eligible("VALIDATION_NOT_PASSED")
    return EligibleCreative(
        creative.study_snapshot_id, creative.study_snapshot_version, creative.post_format
    )


async def _validation_passed(
    db: AsyncSession, ctx: TenantContext, creative_id: UUID, creative_version: int
) -> bool:
    # STUB, owned by Dinesh (NRC-7): replace with the validation-PASS lookup for this exact
    # version and set VALIDATION_GATE_IS_STUB to False in the same change.
    return True


def _not_eligible(reason: str) -> NoReturn:
    raise GenerationNotEligibleError(details={"reason": reason})
