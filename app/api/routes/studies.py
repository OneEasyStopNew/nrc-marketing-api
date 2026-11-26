# Studies, their approved snapshots, and the formats and limits drafts follow.

from typing import Annotated, Any, Final
from uuid import UUID

from fastapi import APIRouter, Query

from app.api.deps import CreativeReader, DbSession, TenantCtx
from app.schemas.common import ErrorBody
from app.schemas.studies import PlatformCapabilities, StudyPage, StudyRead, StudySnapshotPage
from app.services import studies

router = APIRouter(tags=["studies"])

_NOT_FOUND: Final[dict[int | str, dict[str, Any]]] = {
    404: {"model": ErrorBody, "description": "Study not found or not permitted."}
}


@router.get(
    "/studies",
    summary="Studies the caller may use, by title",
    description="Each study says whether it accepts new drafts (availability) and carries its "
    "latest approved snapshot. Closed studies are listed so their drafts stay reachable.",
)
async def list_studies(
    ctx: CreativeReader,
    db: DbSession,
    cursor: Annotated[UUID | None, Query(description="next_cursor from the last page.")] = None,
    limit: Annotated[int, Query(ge=1, le=100)] = 50,
) -> StudyPage:
    return await studies.list_studies(db, ctx, after=cursor, limit=limit)


@router.get(
    "/studies/{study_id}", summary="One study and its latest snapshot", responses=_NOT_FOUND
)
async def get_study(study_id: UUID, ctx: CreativeReader, db: DbSession) -> StudyRead:
    return await studies.get_study(db, ctx, study_id)


@router.get(
    "/studies/{study_id}/snapshots",
    summary="Approved snapshots of a study, newest first",
    description="Snapshots never change. A draft keeps the snapshot it was created with, so a "
    "newer snapshot does not change existing drafts, validation runs or jobs.",
    responses=_NOT_FOUND,
)
async def list_study_snapshots(
    study_id: UUID,
    ctx: CreativeReader,
    db: DbSession,
    cursor: Annotated[
        int | None, Query(ge=1, description="next_cursor from the last page.")
    ] = None,
    limit: Annotated[int, Query(ge=1, le=50)] = 20,
) -> StudySnapshotPage:
    return await studies.list_snapshots(db, ctx, study_id, before=cursor, limit=limit)


@router.get("/capabilities", summary="Languages, post formats and text limits for drafts")
async def get_capabilities(ctx: TenantCtx) -> PlatformCapabilities:
    # Same for every tenant today; tenant-scoped so per-tenant formats can follow.
    return studies.platform_capabilities()
