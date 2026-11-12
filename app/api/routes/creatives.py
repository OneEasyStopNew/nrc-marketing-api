# Creative drafts.

import re
from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, Query, Response, status

from app.api.deps import (
    CreativeReader,
    CreativeWriter,
    DbSession,
    DemoStubsDep,
    IdempotencyKey,
    IfMatch,
)
from app.core.errors import FeatureNotAvailableError, VersionRequiredError
from app.schemas.assets import ExportManifest
from app.schemas.common import ErrorBody
from app.schemas.creatives import CreativeCreate, CreativeRead, CreativeUpdate, CreativeVersionPage
from app.schemas.policies import GenerationEligibility
from app.services import creatives, eligibility

router = APIRouter(prefix="/creatives", tags=["creatives"])

_IF_MATCH = re.compile(r'"([1-9][0-9]{0,8})"')


def _etag(version: int) -> str:
    return f'"{version}"'


def _expected_version(if_match: str | None) -> int:
    if if_match is None:
        raise VersionRequiredError()
    match = _IF_MATCH.fullmatch(if_match.strip())
    if match is None:
        raise VersionRequiredError('If-Match must be one quoted version number, e.g. "3".')
    return int(match.group(1))


@router.post(
    "",
    status_code=status.HTTP_201_CREATED,
    summary="Create a draft creative",
    description="Opening the composer must not call this; create only on an explicit save. "
    "A retry with the same Idempotency-Key and body returns the original creative.",
    responses={
        404: {"model": ErrorBody, "description": "Study not found or not permitted."},
        409: {"model": ErrorBody, "description": "Study closed, or Idempotency-Key reused."},
    },
)
async def create_creative(
    body: CreativeCreate,
    ctx: CreativeWriter,
    db: DbSession,
    idempotency_key: IdempotencyKey,
    response: Response,
) -> CreativeRead:
    creative, replayed = await creatives.create_creative(db, ctx, body, idempotency_key)
    response.headers["ETag"] = _etag(creative.version)
    response.headers["Location"] = f"/v1/marketing/creatives/{creative.id}"
    if replayed:
        response.headers["Idempotent-Replayed"] = "true"
    return creative


@router.get(
    "/{creative_id}",
    summary="Current version of a creative",
    responses={404: {"model": ErrorBody}},
)
async def get_creative(
    creative_id: UUID, ctx: CreativeReader, db: DbSession, response: Response
) -> CreativeRead:
    creative = await creatives.get_creative(db, ctx, creative_id)
    response.headers["ETag"] = _etag(creative.version)
    return creative


@router.patch(
    "/{creative_id}",
    summary="Save an edit as a new version",
    description='Requires If-Match: "<version you edited>". A stale version returns 412 with '
    "details.current_version; the original prompt is never changed.",
    responses={
        404: {"model": ErrorBody},
        412: {"model": ErrorBody, "description": "Someone saved a newer version first."},
        428: {"model": ErrorBody, "description": "If-Match missing or malformed."},
    },
)
async def update_creative(
    creative_id: UUID,
    body: CreativeUpdate,
    ctx: CreativeWriter,
    db: DbSession,
    response: Response,
    if_match: IfMatch = None,
) -> CreativeRead:
    expected = _expected_version(if_match)
    creative = await creatives.update_creative(db, ctx, creative_id, expected, body)
    response.headers["ETag"] = _etag(creative.version)
    return creative


@router.get(
    "/{creative_id}/versions",
    summary="Version history, newest first",
    responses={404: {"model": ErrorBody}},
)
async def list_creative_versions(
    creative_id: UUID,
    ctx: CreativeReader,
    db: DbSession,
    cursor: Annotated[
        int | None, Query(ge=1, description="next_cursor from the last page.")
    ] = None,
    limit: Annotated[int, Query(ge=1, le=50)] = 20,
) -> CreativeVersionPage:
    return await creatives.list_versions(db, ctx, creative_id, before=cursor, limit=limit)


@router.get(
    "/{creative_id}/generation-eligibility",
    summary="Whether image generation may start for a version",
    description="Defaults to the current version. Generation is refused with 409 "
    "GENERATION_NOT_ELIGIBLE unless this returns eligible: true.",
    responses={404: {"model": ErrorBody}},
)
async def get_generation_eligibility(
    creative_id: UUID,
    ctx: CreativeReader,
    db: DbSession,
    version: Annotated[int | None, Query(ge=1, description="Defaults to the current one.")] = None,
) -> GenerationEligibility:
    return await eligibility.check_generation_eligibility(db, ctx, creative_id, version)


@router.get(
    "/{creative_id}/export",
    summary="Download the review draft (D5)",
    responses={
        404: {"model": ErrorBody},
        409: {"model": ErrorBody, "description": "EXPORT_NOT_READY: no image for this version."},
        501: {"model": ErrorBody, "description": "Not implemented yet (DEMO_STUBS is off)."},
    },
)
async def export_creative(
    creative_id: UUID, ctx: CreativeReader, db: DbSession, stubs: DemoStubsDep
) -> ExportManifest:
    if stubs is None:
        raise FeatureNotAvailableError("Draft export is not implemented yet.")
    return await stubs.export(db, ctx, creative_id)
