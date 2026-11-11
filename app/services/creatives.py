# Creative drafts: explicit creation, immutable versions and optimistic concurrency.

from typing import Final
from uuid import UUID, uuid4

from sqlalchemy import ColumnElement, and_, func, select, update
from sqlalchemy.ext.asyncio import AsyncSession

from app.auth.access import TenantContext
from app.core.errors import (
    NotFoundError,
    StudyClosedError,
    StudySnapshotMissingError,
    VersionConflictError,
)
from app.models import Creative, CreativeVersion, StudySnapshot
from app.models.enums import RecruitmentStatus
from app.schemas.creatives import (
    CreativeCreate,
    CreativeRead,
    CreativeUpdate,
    CreativeVersionPage,
    CreativeVersionRead,
)
from app.services import idempotency

CREATE_ROUTE: Final = "POST /creatives"


async def create_creative(
    db: AsyncSession, ctx: TenantContext, data: CreativeCreate, idempotency_key: str
) -> tuple[CreativeRead, bool]:
    """Create a draft at version 1. Returns the creative and whether this was a replay.

    Raises:
        NotFoundError: the study does not exist or the caller has no access to it.
        StudyClosedError / StudySnapshotMissingError: the study cannot start new creatives.
        IdempotencyKeyReusedError: the key was used before with a different payload.
    """
    fingerprint = idempotency.request_hash(ctx.principal.sub, data)
    existing_id = await idempotency.claim(
        db, ctx.tenant_id, CREATE_ROUTE, idempotency_key, fingerprint
    )
    if existing_id is not None:
        return await get_creative(db, ctx, existing_id), True

    snapshot = await _latest_open_snapshot(db, ctx, data.study_id)
    creative_id = uuid4()
    db.add(
        Creative(
            id=creative_id,
            tenant_id=ctx.tenant_id,
            study_id=data.study_id,
            original_prompt=data.prompt,
            current_version=1,
            created_by=ctx.principal.sub,
        )
    )
    # Flush the parent first: there is no ORM relationship to order the two inserts.
    await db.flush()
    db.add(
        CreativeVersion(
            id=uuid4(),
            tenant_id=ctx.tenant_id,
            creative_id=creative_id,
            version=1,
            prompt=data.prompt,
            caption=data.caption,
            study_snapshot_id=snapshot.id,
            language=data.language,
            channel=data.channel,
            post_format=data.post_format,
            created_by=ctx.principal.sub,
        )
    )
    await idempotency.complete(db, ctx.tenant_id, CREATE_ROUTE, idempotency_key, creative_id)
    await db.commit()
    return await get_creative(db, ctx, creative_id), False


async def get_creative(db: AsyncSession, ctx: TenantContext, creative_id: UUID) -> CreativeRead:
    current_view = (
        select(
            Creative.id,
            Creative.study_id,
            Creative.current_version.label("version"),
            Creative.original_prompt,
            CreativeVersion.prompt.label("current_prompt"),
            CreativeVersion.caption,
            CreativeVersion.language,
            CreativeVersion.channel,
            CreativeVersion.post_format,
            CreativeVersion.study_snapshot_id,
            StudySnapshot.version.label("study_snapshot_version"),
            Creative.created_by,
            Creative.created_at,
            Creative.updated_at,
        )
        .join(
            CreativeVersion,
            and_(
                CreativeVersion.creative_id == Creative.id,
                CreativeVersion.version == Creative.current_version,
            ),
        )
        .join(StudySnapshot, StudySnapshot.id == CreativeVersion.study_snapshot_id)
        .where(*visible_to(ctx), Creative.id == creative_id)
    )
    row = (await db.execute(current_view)).one_or_none()
    if row is None:
        raise NotFoundError("Creative not found.")
    return CreativeRead.model_validate(row)


async def update_creative(
    db: AsyncSession,
    ctx: TenantContext,
    creative_id: UUID,
    expected_version: int,
    changes: CreativeUpdate,
) -> CreativeRead:
    """Save the changes as a new immutable version on top of expected_version.

    Raises:
        NotFoundError: no such creative visible to the caller.
        VersionConflictError: someone saved a newer version first (details has current_version).
    """
    new_version = await db.scalar(
        update(Creative)
        .where(
            *visible_to(ctx),
            Creative.id == creative_id,
            Creative.current_version == expected_version,
        )
        .values(current_version=Creative.current_version + 1, updated_at=func.now())
        .returning(Creative.current_version)
        .execution_options(synchronize_session=False)
    )
    if new_version is None:
        current = await db.scalar(
            select(Creative.current_version).where(*visible_to(ctx), Creative.id == creative_id)
        )
        if current is None:
            raise NotFoundError("Creative not found.")
        raise VersionConflictError(details={"current_version": current})

    base = (
        await db.execute(
            select(CreativeVersion).where(
                CreativeVersion.creative_id == creative_id,
                CreativeVersion.version == expected_version,
            )
        )
    ).scalar_one()
    fields = changes.model_dump(exclude_unset=True)
    db.add(
        CreativeVersion(
            id=uuid4(),
            tenant_id=ctx.tenant_id,
            creative_id=creative_id,
            version=new_version,
            prompt=fields.get("current_prompt", base.prompt),
            caption=fields.get("caption", base.caption),
            study_snapshot_id=base.study_snapshot_id,
            language=base.language,
            channel=base.channel,
            post_format=base.post_format,
            created_by=ctx.principal.sub,
        )
    )
    await db.commit()
    return await get_creative(db, ctx, creative_id)


async def list_versions(
    db: AsyncSession, ctx: TenantContext, creative_id: UUID, *, before: int | None, limit: int
) -> CreativeVersionPage:
    # Newest first. Pass the returned next_cursor as `before` to page back through history.
    exists = await db.scalar(
        select(Creative.id).where(*visible_to(ctx), Creative.id == creative_id)
    )
    if exists is None:
        raise NotFoundError("Creative not found.")

    query = (
        select(
            CreativeVersion.version,
            CreativeVersion.prompt.label("current_prompt"),
            CreativeVersion.caption,
            CreativeVersion.study_snapshot_id,
            CreativeVersion.language,
            CreativeVersion.channel,
            CreativeVersion.post_format,
            CreativeVersion.created_by,
            CreativeVersion.created_at,
        )
        .where(CreativeVersion.creative_id == creative_id)
        .order_by(CreativeVersion.version.desc())
        .limit(limit + 1)
    )
    if before is not None:
        query = query.where(CreativeVersion.version < before)
    rows = (await db.execute(query)).all()
    items = [CreativeVersionRead.model_validate(row) for row in rows[:limit]]
    next_cursor = items[-1].version if len(rows) > limit else None
    return CreativeVersionPage(items=items, next_cursor=next_cursor)


def visible_to(ctx: TenantContext) -> tuple[ColumnElement[bool], ...]:
    # RLS already hides other tenants; filtering here too keeps the query correct without it.
    return (Creative.tenant_id == ctx.tenant_id, Creative.study_id.in_(ctx.study_ids))


async def _latest_open_snapshot(
    db: AsyncSession, ctx: TenantContext, study_id: UUID
) -> StudySnapshot:
    if study_id not in ctx.study_ids:
        raise NotFoundError("Study not found.")
    snapshot = await db.scalar(
        select(StudySnapshot)
        .where(StudySnapshot.tenant_id == ctx.tenant_id, StudySnapshot.study_id == study_id)
        .order_by(StudySnapshot.version.desc())
        .limit(1)
    )
    if snapshot is None:
        raise StudySnapshotMissingError()
    if snapshot.recruitment_status is RecruitmentStatus.CLOSED:
        raise StudyClosedError()
    return snapshot
