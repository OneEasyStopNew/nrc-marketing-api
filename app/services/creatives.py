# Creative drafts: explicit creation, immutable versions and optimistic concurrency.

from collections.abc import Mapping
from typing import Any, Final
from uuid import UUID, uuid4

from sqlalchemy import ColumnElement, ScalarSelect, and_, func, select, update
from sqlalchemy.ext.asyncio import AsyncSession

from app.auth.access import TenantContext
from app.core.errors import (
    FieldValidationError,
    NotFoundError,
    StudyClosedError,
    StudySnapshotMissingError,
    VersionConflictError,
)
from app.models import Asset, Creative, CreativeVersion, Membership, StudySnapshot
from app.models.enums import BriefField, RecruitmentStatus
from app.schemas.creatives import (
    CreativeCreate,
    CreativeRead,
    CreativeUpdate,
    CreativeVersionPage,
    CreativeVersionRead,
)
from app.services import briefs, idempotency

CREATE_ROUTE: Final = "POST /creatives"
# Columns of creative_versions that hold the brief fields, named like the API fields.
BRIEF_COLUMNS: Final = (
    CreativeVersion.headline,
    CreativeVersion.call_to_action,
    CreativeVersion.destination_url,
    CreativeVersion.site,
    CreativeVersion.audience,
    CreativeVersion.visual_instructions,
)


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
    brief = _brief_values(data.model_dump(mode="json", include=set(_BRIEF_NAMES)))
    briefs.check_site(brief["site"], snapshot.facts)
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
            image_text=data.image_text,
            **brief,
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
            CreativeVersion.image_text,
            *BRIEF_COLUMNS,
            CreativeVersion.language,
            CreativeVersion.channel,
            CreativeVersion.post_format,
            CreativeVersion.study_snapshot_id,
            StudySnapshot.version.label("study_snapshot_version"),
            _current_asset_id().label("asset_id"),
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
    creative = CreativeRead.model_validate(row)
    values = {field: getattr(creative, field.value) for field in BriefField}
    creative.brief_issues = briefs.brief_issues(creative.post_format, values)
    return creative


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
    fields = changes.model_dump(mode="json", exclude_unset=True)
    brief = _brief_values({name: getattr(base, name) for name in _BRIEF_NAMES} | fields)
    if "site" in fields:
        # Against the snapshot this draft is pinned to, not the study's latest one.
        pinned = (
            await db.execute(
                select(StudySnapshot.facts).where(StudySnapshot.id == base.study_snapshot_id)
            )
        ).scalar_one()
        try:
            briefs.check_site(brief["site"], pinned)
        except FieldValidationError:
            await db.rollback()  # undo the version bump above
            raise
    db.add(
        CreativeVersion(
            id=uuid4(),
            tenant_id=ctx.tenant_id,
            creative_id=creative_id,
            version=new_version,
            prompt=fields.get("current_prompt", base.prompt),
            caption=fields.get("caption", base.caption),
            image_text=fields.get("image_text", base.image_text),
            **brief,
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
            CreativeVersion.image_text,
            *BRIEF_COLUMNS,
            CreativeVersion.study_snapshot_id,
            CreativeVersion.language,
            CreativeVersion.channel,
            CreativeVersion.post_format,
            CreativeVersion.created_by,
            Membership.display_name.label("created_by_name"),
            CreativeVersion.created_at,
        )
        # Outer join: an author who has since left the tenant still has their versions listed.
        .outerjoin(
            Membership,
            and_(
                Membership.tenant_id == CreativeVersion.tenant_id,
                Membership.user_sub == CreativeVersion.created_by,
            ),
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


def _current_asset_id() -> ScalarSelect[UUID]:
    # Newest image generated for the creative's current version; an edit makes it NULL again.
    # Correlated per creative row, so a list built on the same select stays one query.
    return (
        select(Asset.id)
        .where(
            Asset.tenant_id == Creative.tenant_id,
            Asset.creative_id == Creative.id,
            Asset.creative_version == Creative.current_version,
        )
        .order_by(Asset.created_at.desc(), Asset.id.desc())
        .limit(1)
        .correlate(Creative)
        .scalar_subquery()
    )


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


_BRIEF_NAMES: Final = frozenset(field.value for field in BriefField)


def _brief_values(values: Mapping[str, Any]) -> dict[str, Any]:
    # The brief fields of a new version, keyed by column name; missing ones are None.
    return {name: values.get(name) for name in sorted(_BRIEF_NAMES)}
