# Studies a member may use, their approved snapshots, and the formats and limits drafts follow.

from uuid import UUID

from sqlalchemy import select, tuple_
from sqlalchemy.dialects.postgresql import distinct_on
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import aliased

from app.auth.access import TenantContext
from app.core.errors import NotFoundError
from app.models import Study, StudySnapshot
from app.models.enums import FORMAT_SPECS, BriefField, CallToAction, Language, RecruitmentStatus
from app.schemas.common import (
    MAX_AUDIENCE_CHARS,
    MAX_CAPTION_CHARS,
    MAX_DESTINATION_URL_CHARS,
    MAX_GENERATION_TEXT_CHARS,
    MAX_HEADLINE_CHARS,
    MAX_IMAGE_TEXT_CHARS,
    MAX_IMAGE_TEXT_LINES,
    MAX_PROMPT_CHARS,
    MAX_VISUAL_INSTRUCTIONS_CHARS,
)
from app.schemas.studies import (
    PlatformCapabilities,
    StudyAvailability,
    StudyPage,
    StudyRead,
    StudySnapshotPage,
    TextLimits,
)
from app.schemas.tenancy import PostFormatOption, StudyFacts, StudySnapshotRead

STUDY_NOT_FOUND = "Study not found."


async def list_studies(
    db: AsyncSession, ctx: TenantContext, *, after: UUID | None, limit: int
) -> StudyPage:
    # Permitted studies by title, each with its latest approved snapshot (or none yet).
    latest = (
        select(StudySnapshot)
        .where(StudySnapshot.tenant_id == ctx.tenant_id)
        .ext(distinct_on(StudySnapshot.study_id))
        .order_by(StudySnapshot.study_id, StudySnapshot.version.desc())
        .subquery()
    )
    snapshot = aliased(StudySnapshot, latest)
    query = (
        select(Study, snapshot)
        .outerjoin(snapshot, snapshot.study_id == Study.id)
        .where(Study.tenant_id == ctx.tenant_id, Study.id.in_(ctx.study_ids))
        .order_by(Study.title, Study.id)
        .limit(limit + 1)
    )
    if after is not None:
        # Keyset on (title, id). A cursor the caller cannot see gives an empty page.
        if after not in ctx.study_ids:
            return StudyPage(items=[], next_cursor=None)
        cursor_title = await db.scalar(
            select(Study.title).where(Study.tenant_id == ctx.tenant_id, Study.id == after)
        )
        if cursor_title is None:
            return StudyPage(items=[], next_cursor=None)
        query = query.where(tuple_(Study.title, Study.id) > tuple_(cursor_title, after))
    rows = (await db.execute(query)).all()
    items = [_study_read(study, latest_snapshot) for study, latest_snapshot in rows[:limit]]
    next_cursor = items[-1].study_id if len(rows) > limit else None
    return StudyPage(items=items, next_cursor=next_cursor)


async def get_study(db: AsyncSession, ctx: TenantContext, study_id: UUID) -> StudyRead:
    study = await _visible_study(db, ctx, study_id)
    latest = await db.scalar(
        select(StudySnapshot)
        .where(StudySnapshot.tenant_id == ctx.tenant_id, StudySnapshot.study_id == study_id)
        .order_by(StudySnapshot.version.desc())
        .limit(1)
    )
    return _study_read(study, latest)


async def list_snapshots(
    db: AsyncSession, ctx: TenantContext, study_id: UUID, *, before: int | None, limit: int
) -> StudySnapshotPage:
    # Newest first. Snapshots are immutable, so older pages never change.
    study = await _visible_study(db, ctx, study_id)
    query = (
        select(StudySnapshot)
        .where(StudySnapshot.tenant_id == ctx.tenant_id, StudySnapshot.study_id == study_id)
        .order_by(StudySnapshot.version.desc())
        .limit(limit + 1)
    )
    if before is not None:
        query = query.where(StudySnapshot.version < before)
    snapshots = (await db.scalars(query)).all()
    items = [snapshot_read(s, study.title, study.external_ref) for s in snapshots[:limit]]
    next_cursor = items[-1].version if len(snapshots) > limit else None
    return StudySnapshotPage(items=items, next_cursor=next_cursor)


def platform_capabilities() -> PlatformCapabilities:
    return PlatformCapabilities(
        languages=list(Language),
        post_formats=post_format_options(),
        call_to_actions=list(CallToAction),
        text_limits=TextLimits(
            prompt_max_chars=MAX_PROMPT_CHARS,
            caption_max_chars=MAX_CAPTION_CHARS,
            image_text_max_chars=MAX_IMAGE_TEXT_CHARS,
            image_text_max_lines=MAX_IMAGE_TEXT_LINES,
            headline_max_chars=MAX_HEADLINE_CHARS,
            audience_max_chars=MAX_AUDIENCE_CHARS,
            visual_instructions_max_chars=MAX_VISUAL_INSTRUCTIONS_CHARS,
            destination_url_max_chars=MAX_DESTINATION_URL_CHARS,
            generation_text_max_chars=MAX_GENERATION_TEXT_CHARS,
        ),
    )


def post_format_options() -> list[PostFormatOption]:
    return [
        PostFormatOption(
            channel=spec.channel,
            post_format=post_format,
            aspect_ratio=spec.aspect_ratio,
            width_px=spec.width_px,
            height_px=spec.height_px,
            link_fields=[field for field in BriefField if field in spec.link_fields],
        )
        for post_format, spec in FORMAT_SPECS.items()
    ]


def snapshot_read(snapshot: StudySnapshot, title: str, ref: str | None) -> StudySnapshotRead:
    return StudySnapshotRead(
        study_id=snapshot.study_id,
        study_title=title,
        study_external_ref=ref,
        snapshot_id=snapshot.id,
        version=snapshot.version,
        recruitment_status=snapshot.recruitment_status,
        language=snapshot.language,
        facts=StudyFacts.model_validate(snapshot.facts),
        source_ref=snapshot.source_ref,
        approved_by=snapshot.approved_by,
        approved_at=snapshot.approved_at,
        is_synthetic=snapshot.is_synthetic,
    )


async def _visible_study(db: AsyncSession, ctx: TenantContext, study_id: UUID) -> Study:
    # Another tenant's study and one the member has no grant for look the same: not found.
    if study_id not in ctx.study_ids:
        raise NotFoundError(STUDY_NOT_FOUND)
    study = await db.scalar(
        select(Study).where(Study.tenant_id == ctx.tenant_id, Study.id == study_id)
    )
    if study is None:
        raise NotFoundError(STUDY_NOT_FOUND)
    return study


def _study_read(study: Study, latest: StudySnapshot | None) -> StudyRead:
    if latest is None:
        availability = StudyAvailability.NO_APPROVED_SNAPSHOT
    elif latest.recruitment_status is RecruitmentStatus.CLOSED:
        availability = StudyAvailability.CLOSED
    else:
        availability = StudyAvailability.RECRUITING
    return StudyRead(
        study_id=study.id,
        title=study.title,
        external_ref=study.external_ref,
        availability=availability,
        latest_snapshot=(
            snapshot_read(latest, study.title, study.external_ref) if latest is not None else None
        ),
    )
