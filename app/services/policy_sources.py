# Policy sources (NRC-4): register a READY policy upload as a source, and read sources and their
# clauses. Clause extraction runs in the policy worker (app.workers.policy), not here.

from uuid import UUID

from sqlalchemy import func, or_, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from app.auth.access import TenantContext
from app.core.errors import NotFoundError, PolicySourceUploadNotReadyError
from app.models import MediaUpload, PolicySource, PolicySourceClause
from app.models.enums import PolicyLanguage, SourceExtractionState, UploadPurpose, UploadState


async def register_source(
    db: AsyncSession,
    ctx: TenantContext,
    *,
    upload_id: UUID,
    title: str,
    source_ref: str,
    language: PolicyLanguage,
    is_synthetic: bool,
) -> tuple[PolicySource, bool]:
    """Register a READY policy upload as a source; returns (source, created).

    A source with the same checksum or the same upload already in the tenant is returned
    unchanged (created=False): re-importing a file keeps its provenance (AC3). Does not commit.
    """
    upload = await db.scalar(
        select(MediaUpload).where(
            MediaUpload.tenant_id == ctx.tenant_id,
            MediaUpload.purpose == UploadPurpose.POLICY_SOURCE,
            MediaUpload.id == upload_id,
        )
    )
    if upload is None:
        raise NotFoundError("Upload not found.")
    if upload.state is not UploadState.READY:
        raise PolicySourceUploadNotReadyError(details={"status": upload.state.value})
    existing = await _existing(db, ctx, upload)
    if existing is not None:
        if existing.extraction_state is SourceExtractionState.FAILED:
            _retry_from(existing, upload)
            await db.flush()
        return existing, False
    source = PolicySource(
        tenant_id=ctx.tenant_id,
        title=title,
        source_ref=source_ref,
        checksum_sha256=upload.declared_checksum_sha256,
        storage_key=upload.s3_key,
        language=language,
        is_synthetic=is_synthetic,
        created_by=ctx.principal.sub,
        upload_id=upload.id,
        extraction_state=SourceExtractionState.PENDING,
        available_at=func.now(),
    )
    try:
        # A savepoint: a concurrent duplicate loses the unique constraint without ending the
        # caller's transaction.
        async with db.begin_nested():
            db.add(source)
            await db.flush()
    except IntegrityError:
        existing = await _existing(db, ctx, upload)
        if existing is None:
            raise
        return existing, False
    return source, True


def _retry_from(source: PolicySource, upload: MediaUpload) -> None:
    # Registering the file again is the way to retry a failed extraction (after an outage or a
    # configuration fix); otherwise the checksum would tie the file to a dead source forever.
    # The bytes are the same (verified checksum), so title and provenance stay as they were.
    source.upload_id = upload.id
    source.storage_key = upload.s3_key
    source.extraction_state = SourceExtractionState.PENDING
    source.extraction_error_code = None
    source.available_at = func.now()
    source.leased_until = None
    source.lease_owner = None
    source.attempts = 0
    source.failed_attempts = 0


async def get_source(db: AsyncSession, ctx: TenantContext, source_id: UUID) -> PolicySource:
    source = await db.scalar(
        select(PolicySource).where(
            PolicySource.tenant_id == ctx.tenant_id, PolicySource.id == source_id
        )
    )
    if source is None:
        raise NotFoundError("Policy source not found.")
    return source


async def list_source_clauses(
    db: AsyncSession, ctx: TenantContext, source_id: UUID
) -> list[PolicySourceClause]:
    await get_source(db, ctx, source_id)
    result = await db.scalars(
        select(PolicySourceClause)
        .where(
            PolicySourceClause.tenant_id == ctx.tenant_id,
            PolicySourceClause.source_id == source_id,
        )
        # Refs number clauses in reading order (p1-c9 before p1-c10): shorter first, then text.
        .order_by(
            PolicySourceClause.page.asc().nulls_last(),
            func.length(PolicySourceClause.clause_ref),
            PolicySourceClause.clause_ref,
        )
    )
    return list(result)


async def _existing(
    db: AsyncSession, ctx: TenantContext, upload: MediaUpload
) -> PolicySource | None:
    return await db.scalar(
        select(PolicySource)
        .where(
            PolicySource.tenant_id == ctx.tenant_id,
            or_(
                PolicySource.checksum_sha256 == upload.declared_checksum_sha256,
                PolicySource.upload_id == upload.id,
            ),
        )
        .limit(1)
    )
