"""Simulated image validation, generation, assets, media and export, enabled by DEMO_STUBS=true.

The frontend integrates against these until D3-D5 land. Responses use the real contract
schemas. Briefs and captions are not simulated: app.services.validations checks and stores
them, and the generation gate here reads those stored runs.

Nothing here is persisted. Records live in this process's memory, are bounded, and disappear on
restart, so run a single worker. Each check-and-insert runs without an await in between, so
concurrent requests on the one event loop cannot interleave inside it.
"""

import hashlib
import logging
import time
from collections import OrderedDict
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Final
from uuid import UUID, uuid4

from sqlalchemy.ext.asyncio import AsyncSession

from app.auth.access import TenantContext
from app.core.errors import (
    ExportNotReadyError,
    IdempotencyKeyReusedError,
    NoActivePolicyError,
    NotFoundError,
)
from app.models.enums import FORMAT_SPECS, ExecutionState
from app.schemas.assets import AssetOrigin, AssetRead, ExportManifest, MediaItem, MediaPage
from app.schemas.creatives import CreativeRead
from app.schemas.jobs import CreativeJobCreate, JobAccepted, JobKind, JobRead
from app.schemas.policies import PolicyBundleRead
from app.schemas.validations import (
    CoverageGap,
    ImageRegion,
    ValidationAccepted,
    ValidationCreate,
    ValidationInputKind,
    ValidationRunRead,
)
from app.services import (
    creatives,
    generation_checks,
    idempotency,
    policies,
    rules,
    tenancy,
    validations,
)

logger = logging.getLogger(__name__)

MAX_RECORDS: Final = 1000
DOWNLOAD_URL_TTL: Final = timedelta(minutes=15)
DEMO_FILES_PATH: Final = "/demo-files"
STUB_PROVIDER: Final = "demo-stub"
STUB_MODEL: Final = "placeholder-svg"
STUB_MIME_TYPE: Final = "image/svg+xml"
STUB_GAP: Final = CoverageGap(
    reason="Demo stub: the placeholder image has no content, so the brief stands in for it."
)
VALIDATE_ROUTE: Final = "POST /validations"
GENERATE_ROUTE: Final = "POST /creative-jobs"
_PAYMENT_REGION: Final = ImageRegion(x=0.1, y=0.72, width=0.8, height=0.16)

Clock = Callable[[], float]


@dataclass(frozen=True, slots=True)
class StubTimings:
    """How long simulated work takes, so the UI can exercise its progress and polling states."""

    validation_seconds: float = 2.0
    generation_seconds: float = 6.0


class _BoundedMap[K, V]:
    # Insertion-ordered map that forgets its oldest entries beyond `limit`.

    def __init__(self, limit: int) -> None:
        self._items: OrderedDict[K, V] = OrderedDict()
        self._limit = limit

    def get(self, key: K) -> V | None:
        return self._items.get(key)

    def values(self) -> list[V]:
        return list(self._items.values())

    def put(self, key: K, value: V) -> None:
        self._items[key] = value
        self._items.move_to_end(key)
        while len(self._items) > self._limit:
            self._items.popitem(last=False)


@dataclass(frozen=True, slots=True)
class _Replay:
    fingerprint: str
    resource_id: UUID


@dataclass(frozen=True, slots=True)
class _ValidationRecord:
    tenant_id: UUID
    creative_id: UUID
    started: float
    finished: ValidationRunRead


@dataclass(frozen=True, slots=True)
class _GenerationRecord:
    tenant_id: UUID
    study_id: UUID
    creative_id: UUID
    creative_version: int
    created_by: str
    job_id: UUID
    started: float
    created_at: datetime
    asset: AssetRead


class DemoStubs:
    """In-memory stand-in for image validation, generation, assets and export."""

    def __init__(
        self,
        public_base_url: str,
        *,
        timings: StubTimings | None = None,
        clock: Clock = time.monotonic,
    ) -> None:
        self._base_url = public_base_url.rstrip("/")
        self._timings = timings or StubTimings()
        self._clock = clock
        self._replays: _BoundedMap[tuple[UUID, str, str], _Replay] = _BoundedMap(MAX_RECORDS)
        self._runs: _BoundedMap[UUID, _ValidationRecord] = _BoundedMap(MAX_RECORDS)
        self._jobs: _BoundedMap[UUID, _GenerationRecord] = _BoundedMap(MAX_RECORDS)
        self._job_by_asset: _BoundedMap[UUID, UUID] = _BoundedMap(MAX_RECORDS)
        self._latest_jobs: _BoundedMap[tuple[UUID, UUID, int], UUID] = _BoundedMap(MAX_RECORDS)

    # Image validation

    async def start_validation(
        self, db: AsyncSession, ctx: TenantContext, body: ValidationCreate, idempotency_key: str
    ) -> tuple[ValidationAccepted, bool]:
        """Record a simulated image validation. Returns the run and whether this was a replay.

        Raises:
            NotFoundError: unknown creative or version, or no finished image for that version.
            NoActivePolicyError: the tenant has no policy to validate against.
            IdempotencyKeyReusedError: the key was used before with a different body.
        """
        if body.input_kind is not ValidationInputKind.IMAGE:
            raise ValueError("the demo stubs validate images only")
        fingerprint = idempotency.request_hash(ctx.principal.sub, body)
        replayed = self._replayed_run(ctx, idempotency_key, fingerprint)
        if replayed is not None:
            return replayed, True

        creative = await creatives.get_creative(db, ctx, body.creative_id)
        prompt = await _prompt(db, ctx, body.creative_id, body.creative_version)
        asset = self._validation_asset(ctx, body)
        policy = await _active_policy(db, ctx)
        evaluation = _evaluate_image(prompt, policy)

        # No await from here on (see the module docstring).
        replayed = self._replayed_run(ctx, idempotency_key, fingerprint)
        if replayed is not None:
            return replayed, True
        run_id = uuid4()
        finished = ValidationRunRead(
            validation_run_id=run_id,
            input_kind=body.input_kind,
            creative_version=body.creative_version,
            state=ExecutionState.SUCCEEDED,
            outcome=evaluation.outcome,
            is_current=body.creative_version == creative.version,
            evaluated_at=datetime.now(tz=UTC) + timedelta(seconds=self._timings.validation_seconds),
            job_id=uuid4(),
            input_hash=asset.checksum_sha256,
            asset_id=body.asset_id,
            study_snapshot_version=creative.study_snapshot_version,
            policy_bundle_version=policy.version,
            gates=evaluation.gates,
            rule_results=evaluation.rule_results,
            findings=evaluation.findings,
            coverage_gaps=evaluation.coverage_gaps,
        )
        record = _ValidationRecord(ctx.tenant_id, body.creative_id, self._clock(), finished)
        self._runs.put(run_id, record)
        self._replays.put(
            (ctx.tenant_id, VALIDATE_ROUTE, idempotency_key), _Replay(fingerprint, run_id)
        )
        logger.info("demo image validation %s recorded: %s", run_id, evaluation.outcome.value)
        return self._accepted_run(record), False

    async def get_validation(
        self, db: AsyncSession, ctx: TenantContext, validation_run_id: UUID
    ) -> ValidationRunRead:
        """Return the run as it looks now: RUNNING until its simulated duration has passed.

        Raises:
            NotFoundError: no such run visible to the caller.
        """
        record = self._runs.get(validation_run_id)
        if record is None or record.tenant_id != ctx.tenant_id:
            raise NotFoundError("Validation run not found.")
        try:
            # Also enforces study access: a creative the caller cannot see hides its runs.
            creative = await creatives.get_creative(db, ctx, record.creative_id)
        except NotFoundError:
            raise NotFoundError("Validation run not found.") from None
        return self._run_view(record, current_version=creative.version)

    # Generation and assets

    async def start_generation(
        self, db: AsyncSession, ctx: TenantContext, body: CreativeJobCreate, idempotency_key: str
    ) -> tuple[JobAccepted, bool]:
        """Record a generation job for an eligible version. Returns the job and whether replayed.

        The same gate as real jobs (generation_checks), so the demo answers with the same errors.

        Raises:
            NotFoundError: unknown creative.
            VersionConflictError: not the current version (412, details.current_version).
            GenerationNotEligibleError: closed study, language or format; details.reason says which.
            ValidationNotPassedError: no passing brief validation of this version against the
                active policy; details.validation_run_id and details.status.
            IdempotencyKeyReusedError: the key was used before with a different body.
        """
        fingerprint = idempotency.request_hash(ctx.principal.sub, body)
        replayed = self._replayed_job(ctx, idempotency_key, fingerprint)
        if replayed is not None:
            return replayed, True

        await generation_checks.assert_generation_eligible(
            db, ctx, body.creative_id, body.creative_version
        )
        creative = await creatives.get_creative(db, ctx, body.creative_id)

        # No await from here on (see the module docstring).
        replayed = self._replayed_job(ctx, idempotency_key, fingerprint)
        if replayed is not None:
            return replayed, True
        record = self._new_generation(ctx, creative, body.creative_version)
        self._jobs.put(record.job_id, record)
        self._job_by_asset.put(record.asset.asset_id, record.job_id)
        self._latest_jobs.put(
            (ctx.tenant_id, body.creative_id, body.creative_version), record.job_id
        )
        self._replays.put(
            (ctx.tenant_id, GENERATE_ROUTE, idempotency_key), _Replay(fingerprint, record.job_id)
        )
        logger.info("demo generation job %s recorded", record.job_id)
        return JobAccepted(job_id=record.job_id, state=self._job_view(record).state), False

    async def get_job(self, db: AsyncSession, ctx: TenantContext, job_id: UUID) -> JobRead:
        """Return the job's simulated progress: QUEUED, RUNNING, WAITING_PROVIDER, SUCCEEDED.

        Raises:
            NotFoundError: no such job visible to the caller.
        """
        return self._job_view(await self._visible_job(db, ctx, job_id, "Job not found."))

    async def get_asset(self, db: AsyncSession, ctx: TenantContext, asset_id: UUID) -> AssetRead:
        """Return a generated asset once its job has succeeded, with a placeholder image URL.

        Raises:
            NotFoundError: no such asset visible to the caller, or its job is still running.
        """
        job_id = self._job_by_asset.get(asset_id)
        if job_id is None:
            raise NotFoundError("Asset not found.")
        record = await self._visible_job(db, ctx, job_id, "Asset not found.")
        if not self._is_done(record.started, self._timings.generation_seconds):
            raise NotFoundError("Asset not found.")
        return self._asset_view(record)

    async def list_media(
        self, db: AsyncSession, ctx: TenantContext, *, before: datetime | None, limit: int
    ) -> MediaPage:
        """Finished generated images in the caller's studies, newest first.

        Pass the returned next_cursor as `before` to page back. Studies the caller cannot use
        hide their images, the same as GET /assets/{asset_id}.
        """
        visible = sorted(
            (
                record
                for record in self._jobs.values()
                if record.tenant_id == ctx.tenant_id
                and record.study_id in ctx.study_ids
                and self._is_done(record.started, self._timings.generation_seconds)
                and (before is None or record.asset.created_at < before)
            ),
            key=lambda record: record.asset.created_at,
            reverse=True,
        )
        page = visible[:limit]
        names = await tenancy.display_names(db, ctx, {record.created_by for record in page})
        items = [self._media_item(record, names.get(record.created_by)) for record in page]
        next_cursor = page[-1].asset.created_at if len(visible) > limit else None
        return MediaPage(items=items, next_cursor=next_cursor)

    def placeholder_file(self, asset_id: UUID) -> bytes | None:
        # Image bytes behind download_url. Generic content only: no tenant or creative data.
        record = self._finished_job_for_asset(asset_id)
        if record is None:
            return None
        return _placeholder_svg(asset_id, record.asset.width_px, record.asset.height_px)

    # Export

    async def export(
        self, db: AsyncSession, ctx: TenantContext, creative_id: UUID
    ) -> ExportManifest:
        """Bundle the current version, its generated image and its latest stored validation.

        Raises:
            NotFoundError: no such creative visible to the caller.
            ExportNotReadyError: the current version has no finished generated image yet.
        """
        creative = await creatives.get_creative(db, ctx, creative_id)
        job_id = self._latest_jobs.get((ctx.tenant_id, creative_id, creative.version))
        record = None if job_id is None else self._jobs.get(job_id)
        if record is None or not self._is_done(record.started, self._timings.generation_seconds):
            raise ExportNotReadyError()
        latest = await validations.latest_summary(db, ctx, creative_id, creative.version)
        policy = await policies.get_active_policy_summary(db, ctx.tenant_id)
        return ExportManifest(
            status="DRAFT",
            creative_id=creative_id,
            creative_version=creative.version,
            caption=creative.caption,
            image=self._asset_view(record),
            study_snapshot_version=creative.study_snapshot_version,
            policy_bundle_version=None if policy is None else policy.version,
            latest_validation=latest,
            exported_at=datetime.now(tz=UTC),
        )

    # Helpers

    def _is_done(self, started: float, duration: float) -> bool:
        return self._clock() - started >= duration

    def _replay_id(
        self, ctx: TenantContext, route: str, idempotency_key: str, fingerprint: str
    ) -> UUID | None:
        replay = self._replays.get((ctx.tenant_id, route, idempotency_key))
        if replay is None:
            return None
        if replay.fingerprint != fingerprint:
            raise IdempotencyKeyReusedError()
        return replay.resource_id

    def _replayed_run(
        self, ctx: TenantContext, idempotency_key: str, fingerprint: str
    ) -> ValidationAccepted | None:
        run_id = self._replay_id(ctx, VALIDATE_ROUTE, idempotency_key, fingerprint)
        record = None if run_id is None else self._runs.get(run_id)
        return None if record is None else self._accepted_run(record)

    def _replayed_job(
        self, ctx: TenantContext, idempotency_key: str, fingerprint: str
    ) -> JobAccepted | None:
        job_id = self._replay_id(ctx, GENERATE_ROUTE, idempotency_key, fingerprint)
        record = None if job_id is None else self._jobs.get(job_id)
        if record is None:
            return None
        return JobAccepted(job_id=record.job_id, state=self._job_view(record).state)

    def _accepted_run(self, record: _ValidationRecord) -> ValidationAccepted:
        return ValidationAccepted(
            job_id=record.finished.job_id,
            validation_run_id=record.finished.validation_run_id,
            state=self._run_view(record, current_version=None).state,
        )

    def _run_view(
        self, record: _ValidationRecord, *, current_version: int | None
    ) -> ValidationRunRead:
        is_current = record.finished.is_current
        if current_version is not None:
            is_current = record.finished.creative_version == current_version
        if self._is_done(record.started, self._timings.validation_seconds):
            return record.finished.model_copy(update={"is_current": is_current})
        return record.finished.model_copy(
            update={
                "state": ExecutionState.RUNNING,
                "outcome": None,
                "evaluated_at": None,
                "is_current": is_current,
                "gates": [],
                "rule_results": [],
                "findings": [],
                "coverage_gaps": [],
            }
        )

    def _finished_job_for_asset(self, asset_id: UUID) -> _GenerationRecord | None:
        job_id = self._job_by_asset.get(asset_id)
        record = None if job_id is None else self._jobs.get(job_id)
        if record is None or not self._is_done(record.started, self._timings.generation_seconds):
            return None
        return record

    def _validation_asset(self, ctx: TenantContext, body: ValidationCreate) -> AssetRead:
        # The image must be the finished one generated for this exact creative version.
        record = None if body.asset_id is None else self._finished_job_for_asset(body.asset_id)
        if (
            record is None
            or record.tenant_id != ctx.tenant_id
            or record.creative_id != body.creative_id
            or record.creative_version != body.creative_version
        ):
            raise NotFoundError("Asset not found.")
        return record.asset

    def _new_generation(
        self, ctx: TenantContext, creative: CreativeRead, version: int
    ) -> _GenerationRecord:
        spec = FORMAT_SPECS[creative.post_format]
        asset_id = uuid4()
        content = _placeholder_svg(asset_id, spec.width_px, spec.height_px)
        created_at = datetime.now(tz=UTC)
        asset = AssetRead(
            asset_id=asset_id,
            creative_id=creative.id,
            origin=AssetOrigin.GENERATED,
            mime_type=STUB_MIME_TYPE,
            width_px=spec.width_px,
            height_px=spec.height_px,
            size_bytes=len(content),
            checksum_sha256=hashlib.sha256(content).hexdigest(),
            provider=STUB_PROVIDER,
            model=STUB_MODEL,
            created_at=created_at + timedelta(seconds=self._timings.generation_seconds),
            download_url=None,
            download_url_expires_at=None,
        )
        return _GenerationRecord(
            tenant_id=ctx.tenant_id,
            study_id=creative.study_id,
            creative_id=creative.id,
            creative_version=version,
            created_by=ctx.principal.sub,
            job_id=uuid4(),
            started=self._clock(),
            created_at=created_at,
            asset=asset,
        )

    async def _visible_job(
        self, db: AsyncSession, ctx: TenantContext, job_id: UUID, not_found: str
    ) -> _GenerationRecord:
        record = self._jobs.get(job_id)
        if record is None or record.tenant_id != ctx.tenant_id:
            raise NotFoundError(not_found)
        try:
            # Also enforces study access: a creative the caller cannot see hides its jobs.
            await creatives.get_creative(db, ctx, record.creative_id)
        except NotFoundError:
            raise NotFoundError(not_found) from None
        return record

    def _job_view(self, record: _GenerationRecord) -> JobRead:
        total = self._timings.generation_seconds
        elapsed = max(0.0, min(self._clock() - record.started, total))
        if elapsed >= total:
            state, stage = ExecutionState.SUCCEEDED, "stored"
        elif elapsed >= total * 0.5:
            state, stage = ExecutionState.WAITING_PROVIDER, "waiting_provider"
        elif elapsed >= total * 0.15:
            state, stage = ExecutionState.RUNNING, "submitting"
        else:
            state, stage = ExecutionState.QUEUED, None
        done = state is ExecutionState.SUCCEEDED
        submitted = state is not ExecutionState.QUEUED
        return JobRead(
            job_id=record.job_id,
            kind=JobKind.IMAGE_GENERATION,
            creative_id=record.creative_id,
            creative_version=record.creative_version,
            state=state,
            stage=stage,
            provider_request_id=f"demo-{record.job_id.hex[:12]}" if submitted else None,
            failure_code=None,
            failure_message=None,
            result_asset_id=record.asset.asset_id if done else None,
            created_at=record.created_at,
            updated_at=record.created_at + timedelta(seconds=elapsed),
            completed_at=record.asset.created_at if done else None,
        )

    def _asset_view(self, record: _GenerationRecord) -> AssetRead:
        asset_id = record.asset.asset_id
        return record.asset.model_copy(
            update={
                "download_url": f"{self._base_url}{DEMO_FILES_PATH}/{asset_id}",
                "download_url_expires_at": datetime.now(tz=UTC) + DOWNLOAD_URL_TTL,
            }
        )

    def _media_item(self, record: _GenerationRecord, created_by_name: str | None) -> MediaItem:
        asset = self._asset_view(record)
        return MediaItem(
            asset_id=asset.asset_id,
            creative_id=record.creative_id,
            creative_version=record.creative_version,
            origin=asset.origin,
            mime_type=asset.mime_type,
            width_px=asset.width_px,
            height_px=asset.height_px,
            size_bytes=asset.size_bytes,
            created_at=asset.created_at,
            created_by=record.created_by,
            created_by_name=created_by_name,
            download_url=asset.download_url,
            download_url_expires_at=asset.download_url_expires_at,
        )


async def _prompt(db: AsyncSession, ctx: TenantContext, creative_id: UUID, version: int) -> str:
    page = await creatives.list_versions(db, ctx, creative_id, before=version + 1, limit=1)
    if not page.items or page.items[0].version != version:
        raise NotFoundError("Creative version not found.")
    return page.items[0].current_prompt


async def _active_policy(db: AsyncSession, ctx: TenantContext) -> PolicyBundleRead:
    try:
        return await policies.get_active_policy(db, ctx)
    except NotFoundError:
        raise NoActivePolicyError() from None


def _evaluate_image(prompt: str, policy: PolicyBundleRead) -> rules.Evaluation:
    # The placeholder has no content, so the brief decides: a brief that mentions payment gets
    # a payment-emphasis finding on a fixed region; every other image rule counts as checked.
    applicable = rules.applicable_rules(policy, ValidationInputKind.IMAGE)
    hits: dict[str, list[rules.Hit]] = {rule_id: [] for rule_id in applicable}
    payment = rules.PAYMENT_CHECK.rule_id
    if payment in applicable and rules.PAYMENT_CHECK.pattern.search(prompt):
        hits[payment] = [rules.Hit("The payment amount is shown in large type.")]
    return rules.summarise(
        applicable, hits, extra_gaps=[STUB_GAP], regions={payment: _PAYMENT_REGION}
    )


def _placeholder_svg(asset_id: UUID, width: int, height: int) -> bytes:
    # Built only from integers and the asset ID's hex digits, so nothing needs escaping.
    label = asset_id.hex[:8]
    svg = (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}">'
        '<defs><linearGradient id="bg" x1="0" y1="0" x2="1" y2="1">'
        '<stop offset="0" stop-color="#1f4e79"/><stop offset="1" stop-color="#2e8b8b"/>'
        "</linearGradient></defs>"
        f'<rect width="{width}" height="{height}" fill="url(#bg)"/>'
        f'<text x="50%" y="47%" text-anchor="middle" font-family="Arial, sans-serif" '
        f'font-size="{width // 12}" fill="#ffffff">DEMO IMAGE</text>'
        f'<text x="50%" y="55%" text-anchor="middle" font-family="Arial, sans-serif" '
        f'font-size="{width // 32}" fill="#dfe9f3">Placeholder, not AI-generated · {label}</text>'
        "</svg>"
    )
    return svg.encode("utf-8")
