"""Connector worker: leases outbox events and runs the matching job handler.

Run with `uv run python -m app.workers.connector`. Each event is processed in its own session
bound to the event's tenant, so row-level security applies to everything except the lease call.
A crashed worker's lease expires and the event is leased again; handlers are written so a
repeated step is harmless (guarded transitions, the stored provider idempotency key).
"""

import asyncio
import importlib
import logging
import os
import secrets
import signal
import socket
from collections.abc import Mapping
from contextlib import suppress
from dataclasses import dataclass, replace
from datetime import UTC, datetime, timedelta
from types import MappingProxyType
from typing import Final, Protocol, cast
from uuid import UUID

from sqlalchemy import func, select, text, update
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.core.config import Settings, get_settings
from app.core.logging import configure_logging
from app.db.session import bind_tenant, create_engine, create_sessionmaker
from app.models import CreativeVersion, Job, OutboxEvent
from app.models.enums import FORMAT_SPECS, ExecutionState
from app.models.jobs import ReservationState
from app.schemas.jobs import JobKind
from app.services import eligibility
from app.services.jobs import (
    ACTIVE_STATES,
    HIGGSFIELD_NAME,
    TERMINAL_STATES,
    settle_reservation,
    transition,
)
from app.services.providers.base import (
    TERMINAL_PROVIDER_STATUSES,
    FetchedImage,
    ImageProvider,
    ImageRequest,
    ProviderError,
    ProviderJobStatus,
    ProviderOutcomeUnknownError,
)
from app.services.providers.simulator import SIMULATOR_NAME, SimulatedImageProvider

logger = logging.getLogger(__name__)

DEFAULT_OWNER: Final = f"{socket.gethostname()}:{os.getpid()}"
LEASE_BATCH: Final = 10
NO_HANDLER_RETRY_SECONDS: Final = 60.0
STAGE_SUBMITTING: Final = "submitting"
STAGE_RETRY_WAIT: Final = "retry_wait"
STAGE_RECONCILE: Final = "reconcile"
STAGE_WAITING: Final = "waiting_provider"
# Provider finished but no storage is wired (until NRC-3): the job stops at this persisted result.
STAGE_OUTPUT_READY: Final = "output_ready"
STAGE_STORED: Final = "stored"
_LEASE = text(
    "SELECT event_id, tenant_id, job_id, event_type"
    " FROM lease_outbox_events(:owner, :batch, :lease)"
)
_jitter = secrets.SystemRandom()


class WorkerConfigError(Exception):
    # The worker cannot start with this configuration. Messages name settings, never values.
    pass


class GeneratedImageSink(Protocol):
    """Stores a generated image and its asset row (real implementation: NRC-3).

    Called inside the job's transaction with the tenant bound; returns the new asset id, which
    the worker records as the job's result_asset_id in the same transaction.
    """

    async def store(
        self, db: AsyncSession, job: Job, image: FetchedImage, provider_name: str, model: str
    ) -> UUID: ...


@dataclass(frozen=True, slots=True)
class Step:
    # Everything a handler may use besides its session and job.

    providers: Mapping[str, ImageProvider]
    sink: GeneratedImageSink | None
    settings: Settings
    now: datetime
    # How many times this event has been leased, including now (for backoff).
    event_attempts: int


class JobHandler(Protocol):
    """Runs the next step of one kind of job. Register one per JobKind in HANDLERS.

    To add a kind (e.g. validation, NRC-7):
    1. Add the JobKind value. Create the job row with that kind plus one outbox_events row
       (event_type "job.run") in the same transaction, as app.services.jobs.accept_creative_job
       does.
    2. Implement handle(). The session is bound to the job's tenant, so RLS applies. Change state
       only through app.services.jobs.transition, check job.deadline_at, and commit before a slow
       external call when the step must survive a crash. The worker commits after handle returns.
    3. Return None when the job needs no further events, or the seconds after which the same event
       should run again (backoff via available_at instead of sleeping).
    An exception is logged and the event runs again once its lease expires.
    """

    async def handle(self, db: AsyncSession, job: Job, step: Step) -> float | None: ...


def backoff_seconds(attempt: int, *, base: float = 2.0, cap: float = 60.0) -> float:
    # Exponential, with jitter between 50% and 100% so many jobs don't retry in lockstep.
    return min(cap, base * 2.0 ** max(attempt - 1, 0)) * _jitter.uniform(0.5, 1.0)


class ImageGenerationHandler:
    async def handle(self, db: AsyncSession, job: Job, step: Step) -> float | None:
        if job.state in TERMINAL_STATES:
            return None
        if job.deadline_at <= step.now:
            await _give_up(db, job, "DEADLINE_EXCEEDED", "The job did not finish in time.")
            return None
        provider = step.providers.get(job.provider or "")
        if provider is None:
            # Live mode never falls back to the simulator, and the simulator never runs live jobs.
            await _give_up(db, job, "PROVIDER_NOT_CONFIGURED", "No provider for this job.")
            return None
        if job.state in (ExecutionState.QUEUED, ExecutionState.RUNNING):
            return await self._submit(db, job, provider, step, reconcile=False)
        if job.state is ExecutionState.UNKNOWN:
            return await self._submit(db, job, provider, step, reconcile=True)
        if job.state is ExecutionState.WAITING_PROVIDER and job.provider_request_id is not None:
            return await self._poll(db, job, provider, job.provider_request_id, step)
        return None

    async def _submit(
        self, db: AsyncSession, job: Job, provider: ImageProvider, step: Step, *, reconcile: bool
    ) -> float | None:
        # Reconcile resends the identical request with the same key; the provider returns the
        # original request id instead of starting (and billing) a second generation.
        if not reconcile:
            moved = await transition(
                db,
                job.id,
                allowed_from={ExecutionState.QUEUED, ExecutionState.RUNNING},
                to=ExecutionState.RUNNING,
                stage=STAGE_SUBMITTING,
            )
            if not moved:
                return None
            # Committed before the call, so after a crash the job reads as possibly accepted.
            await db.commit()
        request = await _image_request(db, job)
        try:
            submitted = await provider.submit(request)
        except ProviderOutcomeUnknownError as error:
            if reconcile:
                logger.warning("reconcile inconclusive job=%s tenant=%s", job.id, job.tenant_id)
                return None
            moved = await transition(
                db,
                job.id,
                allowed_from={ExecutionState.RUNNING},
                to=ExecutionState.UNKNOWN,
                stage=STAGE_RECONCILE,
                failure_code=error.code,
            )
            if moved:
                await settle_reservation(db, job.id, ReservationState.HELD)
            return backoff_seconds(1)
        except ProviderError as error:
            if reconcile:
                logger.warning(
                    "reconcile failed job=%s tenant=%s code=%s", job.id, job.tenant_id, error.code
                )
                return None
            attempts = job.attempt_count + 1
            if error.retryable and attempts < job.max_attempts:
                await transition(
                    db,
                    job.id,
                    allowed_from={ExecutionState.RUNNING},
                    to=ExecutionState.RUNNING,
                    stage=STAGE_RETRY_WAIT,
                    attempt_count=attempts,
                )
                return max(error.retry_after_seconds or 0.0, backoff_seconds(attempts))
            # Rejected before acceptance, so nothing was billed.
            await _fail(db, job.id, error, accepted=False, attempt_count=attempts)
            return None

        logger.info(
            "provider accepted job=%s tenant=%s provider_request=%s",
            job.id,
            job.tenant_id,
            submitted.provider_request_id,
        )
        moved = await transition(
            db,
            job.id,
            allowed_from={ExecutionState.RUNNING, ExecutionState.UNKNOWN},
            to=ExecutionState.WAITING_PROVIDER,
            stage=STAGE_WAITING,
            provider_request_id=submitted.provider_request_id,
            failure_code=None,
        )
        if not moved:
            return None
        await db.commit()
        return await self._poll(db, job, provider, submitted.provider_request_id, step)

    async def _poll(
        self,
        db: AsyncSession,
        job: Job,
        provider: ImageProvider,
        provider_request_id: str,
        step: Step,
    ) -> float | None:
        try:
            status = await provider.get_status(provider_request_id)
        except ProviderError as error:
            if error.retryable or isinstance(error, ProviderOutcomeUnknownError):
                return max(error.retry_after_seconds or 0.0, backoff_seconds(step.event_attempts))
            await _fail(db, job.id, error, accepted=True)
            return None
        if status.status not in TERMINAL_PROVIDER_STATUSES:
            return backoff_seconds(step.event_attempts)
        if status.status is not ProviderJobStatus.COMPLETED:
            code = status.failure_code or f"PROVIDER_{status.status.value.upper()}"
            await _fail_with(
                db, job.id, code, "The provider did not produce an image.", accepted=True
            )
            return None
        if step.sink is None:
            await transition(
                db,
                job.id,
                allowed_from={ExecutionState.WAITING_PROVIDER},
                to=ExecutionState.WAITING_PROVIDER,
                stage=STAGE_OUTPUT_READY,
            )
            return None
        try:
            image = await provider.fetch_output(status)
        except ProviderError as error:
            if error.retryable:
                return backoff_seconds(step.event_attempts)
            await _fail(db, job.id, error, accepted=True)
            return None
        asset_id = await step.sink.store(db, job, image, provider.name, provider.model)
        moved = await transition(
            db,
            job.id,
            allowed_from={ExecutionState.WAITING_PROVIDER},
            to=ExecutionState.SUCCEEDED,
            stage=STAGE_STORED,
            result_asset_id=asset_id,
        )
        if moved:
            await settle_reservation(db, job.id, ReservationState.COMMITTED)
        return None


HANDLERS: Final[Mapping[JobKind, JobHandler]] = MappingProxyType(
    {JobKind.IMAGE_GENERATION: ImageGenerationHandler()}
)


async def run_once(
    sessions: async_sessionmaker[AsyncSession],
    providers: Mapping[str, ImageProvider],
    sink: GeneratedImageSink | None,
    settings: Settings,
    *,
    now: datetime | None = None,
    owner: str = DEFAULT_OWNER,
    handlers: Mapping[JobKind, JobHandler] = HANDLERS,
) -> int:
    # Lease one batch of due events and run each once. Returns how many events were leased.
    async with sessions() as db:
        events = (
            await db.execute(
                _LEASE,
                {"owner": owner, "batch": LEASE_BATCH, "lease": settings.worker_lease_seconds},
            )
        ).all()
        await db.commit()
    base = Step(providers, sink, settings, now or datetime.now(UTC), event_attempts=1)
    for event in events:
        try:
            await _process(sessions, event.event_id, event.tenant_id, event.job_id, handlers, base)
        except Exception:
            # The lease expires and the event runs again; the handler's deadline bounds retries.
            logger.exception(
                "job event failed job=%s tenant=%s event=%s",
                event.job_id,
                event.tenant_id,
                event.event_id,
            )
    return len(events)


async def _process(
    sessions: async_sessionmaker[AsyncSession],
    event_id: UUID,
    tenant_id: UUID,
    job_id: UUID,
    handlers: Mapping[JobKind, JobHandler],
    base: Step,
) -> None:
    async with sessions() as db:
        await bind_tenant(db, tenant_id)
        job = await db.get(Job, job_id)
        attempts = await db.scalar(select(OutboxEvent.attempts).where(OutboxEvent.id == event_id))
        handler = handlers.get(JobKind(job.kind)) if job and job.kind in JobKind else None
        if job is None or handler is None:
            logger.error("no handler for job=%s tenant=%s", job_id, tenant_id)
            await _finish_event(db, event_id, NO_HANDLER_RETRY_SECONDS, error_code="NO_HANDLER")
        else:
            step = replace(base, event_attempts=attempts or 1)
            await _finish_event(db, event_id, await handler.handle(db, job, step))
        await db.commit()


async def _finish_event(
    db: AsyncSession, event_id: UUID, retry_after: float | None, *, error_code: str | None = None
) -> None:
    values: dict[str, object] = {
        "leased_until": None,
        "lease_owner": None,
        "last_error_code": error_code,
    }
    if retry_after is None:
        values["processed_at"] = func.now()
    else:
        values["available_at"] = func.clock_timestamp() + timedelta(seconds=retry_after)
    await db.execute(
        update(OutboxEvent)
        .where(OutboxEvent.id == event_id)
        .values(**values)
        .execution_options(synchronize_session=False)
    )


async def _image_request(db: AsyncSession, job: Job) -> ImageRequest:
    content = (
        await db.execute(
            select(CreativeVersion.prompt, CreativeVersion.post_format).where(
                CreativeVersion.tenant_id == job.tenant_id,
                CreativeVersion.creative_id == job.creative_id,
                CreativeVersion.version == job.creative_version,
            )
        )
    ).one()
    spec = FORMAT_SPECS[content.post_format]
    return ImageRequest(
        prompt=content.prompt,
        width_px=spec.width_px,
        height_px=spec.height_px,
        aspect_ratio=spec.aspect_ratio,
        idempotency_key=str(job.provider_idempotency_key),
    )


async def _fail(
    db: AsyncSession, job_id: UUID, error: ProviderError, *, accepted: bool, **values: object
) -> None:
    await _fail_with(db, job_id, error.code, error.message, accepted=accepted, **values)


async def _fail_with(
    db: AsyncSession, job_id: UUID, code: str, message: str, *, accepted: bool, **values: object
) -> None:
    # Credits are released only when the provider never accepted the request.
    moved = await transition(
        db,
        job_id,
        allowed_from=ACTIVE_STATES,
        to=ExecutionState.FAILED,
        failure_code=code,
        failure_message=message[:500],
        **values,
    )
    if moved:
        settled = ReservationState.COMMITTED if accepted else ReservationState.RELEASED
        await settle_reservation(db, job_id, settled)


async def _give_up(db: AsyncSession, job: Job, code: str, message: str) -> None:
    # FAILED when the provider certainly never got the request, otherwise UNKNOWN for review.
    if job.state is ExecutionState.UNKNOWN:
        return
    maybe_accepted = job.state is ExecutionState.WAITING_PROVIDER or (
        job.state is ExecutionState.RUNNING and job.stage == STAGE_SUBMITTING
    )
    if not maybe_accepted:
        await _fail_with(db, job.id, code, message, accepted=False)
        return
    moved = await transition(
        db,
        job.id,
        allowed_from={job.state},
        to=ExecutionState.UNKNOWN,
        failure_code=code,
        failure_message=message,
    )
    if moved:
        await settle_reservation(db, job.id, ReservationState.HELD)


def check_live_mode(settings: Settings) -> None:
    # Live mode must have real credentials and a real validation gate; it never falls back.
    missing = [
        name
        for name, value in (
            ("HIGGSFIELD_API_KEY_ID", settings.higgsfield_api_key_id),
            ("HIGGSFIELD_API_KEY_SECRET", settings.higgsfield_api_key_secret),
            ("HIGGSFIELD_MODEL", settings.higgsfield_model),
        )
        if not value
    ]
    problems = [f"missing {', '.join(missing)}"] if missing else []
    if eligibility.VALIDATION_GATE_IS_STUB:
        problems.append("the validation-PASS gate is still the stub (NRC-7)")
    if problems:
        raise WorkerConfigError(f"GENERATION_MODE=live refused: {'; '.join(problems)}.")


def build_providers(settings: Settings) -> dict[str, ImageProvider]:
    if settings.generation_mode == "simulator":
        return {SIMULATOR_NAME: SimulatedImageProvider()}
    check_live_mode(settings)
    return {HIGGSFIELD_NAME: _higgsfield_provider(settings)}


def _higgsfield_provider(settings: Settings) -> ImageProvider:
    # Imported lazily: the adapter ships separately (NRC-10) and only live mode needs it.
    module_name = "app.services.providers.higgsfield"
    try:
        module = importlib.import_module(module_name)
    except ModuleNotFoundError as error:
        if error.name != module_name:
            raise
        raise WorkerConfigError(
            f"GENERATION_MODE=live needs the Higgsfield adapter ({module_name}, NRC-10)."
        ) from error
    factory = getattr(module, "create_provider", None)
    if factory is None:
        raise WorkerConfigError(f"{module_name} has no create_provider(settings).")
    return cast(ImageProvider, factory(settings))


async def run_forever(settings: Settings) -> None:
    providers = build_providers(settings)
    engine = create_engine(settings)
    sessions = create_sessionmaker(engine)
    stop = asyncio.Event()
    _stop_on_signals(stop)
    # Storage is wired in NRC-3; until then completed jobs stop at the persisted provider result.
    sink: GeneratedImageSink | None = None
    logger.info("connector worker started mode=%s", settings.generation_mode)
    try:
        while not stop.is_set():
            try:
                leased = await run_once(sessions, providers, sink, settings)
            except Exception:
                logger.exception("connector iteration failed")
                leased = 0
            if leased == 0:
                with suppress(TimeoutError):
                    await asyncio.wait_for(stop.wait(), settings.worker_poll_interval_seconds)
    finally:
        await engine.dispose()
    logger.info("connector worker stopped")


def _stop_on_signals(stop: asyncio.Event) -> None:
    # Finish the current batch, then exit.
    loop = asyncio.get_running_loop()
    for signum in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(signum, stop.set)
        except NotImplementedError:
            # Windows event loops have no add_signal_handler.
            signal.signal(signum, lambda *_: loop.call_soon_threadsafe(stop.set))


def main() -> None:
    settings = get_settings()
    configure_logging(settings.log_level)
    try:
        asyncio.run(run_forever(settings))
    except WorkerConfigError as error:
        logger.error("connector worker refused to start: %s", error)
        raise SystemExit(1) from error


if __name__ == "__main__":
    main()
