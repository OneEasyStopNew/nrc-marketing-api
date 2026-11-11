"""Connector worker: leases outbox events one at a time and runs the matching job handler.

Run with `uv run python -m app.workers.connector`. Each event is processed in its own session
bound to the event's tenant, so row-level security applies to everything except the lease call.

Only one worker acts on a job at a time:
- each lease covers one event, and a fencing heartbeat renews it before every externally visible
  step (provider calls, storage, finishing the event). If the lease was lost (it expired and
  another worker took the event), the step is abandoned without touching the job;
- a session-level advisory lock per job spans the whole processing, including the commits a
  handler makes mid-way.
A crashed worker's lease expires and the event is leased again; handlers are written so a
repeated step is harmless (guarded transitions, the stored provider idempotency key).
"""

import asyncio
import logging
import os
import secrets
import signal
import socket
from collections.abc import Awaitable, Callable, Mapping
from contextlib import suppress
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from decimal import ROUND_CEILING, Decimal
from types import MappingProxyType
from typing import Final, Protocol, runtime_checkable
from uuid import UUID

from pydantic import SecretStr, ValidationError
from sqlalchemy import ARRAY, Row, Text, Update, bindparam, func, select, text, update
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker
from sqlalchemy.orm.attributes import set_committed_value

from app.core.config import (
    LEASE_MAX_SECONDS,
    LEASE_MIN_SECONDS,
    Settings,
    WorkerDatabaseSettings,
    get_settings,
)
from app.core.logging import configure_logging
from app.db.session import bind_tenant, create_engine, create_sessionmaker
from app.models import CreativeVersion, Job, OutboxEvent
from app.models.enums import FORMAT_SPECS, ExecutionState, JobStage
from app.models.jobs import FAILURE_MESSAGE_MAX_CHARS, ReservationState
from app.schemas.jobs import JobKind
from app.services import generation_checks, spend
from app.services.assets import StoredImageSink, emit_asset_generated
from app.services.jobs import (
    NON_TERMINAL_STATES,
    RUN_EVENT,
    TERMINAL_STATES,
    provider_idempotency_key,
    settle_reservation,
    transition,
)
from app.services.media import configure_pillow
from app.services.providers.base import (
    HIGGSFIELD_NAME,
    TERMINAL_PROVIDER_STATUSES,
    FetchedImage,
    ImageProvider,
    ImageRequest,
    ProviderError,
    ProviderJobStatus,
    ProviderOutcomeUnknownError,
    ProviderOutputExpiredError,
    ProviderOutputHostError,
    StatusResult,
    SubmitResult,
)
from app.services.providers.higgsfield import (
    ProviderSecretError,
    create_provider,
    missing_live_settings,
    with_higgsfield_key,
)
from app.services.providers.simulator import SIMULATOR_NAME, SimulatedImageProvider
from app.services.storage import ObjectStorage

logger = logging.getLogger(__name__)

# The provider saw this key with a different body: resending can never reconcile it.
IDEMPOTENCY_CONFLICT: Final = "PROVIDER_IDEMPOTENCY_CONFLICT"
BUDGET_EXHAUSTED: Final = "BUDGET_EXHAUSTED"
MODEL_CHANGED: Final = "MODEL_CHANGED"
ATTEMPTS_EXHAUSTED: Final = "ATTEMPTS_EXHAUSTED"
NO_HANDLER: Final = "NO_HANDLER"
# The provider billed a completed generation that could not be stored.
OUTPUT_UNAVAILABLE: Final = "OUTPUT_UNAVAILABLE"
# The provider billed a completed generation whose files expired before they were stored.
OUTPUT_EXPIRED: Final = ProviderOutputExpiredError.code
# Accepted, but still queued or in progress at the provider when the grace window ended.
PROVIDER_STILL_RUNNING: Final = "PROVIDER_STILL_RUNNING"
_LEASE = text(
    "SELECT event_id, tenant_id, job_id, event_type"
    " FROM lease_outbox_events(:owner, 1, :lease, :event_types)"
).bindparams(bindparam("event_types", type_=ARRAY(Text)))
_jitter = secrets.SystemRandom()


def process_owner() -> str:
    # Hostname and PID can repeat across containers (both PID 1), so add a random suffix.
    return f"{socket.gethostname()}:{os.getpid()}:{secrets.token_hex(4)}"


# Computed once, so every lease this process takes carries the same owner.
DEFAULT_OWNER: Final = process_owner()


class WorkerConfigError(Exception):
    # The worker cannot start with this configuration. Messages name settings, never values.
    pass


@runtime_checkable
class _Closable(Protocol):
    # Providers that hold network clients (Higgsfield) close them on shutdown.
    async def aclose(self) -> None: ...


class LeaseLostError(Exception):
    # Another worker owns the event now. The current step is rolled back and the job left alone.
    pass


class NoHandlerError(Exception):
    # Raised by an event handler that cannot run this event (e.g. no handler for the job's kind).
    # Counts as a failed attempt; the event is retried after WORKER_NO_HANDLER_RETRY_SECONDS.
    pass


class GeneratedImageSink(Protocol):
    """Stores a generated image and its asset row (implementation: app.services.assets).

    Called inside the job's transaction with the tenant bound; returns the new asset id, which
    the worker records as the job's result_asset_id in the same transaction. Raise a
    ProviderError subclass (e.g. ProviderOutputError) when the image is unusable.
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
    # Renews this worker's lease; raises LeaseLostError if another worker owns the event.
    heartbeat: Callable[[AsyncSession], Awaitable[None]]


@dataclass(frozen=True, slots=True)
class LeasedEvent:
    # One outbox event, read under its tenant after this worker leased it.

    id: UUID
    tenant_id: UUID
    job_id: UUID
    event_type: str
    # Ids only (never prompts or URLs); its shape is the event type's contract.
    payload: Mapping[str, object]
    # Leases so far, including this one; failed_attempts counts only the failed ones.
    attempts: int
    failed_attempts: int


class EventHandler(Protocol):
    """Handles every outbox event of one type. Subscribe by registering it in EVENT_HANDLERS.

    The worker leases exactly the event types registered there, so subscribing is one line, e.g.
    `ASSET_GENERATED_EVENT: ImageCheckHandler(),`; nothing else in the worker changes.

    Delivery is at least once: after a crash, a hang or a lost lease the same event runs again,
    possibly on another worker, so handle() must be idempotent (guarded transitions, unique
    keys, idempotency keys on external calls). Events of one job never run concurrently (an
    advisory lock per job_id).

    handle(): the session is bound to the event's tenant, so RLS applies. Call
    `await step.heartbeat(db)` before every external call or slow step; it raises LeaseLostError
    when another worker took the event over. The worker commits after handle returns. Return
    None when the event is done, or the seconds after which it should run again (backoff via
    available_at, never sleeping). Raise NoHandlerError when this event cannot be handled here.
    An exception leaves the lease to expire, which counts as a failed attempt.

    give_up(): called once, in the event's transaction, when the event has failed
    GENERATION_MAX_ATTEMPTS times (code ATTEMPTS_EXHAUSTED or NO_HANDLER). Leave the related
    records in a state an operator can see; the event is then closed and never leased again.
    """

    async def handle(self, db: AsyncSession, event: LeasedEvent, step: Step) -> float | None: ...

    async def give_up(self, db: AsyncSession, event: LeasedEvent, code: str) -> None: ...


class JobHandler(Protocol):
    """Runs the next step of one kind of job. Register one per JobKind in JOB_HANDLERS.

    To add a kind (e.g. validation, NRC-7):
    1. Add the JobKind value. Create the job row with that kind plus one outbox_events row
       (event_type "job.run") in the same transaction, as app.services.jobs.accept_creative_job
       does.
    2. Implement handle(). The session is bound to the job's tenant, so RLS applies. Change state
       only through app.services.jobs.transition and check job.deadline_at. Call
       `await step.heartbeat(db)` before every external call or slow step, and commit before an
       external call when the step must survive a crash. The worker commits after handle returns.
    3. Return None when the job needs no further events, or the seconds after which the same event
       should run again (backoff via available_at instead of sleeping).
    An exception is logged and the event runs again once its lease expires. Each expired lease
    counts as a failure; after GENERATION_MAX_ATTEMPTS failures the job is given up (_give_up).
    "job.run" events reach these handlers through JobRunHandler, the EVENT_HANDLERS entry.
    """

    async def handle(self, db: AsyncSession, job: Job, step: Step) -> float | None: ...


def backoff_seconds(attempt: int, settings: Settings) -> float:
    # Exponential from WORKER_BACKOFF_BASE_SECONDS up to WORKER_BACKOFF_MAX_SECONDS, with jitter
    # between 50% and 100% so many jobs don't retry in lockstep.
    delay = settings.worker_backoff_base_seconds * 2.0 ** max(attempt - 1, 0)
    return min(settings.worker_backoff_max_seconds, delay) * _jitter.uniform(0.5, 1.0)


class ImageGenerationHandler:
    async def handle(self, db: AsyncSession, job: Job, step: Step) -> float | None:
        if job.state in TERMINAL_STATES:
            return None
        accepted = (
            job.state is ExecutionState.WAITING_PROVIDER and job.provider_request_id is not None
        )
        # An accepted job is billed when it completes, so the deadline alone never stops its
        # polling: _poll parks it as UNKNOWN once it is still unfinished past the grace window,
        # and a completed one is fetched (stage fetching_output) until its output expires.
        if job.deadline_at <= step.now and not accepted:
            await _give_up(db, job, "DEADLINE_EXCEEDED", "The job did not finish in time.")
            return None
        provider = step.providers.get(job.provider or "")
        if provider is None:
            # Live mode never falls back to the simulator, and the simulator never runs live jobs.
            await _give_up(db, job, "PROVIDER_NOT_CONFIGURED", "No provider for this job.")
            return None
        if job.state is ExecutionState.QUEUED or (
            job.state is ExecutionState.RUNNING and job.stage != JobStage.SUBMITTING
        ):
            return await self._submit(db, job, provider, step)
        if job.state in (ExecutionState.RUNNING, ExecutionState.UNKNOWN):
            # RUNNING/submitting means a worker died after recording the submit: the provider may
            # have the request, so only the same-key resend is safe.
            return await self._reconcile(db, job, provider, step)
        if job.state is ExecutionState.WAITING_PROVIDER and job.provider_request_id is not None:
            return await self._poll(db, job, provider, job.provider_request_id, step)
        return None

    async def _submit(
        self, db: AsyncSession, job: Job, provider: ImageProvider, step: Step
    ) -> float | None:
        if provider.model != job.model:
            # Never sent (QUEUED or a retry before acceptance), so nothing was billed.
            await _fail_with(
                db,
                job.id,
                MODEL_CHANGED,
                "The configured model changed after the job was accepted.",
                billed=False,
            )
            return None
        request = await _image_request(db, job)
        values: dict[str, object] = {}
        # Priced once, before the first submit; retries and reconcile reuse the estimate.
        if job.provider_cost_estimate_usd_micros is None:
            await step.heartbeat(db)
            try:
                micros = await _estimate_usd_micros(provider, request)
            except ProviderError as error:
                return await _retry_or_fail(db, job, error, step.settings)
            _log_call("estimate", job, provider, None, micros)
            if micros > _usd_micros(step.settings.generation_max_cost_usd_per_image):
                await _fail_with(
                    db,
                    job.id,
                    "COST_ABOVE_LIMIT",
                    "The provider's price is above the per-image limit.",
                    billed=False,
                    provider_cost_estimate_usd_micros=micros,
                )
                return None
            values["provider_cost_estimate_usd_micros"] = micros
            set_committed_value(job, "provider_cost_estimate_usd_micros", micros)
        await step.heartbeat(db)
        moved = await transition(
            db,
            job.id,
            allowed_from={ExecutionState.QUEUED, ExecutionState.RUNNING},
            to=ExecutionState.RUNNING,
            stage=JobStage.SUBMITTING,
            **values,
        )
        if not moved:
            return None
        # Same transaction as the move to submitting: past the global budget, nothing is sent.
        estimate = job.provider_cost_estimate_usd_micros
        budget = _usd_micros(step.settings.generation_global_budget_usd)
        if estimate is None or not await spend.reserve(
            db, job.id, str(job.provider), estimate, budget
        ):
            await _fail_with(
                db,
                job.id,
                BUDGET_EXHAUSTED,
                "The global provider budget is used up.",
                billed=False,
            )
            return None
        # Committed before the call, so after a crash the job is reconciled, never resubmitted.
        await db.commit()
        await step.heartbeat(db)
        try:
            submitted = await provider.submit(request)
        except ProviderOutcomeUnknownError as error:
            # Timeout after sending, 5xx or an unreadable answer: the provider may have the
            # request, so only the same-key resend (reconcile) is safe from here on.
            await _park_unknown(db, job.id, error.code)
            if error.code == IDEMPOTENCY_CONFLICT:
                _log_review_stop(job, error.code)
                return None
            return max(error.retry_after_seconds or 0.0, backoff_seconds(1, step.settings))
        except ProviderError as error:
            return await _retry_or_fail(db, job, error, step.settings)
        return await self._accepted(db, job, provider, submitted, step)

    async def _reconcile(
        self, db: AsyncSession, job: Job, provider: ImageProvider, step: Step
    ) -> float | None:
        # Resend the identical request with the same key: the provider returns the original
        # request id instead of starting (and billing) a second generation. Never re-estimated.
        if provider.model != job.model:
            # A different model changes the request body, so the key could never match it.
            await _park_unknown(db, job.id, MODEL_CHANGED)
            _log_review_stop(job, MODEL_CHANGED)
            return None
        request = await _image_request(db, job)
        await step.heartbeat(db)
        try:
            submitted = await provider.submit(request)
        except ProviderError as error:
            await _park_unknown(db, job.id, error.code)
            transient = error.retryable or isinstance(error, ProviderOutcomeUnknownError)
            if transient and error.code != IDEMPOTENCY_CONFLICT:
                return max(
                    error.retry_after_seconds or 0.0,
                    backoff_seconds(step.event_attempts, step.settings),
                )
            _log_review_stop(job, error.code)
            return None
        return await self._accepted(db, job, provider, submitted, step)

    async def _accepted(
        self,
        db: AsyncSession,
        job: Job,
        provider: ImageProvider,
        submitted: SubmitResult,
        step: Step,
    ) -> float | None:
        _log_call(
            "submit",
            job,
            provider,
            submitted.provider_request_id,
            job.provider_cost_estimate_usd_micros,
        )
        moved = await transition(
            db,
            job.id,
            allowed_from={ExecutionState.RUNNING, ExecutionState.UNKNOWN},
            to=ExecutionState.WAITING_PROVIDER,
            stage=JobStage.WAITING_PROVIDER,
            provider_request_id=submitted.provider_request_id,
            failure_code=None,
        )
        if not moved:
            return None
        # The sink records it on the asset; the job was loaded before the id was known.
        set_committed_value(job, "provider_request_id", submitted.provider_request_id)
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
        await step.heartbeat(db)
        _log_call(
            "status", job, provider, provider_request_id, job.provider_cost_estimate_usd_micros
        )
        try:
            status = await provider.get_status(provider_request_id)
        except ProviderError as error:
            if error.retryable or isinstance(error, ProviderOutcomeUnknownError):
                return max(
                    error.retry_after_seconds or 0.0,
                    backoff_seconds(step.event_attempts, step.settings),
                )
            # Accepted, but no longer visible (e.g. 401 after a key rotation): hold the credits
            # and reconcile later, which stops for review if the cause persists.
            await _park_unknown(db, job.id, error.code)
            return backoff_seconds(step.event_attempts, step.settings)
        if status.status not in TERMINAL_PROVIDER_STATUSES:
            grace = timedelta(seconds=step.settings.generation_accepted_grace_seconds)
            if step.now >= job.deadline_at + grace:
                # Still running long after the deadline: stop polling and hold the credits and
                # the spend reservation, since the provider may yet complete (and bill) it.
                await _park_unknown(db, job.id, PROVIDER_STILL_RUNNING)
                _log_stuck_unknown(job, PROVIDER_STILL_RUNNING)
                return None
            return backoff_seconds(step.event_attempts, step.settings)
        if status.status is not ProviderJobStatus.COMPLETED:
            code = status.failure_code or f"PROVIDER_{status.status.value.upper()}"
            # The provider bills completed generations only.
            await _fail_with(
                db, job.id, code, "The provider did not produce an image.", billed=False
            )
            return None
        actual = None if status.actual_cost_usd is None else _usd_micros(status.actual_cost_usd)
        logger.info(
            "provider cost job=%s provider_request=%s model=%s estimate_usd_micros=%s "
            "actual_usd_micros=%s",
            job.id,
            provider_request_id,
            provider.model,
            job.provider_cost_estimate_usd_micros,
            actual,
        )
        # From here on the grace window no longer applies: the provider billed, so the output is
        # fetched until it expires. An expired output is never fetched.
        if _output_expires_at(job, status, step) <= step.now:
            # Billed for a completed generation that can no longer be kept.
            await _fail_with(
                db,
                job.id,
                OUTPUT_EXPIRED,
                "The provider's output expired.",
                billed=True,
                actual_usd_micros=actual,
            )
            return None
        if step.sink is None:
            await transition(
                db,
                job.id,
                allowed_from={ExecutionState.WAITING_PROVIDER},
                to=ExecutionState.WAITING_PROVIDER,
                stage=JobStage.OUTPUT_READY,
            )
            return None
        if job.stage != JobStage.FETCHING_OUTPUT:
            moved = await transition(
                db,
                job.id,
                allowed_from={ExecutionState.WAITING_PROVIDER},
                to=ExecutionState.WAITING_PROVIDER,
                stage=JobStage.FETCHING_OUTPUT,
            )
            if not moved:
                return None
            set_committed_value(job, "stage", JobStage.FETCHING_OUTPUT)
            # Committed before the download: if it never succeeds, the job is known to be billed.
            await db.commit()
        await step.heartbeat(db)
        _log_call(
            "fetch", job, provider, provider_request_id, job.provider_cost_estimate_usd_micros
        )
        try:
            image = await provider.fetch_output(status)
            await step.heartbeat(db)
            # The asset row is kept only together with the job's SUCCEEDED transition.
            stored = await db.begin_nested()
            try:
                asset_id = await step.sink.store(db, job, image, provider.name, provider.model)
            except BaseException:
                await stored.rollback()
                raise
        except ProviderError as error:
            if isinstance(error, ProviderOutputHostError):
                _log_output_host_refused(job, provider_request_id, error.host)
            if error.retryable:
                # Never counted as a failed attempt: the output's expiry (checked on every poll
                # above) is what ends these retries.
                return backoff_seconds(step.event_attempts, step.settings)
            # The provider's own expiry check at fetch time ends the job like the check above.
            expired = isinstance(error, ProviderOutputExpiredError)
            # Billed for a completed generation that can't be kept.
            await _fail_with(
                db,
                job.id,
                OUTPUT_EXPIRED if expired else OUTPUT_UNAVAILABLE,
                error.message,
                billed=True,
                actual_usd_micros=actual,
            )
            return None
        moved = await transition(
            db,
            job.id,
            allowed_from={ExecutionState.WAITING_PROVIDER},
            to=ExecutionState.SUCCEEDED,
            stage=JobStage.STORED,
            result_asset_id=asset_id,
            provider_cost_actual_usd_micros=actual,
        )
        if not moved:
            # Already finished elsewhere: drop this asset row (its object becomes an orphan).
            await stored.rollback()
            logger.warning(
                "job already terminal, asset discarded job=%s tenant=%s", job.id, job.tenant_id
            )
            return None
        await stored.commit()
        await settle_reservation(db, job.id, ReservationState.COMMITTED)
        await spend.settle(db, job.id, billed=True, actual_usd_micros=actual)
        # Hand-off to image validation (NRC-7), in the same commit as SUCCEEDED.
        await emit_asset_generated(db, asset_id)
        return None


JOB_HANDLERS: Final[Mapping[JobKind, JobHandler]] = MappingProxyType(
    {JobKind.IMAGE_GENERATION: ImageGenerationHandler()}
)


class JobRunHandler:
    # "job.run": runs the next step of the event's job with the handler for its kind.

    def __init__(self, job_handlers: Mapping[JobKind, JobHandler]) -> None:
        self._job_handlers = job_handlers

    async def handle(self, db: AsyncSession, event: LeasedEvent, step: Step) -> float | None:
        job = await db.get(Job, event.job_id)
        handler = self._job_handlers.get(JobKind(job.kind)) if job and job.kind in JobKind else None
        if job is None or handler is None:
            raise NoHandlerError
        return await handler.handle(db, job, step)

    async def give_up(self, db: AsyncSession, event: LeasedEvent, code: str) -> None:
        job = await db.get(Job, event.job_id)
        if job is not None:
            await _give_up(db, job, code, _GIVE_UP_MESSAGES[code])


_GIVE_UP_MESSAGES: Final = MappingProxyType(
    {
        ATTEMPTS_EXHAUSTED: "The job failed too many times.",
        NO_HANDLER: "No worker can run this kind of job.",
    }
)

# The one subscription registry, keyed by outbox event type. The worker leases exactly these
# types; others (asset.generated until NRC-7 registers its handler) wait untouched.
EVENT_HANDLERS: Final[Mapping[str, EventHandler]] = MappingProxyType(
    {RUN_EVENT: JobRunHandler(JOB_HANDLERS)}
)


@dataclass(frozen=True, slots=True)
class _Run:
    sessions: async_sessionmaker[AsyncSession]
    providers: Mapping[str, ImageProvider]
    sink: GeneratedImageSink | None
    settings: Settings
    owner: str
    lease_seconds: int
    event_handlers: Mapping[str, EventHandler]
    now: datetime | None


async def run_once(
    sessions: async_sessionmaker[AsyncSession],
    providers: Mapping[str, ImageProvider],
    sink: GeneratedImageSink | None,
    settings: Settings,
    *,
    now: datetime | None = None,
    owner: str = DEFAULT_OWNER,
    event_handlers: Mapping[str, EventHandler] = EVENT_HANDLERS,
    max_events: int | None = None,
) -> int:
    # Lease and run up to max_events (default WORKER_MAX_EVENTS_PER_RUN) due events, one lease
    # per event. Returns how many ran.
    if max_events is None:
        max_events = settings.worker_max_events_per_run
    run = _Run(
        sessions, providers, sink, settings, owner, lease_seconds(settings), event_handlers, now
    )
    leased = 0
    while leased < max_events:
        async with sessions() as db:
            event = (
                await db.execute(
                    _LEASE,
                    {
                        "owner": owner,
                        "lease": run.lease_seconds,
                        "event_types": list(event_handlers),
                    },
                )
            ).one_or_none()
            await db.commit()
        if event is None:
            break
        leased += 1
        try:
            await _process(run, event)
        except LeaseLostError:
            logger.warning(
                "lease lost, event abandoned job=%s tenant=%s event=%s",
                event.job_id,
                event.tenant_id,
                event.event_id,
            )
        except Exception:
            # The lease expires unfinished, which the next lease counts as a failed attempt.
            logger.exception(
                "job event failed job=%s tenant=%s event=%s",
                event.job_id,
                event.tenant_id,
                event.event_id,
            )
    return leased


def lease_seconds(settings: Settings) -> int:
    # Settings validate this range; the clamp also covers copies made without validation.
    return min(max(settings.worker_lease_seconds, LEASE_MIN_SECONDS), LEASE_MAX_SECONDS)


async def _process(run: _Run, event: Row[tuple[UUID, UUID, UUID, str]]) -> None:
    lock_key = func.hashtextextended(f"job:{event.job_id}", 0)
    # Session-level lock on its own connection, so it spans the handler's mid-step commits.
    async with run.sessions() as lock_db:
        if not await lock_db.scalar(select(func.pg_try_advisory_lock(lock_key))):
            # Another worker is still on this job: hand the event back for a later try.
            async with run.sessions() as db:
                await bind_tenant(db, event.tenant_id)
                await _finish_event(db, run, event.event_id, backoff_seconds(1, run.settings))
                await db.commit()
            return
        try:
            await _run_handler(run, event)
        finally:
            await lock_db.scalar(select(func.pg_advisory_unlock(lock_key)))
            await lock_db.commit()


async def _run_handler(run: _Run, event: Row[tuple[UUID, UUID, UUID, str]]) -> None:
    async with run.sessions() as db:
        await bind_tenant(db, event.tenant_id)

        async def heartbeat(session: AsyncSession) -> None:
            renewed = await session.scalar(
                _fenced(event.event_id, run.owner)
                .values(leased_until=func.clock_timestamp() + timedelta(seconds=run.lease_seconds))
                .returning(OutboxEvent.id)
            )
            if renewed is None:
                raise LeaseLostError

        await heartbeat(db)
        row = (
            await db.execute(
                select(
                    OutboxEvent.payload, OutboxEvent.attempts, OutboxEvent.failed_attempts
                ).where(OutboxEvent.id == event.event_id)
            )
        ).one()
        leased = LeasedEvent(
            event.event_id,
            event.tenant_id,
            event.job_id,
            event.event_type,
            MappingProxyType(row.payload),
            row.attempts,
            row.failed_attempts,
        )
        handler = run.event_handlers.get(leased.event_type)
        max_failures = run.settings.generation_max_attempts
        if handler is not None and leased.failed_attempts >= max_failures:
            await handler.give_up(db, leased, ATTEMPTS_EXHAUSTED)
            await _finish_event(db, run, leased.id, None, error_code=ATTEMPTS_EXHAUSTED)
        else:
            step = Step(
                run.providers,
                run.sink,
                run.settings,
                run.now or datetime.now(UTC),
                leased.attempts,
                heartbeat,
            )
            try:
                if handler is None:
                    raise NoHandlerError
                retry = await handler.handle(db, leased, step)
            except NoHandlerError:
                await _no_handler(db, run, leased, handler)
            else:
                await _finish_event(db, run, leased.id, retry)
        await db.commit()


async def _no_handler(
    db: AsyncSession, run: _Run, event: LeasedEvent, handler: EventHandler | None
) -> None:
    # Counts as a failure, so an event nobody can run is not retried forever.
    failures = event.failed_attempts + 1
    logger.error(
        "no handler for event=%s type=%s job=%s tenant=%s failures=%s",
        event.id,
        event.event_type,
        event.job_id,
        event.tenant_id,
        failures,
    )
    if failures < run.settings.generation_max_attempts:
        retry: float | None = run.settings.worker_no_handler_retry_seconds
    else:
        retry = None
        if handler is not None:
            await handler.give_up(db, event, NO_HANDLER)
    await _finish_event(db, run, event.id, retry, error_code=NO_HANDLER, failed=True)


def _fenced(event_id: UUID, owner: str) -> Update:
    # Only the worker whose lease is still valid may change the event.
    return (
        update(OutboxEvent)
        .where(
            OutboxEvent.id == event_id,
            OutboxEvent.lease_owner == owner,
            OutboxEvent.leased_until > func.clock_timestamp(),
        )
        .execution_options(synchronize_session=False)
    )


async def _finish_event(
    db: AsyncSession,
    run: _Run,
    event_id: UUID,
    retry_after: float | None,
    *,
    error_code: str | None = None,
    failed: bool = False,
) -> None:
    # Clearing leased_until marks the lease as finished, so the next lease counts no failure.
    values: dict[str, object] = {
        "leased_until": None,
        "lease_owner": None,
        "last_error_code": error_code,
    }
    if failed:
        values["failed_attempts"] = OutboxEvent.failed_attempts + 1
    if retry_after is None:
        values["processed_at"] = func.now()
    else:
        values["available_at"] = func.clock_timestamp() + timedelta(seconds=retry_after)
    finished = await db.scalar(
        _fenced(event_id, run.owner).values(**values).returning(OutboxEvent.id)
    )
    if finished is None:
        raise LeaseLostError


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
        idempotency_key=job.provider_idempotency_key or provider_idempotency_key(job.id),
    )


async def _estimate_usd_micros(provider: ImageProvider, request: ImageRequest) -> int:
    usd = await provider.estimate_cost_usd(request)
    if usd is None:
        # Never submit without a price: a missing estimate is a non-retryable provider error.
        raise ProviderError("The provider returned no cost estimate.")
    return _usd_micros(usd)


def _usd_micros(usd: Decimal) -> int:
    # Money stays in integer units; round up so the cap is never undercut.
    return int((usd * 1_000_000).to_integral_value(rounding=ROUND_CEILING))


def _log_call(
    call: str,
    job: Job,
    provider: ImageProvider,
    provider_request_id: str | None,
    cost_usd_micros: int | None,
) -> None:
    # Ids and numbers only: never the key, the prompt or output URLs.
    logger.info(
        "provider call=%s job=%s tenant=%s provider_request=%s model=%s cost_usd_micros=%s",
        call,
        job.id,
        job.tenant_id,
        provider_request_id,
        provider.model,
        cost_usd_micros,
    )


async def _retry_or_fail(
    db: AsyncSession, job: Job, error: ProviderError, settings: Settings
) -> float | None:
    """For errors where the provider certainly did not accept the request.

    Retryable ones (429, 423, 5xx on estimate or status, connect failures) back off and try
    again with the same key until the job's deadline, which the handler checks first. Anything
    else (400, 401, 402, 403, 404, 422) fails at once without billing.
    """
    attempts = job.attempt_count + 1
    if error.retryable:
        await transition(
            db,
            job.id,
            allowed_from={ExecutionState.QUEUED, ExecutionState.RUNNING},
            to=ExecutionState.RUNNING,
            stage=JobStage.RETRY_WAIT,
            attempt_count=attempts,
        )
        return max(error.retry_after_seconds or 0.0, backoff_seconds(attempts, settings))
    logger.warning(
        "provider refused job=%s tenant=%s code=%s reason=%s",
        job.id,
        job.tenant_id,
        error.code,
        error.message,
    )
    await _fail_with(db, job.id, error.code, error.message, billed=False, attempt_count=attempts)
    return None


def _log_stuck_unknown(job: Job, reason: str) -> None:
    # Given up while the provider may have the request: no worker touches it again and its
    # credits stay HELD, so alerting needs to see it. Ids only.
    logger.error(
        "job stuck in UNKNOWN job=%s tenant=%s provider_request=%s stage=%s reason=%s",
        job.id,
        job.tenant_id,
        job.provider_request_id,
        job.stage,
        reason,
    )


def _output_expires_at(job: Job, status: StatusResult, step: Step) -> datetime:
    """When a completed output stops being downloadable.

    The provider's expiry if it gave one; otherwise PROVIDER_OUTPUT_RETENTION_FALLBACK_SECONDS
    after the worker first saw the output completed. That moment is the job's updated_at once it
    is in stage fetching_output: the transition into that stage is committed before the first
    download, and nothing updates the job again until it leaves the stage, so it survives worker
    restarts without a new column. Before that transition, it is now.
    """
    if status.outputs_expire_at is not None:
        return status.outputs_expire_at
    logger.warning(
        "provider output has no expires_at, assuming the retention fallback job=%s tenant=%s "
        "provider_request=%s",
        job.id,
        job.tenant_id,
        status.provider_request_id,
    )
    seen_completed = job.updated_at if job.stage == JobStage.FETCHING_OUTPUT else step.now
    fallback = timedelta(seconds=step.settings.provider_output_retention_fallback_seconds)
    return seen_completed + fallback


def _log_output_host_refused(job: Job, provider_request_id: str, host: str) -> None:
    # Ops: add the host to HIGGSFIELD_OUTPUT_HOSTS and restart the worker; the job keeps
    # retrying until the output expires. The host only, never the signed URL's path or query.
    logger.error(
        "provider output host not allowed job=%s tenant=%s provider_request=%s host=%s",
        job.id,
        job.tenant_id,
        provider_request_id,
        host,
    )


def _log_review_stop(job: Job, code: str) -> None:
    # The job stays UNKNOWN with its credits HELD; scripts/list_unknown_jobs.py lists it.
    logger.warning(
        "job stopped for operator review job=%s tenant=%s code=%s", job.id, job.tenant_id, code
    )


async def _fail_with(
    db: AsyncSession,
    job_id: UUID,
    code: str,
    message: str,
    *,
    billed: bool,
    actual_usd_micros: int | None = None,
    **values: object,
) -> None:
    # Credits and provider spend are committed only when the provider billed the request.
    if actual_usd_micros is not None:
        values["provider_cost_actual_usd_micros"] = actual_usd_micros
    moved = await transition(
        db,
        job_id,
        allowed_from=NON_TERMINAL_STATES,
        to=ExecutionState.FAILED,
        failure_code=code,
        failure_message=message[:FAILURE_MESSAGE_MAX_CHARS],
        **values,
    )
    if moved:
        settled = ReservationState.COMMITTED if billed else ReservationState.RELEASED
        await settle_reservation(db, job_id, settled)
        await spend.settle(db, job_id, billed=billed, actual_usd_micros=actual_usd_micros)


async def _park_unknown(db: AsyncSession, job_id: UUID, code: str) -> None:
    # The provider may have the request: hold the credits until reconcile or an operator settles.
    moved = await transition(
        db,
        job_id,
        allowed_from={
            ExecutionState.RUNNING,
            ExecutionState.WAITING_PROVIDER,
            ExecutionState.UNKNOWN,
        },
        to=ExecutionState.UNKNOWN,
        stage=JobStage.RECONCILE,
        failure_code=code,
    )
    if moved:
        await settle_reservation(db, job_id, ReservationState.HELD)


async def _give_up(db: AsyncSession, job: Job, code: str, message: str) -> None:
    """Stop working on a job past its deadline or attempt cap. The worker never leases it again.

    - The provider completed (billed) the generation but the output was never stored: FAILED
      OUTPUT_UNAVAILABLE with the credits and spend committed.
    - The provider may have the request (submitted, accepted, or already UNKNOWN): UNKNOWN with
      the credits HELD, for an operator (scripts/list_unknown_jobs.py lists these jobs).
    - Otherwise the provider never got it: FAILED with everything released.
    """
    if job.state is ExecutionState.UNKNOWN:
        _log_stuck_unknown(job, code)
        return
    if job.state is ExecutionState.WAITING_PROVIDER and job.stage == JobStage.FETCHING_OUTPUT:
        await _fail_with(
            db,
            job.id,
            OUTPUT_UNAVAILABLE,
            "The provider's image could not be stored in time.",
            billed=True,
        )
        return
    maybe_accepted = job.state is ExecutionState.WAITING_PROVIDER or (
        job.state is ExecutionState.RUNNING and job.stage == JobStage.SUBMITTING
    )
    if maybe_accepted:
        await _park_unknown(db, job.id, code)
        _log_stuck_unknown(job, code)
    else:
        await _fail_with(db, job.id, code, message, billed=False)


def check_worker_settings(settings: Settings) -> None:
    # A provider call must end before the lease that covers it expires.
    if settings.http_timeout_seconds >= lease_seconds(settings):
        raise WorkerConfigError("HTTP_TIMEOUT_SECONDS must be below WORKER_LEASE_SECONDS.")


def check_live_mode(settings: Settings) -> None:
    # Live mode must have real credentials and a real validation gate; it never falls back.
    problems = missing_live_settings(settings)
    if not settings.aws_s3_bucket:
        # Without storage, finished jobs would park forever; production never runs that way.
        problems.append("missing AWS_S3_BUCKET (generated images need storage)")
    if generation_checks.VALIDATION_GATE_IS_STUB:
        problems.append("the validation-PASS gate is still the stub (NRC-7)")
    if problems:
        raise WorkerConfigError(f"GENERATION_MODE=live refused: {'; '.join(problems)}")


def build_providers(settings: Settings) -> dict[str, ImageProvider]:
    check_worker_settings(settings)
    if settings.generation_mode == "simulator":
        return {SIMULATOR_NAME: SimulatedImageProvider()}
    check_live_mode(settings)
    return {HIGGSFIELD_NAME: create_provider(settings)}


def load_provider_secrets(settings: Settings) -> Settings:
    # Staging/production read the Higgsfield key from Secrets Manager; it stays in memory only.
    try:
        return with_higgsfield_key(settings)
    except ProviderSecretError as error:
        raise WorkerConfigError(str(error)) from None


def worker_database_url() -> SecretStr:
    # The worker connects as marketing_worker, never as the API's role.
    try:
        return WorkerDatabaseSettings().worker_database_url
    except ValidationError:
        raise WorkerConfigError("WORKER_DATABASE_URL is missing or invalid.") from None


async def run_forever(settings: Settings) -> None:
    database_url = worker_database_url()
    settings = await asyncio.to_thread(load_provider_secrets, settings)
    providers = build_providers(settings)
    try:
        await _serve(settings, providers, database_url)
    finally:
        await close_providers(providers)
    logger.info("connector worker stopped")


async def _serve(
    settings: Settings, providers: Mapping[str, ImageProvider], database_url: SecretStr
) -> None:
    engine = create_engine(settings, database_url)
    sessions = create_sessionmaker(engine)
    stop = asyncio.Event()
    _stop_on_signals(stop)
    # Simulator mode may run without storage: completed jobs then park at the provider result.
    sink = StoredImageSink(ObjectStorage(settings), settings) if settings.aws_s3_bucket else None
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


async def close_providers(providers: Mapping[str, ImageProvider]) -> None:
    for provider in providers.values():
        if isinstance(provider, _Closable):
            await provider.aclose()


def _stop_on_signals(stop: asyncio.Event) -> None:
    # Finish the current event, then exit.
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
    configure_pillow(settings)
    try:
        asyncio.run(run_forever(settings))
    except WorkerConfigError as error:
        logger.error("connector worker refused to start: %s", error)
        raise SystemExit(1) from error


if __name__ == "__main__":
    main()
