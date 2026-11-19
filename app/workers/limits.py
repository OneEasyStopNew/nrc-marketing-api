"""Provider limits in the connector worker (NRC-6): a concurrency cap and a circuit breaker.

Concurrency: at most GENERATION_MAX_IN_FLIGHT_PER_PROVIDER generations per provider at once,
across all tenants and worker processes, so autoscaling the worker never exceeds the provider's
quota. `submit_slot_free` runs in the transaction that moves a job to its submitting stage: a
per-provider advisory lock held until that transaction commits serialises the check and the move,
so two workers can never both take the last slot. The lock never spans a provider call.

Circuit breaker: per provider in each worker process. After PROVIDER_CIRCUIT_FAILURE_THRESHOLD
transient failures in a row on estimate or submit, new estimates and submits fail fast with a
retryable ProviderUnavailableError for PROVIDER_CIRCUIT_OPEN_SECONDS instead of calling the
provider; the job's normal retry backoff and deadline apply. Status polls and output fetches are
never blocked: an accepted generation is billed whether or not we look at it.
"""

import logging
import time
from collections.abc import Callable, Mapping
from decimal import Decimal
from typing import Final

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import Settings
from app.services.providers.base import (
    FetchedImage,
    ImageProvider,
    ImageRequest,
    ProviderError,
    ProviderOutcomeUnknownError,
    ProviderUnavailableError,
    StatusResult,
    SubmitResult,
)

logger = logging.getLogger(__name__)

_SUBMIT_LOCK_PREFIX: Final = "provider-submit:"


async def submit_slot_free(db: AsyncSession, provider: str, settings: Settings) -> bool:
    """Whether the provider has room for one more generation. Call it in the transaction that
    moves the job to its submitting stage, right before that move.

    Takes a transaction-level advisory lock per provider, released when that transaction ends,
    and counts in-flight jobs across tenants through provider_in_flight() (a SECURITY DEFINER
    function that returns a count only).
    """
    await db.execute(
        select(func.pg_advisory_xact_lock(func.hashtextextended(_SUBMIT_LOCK_PREFIX + provider, 0)))
    )
    in_flight = await db.scalar(select(func.provider_in_flight(provider)))
    return (in_flight or 0) < settings.generation_max_in_flight_per_provider


class CircuitBreaker:
    # Consecutive transient failures of one provider, in this process only.

    def __init__(
        self,
        name: str,
        settings: Settings,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._name = name
        self._threshold = settings.provider_circuit_failure_threshold
        self._open_seconds = settings.provider_circuit_open_seconds
        self._clock = clock
        self._failures = 0
        self._open_until = 0.0

    def check(self) -> None:
        # Raises while open; after the open period one request is let through to probe.
        remaining = self._open_until - self._clock()
        if remaining > 0:
            raise ProviderUnavailableError(
                "The provider is failing; new requests are paused briefly.",
                retry_after_seconds=remaining,
            )

    def record(self, error: ProviderError | None) -> None:
        if error is None:
            self._failures = 0
            return
        if not (error.retryable or isinstance(error, ProviderOutcomeUnknownError)):
            # A refusal (400, 402, ...) says the provider is up.
            self._failures = 0
            return
        self._failures += 1
        if self._failures >= self._threshold:
            self._open_until = self._clock() + self._open_seconds
            self._failures = 0
            logger.warning(
                "provider circuit open provider=%s seconds=%s", self._name, self._open_seconds
            )


class GuardedProvider:
    # An ImageProvider whose estimates and submits go through a circuit breaker.

    def __init__(self, inner: ImageProvider, breaker: CircuitBreaker) -> None:
        self.inner = inner
        self.name = inner.name
        self.model = inner.model
        self.is_simulated = inner.is_simulated
        self._breaker = breaker

    async def estimate_cost_usd(self, request: ImageRequest) -> Decimal | None:
        self._breaker.check()
        try:
            estimate = await self.inner.estimate_cost_usd(request)
        except ProviderError as error:
            self._breaker.record(error)
            raise
        self._breaker.record(None)
        return estimate

    async def submit(self, request: ImageRequest) -> SubmitResult:
        self._breaker.check()
        try:
            result = await self.inner.submit(request)
        except ProviderError as error:
            self._breaker.record(error)
            raise
        self._breaker.record(None)
        return result

    async def get_status(self, provider_request_id: str) -> StatusResult:
        return await self.inner.get_status(provider_request_id)

    async def fetch_output(self, result: StatusResult) -> FetchedImage:
        return await self.inner.fetch_output(result)


def guard_providers(
    providers: Mapping[str, ImageProvider],
    settings: Settings,
    clock: Callable[[], float] = time.monotonic,
) -> dict[str, ImageProvider]:
    # One breaker per provider name, shared by its models: an outage is the provider's, not a
    # model's. The originals stay with the caller, which closes them.
    breakers: dict[str, CircuitBreaker] = {}
    guarded: dict[str, ImageProvider] = {}
    for key, provider in providers.items():
        breaker = breakers.setdefault(provider.name, CircuitBreaker(provider.name, settings, clock))
        guarded[key] = GuardedProvider(provider, breaker)
    return guarded
