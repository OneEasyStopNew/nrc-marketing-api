# Helpers shared by the worker processes (connector and media): identity, leases, backoff,
# start-up configuration and graceful stop. No provider or job code, so each worker imports only
# what it runs.

import asyncio
import os
import secrets
import signal
import socket
from typing import Final

from pydantic import SecretStr, ValidationError

from app.core.config import LEASE_MAX_SECONDS, LEASE_MIN_SECONDS, Settings, WorkerDatabaseSettings

_jitter = secrets.SystemRandom()


def process_owner() -> str:
    # Hostname and PID can repeat across containers (both PID 1), so add a random suffix.
    return f"{socket.gethostname()}:{os.getpid()}:{secrets.token_hex(4)}"


# Computed once, so every lease this process takes carries the same owner.
DEFAULT_OWNER: Final = process_owner()


class WorkerConfigError(Exception):
    # The worker cannot start with this configuration. Messages name settings, never values.
    pass


def backoff_seconds(attempt: int, settings: Settings) -> float:
    # Exponential from WORKER_BACKOFF_BASE_SECONDS up to WORKER_BACKOFF_MAX_SECONDS, with jitter
    # between 50% and 100% so many jobs don't retry in lockstep.
    delay = settings.worker_backoff_base_seconds * 2.0 ** max(attempt - 1, 0)
    return min(settings.worker_backoff_max_seconds, delay) * _jitter.uniform(0.5, 1.0)


def lease_seconds(settings: Settings) -> int:
    # Settings validate this range; the clamp also covers copies made without validation.
    return min(max(settings.worker_lease_seconds, LEASE_MIN_SECONDS), LEASE_MAX_SECONDS)


def worker_database_url() -> SecretStr:
    # The worker connects as marketing_worker, never as the API's role.
    try:
        return WorkerDatabaseSettings().worker_database_url
    except ValidationError:
        raise WorkerConfigError("WORKER_DATABASE_URL is missing or invalid.") from None


def stop_on_signals(stop: asyncio.Event) -> None:
    # Finish the current unit of work, then exit.
    loop = asyncio.get_running_loop()
    for signum in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(signum, stop.set)
        except NotImplementedError:
            # Windows event loops have no add_signal_handler.
            signal.signal(signum, lambda *_: loop.call_soon_threadsafe(stop.set))
