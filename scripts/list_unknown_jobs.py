"""List generation jobs that wait for an operator: `tenant=<id> job=<id> state=<state> code=<code>`.

These are UNKNOWN jobs the worker stopped on, and CANCELLED jobs whose provider outcome was
unknown when they were cancelled (failure_code CANCELLED_OUTCOME_UNKNOWN). The code is the job's
failure_code: why it stopped (e.g. ATTEMPTS_EXHAUSTED, DEADLINE_EXCEEDED), or "-" when none.

Usage (runs as the database owner, MIGRATION_DATABASE_URL, in a read-only transaction):
    uv run python -m scripts.list_unknown_jobs

No worker acts on these jobs again and their credits stay HELD: check each one with the
provider (by job id, which is in its idempotency key), then replay or settle it with
scripts/job_operations.py. Prints ids, the state and the code only.
"""

import asyncio
import sys

from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, create_async_engine
from sqlalchemy.pool import NullPool

from app.core.config import OwnerDatabaseSettings
from app.db.session import TenantSession
from app.services.jobs import awaiting_review_details


def read_only_engine(database_url: str) -> AsyncEngine:
    # The server rejects any write made through this engine.
    return create_async_engine(
        database_url,
        poolclass=NullPool,
        connect_args={"server_settings": {"default_transaction_read_only": "on"}},
    )


async def list_unknown_jobs(database_url: str) -> list[str]:
    engine = read_only_engine(database_url)
    try:
        async with AsyncSession(engine, sync_session_class=TenantSession) as db:
            found = await awaiting_review_details(db)
    finally:
        await engine.dispose()
    return [
        f"tenant={item.tenant_id} job={item.job_id} state={item.state.value}"
        f" code={item.failure_code or '-'}"
        for item in found
    ]


def main() -> int:
    url = OwnerDatabaseSettings().migration_database_url.get_secret_value()
    for line in asyncio.run(list_unknown_jobs(url)):
        print(line)
    return 0


if __name__ == "__main__":
    sys.exit(main())
