"""List UNKNOWN generation jobs that wait for an operator, one `tenant=<id> job=<id>` per line.

Usage (runs as the database owner, MIGRATION_DATABASE_URL, in a read-only transaction):
    uv run python -m scripts.list_unknown_jobs

The worker has stopped on these jobs and their credits stay HELD: check each one with the
provider (by job id, which is in its idempotency key), then settle it. Prints ids only.
"""

import asyncio
import sys

from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, create_async_engine
from sqlalchemy.pool import NullPool

from app.core.config import OwnerDatabaseSettings
from app.db.session import TenantSession
from app.services.jobs import jobs_awaiting_review


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
            found = await jobs_awaiting_review(db)
    finally:
        await engine.dispose()
    return [f"tenant={tenant_id} job={job_id}" for tenant_id, job_id in found]


def main() -> int:
    url = OwnerDatabaseSettings().migration_database_url.get_secret_value()
    for line in asyncio.run(list_unknown_jobs(url)):
        print(line)
    return 0


if __name__ == "__main__":
    sys.exit(main())
