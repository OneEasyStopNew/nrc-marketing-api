# Engine and session factories, plus the tenant binding that row-level security relies on.

from uuid import UUID

from pydantic import SecretStr
from sqlalchemy import Connection, event, text
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.orm import Session, SessionTransaction

from app.core.config import Settings
from app.core.errors import TenantNotBoundError

TENANT_INFO_KEY = "tenant_id"
_SET_TENANT = text("SELECT set_config('app.tenant_id', :tenant_id, true)")


class TenantSession(Session):
    # Session that re-applies its bound tenant at the start of every transaction.
    pass


@event.listens_for(TenantSession, "after_begin")
def _apply_tenant(
    session: Session, transaction: SessionTransaction, connection: Connection
) -> None:
    # set_config(..., true) only lasts for one transaction, so it is re-applied after each commit.
    tenant_id = session.info.get(TENANT_INFO_KEY)
    if tenant_id is not None:
        connection.execute(_SET_TENANT, {"tenant_id": str(tenant_id)})


def bound_tenant(db: AsyncSession) -> UUID:
    """The tenant this session is bound to (bind_tenant), for code that takes the session only.

    Raises:
        TenantNotBoundError: the session was never bound, so RLS would hide every row.
    """
    tenant_id = db.info.get(TENANT_INFO_KEY)
    if not isinstance(tenant_id, UUID):
        raise TenantNotBoundError()
    return tenant_id


async def bind_tenant(db: AsyncSession, tenant_id: UUID) -> None:
    # Scope the session to one tenant; the database's RLS policies hide every other tenant.
    db.info[TENANT_INFO_KEY] = tenant_id
    await db.execute(_SET_TENANT, {"tenant_id": str(tenant_id)})


def create_engine(settings: Settings, url: SecretStr | None = None) -> AsyncEngine:
    # url: another role's connection (the worker's); the pool settings stay the same.
    return create_async_engine(
        (url or settings.database_url).get_secret_value(),
        pool_size=settings.db_pool_size,
        max_overflow=settings.db_max_overflow,
        pool_pre_ping=True,
        # Database errors (and their logs) show the SQL but never the bound values, which can be
        # post text or other customer content.
        hide_parameters=True,
        # asyncpg: "timeout" bounds connecting, "command_timeout" bounds each statement.
        connect_args={
            "timeout": settings.db_timeout_seconds,
            "command_timeout": settings.db_timeout_seconds,
        },
    )


def create_sessionmaker(engine: AsyncEngine) -> async_sessionmaker[AsyncSession]:
    return async_sessionmaker(engine, expire_on_commit=False, sync_session_class=TenantSession)
