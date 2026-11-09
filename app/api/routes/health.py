# Liveness and readiness probes (not part of the public API contract).

import asyncio

from fastapi import APIRouter
from sqlalchemy import text
from sqlalchemy.exc import SQLAlchemyError

from app.api.deps import DbSession
from app.core.errors import DependencyUnavailableError

READINESS_TIMEOUT_SECONDS = 2

router = APIRouter(include_in_schema=False)


@router.get("/livez")
async def livez() -> dict[str, str]:
    return {"status": "ok"}


@router.get("/readyz")
async def readyz(db: DbSession) -> dict[str, str]:
    try:
        async with asyncio.timeout(READINESS_TIMEOUT_SECONDS):
            await db.execute(text("SELECT 1"))
    except (TimeoutError, SQLAlchemyError) as err:
        raise DependencyUnavailableError("Database is not reachable.") from err
    return {"status": "ok"}
