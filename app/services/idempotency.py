# Idempotency keys for mutating commands, scoped by tenant and route.

import hashlib
import json
from uuid import UUID

from pydantic import BaseModel
from sqlalchemy import select, update
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.errors import ConflictError, IdempotencyKeyReusedError
from app.models import IdempotencyRecord


def request_hash(subject: str, body: BaseModel) -> str:
    # Stable fingerprint of who sent which payload.
    canonical = json.dumps(
        {"sub": subject, "body": body.model_dump(mode="json")},
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )
    return hashlib.sha256(canonical.encode()).hexdigest()


async def claim(
    db: AsyncSession, tenant_id: UUID, route: str, key: str, fingerprint: str
) -> UUID | None:
    """Claim the key inside the caller's transaction.

    Returns None when the key is new and the caller should do the work, or the resource ID
    created by the earlier identical request. A concurrent duplicate waits on the unique key
    until the first transaction commits.

    Raises:
        IdempotencyKeyReusedError: the key was used before with a different payload.
    """
    claimed = await db.scalar(
        insert(IdempotencyRecord)
        .values(tenant_id=tenant_id, route=route, key=key, request_hash=fingerprint)
        .on_conflict_do_nothing(index_elements=["tenant_id", "route", "key"])
        .returning(IdempotencyRecord.key)
    )
    if claimed is not None:
        return None

    existing = (
        await db.execute(
            select(IdempotencyRecord).where(
                IdempotencyRecord.tenant_id == tenant_id,
                IdempotencyRecord.route == route,
                IdempotencyRecord.key == key,
            )
        )
    ).scalar_one()
    if existing.request_hash != fingerprint:
        raise IdempotencyKeyReusedError()
    if existing.resource_id is None:
        raise ConflictError("A request with this Idempotency-Key is still being processed.")
    return existing.resource_id


async def complete(
    db: AsyncSession, tenant_id: UUID, route: str, key: str, resource_id: UUID
) -> None:
    await db.execute(
        update(IdempotencyRecord)
        .where(
            IdempotencyRecord.tenant_id == tenant_id,
            IdempotencyRecord.route == route,
            IdempotencyRecord.key == key,
        )
        .values(resource_id=resource_id)
        .execution_options(synchronize_session=False)
    )
