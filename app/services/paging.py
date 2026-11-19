# Newest-first pages over a timestamp that can tie, shared by GET /media and GET /assets/uploads.

from collections.abc import Callable, Sequence
from datetime import datetime
from uuid import UUID

from sqlalchemy import Row, Select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import InstrumentedAttribute


async def newest_first_page[*Ts](
    db: AsyncSession,
    query: Select[*Ts],
    *,
    created_at: InstrumentedAttribute[datetime],
    row_id: InstrumentedAttribute[UUID],
    sort_key: Callable[[Row[*Ts]], tuple[datetime, UUID]],
    before: datetime | None,
    limit: int,
) -> tuple[Sequence[Row[*Ts]], datetime | None]:
    """One page of `query`, newest first, and the cursor for the next (older) page.

    The cursor is the page's last timestamp and the next page uses a strict `<` on it, so every
    row tied at that timestamp is included here (FETCH ... WITH TIES), which may exceed `limit`.
    Tied rows are ordered by `sort_key` for a stable display. Older rows are checked with a
    separate query: even limit + 1 rows could all fall inside one tie.
    """
    query = query.order_by(created_at.desc())
    if before is not None:
        query = query.where(created_at < before)
    rows = (await db.execute(query.fetch(limit, with_ties=True))).all()
    page = sorted(rows, key=sort_key, reverse=True)
    if not page:
        return page, None
    boundary = sort_key(page[-1])[0]
    older = await db.scalar(query.with_only_columns(row_id).where(created_at < boundary).limit(1))
    return page, boundary if older is not None else None
