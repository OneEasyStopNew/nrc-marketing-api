# Pages for list endpoints: newest first over a timestamp that can tie (GET /media,
# GET /assets/uploads), or in a fixed order after a cursor row (policy clauses, nodes, edges,
# rules).

from collections.abc import Callable, Sequence
from datetime import datetime
from typing import Any
from uuid import UUID

from sqlalchemy import ColumnElement, Row, Select, tuple_
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import InstrumentedAttribute

from app.core.errors import NotFoundError


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


async def page_after[*Ts](
    db: AsyncSession,
    query: Select[*Ts],
    *,
    order: Sequence[ColumnElement[Any] | InstrumentedAttribute[Any]],
    row_id: InstrumentedAttribute[UUID],
    id_of: Callable[[Row[*Ts]], UUID],
    after: UUID | None,
    limit: int,
) -> tuple[Sequence[Row[*Ts]], UUID | None]:
    """One page of `query` in `order`, after the row whose id is `after`, and the next cursor.

    `order` ends with a unique column (the id), so rows never tie and none is skipped or repeated.
    The cursor is the last row's id; its sort values are read back from `query`, so a cursor
    from another list or tenant names no row there (404).
    """
    query = query.order_by(*order)
    if after is not None:
        anchor = (
            await db.execute(query.with_only_columns(*order).where(row_id == after))
        ).one_or_none()
        if anchor is None:
            raise NotFoundError("The cursor names no item in this list.")
        query = query.where(tuple_(*order) > tuple_(*anchor))
    rows = (await db.execute(query.limit(limit + 1))).all()
    page = rows[:limit]
    return page, id_of(page[-1]) if len(rows) > limit else None
