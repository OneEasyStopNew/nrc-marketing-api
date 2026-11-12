# Migration details that a schema comparison (alembic check) does not cover.

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncConnection


async def test_due_index_leads_with_the_event_type(connection: AsyncConnection) -> None:
    definition = await connection.scalar(
        text("SELECT indexdef FROM pg_indexes WHERE indexname = 'ix_outbox_events_due'")
    )

    assert "(event_type, available_at)" in str(definition)
    assert "WHERE (processed_at IS NULL)" in str(definition)
