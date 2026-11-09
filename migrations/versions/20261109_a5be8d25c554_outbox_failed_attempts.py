"""Count failed processing attempts per outbox event, separately from leases.

outbox_events.attempts counts every lease, including each poll of a long provider job, so it
cannot bound retries. failed_attempts counts failures only: the lease function adds one when it
takes over an event whose previous lease expired unfinished (the worker crashed, hung or raised),
and the worker adds one for an event it has no handler for. Polls and advisory-lock misses
finish their lease normally, so they never count. Same clamps, SECURITY DEFINER, search_path
and grants as before.

Revision ID: a5be8d25c554
Revises: da90399919d2
Create Date: 2026-11-09 23:10:00.000000
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "a5be8d25c554"
down_revision: str | None = "da90399919d2"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

# batch is clamped to 1..50 and lease_seconds to 5..600, whatever the caller passes. A finished
# lease is always cleared (leased_until NULL), so a set one here means it expired unfinished.
LEASE_FUNCTION = """
CREATE OR REPLACE FUNCTION lease_outbox_events(
    owner text, batch integer, lease_seconds integer, event_types text[]
)
RETURNS TABLE (event_id uuid, tenant_id uuid, job_id uuid, event_type text)
LANGUAGE sql VOLATILE SECURITY DEFINER
SET search_path = pg_catalog, public, pg_temp
AS $$
    WITH due AS (
        SELECT o.id FROM public.outbox_events AS o
        WHERE o.processed_at IS NULL
          AND o.available_at <= clock_timestamp()
          AND (o.leased_until IS NULL OR o.leased_until <= clock_timestamp())
          AND o.event_type = ANY(event_types)
        ORDER BY o.available_at
        LIMIT LEAST(GREATEST(batch, 1), 50)
        FOR UPDATE SKIP LOCKED
    )
    UPDATE public.outbox_events AS o
    SET leased_until = clock_timestamp()
            + make_interval(secs => LEAST(GREATEST(lease_seconds, 5), 600)),
        lease_owner = owner,
        attempts = o.attempts + 1{count_failure}
    FROM due
    WHERE o.id = due.id
    RETURNING o.id, o.tenant_id, o.job_id, o.event_type::text
$$
"""
COUNT_FAILURE = """,
        failed_attempts = o.failed_attempts + CASE WHEN o.leased_until IS NULL THEN 0 ELSE 1 END"""


def upgrade() -> None:
    op.add_column(
        "outbox_events",
        sa.Column("failed_attempts", sa.Integer(), server_default=sa.text("0"), nullable=False),
    )
    op.execute(LEASE_FUNCTION.format(count_failure=COUNT_FAILURE))


def downgrade() -> None:
    # The previous body, which does not know the column.
    op.execute(LEASE_FUNCTION.format(count_failure=""))
    op.drop_column("outbox_events", "failed_attempts")
