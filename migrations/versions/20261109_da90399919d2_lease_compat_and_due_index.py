"""Keep the three-argument lease function for older workers, and index due events by type.

db1fe56b1441 dropped lease_outbox_events(owner, batch, lease_seconds), which workers deployed
before it still call during a rolling deploy. It returns as a thin wrapper over the
four-argument function, leasing only job.run: the one event type those workers handle.
ix_outbox_events_due gains event_type first, so the lease (event_type = ANY(...)) skips
unprocessed events of types no worker handles yet (asset.generated) instead of scanning them.

Revision ID: da90399919d2
Revises: db1fe56b1441
Create Date: 2026-11-09 23:00:00.000000
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "da90399919d2"
down_revision: str | None = "db1fe56b1441"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

APP_ROLE = "marketing_app"
WRAPPER = "lease_outbox_events(text, integer, integer)"

# TODO(NRC-6): drop this wrapper in the release after NRC-10, once no older worker runs.
# SECURITY INVOKER: the caller needs EXECUTE on the four-argument function as well.
CREATE_WRAPPER = """
CREATE FUNCTION lease_outbox_events(owner text, batch integer, lease_seconds integer)
RETURNS TABLE (event_id uuid, tenant_id uuid, job_id uuid, event_type text)
LANGUAGE sql VOLATILE SECURITY INVOKER
SET search_path = pg_catalog, public, pg_temp
AS $$
    SELECT * FROM public.lease_outbox_events(owner, batch, lease_seconds, ARRAY['job.run'])
$$
"""
DUE = sa.text("processed_at IS NULL")


def upgrade() -> None:
    op.execute(CREATE_WRAPPER)
    op.execute(f"REVOKE ALL ON FUNCTION {WRAPPER} FROM PUBLIC")
    op.execute(f"GRANT EXECUTE ON FUNCTION {WRAPPER} TO {APP_ROLE}")
    op.drop_index("ix_outbox_events_due", table_name="outbox_events", postgresql_where=DUE)
    op.create_index(
        "ix_outbox_events_due",
        "outbox_events",
        ["event_type", "available_at"],
        unique=False,
        postgresql_where=DUE,
    )


def downgrade() -> None:
    op.drop_index("ix_outbox_events_due", table_name="outbox_events", postgresql_where=DUE)
    op.create_index(
        "ix_outbox_events_due",
        "outbox_events",
        ["available_at"],
        unique=False,
        postgresql_where=DUE,
    )
    op.execute(f"DROP FUNCTION {WRAPPER}")
