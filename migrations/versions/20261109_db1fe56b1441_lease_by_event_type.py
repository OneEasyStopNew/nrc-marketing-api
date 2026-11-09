"""The worker leases only the outbox event types it has handlers for.

lease_outbox_events() gains an event_types argument. Events of other types (asset.generated,
until the NRC-7 validation handler subscribes) stay untouched: never leased, so never retried
or given up. Same clamps, SECURITY DEFINER, search_path and grants as before. A new argument
changes the function's signature, so the old one is dropped rather than left as an overload
that leases everything.

Revision ID: db1fe56b1441
Revises: 440a8d5dba9a
Create Date: 2026-11-09 22:00:00.000000
"""

from collections.abc import Sequence

from alembic import op

revision: str = "db1fe56b1441"
down_revision: str | None = "440a8d5dba9a"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

APP_ROLE = "marketing_app"
OLD_FUNCTION = "lease_outbox_events(text, integer, integer)"
NEW_FUNCTION = "lease_outbox_events(text, integer, integer, text[])"

# batch is clamped to 1..50 and lease_seconds to 5..600, whatever the caller passes.
LEASE_BODY = """
    WITH due AS (
        SELECT o.id FROM public.outbox_events AS o
        WHERE o.processed_at IS NULL
          AND o.available_at <= clock_timestamp()
          AND (o.leased_until IS NULL OR o.leased_until <= clock_timestamp())
          {event_type_filter}
        ORDER BY o.available_at
        LIMIT LEAST(GREATEST(batch, 1), 50)
        FOR UPDATE SKIP LOCKED
    )
    UPDATE public.outbox_events AS o
    SET leased_until = clock_timestamp()
            + make_interval(secs => LEAST(GREATEST(lease_seconds, 5), 600)),
        lease_owner = owner,
        attempts = o.attempts + 1
    FROM due
    WHERE o.id = due.id
    RETURNING o.id, o.tenant_id, o.job_id, o.event_type::text
"""

CREATE_NEW = f"""
CREATE OR REPLACE FUNCTION lease_outbox_events(
    owner text, batch integer, lease_seconds integer, event_types text[]
)
RETURNS TABLE (event_id uuid, tenant_id uuid, job_id uuid, event_type text)
LANGUAGE sql VOLATILE SECURITY DEFINER
SET search_path = pg_catalog, public, pg_temp
AS $$
{LEASE_BODY.format(event_type_filter="AND o.event_type = ANY(event_types)")}
$$
"""

CREATE_OLD = f"""
CREATE OR REPLACE FUNCTION lease_outbox_events(owner text, batch integer, lease_seconds integer)
RETURNS TABLE (event_id uuid, tenant_id uuid, job_id uuid, event_type text)
LANGUAGE sql VOLATILE SECURITY DEFINER
SET search_path = pg_catalog, public, pg_temp
AS $$
{LEASE_BODY.format(event_type_filter="")}
$$
"""


def upgrade() -> None:
    op.execute(f"DROP FUNCTION {OLD_FUNCTION}")
    op.execute(CREATE_NEW)
    op.execute(f"REVOKE ALL ON FUNCTION {NEW_FUNCTION} FROM PUBLIC")
    op.execute(f"GRANT EXECUTE ON FUNCTION {NEW_FUNCTION} TO {APP_ROLE}")


def downgrade() -> None:
    op.execute(f"DROP FUNCTION {NEW_FUNCTION}")
    op.execute(CREATE_OLD)
    op.execute(f"REVOKE ALL ON FUNCTION {OLD_FUNCTION} FROM PUBLIC")
    op.execute(f"GRANT EXECUTE ON FUNCTION {OLD_FUNCTION} TO {APP_ROLE}")
