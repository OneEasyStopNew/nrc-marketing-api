"""Generation jobs: jobs, outbox events, credit reservations and the worker's lease function.

All three tables follow the foundation's tenant pattern (RLS on app.tenant_id, explicit grants).
The connector worker is the only cross-tenant reader: it leases due outbox rows through
lease_outbox_events(), a SECURITY DEFINER function that returns ids only, and then binds the
tenant of each event, so every other read and write stays under RLS.

Revision ID: ffd6ebe6b868
Revises: de579cd5c63a
Create Date: 2026-11-09 15:31:24.080734
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "ffd6ebe6b868"
down_revision: str | None = "de579cd5c63a"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

APP_ROLE = "marketing_app"
TABLES = ("jobs", "outbox_events", "credit_reservations")
CURRENT_TENANT = "NULLIF(current_setting('app.tenant_id', true), '')::uuid"
LEASE_FUNCTION = "lease_outbox_events(text, integer, integer)"

# Runs as the migration owner. FORCE RLS applies to the owner too, so the outbox_lease policy
# lets only this role (the function's owner) see every tenant's outbox rows.
CREATE_LEASE_FUNCTION = """
CREATE FUNCTION lease_outbox_events(owner text, batch integer, lease_seconds integer)
RETURNS TABLE (event_id uuid, tenant_id uuid, job_id uuid, event_type text)
LANGUAGE sql VOLATILE SECURITY DEFINER
SET search_path = pg_catalog, public, pg_temp
AS $$
    WITH due AS (
        SELECT o.id FROM public.outbox_events AS o
        WHERE o.processed_at IS NULL
          AND o.available_at <= clock_timestamp()
          AND (o.leased_until IS NULL OR o.leased_until <= clock_timestamp())
        ORDER BY o.available_at
        LIMIT batch
        FOR UPDATE SKIP LOCKED
    )
    UPDATE public.outbox_events AS o
    SET leased_until = clock_timestamp() + make_interval(secs => lease_seconds),
        lease_owner = owner,
        attempts = o.attempts + 1
    FROM due
    WHERE o.id = due.id
    RETURNING o.id, o.tenant_id, o.job_id, o.event_type::text
$$
"""


def upgrade() -> None:
    op.create_table(
        "jobs",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("tenant_id", sa.Uuid(), nullable=False),
        sa.Column("kind", sa.String(length=32), nullable=False),
        sa.Column("creative_id", sa.Uuid(), nullable=False),
        sa.Column("creative_version", sa.Integer(), nullable=False),
        sa.Column("study_snapshot_id", sa.Uuid(), nullable=False),
        sa.Column("study_snapshot_version", sa.Integer(), nullable=False),
        sa.Column("policy_bundle_version", sa.String(length=64), nullable=True),
        sa.Column(
            "state",
            sa.Enum(
                "QUEUED",
                "RUNNING",
                "WAITING_PROVIDER",
                "NEEDS_REVIEW",
                "SUCCEEDED",
                "FAILED",
                "CANCEL_REQUESTED",
                "CANCELLED",
                "UNKNOWN",
                name="executionstate",
                native_enum=False,
                length=32,
            ),
            nullable=False,
        ),
        sa.Column("stage", sa.String(length=32), nullable=True),
        sa.Column("attempt_count", sa.Integer(), nullable=False),
        sa.Column("max_attempts", sa.Integer(), nullable=False),
        sa.Column("deadline_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("provider", sa.String(length=32), nullable=True),
        sa.Column("model", sa.String(length=100), nullable=True),
        sa.Column("provider_idempotency_key", sa.Uuid(), nullable=True),
        sa.Column("provider_request_id", sa.String(length=200), nullable=True),
        sa.Column("provider_cost_estimate_usd_micros", sa.BigInteger(), nullable=True),
        sa.Column("failure_code", sa.String(length=64), nullable=True),
        sa.Column("failure_message", sa.String(length=500), nullable=True),
        sa.Column("result_asset_id", sa.Uuid(), nullable=True),
        sa.Column("created_by", sa.String(length=128), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column("completed_at", sa.DateTime(timezone=True), nullable=True),
        sa.ForeignKeyConstraint(
            ["tenant_id", "creative_id"],
            ["creatives.tenant_id", "creatives.id"],
            name=op.f("fk_jobs_tenant_id_creatives"),
        ),
        sa.ForeignKeyConstraint(
            ["tenant_id", "study_snapshot_id"],
            ["study_snapshots.tenant_id", "study_snapshots.id"],
            name=op.f("fk_jobs_tenant_id_study_snapshots"),
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_jobs")),
        sa.UniqueConstraint("tenant_id", "id", name=op.f("uq_jobs_tenant_id_id")),
    )
    op.create_index("ix_jobs_tenant_id_state", "jobs", ["tenant_id", "state"], unique=False)
    op.create_table(
        "credit_reservations",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("tenant_id", sa.Uuid(), nullable=False),
        sa.Column("job_id", sa.Uuid(), nullable=False),
        sa.Column("units_reserved", sa.Integer(), nullable=False),
        sa.Column("units_committed", sa.Integer(), nullable=False),
        sa.Column(
            "state",
            sa.Enum(
                "RESERVED",
                "HELD",
                "COMMITTED",
                "RELEASED",
                name="reservationstate",
                native_enum=False,
                length=32,
            ),
            nullable=False,
        ),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.ForeignKeyConstraint(
            ["tenant_id", "job_id"],
            ["jobs.tenant_id", "jobs.id"],
            name=op.f("fk_credit_reservations_tenant_id_jobs"),
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_credit_reservations")),
        sa.UniqueConstraint("job_id", name=op.f("uq_credit_reservations_job_id")),
    )
    op.create_index(
        "ix_credit_reservations_tenant_id_created_at",
        "credit_reservations",
        ["tenant_id", "created_at"],
        unique=False,
    )
    op.create_table(
        "outbox_events",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("tenant_id", sa.Uuid(), nullable=False),
        sa.Column("job_id", sa.Uuid(), nullable=False),
        sa.Column("event_type", sa.String(length=64), nullable=False),
        sa.Column("payload", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column(
            "available_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column("leased_until", sa.DateTime(timezone=True), nullable=True),
        sa.Column("lease_owner", sa.String(length=128), nullable=True),
        sa.Column("attempts", sa.Integer(), nullable=False),
        sa.Column("processed_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("last_error_code", sa.String(length=64), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.ForeignKeyConstraint(
            ["tenant_id", "job_id"],
            ["jobs.tenant_id", "jobs.id"],
            name=op.f("fk_outbox_events_tenant_id_jobs"),
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_outbox_events")),
    )
    op.create_index(
        "ix_outbox_events_due",
        "outbox_events",
        ["available_at"],
        unique=False,
        postgresql_where=sa.text("processed_at IS NULL"),
    )

    # Table names and privileges are the constants above, never user input.
    for table in TABLES:
        op.execute(f"ALTER TABLE {table} ENABLE ROW LEVEL SECURITY")
        op.execute(f"ALTER TABLE {table} FORCE ROW LEVEL SECURITY")
        op.execute(
            f"CREATE POLICY tenant_isolation ON {table} "
            f"USING (tenant_id = {CURRENT_TENANT}) WITH CHECK (tenant_id = {CURRENT_TENANT})"
        )
        op.execute(f"GRANT SELECT, INSERT, UPDATE ON {table} TO {APP_ROLE}")
    op.execute(
        "CREATE POLICY outbox_lease ON outbox_events TO CURRENT_USER USING (true) WITH CHECK (true)"
    )
    op.execute(CREATE_LEASE_FUNCTION)
    op.execute(f"REVOKE ALL ON FUNCTION {LEASE_FUNCTION} FROM PUBLIC")
    op.execute(f"GRANT EXECUTE ON FUNCTION {LEASE_FUNCTION} TO {APP_ROLE}")


def downgrade() -> None:
    # Dropping a table also drops its policies and grants.
    op.execute(f"DROP FUNCTION {LEASE_FUNCTION}")
    op.drop_index(
        "ix_outbox_events_due",
        table_name="outbox_events",
        postgresql_where=sa.text("processed_at IS NULL"),
    )
    op.drop_table("outbox_events")
    op.drop_index("ix_credit_reservations_tenant_id_created_at", table_name="credit_reservations")
    op.drop_table("credit_reservations")
    op.drop_index("ix_jobs_tenant_id_state", table_name="jobs")
    op.drop_table("jobs")
