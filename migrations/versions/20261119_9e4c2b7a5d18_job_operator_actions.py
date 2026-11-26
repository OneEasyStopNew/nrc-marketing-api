"""Operator audit trail for jobs that wait for an operator (NRC-6): job_operator_actions.

One row per operator replay or settlement (scripts/job_operations.py), written in the same
transaction as the change. Insert-only: neither role may update or delete a row, so the trail
cannot be rewritten. The operator script connects as marketing_worker, which gets SELECT and
INSERT. marketing_app gets SELECT only: a role gets only what its code uses, and no API code path
writes these rows, so the API must not be able to forge audit rows (the repo's usual
"SELECT, INSERT for immutable tables" grant would allow it).

Revision ID: 9e4c2b7a5d18
Revises: a1a439290791
Create Date: 2026-11-19 20:00:00.000000
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "9e4c2b7a5d18"
down_revision: str | None = "a1a439290791"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

APP_ROLE = "marketing_app"
WORKER_ROLE = "marketing_worker"
TABLE = "job_operator_actions"
# Unset or reset settings read as '' after a transaction ends; NULLIF makes that match nothing.
CURRENT_TENANT = "NULLIF(current_setting('app.tenant_id', true), '')::uuid"
EXECUTION_STATES = (
    "QUEUED",
    "RUNNING",
    "WAITING_PROVIDER",
    "NEEDS_REVIEW",
    "SUCCEEDED",
    "FAILED",
    "CANCEL_REQUESTED",
    "CANCELLED",
    "UNKNOWN",
)
RESERVATION_STATES = ("RESERVED", "HELD", "COMMITTED", "RELEASED")
OPERATOR_ACTIONS = ("replay", "settle_billed", "settle_not_billed")


def _enum(name: str, *values: str) -> sa.Enum:
    return sa.Enum(*values, name=name, native_enum=False, length=32)


def upgrade() -> None:
    op.create_table(
        TABLE,
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("tenant_id", sa.Uuid(), nullable=False),
        sa.Column("job_id", sa.Uuid(), nullable=False),
        sa.Column("action", _enum("operatoraction", *OPERATOR_ACTIONS), nullable=False),
        sa.Column("operator", sa.String(length=128), nullable=False),
        sa.Column("reason", sa.String(length=500), nullable=False),
        sa.Column("job_state_before", _enum("executionstate", *EXECUTION_STATES), nullable=False),
        sa.Column("job_state_after", _enum("executionstate", *EXECUTION_STATES), nullable=False),
        sa.Column(
            "reservation_state_before",
            _enum("reservationstate", *RESERVATION_STATES),
            nullable=False,
        ),
        sa.Column(
            "reservation_state_after",
            _enum("reservationstate", *RESERVATION_STATES),
            nullable=False,
        ),
        sa.Column("outbox_event_id", sa.Uuid(), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.ForeignKeyConstraint(
            ["tenant_id", "job_id"],
            ["jobs.tenant_id", "jobs.id"],
            name=op.f("fk_job_operator_actions_tenant_id_jobs"),
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_job_operator_actions")),
    )
    op.create_index(
        "ix_job_operator_actions_tenant_id_job_id", TABLE, ["tenant_id", "job_id"], unique=False
    )

    # Table names and privileges are the constants above, never user input.
    op.execute(f"ALTER TABLE {TABLE} ENABLE ROW LEVEL SECURITY")
    op.execute(f"ALTER TABLE {TABLE} FORCE ROW LEVEL SECURITY")
    op.execute(
        f"CREATE POLICY tenant_isolation ON {TABLE} "
        f"USING (tenant_id = {CURRENT_TENANT}) WITH CHECK (tenant_id = {CURRENT_TENANT})"
    )
    op.execute(f"GRANT SELECT ON {TABLE} TO {APP_ROLE}")
    op.execute(f"GRANT SELECT, INSERT ON {TABLE} TO {WORKER_ROLE}")


def downgrade() -> None:
    op.drop_index("ix_job_operator_actions_tenant_id_job_id", table_name=TABLE)
    # Dropping the table also drops its policy and grants.
    op.drop_table(TABLE)
