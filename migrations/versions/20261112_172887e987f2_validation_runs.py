"""Validation runs: stored, immutable results of checking a creative version (NRC-7).

Each run pins the creative version, policy version and study snapshot version it was checked
against. The API role may insert and read runs, never update or delete them.

Revision ID: 172887e987f2
Revises: 39d93abf9900
Create Date: 2026-11-12 12:00:00.000000
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "172887e987f2"
down_revision: str | None = "39d93abf9900"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

APP_ROLE = "marketing_app"
TABLE = "validation_runs"
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


def upgrade() -> None:
    op.create_table(
        TABLE,
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("tenant_id", sa.Uuid(), nullable=False),
        sa.Column("creative_id", sa.Uuid(), nullable=False),
        sa.Column("creative_version", sa.Integer(), nullable=False),
        sa.Column(
            "input_kind",
            sa.Enum(
                "brief", "text", "image", name="validationinputkind", native_enum=False, length=32
            ),
            nullable=False,
        ),
        sa.Column(
            "state",
            sa.Enum(*EXECUTION_STATES, name="executionstate", native_enum=False, length=32),
            nullable=False,
        ),
        sa.Column(
            "outcome",
            sa.Enum("PASS", "FAIL", "NEEDS_REVIEW", name="outcome", native_enum=False, length=32),
            nullable=True,
        ),
        sa.Column("input_hash", sa.String(length=64), nullable=False),
        sa.Column("policy_bundle_id", sa.Uuid(), nullable=False),
        sa.Column("policy_bundle_version", sa.Integer(), nullable=False),
        sa.Column("study_snapshot_version", sa.Integer(), nullable=False),
        sa.Column("result", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column("created_by", sa.String(length=128), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column("evaluated_at", sa.DateTime(timezone=True), nullable=True),
        sa.ForeignKeyConstraint(
            ["tenant_id", "creative_id"],
            ["creatives.tenant_id", "creatives.id"],
            name=op.f("fk_validation_runs_tenant_id_creatives"),
        ),
        sa.ForeignKeyConstraint(
            ["tenant_id", "policy_bundle_id"],
            ["policy_bundles.tenant_id", "policy_bundles.id"],
            name=op.f("fk_validation_runs_tenant_id_policy_bundles"),
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_validation_runs")),
        sa.UniqueConstraint("tenant_id", "id", name=op.f("uq_validation_runs_tenant_id_id")),
    )
    op.create_index(
        "ix_validation_runs_latest",
        TABLE,
        ["tenant_id", "creative_id", "creative_version", "input_kind", "created_at"],
        unique=False,
    )

    # The table name and role are the constants above, never user input.
    op.execute(f"ALTER TABLE {TABLE} ENABLE ROW LEVEL SECURITY")
    op.execute(f"ALTER TABLE {TABLE} FORCE ROW LEVEL SECURITY")
    op.execute(
        f"CREATE POLICY tenant_isolation ON {TABLE} "
        f"USING (tenant_id = {CURRENT_TENANT}) WITH CHECK (tenant_id = {CURRENT_TENANT})"
    )
    op.execute(f"GRANT SELECT, INSERT ON {TABLE} TO {APP_ROLE}")


def downgrade() -> None:
    # Dropping the table also drops its index, policy and grant.
    op.drop_table(TABLE)
