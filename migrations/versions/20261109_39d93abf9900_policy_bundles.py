"""Policy bundles: versioned, tenant-scoped rules that validation checks creatives against.

The highest version per tenant is the active policy. Rows are immutable and loaded by the
owner (seed script or a later policy-admin flow), so the API role may only read them.

Revision ID: 39d93abf9900
Revises: de579cd5c63a
Create Date: 2026-11-09 12:00:00.000000
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "39d93abf9900"
down_revision: str | None = "de579cd5c63a"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

APP_ROLE = "marketing_app"
TABLE = "policy_bundles"
# Unset or reset settings read as '' after a transaction ends; NULLIF makes that match nothing.
CURRENT_TENANT = "NULLIF(current_setting('app.tenant_id', true), '')::uuid"


def upgrade() -> None:
    op.create_table(
        TABLE,
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("tenant_id", sa.Uuid(), nullable=False),
        sa.Column("version", sa.Integer(), nullable=False),
        sa.Column("name", sa.String(length=200), nullable=False),
        sa.Column("source_ref", sa.String(length=300), nullable=False),
        sa.Column("rules", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column("is_synthetic", sa.Boolean(), nullable=False),
        sa.Column("approved_by", sa.String(length=200), nullable=False),
        sa.Column("approved_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.ForeignKeyConstraint(
            ["tenant_id"], ["tenants.id"], name=op.f("fk_policy_bundles_tenant_id_tenants")
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_policy_bundles")),
        sa.UniqueConstraint("tenant_id", "id", name=op.f("uq_policy_bundles_tenant_id_id")),
        sa.UniqueConstraint(
            "tenant_id", "version", name=op.f("uq_policy_bundles_tenant_id_version")
        ),
    )

    # The table name and role are the constants above, never user input.
    op.execute(f"ALTER TABLE {TABLE} ENABLE ROW LEVEL SECURITY")
    op.execute(f"ALTER TABLE {TABLE} FORCE ROW LEVEL SECURITY")
    op.execute(
        f"CREATE POLICY tenant_isolation ON {TABLE} "
        f"USING (tenant_id = {CURRENT_TENANT}) WITH CHECK (tenant_id = {CURRENT_TENANT})"
    )
    op.execute(f"GRANT SELECT ON {TABLE} TO {APP_ROLE}")


def downgrade() -> None:
    # Dropping the table also drops its policy and grant.
    op.drop_table(TABLE)
