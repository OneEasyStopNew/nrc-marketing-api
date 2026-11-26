"""Context manifests (NRC-5): the immutable record of what one policy retrieval decided.

One row per distinct set of inputs. cache_key hashes every dependency (bundle, study snapshot,
scope and retrieval settings), so the same request reuses its row and a changed dependency gets
a new one. Rows are never updated, so the API role gets SELECT and INSERT only, like
validation_runs. The worker gets nothing yet: it holds exactly what it uses.

Revision ID: 2e16028f516f
Revises: 7d2f9a41c6e8
Create Date: 2026-11-18 10:00:00.000000
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "2e16028f516f"
down_revision: str | None = "7d2f9a41c6e8"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

APP_ROLE = "marketing_app"
TABLE = "context_manifests"
# Unset or reset settings read as '' after a transaction ends; NULLIF makes that match nothing.
CURRENT_TENANT = "NULLIF(current_setting('app.tenant_id', true), '')::uuid"


def upgrade() -> None:
    op.create_table(
        TABLE,
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("tenant_id", sa.Uuid(), nullable=False),
        sa.Column("cache_key", sa.String(length=64), nullable=False),
        sa.Column("policy_bundle_id", sa.Uuid(), nullable=False),
        sa.Column("policy_bundle_version", sa.Integer(), nullable=False),
        sa.Column("study_snapshot_id", sa.Uuid(), nullable=False),
        sa.Column("study_snapshot_version", sa.Integer(), nullable=False),
        sa.Column("retrieval_config_version", sa.Integer(), nullable=False),
        sa.Column("manifest", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column("created_by", sa.String(length=128), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.ForeignKeyConstraint(
            ["tenant_id", "policy_bundle_id"],
            ["policy_bundles.tenant_id", "policy_bundles.id"],
            name=op.f("fk_context_manifests_tenant_id_policy_bundles"),
        ),
        sa.ForeignKeyConstraint(
            ["tenant_id", "study_snapshot_id"],
            ["study_snapshots.tenant_id", "study_snapshots.id"],
            name=op.f("fk_context_manifests_tenant_id_study_snapshots"),
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_context_manifests")),
        sa.UniqueConstraint("tenant_id", "id", name=op.f("uq_context_manifests_tenant_id_id")),
        sa.UniqueConstraint(
            "tenant_id", "cache_key", name=op.f("uq_context_manifests_tenant_id_cache_key")
        ),
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
    # Dropping the table also drops its policy and grant.
    op.drop_table(TABLE)
