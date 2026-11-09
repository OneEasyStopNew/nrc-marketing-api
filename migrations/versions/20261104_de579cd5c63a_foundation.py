"""Foundation: tenancy, study snapshots, creatives with immutable versions, idempotency.

Every tenant-owned table gets row-level security keyed on the transaction-local setting
app.tenant_id. The API connects as marketing_app, which the deployment must create first
(see docker/postgres-init.sql); that role cannot update or delete creative versions.

Revision ID: de579cd5c63a
Revises:
Create Date: 2026-11-04 14:09:30.812165
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "de579cd5c63a"
down_revision: str | None = None
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

APP_ROLE = "marketing_app"
TENANT_SCOPED_TABLES = (
    "studies",
    "study_snapshots",
    "membership_studies",
    "creatives",
    "creative_versions",
    "idempotency_records",
)
GRANTS = {
    "tenants": "SELECT",
    "memberships": "SELECT",
    "studies": "SELECT",
    "study_snapshots": "SELECT",
    "membership_studies": "SELECT",
    "creatives": "SELECT, INSERT, UPDATE",
    "creative_versions": "SELECT, INSERT",
    "idempotency_records": "SELECT, INSERT, UPDATE",
}
# Unset or reset settings read as '' after a transaction ends; NULLIF makes that match nothing.
CURRENT_TENANT = "NULLIF(current_setting('app.tenant_id', true), '')::uuid"


def upgrade() -> None:
    op.create_table(
        "tenants",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("name", sa.String(length=200), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_tenants")),
    )
    op.create_table(
        "idempotency_records",
        sa.Column("tenant_id", sa.Uuid(), nullable=False),
        sa.Column("route", sa.String(length=100), nullable=False),
        sa.Column("key", sa.String(length=128), nullable=False),
        sa.Column("request_hash", sa.String(length=64), nullable=False),
        sa.Column("resource_id", sa.Uuid(), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.ForeignKeyConstraint(
            ["tenant_id"], ["tenants.id"], name=op.f("fk_idempotency_records_tenant_id_tenants")
        ),
        sa.PrimaryKeyConstraint("tenant_id", "route", "key", name=op.f("pk_idempotency_records")),
    )
    op.create_table(
        "memberships",
        sa.Column("tenant_id", sa.Uuid(), nullable=False),
        sa.Column("user_sub", sa.String(length=128), nullable=False),
        sa.Column("roles", sa.ARRAY(sa.String(length=32)), nullable=False),
        sa.Column("is_active", sa.Boolean(), server_default=sa.text("true"), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.ForeignKeyConstraint(
            ["tenant_id"], ["tenants.id"], name=op.f("fk_memberships_tenant_id_tenants")
        ),
        sa.PrimaryKeyConstraint("tenant_id", "user_sub", name=op.f("pk_memberships")),
    )
    op.create_index(op.f("ix_memberships_user_sub"), "memberships", ["user_sub"], unique=False)
    op.create_table(
        "studies",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("tenant_id", sa.Uuid(), nullable=False),
        sa.Column("title", sa.String(length=300), nullable=False),
        sa.Column("external_ref", sa.String(length=100), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.ForeignKeyConstraint(
            ["tenant_id"], ["tenants.id"], name=op.f("fk_studies_tenant_id_tenants")
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_studies")),
        sa.UniqueConstraint("tenant_id", "id", name=op.f("uq_studies_tenant_id_id")),
    )
    op.create_table(
        "creatives",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("tenant_id", sa.Uuid(), nullable=False),
        sa.Column("study_id", sa.Uuid(), nullable=False),
        sa.Column("original_prompt", sa.Text(), nullable=False),
        sa.Column("current_version", sa.Integer(), nullable=False),
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
        sa.ForeignKeyConstraint(
            ["tenant_id", "study_id"],
            ["studies.tenant_id", "studies.id"],
            name=op.f("fk_creatives_tenant_id_studies"),
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_creatives")),
        sa.UniqueConstraint("tenant_id", "id", name=op.f("uq_creatives_tenant_id_id")),
    )
    op.create_table(
        "membership_studies",
        sa.Column("tenant_id", sa.Uuid(), nullable=False),
        sa.Column("user_sub", sa.String(length=128), nullable=False),
        sa.Column("study_id", sa.Uuid(), nullable=False),
        sa.ForeignKeyConstraint(
            ["tenant_id", "study_id"],
            ["studies.tenant_id", "studies.id"],
            name=op.f("fk_membership_studies_tenant_id_studies"),
        ),
        sa.ForeignKeyConstraint(
            ["tenant_id", "user_sub"],
            ["memberships.tenant_id", "memberships.user_sub"],
            name=op.f("fk_membership_studies_tenant_id_memberships"),
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint(
            "tenant_id", "user_sub", "study_id", name=op.f("pk_membership_studies")
        ),
    )
    op.create_table(
        "study_snapshots",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("tenant_id", sa.Uuid(), nullable=False),
        sa.Column("study_id", sa.Uuid(), nullable=False),
        sa.Column("version", sa.Integer(), nullable=False),
        sa.Column(
            "recruitment_status",
            sa.Enum("recruiting", "closed", name="recruitmentstatus", native_enum=False, length=32),
            nullable=False,
        ),
        sa.Column(
            "language", sa.Enum("en", name="language", native_enum=False, length=32), nullable=False
        ),
        sa.Column("facts", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column("source_ref", sa.String(length=300), nullable=False),
        sa.Column("approved_by", sa.String(length=200), nullable=False),
        sa.Column("approved_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("is_synthetic", sa.Boolean(), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.ForeignKeyConstraint(
            ["tenant_id", "study_id"],
            ["studies.tenant_id", "studies.id"],
            name=op.f("fk_study_snapshots_tenant_id_studies"),
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_study_snapshots")),
        sa.UniqueConstraint(
            "study_id", "version", name=op.f("uq_study_snapshots_study_id_version")
        ),
        sa.UniqueConstraint("tenant_id", "id", name=op.f("uq_study_snapshots_tenant_id_id")),
    )
    op.create_table(
        "creative_versions",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("tenant_id", sa.Uuid(), nullable=False),
        sa.Column("creative_id", sa.Uuid(), nullable=False),
        sa.Column("version", sa.Integer(), nullable=False),
        sa.Column("prompt", sa.Text(), nullable=False),
        sa.Column("caption", sa.Text(), nullable=True),
        sa.Column("study_snapshot_id", sa.Uuid(), nullable=False),
        sa.Column(
            "language", sa.Enum("en", name="language", native_enum=False, length=32), nullable=False
        ),
        sa.Column(
            "channel",
            sa.Enum("facebook", name="channel", native_enum=False, length=32),
            nullable=False,
        ),
        sa.Column(
            "post_format",
            sa.Enum("feed_image", name="postformat", native_enum=False, length=32),
            nullable=False,
        ),
        sa.Column("created_by", sa.String(length=128), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.ForeignKeyConstraint(
            ["tenant_id", "creative_id"],
            ["creatives.tenant_id", "creatives.id"],
            name=op.f("fk_creative_versions_tenant_id_creatives"),
        ),
        sa.ForeignKeyConstraint(
            ["tenant_id", "study_snapshot_id"],
            ["study_snapshots.tenant_id", "study_snapshots.id"],
            name=op.f("fk_creative_versions_tenant_id_study_snapshots"),
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_creative_versions")),
        sa.UniqueConstraint(
            "creative_id", "version", name=op.f("uq_creative_versions_creative_id_version")
        ),
    )

    # Table names and privileges are the constants above, never user input.
    for table in TENANT_SCOPED_TABLES:
        op.execute(f"ALTER TABLE {table} ENABLE ROW LEVEL SECURITY")
        op.execute(f"ALTER TABLE {table} FORCE ROW LEVEL SECURITY")
        op.execute(
            f"CREATE POLICY tenant_isolation ON {table} "
            f"USING (tenant_id = {CURRENT_TENANT}) WITH CHECK (tenant_id = {CURRENT_TENANT})"
        )
    for table, privileges in GRANTS.items():
        op.execute(f"GRANT {privileges} ON {table} TO {APP_ROLE}")


def downgrade() -> None:
    # Dropping a table also drops its policies and grants.
    op.drop_table("creative_versions")
    op.drop_table("study_snapshots")
    op.drop_table("membership_studies")
    op.drop_table("creatives")
    op.drop_table("studies")
    op.drop_index(op.f("ix_memberships_user_sub"), table_name="memberships")
    op.drop_table("memberships")
    op.drop_table("idempotency_records")
    op.drop_table("tenants")
