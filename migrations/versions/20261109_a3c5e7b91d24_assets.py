"""Generated-image assets: immutable metadata rows, plus the jobs.result_asset_id foreign key.

Same tenant pattern as the other tables (RLS on app.tenant_id). The app role gets SELECT and
INSERT only, so a stored asset can never be rewritten.

Revision ID: a3c5e7b91d24
Revises: ffd6ebe6b868
Create Date: 2026-11-09 18:10:00.000000
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "a3c5e7b91d24"
down_revision: str | None = "ffd6ebe6b868"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

APP_ROLE = "marketing_app"
CURRENT_TENANT = "NULLIF(current_setting('app.tenant_id', true), '')::uuid"


def upgrade() -> None:
    op.create_table(
        "assets",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("tenant_id", sa.Uuid(), nullable=False),
        sa.Column("creative_id", sa.Uuid(), nullable=False),
        sa.Column("creative_version", sa.Integer(), nullable=False),
        sa.Column("job_id", sa.Uuid(), nullable=False),
        sa.Column("origin", sa.String(length=16), nullable=False),
        sa.Column("bucket", sa.String(length=63), nullable=False),
        sa.Column("s3_key", sa.String(length=1024), nullable=False),
        sa.Column("s3_version_id", sa.String(length=1024), nullable=True),
        sa.Column("mime_type", sa.String(length=64), nullable=False),
        sa.Column("width_px", sa.Integer(), nullable=False),
        sa.Column("height_px", sa.Integer(), nullable=False),
        sa.Column("size_bytes", sa.BigInteger(), nullable=False),
        sa.Column("checksum_sha256", sa.String(length=64), nullable=False),
        sa.Column("provider", sa.String(length=32), nullable=True),
        sa.Column("model", sa.String(length=100), nullable=True),
        sa.Column("provider_request_id", sa.String(length=200), nullable=True),
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
            name=op.f("fk_assets_tenant_id_creatives"),
        ),
        sa.ForeignKeyConstraint(
            ["tenant_id", "job_id"],
            ["jobs.tenant_id", "jobs.id"],
            name=op.f("fk_assets_tenant_id_jobs"),
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_assets")),
        sa.UniqueConstraint("job_id", name=op.f("uq_assets_job_id")),
        sa.UniqueConstraint("tenant_id", "id", name=op.f("uq_assets_tenant_id_id")),
    )
    op.create_foreign_key(
        op.f("fk_jobs_tenant_id_assets"),
        "jobs",
        "assets",
        ["tenant_id", "result_asset_id"],
        ["tenant_id", "id"],
    )

    op.execute("ALTER TABLE assets ENABLE ROW LEVEL SECURITY")
    op.execute("ALTER TABLE assets FORCE ROW LEVEL SECURITY")
    op.execute(
        "CREATE POLICY tenant_isolation ON assets "
        f"USING (tenant_id = {CURRENT_TENANT}) WITH CHECK (tenant_id = {CURRENT_TENANT})"
    )
    op.execute(f"GRANT SELECT, INSERT ON assets TO {APP_ROLE}")


def downgrade() -> None:
    op.drop_constraint(op.f("fk_jobs_tenant_id_assets"), "jobs", type_="foreignkey")
    # Dropping the table also drops its policy and grants.
    op.drop_table("assets")
