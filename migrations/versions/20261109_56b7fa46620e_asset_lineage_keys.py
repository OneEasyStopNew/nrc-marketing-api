"""Tie an asset's creative and version to its job, and a job's version to creative_versions.

assets(job_id, tenant_id, creative_id, creative_version) now references the same four columns of
its job, so an asset can only record the creative version its job generated. jobs(creative_id,
creative_version) references creative_versions(creative_id, version), its existing unique key;
the job's (tenant_id, creative_id) key to creatives keeps the tenant consistent.

Revision ID: 56b7fa46620e
Revises: 440a8d5dba9a
Create Date: 2026-11-09 23:20:00.000000
"""

from collections.abc import Sequence

from alembic import op

revision: str = "56b7fa46620e"
down_revision: str | None = "440a8d5dba9a"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

JOB_LINEAGE = ["id", "tenant_id", "creative_id", "creative_version"]


def upgrade() -> None:
    op.create_unique_constraint(
        op.f("uq_jobs_id_tenant_id_creative_id_creative_version"), "jobs", JOB_LINEAGE
    )
    op.create_foreign_key(
        op.f("fk_jobs_creative_id_creative_versions"),
        "jobs",
        "creative_versions",
        ["creative_id", "creative_version"],
        ["creative_id", "version"],
    )
    op.create_foreign_key(
        op.f("fk_assets_job_id_jobs"),
        "assets",
        "jobs",
        ["job_id", "tenant_id", "creative_id", "creative_version"],
        JOB_LINEAGE,
    )


def downgrade() -> None:
    op.drop_constraint(op.f("fk_assets_job_id_jobs"), "assets", type_="foreignkey")
    op.drop_constraint(op.f("fk_jobs_creative_id_creative_versions"), "jobs", type_="foreignkey")
    op.drop_constraint(
        op.f("uq_jobs_id_tenant_id_creative_id_creative_version"), "jobs", type_="unique"
    )
