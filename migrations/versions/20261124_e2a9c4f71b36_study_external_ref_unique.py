"""One study per source id in a tenant (NRC-2 snapshot import).

scripts.import_study_snapshot finds a study by (tenant_id, external_ref), so a source id may
name only one study per tenant. Partial: studies without a source id (synthetic seeds) are not
affected. studies is small (a few rows per tenant), so a plain CREATE INDEX is fine here.
Downgrade drops the index only; no data is lost.

Revision ID: e2a9c4f71b36
Revises: 7c41d2e9a8b0
Create Date: 2026-11-24 12:00:00.000000
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "e2a9c4f71b36"
down_revision: str | None = "7c41d2e9a8b0"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

INDEX = "uq_studies_tenant_id_external_ref"


def upgrade() -> None:
    op.create_index(
        INDEX,
        "studies",
        ["tenant_id", "external_ref"],
        unique=True,
        postgresql_where=sa.text("external_ref IS NOT NULL"),
    )


def downgrade() -> None:
    op.drop_index(INDEX, table_name="studies")
