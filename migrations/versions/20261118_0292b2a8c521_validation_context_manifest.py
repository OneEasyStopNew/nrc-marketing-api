"""Link validation runs to the policy context gathered for them (NRC-5).

Adds a nullable validation_runs.context_manifest_id with a composite foreign key that includes
tenant_id, so a run can never point at another tenant's manifest. Existing runs keep NULL (no
context was gathered for them). Expand-only: no existing column changes, and the API role already
has SELECT and INSERT on both tables.

Revision ID: 0292b2a8c521
Revises: 2e16028f516f
Create Date: 2026-11-18 16:00:00.000000
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0292b2a8c521"
down_revision: str | None = "2e16028f516f"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column("validation_runs", sa.Column("context_manifest_id", sa.Uuid(), nullable=True))
    op.create_foreign_key(
        op.f("fk_validation_runs_tenant_id_context_manifests"),
        "validation_runs",
        "context_manifests",
        ["tenant_id", "context_manifest_id"],
        ["tenant_id", "id"],
    )


def downgrade() -> None:
    op.drop_constraint(
        op.f("fk_validation_runs_tenant_id_context_manifests"),
        "validation_runs",
        type_="foreignkey",
    )
    op.drop_column("validation_runs", "context_manifest_id")
