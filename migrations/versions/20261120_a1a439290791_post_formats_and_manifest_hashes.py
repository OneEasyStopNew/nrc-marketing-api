"""Post-format applicability on policy nodes, and manifest content hashes (NRC-5 review).

- policy_nodes.post_formats: like channels, the post formats a rule or exception is limited to; an
  empty array (the default for existing rows) means no restriction.
- context_manifests.content_sha256: sha256 of the manifest's canonical JSON (sorted keys, no
  whitespace), so the stored JSON can be checked later. Nullable: older rows keep NULL and are
  hashed from their JSON when read.
- validation_runs.context_manifest_sha256: the hash the run used, recorded with the run.

Expand-only: new defaulted or nullable columns; the existing grants cover them.

Revision ID: a1a439290791
Revises: 0292b2a8c521
Create Date: 2026-11-20 12:00:00.000000
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "a1a439290791"
down_revision: str | None = "0292b2a8c521"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

EMPTY_ARRAY = sa.text("'{}'")


def upgrade() -> None:
    op.add_column(
        "policy_nodes",
        sa.Column(
            "post_formats",
            sa.ARRAY(sa.String(length=64)),
            server_default=EMPTY_ARRAY,
            nullable=False,
        ),
    )
    op.add_column(
        "context_manifests", sa.Column("content_sha256", sa.String(length=64), nullable=True)
    )
    op.add_column(
        "validation_runs",
        sa.Column("context_manifest_sha256", sa.String(length=64), nullable=True),
    )


def downgrade() -> None:
    op.drop_column("validation_runs", "context_manifest_sha256")
    op.drop_column("context_manifests", "content_sha256")
    op.drop_column("policy_nodes", "post_formats")
