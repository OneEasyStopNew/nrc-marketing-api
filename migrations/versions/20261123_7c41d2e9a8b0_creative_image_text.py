"""Text to render inside the image, per creative version (NRC-2).

Nullable: existing versions have no text in the image. The app role's table-level SELECT and
INSERT grants on creative_versions cover the new column; it still cannot UPDATE it.
Downgrade drops the column and every version's image text (see docs/migrations.md).

Revision ID: 7c41d2e9a8b0
Revises: b3e6bfed9303
Create Date: 2026-11-23 12:00:00.000000
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "7c41d2e9a8b0"
down_revision: str | None = "b3e6bfed9303"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

TABLE = "creative_versions"


def upgrade() -> None:
    op.add_column(TABLE, sa.Column("image_text", sa.Text(), nullable=True))


def downgrade() -> None:
    op.drop_column(TABLE, "image_text")
