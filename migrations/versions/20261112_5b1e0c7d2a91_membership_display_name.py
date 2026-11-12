"""Membership display names, so the UI can say who saved a version instead of a Cognito sub.

Nullable: existing members keep working and show their sub until a name is set with
scripts.grant_access --display-name.

Revision ID: 5b1e0c7d2a91
Revises: 172887e987f2
Create Date: 2026-11-12 18:00:00.000000
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "5b1e0c7d2a91"
down_revision: str | None = "172887e987f2"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

TABLE = "memberships"


def upgrade() -> None:
    op.add_column(TABLE, sa.Column("display_name", sa.String(length=200), nullable=True))


def downgrade() -> None:
    op.drop_column(TABLE, "display_name")
