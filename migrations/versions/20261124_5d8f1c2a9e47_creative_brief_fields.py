"""Brief fields per creative version: headline, CTA, link, site, audience, visual instructions.

NRC-2. All nullable, so existing versions keep working with the fields empty. Adding nullable
columns without a default only changes the catalogue, so creative_versions is not rewritten.
The app role's table-level SELECT and INSERT grants cover the new columns; it still cannot
UPDATE a version. Downgrade drops the columns and every version's values in them.

Revision ID: 5d8f1c2a9e47
Revises: e2a9c4f71b36
Create Date: 2026-11-24 15:00:00.000000
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "5d8f1c2a9e47"
down_revision: str | None = "e2a9c4f71b36"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

TABLE = "creative_versions"
TEXT_COLUMNS = ("headline", "destination_url", "site", "audience", "visual_instructions")


def upgrade() -> None:
    for name in TEXT_COLUMNS:
        op.add_column(TABLE, sa.Column(name, sa.Text(), nullable=True))
    op.add_column(
        TABLE,
        sa.Column(
            "call_to_action",
            sa.Enum(
                "learn_more",
                "sign_up",
                "apply_now",
                "contact_us",
                name="calltoaction",
                native_enum=False,
                length=32,
            ),
            nullable=True,
        ),
    )


def downgrade() -> None:
    op.drop_column(TABLE, "call_to_action")
    for name in reversed(TEXT_COLUMNS):
        op.drop_column(TABLE, name)
