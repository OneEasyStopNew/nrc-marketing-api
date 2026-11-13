"""Join the generation/storage chain (NRC-6, NRC-3, NRC-10) and the validation chain (NRC-7).

No schema change: the two chains touch different tables. A merge revision rather than
re-parenting 39d93abf9900, so a database already at 5b1e0c7d2a91 upgrades with
`alembic upgrade head` instead of needing the other chain applied by hand.

Revision ID: b3e6bfed9303
Revises: 3fb9adf5dfbe, 5b1e0c7d2a91
Create Date: 2026-11-12 21:31:25.505562
"""

from collections.abc import Sequence

revision: str = "b3e6bfed9303"
down_revision: str | tuple[str, ...] | None = ("3fb9adf5dfbe", "5b1e0c7d2a91")
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    pass


def downgrade() -> None:
    pass
