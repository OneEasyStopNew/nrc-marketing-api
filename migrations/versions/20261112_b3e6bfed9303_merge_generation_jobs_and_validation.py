"""Join the generation-jobs branch (NRC-6) and the validation branch (NRC-7) into one head.

No schema change: the two branches touch different tables.

Revision ID: b3e6bfed9303
Revises: ffd6ebe6b868, 5b1e0c7d2a91
Create Date: 2026-11-12 21:31:25.505562
"""

from collections.abc import Sequence

revision: str = "b3e6bfed9303"
down_revision: str | tuple[str, ...] | None = ("ffd6ebe6b868", "5b1e0c7d2a91")
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    pass


def downgrade() -> None:
    pass
