"""Join the policy retrieval chain (NRC-5) and the client uploads chain (NRC-3) for NRC-4.

No schema change: the two chains touch different tables. A merge revision rather than
re-parenting either head, so a database already at 0292b2a8c521 or deb7a7225553 upgrades with
`alembic upgrade head`.

Revision ID: 5c3a9e1f7b24
Revises: 0292b2a8c521, deb7a7225553
Create Date: 2026-11-20 12:00:00.000000
"""

from collections.abc import Sequence

revision: str = "5c3a9e1f7b24"
down_revision: str | tuple[str, ...] | None = ("0292b2a8c521", "deb7a7225553")
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    pass


def downgrade() -> None:
    pass
