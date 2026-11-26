"""Demo branch only: join the NRC-4 (policy ingestion) and NRC-7 (validation jobs) heads.

Both PRs add a migration on top of NRC-5's head, so combining them gives two heads. This empty
revision exists only on demo/NRC-29-validation-flow; on main, whichever PR merges second re-points
its first migration instead. Both sides grant marketing_worker privileges on the policy tables;
Postgres keeps the union, which is what each worker needs.

Revision ID: 12ee075b6035
Revises: 8e41b6d0a7c3, c4e81d27a9b3
Create Date: 2026-11-26 12:46:48.344942
"""

from collections.abc import Sequence

revision: str = "12ee075b6035"
down_revision: tuple[str, str] = ("8e41b6d0a7c3", "c4e81d27a9b3")
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    pass


def downgrade() -> None:
    pass
