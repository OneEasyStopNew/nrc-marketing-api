"""Provider idempotency keys become text, so new jobs can use "nrc-job-<job_id>".

A widening change: every stored UUID converts to its usual text form, which is exactly the key
those jobs already sent, so jobs in flight keep reconciling with the same key.

Revision ID: 8f8284b27c6f
Revises: a3c5e7b91d24
Create Date: 2026-11-09 21:00:00.000000
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "8f8284b27c6f"
down_revision: str | None = "a3c5e7b91d24"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.alter_column(
        "jobs",
        "provider_idempotency_key",
        existing_type=sa.Uuid(),
        type_=sa.String(length=255),
        existing_nullable=True,
        postgresql_using="provider_idempotency_key::text",
    )


def downgrade() -> None:
    # Only valid while every key is still a UUID; derived keys cannot be converted back.
    op.alter_column(
        "jobs",
        "provider_idempotency_key",
        existing_type=sa.String(length=255),
        type_=sa.Uuid(),
        existing_nullable=True,
        postgresql_using="provider_idempotency_key::uuid",
    )
