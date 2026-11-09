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

UUID_PATTERN = "^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$"
# Terminal states as app.services.jobs.TERMINAL_STATES had them at this revision.
UNFINISHED_DERIVED_KEYS = sa.text(
    "SELECT count(*) FROM jobs"
    " WHERE state NOT IN ('SUCCEEDED', 'FAILED', 'CANCELLED')"
    " AND provider_idempotency_key IS NOT NULL"
    " AND provider_idempotency_key !~* :uuid_pattern"
)


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
    # Derived "nrc-job-..." keys have no UUID form. The older code would resend a missing key as
    # "None" and could be billed twice, so refuse while an unfinished job holds one. row_security
    # off makes Postgres raise instead of silently hiding rows when RLS applies to this role.
    bind = op.get_bind()
    bind.execute(sa.text("SET LOCAL row_security = off"))
    unfinished = bind.scalar(UNFINISHED_DERIVED_KEYS, {"uuid_pattern": UUID_PATTERN})
    bind.execute(sa.text("RESET row_security"))
    if unfinished:
        raise RuntimeError(
            f"Refusing to downgrade 8f8284b27c6f: {unfinished} unfinished job(s) hold a derived "
            "provider idempotency key. Let them finish (or fail them) and run it again."
        )
    # Only finished jobs can hold a derived key here, and they are never resent: those become NULL.
    op.alter_column(
        "jobs",
        "provider_idempotency_key",
        existing_type=sa.String(length=255),
        type_=sa.Uuid(),
        existing_nullable=True,
        postgresql_using=(
            f"CASE WHEN provider_idempotency_key ~* '{UUID_PATTERN}' "
            "THEN provider_idempotency_key::uuid END"
        ),
    )
