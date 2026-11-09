"""Global provider spend guard: provider_spend counters, plus per-job spend state and actual cost.

provider_spend holds one row per provider and no tenant data, so it has no tenant_id and no RLS:
an exception to the tenant_id + RLS rule, approved by Dinesh. Rows are created and seeded here
only; the app role gets SELECT and UPDATE, never INSERT or DELETE. 3fb9adf5dfbe moves those two
privileges to the worker role (marketing_worker), the only one that uses the table.

Revision ID: 440a8d5dba9a
Revises: 8f8284b27c6f
Create Date: 2026-11-09 21:30:00.000000
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "440a8d5dba9a"
down_revision: str | None = "8f8284b27c6f"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

APP_ROLE = "marketing_app"
PROVIDERS = ("higgsfield", "simulator")


def upgrade() -> None:
    op.create_table(
        "provider_spend",
        sa.Column("provider", sa.Text(), nullable=False),
        sa.Column(
            "reserved_usd_micros", sa.BigInteger(), server_default=sa.text("0"), nullable=False
        ),
        sa.Column(
            "committed_usd_micros", sa.BigInteger(), server_default=sa.text("0"), nullable=False
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.CheckConstraint(
            "reserved_usd_micros >= 0", name=op.f("ck_provider_spend_reserved_not_negative")
        ),
        sa.CheckConstraint(
            "committed_usd_micros >= 0", name=op.f("ck_provider_spend_committed_not_negative")
        ),
        sa.PrimaryKeyConstraint("provider", name=op.f("pk_provider_spend")),
    )
    op.bulk_insert(
        sa.table("provider_spend", sa.column("provider", sa.Text())),
        [{"provider": provider} for provider in PROVIDERS],
    )
    op.execute(f"GRANT SELECT, UPDATE ON provider_spend TO {APP_ROLE}")

    op.add_column(
        "jobs", sa.Column("provider_cost_actual_usd_micros", sa.BigInteger(), nullable=True)
    )
    op.add_column(
        "jobs",
        sa.Column(
            "provider_spend_state",
            sa.Enum(
                "RESERVED",
                "COMMITTED",
                "RELEASED",
                name="spendstate",
                native_enum=False,
                length=32,
            ),
            nullable=True,
        ),
    )


def downgrade() -> None:
    op.drop_column("jobs", "provider_spend_state")
    op.drop_column("jobs", "provider_cost_actual_usd_micros")
    # Dropping the table also drops its grants.
    op.drop_table("provider_spend")
