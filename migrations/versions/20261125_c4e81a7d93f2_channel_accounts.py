"""Connected social accounts and Meta login sessions (NRC-13).

Same tenant pattern as the other tables (RLS on app.tenant_id). Provider tokens are stored
encrypted by the API; the database never sees them in clear. The app role may not delete either
table: disconnecting an account keeps its row.

Revision ID: c4e81a7d93f2
Revises: b3e6bfed9303
Create Date: 2026-11-25 12:00:00.000000
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "c4e81a7d93f2"
down_revision: str | None = "b3e6bfed9303"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

APP_ROLE = "marketing_app"
CURRENT_TENANT = "NULLIF(current_setting('app.tenant_id', true), '')::uuid"
TABLES = ("meta_connect_sessions", "channel_accounts")


def _status(name: str, *values: str) -> sa.Enum:
    return sa.Enum(*values, name=name, native_enum=False, length=32)


def upgrade() -> None:
    op.create_table(
        "meta_connect_sessions",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("tenant_id", sa.Uuid(), nullable=False),
        sa.Column("state_hash", sa.String(length=64), nullable=False),
        sa.Column("initiated_by", sa.String(length=128), nullable=False),
        sa.Column("reconnect_account_id", sa.Uuid(), nullable=True),
        sa.Column("return_path", sa.String(length=200), nullable=False),
        sa.Column(
            "status",
            _status(
                "connectstatus", "WAITING_FOR_META", "AWAITING_SELECTION", "COMPLETED", "FAILED"
            ),
            nullable=False,
        ),
        sa.Column("error_code", sa.String(length=64), nullable=True),
        sa.Column("meta_user_id", sa.String(length=64), nullable=True),
        sa.Column("user_token_encrypted", sa.Text(), nullable=True),
        sa.Column("granted_scopes", postgresql.ARRAY(sa.String(length=64)), nullable=True),
        sa.Column("candidates", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column("expires_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("callback_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("completed_at", sa.DateTime(timezone=True), nullable=True),
        sa.ForeignKeyConstraint(
            ["tenant_id"], ["tenants.id"], name=op.f("fk_meta_connect_sessions_tenant_id_tenants")
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_meta_connect_sessions")),
        sa.UniqueConstraint("state_hash", name=op.f("uq_meta_connect_sessions_state_hash")),
    )
    op.create_table(
        "channel_accounts",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("tenant_id", sa.Uuid(), nullable=False),
        sa.Column("channel", _status("accountchannel", "facebook", "instagram"), nullable=False),
        sa.Column("provider_account_id", sa.String(length=64), nullable=False),
        sa.Column("page_id", sa.String(length=64), nullable=False),
        sa.Column("name", sa.String(length=300), nullable=False),
        sa.Column("handle", sa.String(length=300), nullable=True),
        sa.Column(
            "state",
            _status(
                "accountstate",
                "CONNECTED",
                "TEST_ONLY",
                "PENDING_APP_REVIEW",
                "PERMISSION_MISSING",
                "EXPIRED",
                "REVOKED",
                "DISCONNECTED",
            ),
            nullable=False,
        ),
        sa.Column("state_reason", sa.Text(), nullable=True),
        sa.Column(
            "granted_scopes",
            postgresql.ARRAY(sa.String(length=64)),
            server_default=sa.text("'{}'"),
            nullable=False,
        ),
        sa.Column(
            "missing_permissions",
            postgresql.ARRAY(sa.String(length=64)),
            server_default=sa.text("'{}'"),
            nullable=False,
        ),
        sa.Column("token_encrypted", sa.Text(), nullable=True),
        sa.Column("token_expires_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("capability_version", sa.String(length=64), nullable=False),
        sa.Column("connected_by", sa.String(length=128), nullable=False),
        sa.Column("connected_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("last_checked_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("disconnected_by", sa.String(length=128), nullable=True),
        sa.Column("disconnected_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.ForeignKeyConstraint(
            ["tenant_id"], ["tenants.id"], name=op.f("fk_channel_accounts_tenant_id_tenants")
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_channel_accounts")),
        sa.UniqueConstraint(
            "tenant_id",
            "channel",
            "provider_account_id",
            name=op.f("uq_channel_accounts_tenant_id_channel_provider_account_id"),
        ),
        sa.UniqueConstraint("tenant_id", "id", name=op.f("uq_channel_accounts_tenant_id_id")),
    )
    for table in TABLES:
        op.execute(f"ALTER TABLE {table} ENABLE ROW LEVEL SECURITY")
        op.execute(f"ALTER TABLE {table} FORCE ROW LEVEL SECURITY")
        op.execute(
            f"CREATE POLICY tenant_isolation ON {table} "
            f"USING (tenant_id = {CURRENT_TENANT}) WITH CHECK (tenant_id = {CURRENT_TENANT})"
        )
        op.execute(f"GRANT SELECT, INSERT, UPDATE ON {table} TO {APP_ROLE}")


def downgrade() -> None:
    # Destroys every connected account and its encrypted tokens; reconnecting is the only way back.
    op.drop_table("channel_accounts")
    op.drop_table("meta_connect_sessions")
