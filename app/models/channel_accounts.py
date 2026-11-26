# Social accounts a tenant has connected (NRC-13) and the Meta login sessions that bind them.

from datetime import datetime
from enum import StrEnum
from uuid import UUID, uuid4

from sqlalchemy import (
    ARRAY,
    DateTime,
    ForeignKey,
    String,
    Text,
    UniqueConstraint,
    func,
    text,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column

from app.db.base import Base
from app.models.enums import enum_column


class AccountChannel(StrEnum):
    FACEBOOK = "facebook"
    INSTAGRAM = "instagram"


class AccountState(StrEnum):
    # Publishing is allowed only in CONNECTED, and in TEST_ONLY to the Meta app's own test
    # accounts. Every other state names what has to happen first.
    CONNECTED = "CONNECTED"
    TEST_ONLY = "TEST_ONLY"
    PENDING_APP_REVIEW = "PENDING_APP_REVIEW"
    PERMISSION_MISSING = "PERMISSION_MISSING"
    EXPIRED = "EXPIRED"
    REVOKED = "REVOKED"
    DISCONNECTED = "DISCONNECTED"


class ConnectStatus(StrEnum):
    WAITING_FOR_META = "WAITING_FOR_META"
    AWAITING_SELECTION = "AWAITING_SELECTION"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"


class MetaConnectSession(Base):
    """One Meta login started by one user in one tenant.

    The browser carries only "<tenant>.<secret>" as the OAuth state; the row keeps the secret's
    SHA-256. The callback stores the user token encrypted and the Pages/Instagram accounts it can
    see; nothing is bound until the same user picks accounts. The token is wiped on completion.
    """

    __tablename__ = "meta_connect_sessions"
    __table_args__ = (UniqueConstraint("state_hash"),)

    id: Mapped[UUID] = mapped_column(primary_key=True, default=uuid4)
    tenant_id: Mapped[UUID] = mapped_column(ForeignKey("tenants.id"))
    state_hash: Mapped[str] = mapped_column(String(64))
    initiated_by: Mapped[str] = mapped_column(String(128))
    # Set when the login refreshes one existing account instead of adding new ones.
    reconnect_account_id: Mapped[UUID | None]
    # Web app path the browser returns to; validated as a bare path when the login starts.
    return_path: Mapped[str] = mapped_column(String(200))
    status: Mapped[ConnectStatus] = mapped_column(enum_column(ConnectStatus))
    error_code: Mapped[str | None] = mapped_column(String(64))
    meta_user_id: Mapped[str | None] = mapped_column(String(64))
    user_token_encrypted: Mapped[str | None] = mapped_column(Text)
    granted_scopes: Mapped[list[str] | None] = mapped_column(ARRAY(String(64)))
    # Safe fields only (ids, names, handles, missing permissions); never a token.
    candidates: Mapped[list[dict[str, object]] | None] = mapped_column(JSONB)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    expires_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    callback_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))


class ChannelAccount(Base):
    """A Facebook Page or Instagram professional account the tenant explicitly bound.

    Disconnecting keeps the row (state DISCONNECTED, token wiped) so history that names the
    account stays readable; connecting the same account again reuses it.
    """

    __tablename__ = "channel_accounts"
    __table_args__ = (
        UniqueConstraint("tenant_id", "id"),
        UniqueConstraint("tenant_id", "channel", "provider_account_id"),
    )

    id: Mapped[UUID] = mapped_column(primary_key=True, default=uuid4)
    tenant_id: Mapped[UUID] = mapped_column(ForeignKey("tenants.id"))
    channel: Mapped[AccountChannel] = mapped_column(enum_column(AccountChannel))
    # Page id for Facebook, Instagram user id for Instagram.
    provider_account_id: Mapped[str] = mapped_column(String(64))
    # The Facebook Page whose token publishes for this account (the linked Page for Instagram).
    page_id: Mapped[str] = mapped_column(String(64))
    name: Mapped[str] = mapped_column(String(300))
    handle: Mapped[str | None] = mapped_column(String(300))
    state: Mapped[AccountState] = mapped_column(enum_column(AccountState))
    state_reason: Mapped[str | None] = mapped_column(Text)
    granted_scopes: Mapped[list[str]] = mapped_column(
        ARRAY(String(64)), server_default=text("'{}'")
    )
    missing_permissions: Mapped[list[str]] = mapped_column(
        ARRAY(String(64)), server_default=text("'{}'")
    )
    token_encrypted: Mapped[str | None] = mapped_column(Text)
    token_expires_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    capability_version: Mapped[str] = mapped_column(String(64))
    connected_by: Mapped[str] = mapped_column(String(128))
    connected_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    last_checked_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    disconnected_by: Mapped[str | None] = mapped_column(String(128))
    disconnected_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
