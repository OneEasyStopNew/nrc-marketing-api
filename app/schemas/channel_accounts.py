# Connected social accounts and the Meta connect flow (NRC-13).

from datetime import datetime
from typing import Annotated
from uuid import UUID

from pydantic import Field, StringConstraints

from app.models.channel_accounts import AccountChannel, AccountState, ConnectStatus
from app.models.enums import PostFormat
from app.schemas.common import ApiModel, RequestModel


class ChannelCapabilities(ApiModel):
    """What this account can publish now. Checked again at dispatch; not a promise."""

    version: str = Field(description="Capability matrix version; publication manifests keep it.")
    text_only: bool = Field(description="A post with no media is allowed (Facebook Pages only).")
    requires_media: bool
    image: bool
    video: bool = Field(description="Video publishing is not supported yet on any channel.")
    formats: list[PostFormat] = Field(description="Post formats this app can publish here now.")
    image_mime_types: list[str]
    max_image_bytes: int | None
    min_aspect_ratio: str | None = Field(description="Narrowest width:height, e.g. 4:5.")
    max_aspect_ratio: str | None = Field(description="Widest width:height, e.g. 1.91:1.")
    caption_max_chars: int | None
    schedule: bool = Field(description="The provider can hold a post until a set time.")
    delete: bool = Field(description="This app can remove a published post here.")


class ChannelAccountRead(ApiModel):
    id: UUID
    channel: AccountChannel
    provider_account_id: str = Field(description="Page id (Facebook) or Instagram user id.")
    name: str
    handle: str | None = Field(description="Instagram username; null for Facebook Pages.")
    state: AccountState
    state_reason: str | None = Field(description="What to do next, for states other than ready.")
    can_publish: bool = Field(description="True only in CONNECTED, or TEST_ONLY for test accounts.")
    permissions: list[str] = Field(description="Meta permissions granted for this account.")
    missing_permissions: list[str]
    connected_by: str
    connected_at: datetime
    expires_at: datetime | None = Field(description="When the token stops working, if Meta says.")
    last_checked_at: datetime | None
    disconnected_at: datetime | None
    capabilities: ChannelCapabilities


class ChannelAccountList(ApiModel):
    items: list[ChannelAccountRead]


class ChannelAccountPage(ApiModel):
    # Facebook accounts first, then Instagram; by name within each.
    items: list[ChannelAccountRead]
    next_cursor: UUID | None = Field(description="Pass as cursor for the next page.")


ReturnPath = Annotated[
    str,
    # "/" or "/seg/seg": no host, no "//", no query; the web app URL is always prefixed.
    StringConstraints(max_length=200, pattern=r"^(/[A-Za-z0-9_-]+)*/?$"),
]


class MetaConnectStart(RequestModel):
    return_path: ReturnPath = Field(
        default="/accounts",
        description="Web app path to land on after Meta, e.g. /accounts. Paths only, no hosts.",
    )


class MetaConnectStarted(ApiModel):
    connection_id: UUID
    authorization_url: str = Field(description="Send the browser here (top-level navigation).")
    expires_at: datetime


class ConnectCandidate(ApiModel):
    channel: AccountChannel
    provider_account_id: str
    name: str
    handle: str | None
    page_id: str
    can_publish: bool = Field(description="False when a Page task or permission is missing.")
    missing_permissions: list[str]
    already_connected: bool


class MetaConnectionRead(ApiModel):
    connection_id: UUID
    status: ConnectStatus
    error_code: str | None = Field(
        description="Set when status is FAILED: login_cancelled, state_expired, "
        "meta_error or no_pages."
    )
    expires_at: datetime
    candidates: list[ConnectCandidate]


class AccountSelection(RequestModel):
    channel: AccountChannel
    provider_account_id: Annotated[str, StringConstraints(pattern=r"^[0-9]{1,64}$")]


class MetaBindRequest(RequestModel):
    accounts: list[AccountSelection] = Field(min_length=1, max_length=50)
