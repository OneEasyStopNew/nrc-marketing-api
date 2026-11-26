# Meta Graph API calls for connecting accounts (NRC-13). Server-side only; tokens never leave here
# except encrypted, and no exception or log line carries one.

import hashlib
import hmac
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any, Final, Protocol

import httpx
from pydantic import BaseModel, ConfigDict, Field, SecretStr, ValidationError

# Minimum permissions for first-pass organic publishing to Pages and Instagram professional
# accounts through Facebook Login for Business. No inbox, ads or insights permissions.
FACEBOOK_SCOPES: Final = ("pages_show_list", "pages_read_engagement", "pages_manage_posts")
INSTAGRAM_SCOPES: Final = ("instagram_basic", "instagram_content_publish")
# Needed to list Pages the person reaches through a Business portfolio.
SHARED_SCOPES: Final = ("business_management",)
REQUESTED_SCOPES: Final = SHARED_SCOPES + FACEBOOK_SCOPES + INSTAGRAM_SCOPES
# Page task needed to publish as the Page.
PUBLISH_TASK: Final = "CREATE_CONTENT"


class MetaGraphError(Exception):
    """A Graph API failure. code/subcode are Meta's numbers; message is Meta's text, no token."""

    def __init__(self, message: str, *, code: int | None = None, subcode: int | None = None):
        super().__init__(message)
        self.code = code
        self.subcode = subcode

    @property
    def token_invalid(self) -> bool:
        # 190: invalid or expired access token. 102: session invalid.
        return self.code in (102, 190)


@dataclass(frozen=True, slots=True)
class UserToken:
    access_token: SecretStr
    expires_at: datetime | None


@dataclass(frozen=True, slots=True)
class MetaInstagram:
    id: str
    username: str | None
    name: str | None


@dataclass(frozen=True, slots=True)
class MetaPage:
    id: str
    name: str
    access_token: SecretStr
    tasks: frozenset[str]
    instagram: MetaInstagram | None


@dataclass(frozen=True, slots=True)
class TokenInfo:
    is_valid: bool
    expires_at: datetime | None
    scopes: frozenset[str]


class _Meta(BaseModel):
    # Meta adds fields over time; keep what we read, ignore the rest.
    model_config = ConfigDict(extra="ignore")


class _TokenBody(_Meta):
    access_token: SecretStr
    expires_in: int | None = None


class _MeBody(_Meta):
    id: str = Field(pattern=r"^[0-9]{1,64}$")


class _Permission(_Meta):
    permission: str
    status: str


class _PermissionsBody(_Meta):
    data: list[_Permission] = []


class _InstagramBody(_Meta):
    id: str = Field(pattern=r"^[0-9]{1,64}$")
    username: str | None = None
    name: str | None = None


class _PageBody(_Meta):
    id: str = Field(pattern=r"^[0-9]{1,64}$")
    name: str = Field(max_length=300)
    access_token: SecretStr
    tasks: list[str] = []
    instagram_business_account: _InstagramBody | None = None


class _Cursors(_Meta):
    after: str | None = None


class _Paging(_Meta):
    cursors: _Cursors = _Cursors()
    next: str | None = None


class _PagesBody(_Meta):
    data: list[_PageBody] = []
    paging: _Paging = _Paging()


class _DebugData(_Meta):
    is_valid: bool = False
    expires_at: int | None = None
    scopes: list[str] = []


class _DebugBody(_Meta):
    data: _DebugData


class _ErrorDetails(_Meta):
    message: str = ""
    code: int | None = None
    error_subcode: int | None = None


class _ErrorBody(_Meta):
    error: _ErrorDetails | None = None


def _parse[M: _Meta](model: type[M], body: dict[str, Any]) -> M:
    try:
        return model.model_validate(body)
    except ValidationError:
        # The validation error can echo input values (tokens); keep only the shape's name.
        raise MetaGraphError(f"Meta answered with an unexpected {model.__name__}") from None


class MetaGraph(Protocol):
    async def exchange_code(self, code: str, redirect_uri: str) -> UserToken: ...

    async def long_lived_token(self, token: SecretStr) -> UserToken: ...

    async def me(self, token: SecretStr) -> str: ...

    async def granted_scopes(self, token: SecretStr) -> frozenset[str]: ...

    async def pages(self, token: SecretStr) -> list[MetaPage]: ...

    async def debug_token(self, token: SecretStr) -> TokenInfo: ...


def _expiry(seconds: int | None) -> datetime | None:
    # expires_in is seconds from now; missing or 0 means the token does not expire.
    return datetime.now(UTC) + timedelta(seconds=seconds) if seconds else None


class HttpMetaGraph:
    def __init__(
        self,
        http: httpx.AsyncClient,
        *,
        graph_base_url: str,
        version: str,
        app_id: str,
        app_secret: SecretStr,
    ) -> None:
        self._http = http
        self._base = f"{graph_base_url.rstrip('/')}/{version}"
        self._app_id = app_id
        self._app_secret = app_secret

    def _proof(self, token: SecretStr) -> str:
        # appsecret_proof: a stolen token alone cannot call the API as this app.
        secret = self._app_secret.get_secret_value().encode()
        return hmac.new(secret, token.get_secret_value().encode(), hashlib.sha256).hexdigest()

    async def _get(
        self, path: str, params: dict[str, str], token: SecretStr | None = None
    ) -> dict[str, Any]:
        headers = {}
        if token is not None:
            headers["Authorization"] = f"Bearer {token.get_secret_value()}"
            params = {**params, "appsecret_proof": self._proof(token)}
        try:
            response = await self._http.get(f"{self._base}/{path}", params=params, headers=headers)
        except httpx.HTTPError as exc:
            # str(exc) can include the request URL; keep only the type.
            raise MetaGraphError(f"Meta request failed ({type(exc).__name__})") from None
        try:
            body = response.json()
        except ValueError:
            raise MetaGraphError(f"Meta answered {response.status_code} without JSON") from None
        if not isinstance(body, dict):
            raise MetaGraphError("Meta answered with an unexpected body")
        error = _parse(_ErrorBody, body).error
        if response.is_error or error is not None:
            details = error or _ErrorDetails()
            raise MetaGraphError(
                (details.message or f"Meta answered {response.status_code}")[:300],
                code=details.code,
                subcode=details.error_subcode,
            )
        return body

    async def exchange_code(self, code: str, redirect_uri: str) -> UserToken:
        body = await self._get(
            "oauth/access_token",
            {
                "client_id": self._app_id,
                "client_secret": self._app_secret.get_secret_value(),
                "redirect_uri": redirect_uri,
                "code": code,
            },
        )
        answer = _parse(_TokenBody, body)
        return UserToken(answer.access_token, _expiry(answer.expires_in))

    async def long_lived_token(self, token: SecretStr) -> UserToken:
        body = await self._get(
            "oauth/access_token",
            {
                "grant_type": "fb_exchange_token",
                "client_id": self._app_id,
                "client_secret": self._app_secret.get_secret_value(),
                "fb_exchange_token": token.get_secret_value(),
            },
        )
        answer = _parse(_TokenBody, body)
        return UserToken(answer.access_token, _expiry(answer.expires_in))

    async def me(self, token: SecretStr) -> str:
        return _parse(_MeBody, await self._get("me", {"fields": "id"}, token)).id

    async def granted_scopes(self, token: SecretStr) -> frozenset[str]:
        body = _parse(_PermissionsBody, await self._get("me/permissions", {}, token))
        return frozenset(item.permission for item in body.data if item.status == "granted")

    async def pages(self, token: SecretStr) -> list[MetaPage]:
        fields = "id,name,access_token,tasks,instagram_business_account{id,username,name}"
        params = {"fields": fields, "limit": "100"}
        pages: list[MetaPage] = []
        path = "me/accounts"
        # Follow at most 10 pages of results (1,000 Pages); more is not a real tenant.
        for _ in range(10):
            body = _parse(_PagesBody, await self._get(path, params, token))
            pages.extend(
                MetaPage(
                    id=item.id,
                    name=item.name,
                    access_token=item.access_token,
                    tasks=frozenset(item.tasks),
                    instagram=MetaInstagram(id=ig.id, username=ig.username, name=ig.name)
                    if (ig := item.instagram_business_account) is not None
                    else None,
                )
                for item in body.data
            )
            after = body.paging.cursors.after
            if not body.paging.next or not after:
                break
            params = {**params, "after": after}
        return pages

    async def debug_token(self, token: SecretStr) -> TokenInfo:
        app_token = SecretStr(f"{self._app_id}|{self._app_secret.get_secret_value()}")
        body = await self._get("debug_token", {"input_token": token.get_secret_value()}, app_token)
        data = _parse(_DebugBody, body).data
        expires = data.expires_at
        return TokenInfo(
            is_valid=data.is_valid,
            # 0 means the token does not expire.
            expires_at=datetime.fromtimestamp(expires, UTC) if expires else None,
            scopes=frozenset(data.scopes),
        )
