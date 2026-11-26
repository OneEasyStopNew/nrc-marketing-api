# Connecting Facebook Pages and Instagram accounts through Meta login (NRC-13). Meta is faked;
# nothing here calls the real Graph API.

import base64
import logging
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from urllib.parse import parse_qs, urlsplit
from uuid import UUID, uuid4

import httpx
import pytest
from pydantic import SecretStr
from sqlalchemy import select, text
from sqlalchemy.ext.asyncio import AsyncConnection

from app.api.routes.channel_accounts import get_meta_config, get_meta_graph
from app.core.crypto import TokenCipher, TokenDecryptError, TokenKeyError
from app.models import ChannelAccount, Membership, MetaConnectSession
from app.services.channel_accounts import MetaConfig, require_config
from app.services.meta_graph import (
    REQUESTED_SCOPES,
    HttpMetaGraph,
    MetaGraphError,
    MetaInstagram,
    MetaPage,
    TokenInfo,
    UserToken,
)
from tests.conftest import AuthHeaders, World, _session

KEY = "k1:" + base64.b64encode(b"0" * 32).decode()
WEB = "https://app.example.org"
CALLBACK = "/webhooks/meta/oauth/callback"
USER_TOKEN = "user-token-SECRET-1"  # noqa: S105 (fake)
PAGE_TOKEN = "page-token-SECRET-1"  # noqa: S105 (fake)
ALL_SCOPES = frozenset(REQUESTED_SCOPES)
PAGE = MetaPage(
    id="1001",
    name="Synthetic Research Clinic",
    access_token=SecretStr(PAGE_TOKEN),
    tasks=frozenset({"CREATE_CONTENT", "MODERATE"}),
    instagram=MetaInstagram(id="1789", username="synthetic.clinic", name="Synthetic Clinic"),
)
OTHER_PAGE = MetaPage(
    id="2002",
    name="Synthetic Second Page",
    access_token=SecretStr("page-token-SECRET-2"),
    tasks=frozenset({"ANALYZE"}),
    instagram=None,
)


@dataclass
class FakeGraph:
    pages_list: list[MetaPage] = field(default_factory=lambda: [PAGE, OTHER_PAGE])
    granted: frozenset[str] = ALL_SCOPES
    token_info: TokenInfo | Exception = field(
        default_factory=lambda: TokenInfo(is_valid=True, expires_at=None, scopes=ALL_SCOPES)
    )
    codes: list[str] = field(default_factory=list)

    async def exchange_code(self, code: str, redirect_uri: str) -> UserToken:
        self.codes.append(code)
        if code != "good-code":
            raise MetaGraphError("Invalid verification code", code=100)
        return UserToken(SecretStr("short-lived"), None)

    async def long_lived_token(self, token: SecretStr) -> UserToken:
        return UserToken(SecretStr(USER_TOKEN), None)

    async def me(self, token: SecretStr) -> str:
        return "meta-user-1"

    async def granted_scopes(self, token: SecretStr) -> frozenset[str]:
        return self.granted

    async def pages(self, token: SecretStr) -> list[MetaPage]:
        assert token.get_secret_value() == USER_TOKEN
        return self.pages_list

    async def debug_token(self, token: SecretStr) -> TokenInfo:
        if isinstance(self.token_info, Exception):
            raise self.token_info
        return self.token_info


def _config(mode: str = "development") -> MetaConfig:
    return MetaConfig(
        app_id="123456789",
        redirect_uri=f"https://api.example.org{CALLBACK}",
        login_base_url="https://www.facebook.com",
        version="v26.0",
        web_app_url=WEB,
        ttl=timedelta(minutes=10),
        mode=mode,
        cipher=TokenCipher(SecretStr(KEY)),
    )


@pytest.fixture
def graph() -> FakeGraph:
    return FakeGraph()


@pytest.fixture
def config() -> list[MetaConfig | None]:
    # A box, so a test can change the mode (or unset Meta) after the client is built.
    return [_config()]


@pytest.fixture
def app_overrides(
    graph: FakeGraph, config: list[MetaConfig | None]
) -> dict[Callable[..., object], Callable[..., object]]:
    return {get_meta_graph: lambda: graph, get_meta_config: lambda: require_config(config[0])}


@asynccontextmanager
async def _as_owner(connection: AsyncConnection) -> AsyncIterator[None]:
    await connection.execute(text("RESET ROLE"))
    try:
        yield
    finally:
        await connection.execute(text("SET LOCAL ROLE marketing_app"))


@pytest.fixture(autouse=True)
async def admins(connection: AsyncConnection, world: World) -> None:
    async with _as_owner(connection), _session(connection) as db:
        db.add_all(
            [
                Membership(tenant_id=world.a.tenant_id, user_sub="admin-a", roles=["admin"]),
                Membership(tenant_id=world.a.tenant_id, user_sub="admin-a2", roles=["admin"]),
                Membership(tenant_id=world.b.tenant_id, user_sub="admin-b", roles=["admin"]),
            ]
        )
        await db.commit()


async def _start(
    client: httpx.AsyncClient, auth: AuthHeaders, tenant: UUID, sub: str = "admin-a"
) -> tuple[str, str]:
    response = await client.post(
        "/v1/marketing/integrations/meta/connect",
        headers=auth(sub, tenant),
        json={"return_path": "/accounts"},
    )
    assert response.status_code == 201, response.text
    url = urlsplit(response.json()["authorization_url"])
    return response.json()["connection_id"], parse_qs(url.query)["state"][0]


async def _callback(client: httpx.AsyncClient, **params: str) -> dict[str, list[str]]:
    response = await client.get(CALLBACK, params=params)
    assert response.status_code == 303, response.text
    location = response.headers["location"]
    assert location.startswith(f"{WEB}/accounts?")
    assert response.headers["cache-control"] == "no-store"
    return parse_qs(urlsplit(location).query)


async def _connect(
    client: httpx.AsyncClient,
    auth: AuthHeaders,
    world: World,
    picks: list[dict[str, str]] | None = None,
) -> httpx.Response:
    connection_id, state = await _start(client, auth, world.a.tenant_id)
    await _callback(client, state=state, code="good-code")
    return await client.post(
        f"/v1/marketing/integrations/meta/connections/{connection_id}/accounts",
        headers=auth("admin-a", world.a.tenant_id),
        json={
            "accounts": picks
            or [
                {"channel": "facebook", "provider_account_id": "1001"},
                {"channel": "instagram", "provider_account_id": "1789"},
            ]
        },
    )


async def test_admin_connects_a_page_and_its_instagram_account_without_exposing_tokens(
    client: httpx.AsyncClient,
    connection: AsyncConnection,
    world: World,
    auth: AuthHeaders,
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level(logging.DEBUG)
    connection_id, state = await _start(client, auth, world.a.tenant_id)
    landing = await _callback(client, state=state, code="good-code")
    found = await client.get(
        f"/v1/marketing/integrations/meta/connections/{connection_id}",
        headers=auth("admin-a", world.a.tenant_id),
    )
    bound = await client.post(
        f"/v1/marketing/integrations/meta/connections/{connection_id}/accounts",
        headers=auth("admin-a", world.a.tenant_id),
        json={
            "accounts": [
                {"channel": "facebook", "provider_account_id": "1001"},
                {"channel": "instagram", "provider_account_id": "1789"},
            ]
        },
    )
    listed = await client.get(
        "/v1/marketing/channel-accounts", headers=auth("creator-a", world.a.tenant_id)
    )

    assert landing == {"meta_connection": [connection_id], "meta_connect": ["select_accounts"]}
    assert [
        (c["channel"], c["provider_account_id"], c["can_publish"])
        for c in found.json()["candidates"]
    ] == [
        ("facebook", "1001", True),
        ("instagram", "1789", True),
        ("facebook", "2002", False),
    ]
    assert found.json()["candidates"][2]["missing_permissions"] == ["page_task:CREATE_CONTENT"]
    assert bound.status_code == 200, bound.text
    items = listed.json()["items"]
    assert [
        (a["channel"], a["name"], a["handle"], a["state"], a["can_publish"]) for a in items
    ] == [
        ("facebook", "Synthetic Research Clinic", None, "TEST_ONLY", True),
        ("instagram", "Synthetic Clinic", "synthetic.clinic", "TEST_ONLY", True),
    ]
    assert items[0]["capabilities"]["text_only"] is True
    assert items[1]["capabilities"]["requires_media"] is True
    assert items[1]["capabilities"]["version"] == items[0]["capabilities"]["version"]
    for body in (found.text, bound.text, listed.text, caplog.text):
        assert USER_TOKEN not in body
        assert PAGE_TOKEN not in body
        assert "short-lived" not in body

    await bind_tenant_on(connection, world.a.tenant_id)
    rows = (
        await connection.execute(
            select(ChannelAccount.id, ChannelAccount.tenant_id, ChannelAccount.token_encrypted)
        )
    ).all()
    session = (
        await connection.execute(
            select(
                MetaConnectSession.state_hash,
                MetaConnectSession.user_token_encrypted,
                MetaConnectSession.status,
            )
        )
    ).one()
    assert len(rows) == 2
    cipher = TokenCipher(SecretStr(KEY))
    for row in rows:
        assert row.token_encrypted is not None
        assert PAGE_TOKEN not in row.token_encrypted
        context = f"channel-account:{row.tenant_id}:{row.id}"
        assert cipher.decrypt(row.token_encrypted, context=context) == PAGE_TOKEN
    # The state's secret is never stored, and the user token is gone once accounts are bound.
    assert state.split(".", 1)[1] not in session.state_hash
    assert session.user_token_encrypted is None
    assert session.status == "COMPLETED"


async def bind_tenant_on(connection: AsyncConnection, tenant_id: UUID) -> None:
    await connection.execute(
        text("SELECT set_config('app.tenant_id', :t, true)"), {"t": str(tenant_id)}
    )


async def test_the_login_url_asks_only_for_publishing_permissions(
    client: httpx.AsyncClient, world: World, auth: AuthHeaders
) -> None:
    response = await client.post(
        "/v1/marketing/integrations/meta/connect",
        headers=auth("admin-a", world.a.tenant_id),
        json={},
    )
    url = urlsplit(response.json()["authorization_url"])
    query = parse_qs(url.query)

    assert f"{url.scheme}://{url.netloc}{url.path}" == "https://www.facebook.com/v26.0/dialog/oauth"
    assert query["client_id"] == ["123456789"]
    assert query["redirect_uri"] == [f"https://api.example.org{CALLBACK}"]
    assert set(query["scope"][0].split(",")) == {
        "business_management",
        "pages_show_list",
        "pages_read_engagement",
        "pages_manage_posts",
        "instagram_basic",
        "instagram_content_publish",
    }
    assert query["state"][0].startswith(f"{world.a.tenant_id.hex}.")


@pytest.mark.parametrize(
    "forge",
    [
        lambda s: s,  # replayed
        lambda s: s.split(".")[0] + ".not-the-secret",
        lambda s: uuid4().hex + "." + s.split(".")[1],  # right secret, another tenant
        lambda s: "garbage",
        lambda s: "",
    ],
    ids=["replay", "wrong-secret", "other-tenant", "malformed", "empty"],
)
async def test_a_replayed_or_forged_state_is_refused_and_changes_nothing(
    client: httpx.AsyncClient,
    world: World,
    auth: AuthHeaders,
    graph: FakeGraph,
    forge: Callable[[str], str],
) -> None:
    connection_id, state = await _start(client, auth, world.a.tenant_id)
    await _callback(client, state=state, code="good-code")
    codes_before = len(graph.codes)

    landing = await _callback(client, state=forge(state), code="good-code")
    after = await client.get(
        f"/v1/marketing/integrations/meta/connections/{connection_id}",
        headers=auth("admin-a", world.a.tenant_id),
    )

    assert landing == {"meta_connect": ["failed"], "reason": ["invalid_state"]}
    assert len(graph.codes) == codes_before  # Meta was never called
    assert after.json()["status"] == "AWAITING_SELECTION"


async def test_cancelled_expired_and_failed_logins_land_with_a_reason(
    client: httpx.AsyncClient,
    connection: AsyncConnection,
    world: World,
    auth: AuthHeaders,
    graph: FakeGraph,
) -> None:
    cancelled_id, cancelled = await _start(client, auth, world.a.tenant_id)
    bad_code_id, bad_code = await _start(client, auth, world.a.tenant_id)
    expired_id, expired = await _start(client, auth, world.a.tenant_id)
    await bind_tenant_on(connection, world.a.tenant_id)
    await connection.execute(
        text(
            "UPDATE meta_connect_sessions SET expires_at = now() - interval '1 second' "
            "WHERE id = :id"
        ),
        {"id": expired_id},
    )

    results = [
        await _callback(client, state=cancelled, error="access_denied"),
        await _callback(client, state=bad_code, code="stolen-code"),
        await _callback(client, state=expired, code="good-code"),
    ]
    graph.pages_list = []
    _, no_pages = await _start(client, auth, world.a.tenant_id)
    results.append(await _callback(client, state=no_pages, code="good-code"))
    reasons = [
        (
            await client.get(
                f"/v1/marketing/integrations/meta/connections/{cid}",
                headers=auth("admin-a", world.a.tenant_id),
            )
        ).json()["error_code"]
        for cid in (cancelled_id, bad_code_id, expired_id)
    ]

    assert [r["meta_connect"] for r in results] == [["failed"]] * 4
    assert reasons == ["login_cancelled", "meta_error", "state_expired"]


async def test_only_the_admin_who_started_the_login_can_see_or_bind_it(
    client: httpx.AsyncClient, world: World, auth: AuthHeaders
) -> None:
    connection_id, state = await _start(client, auth, world.a.tenant_id)
    await _callback(client, state=state, code="good-code")
    pick = {"accounts": [{"channel": "facebook", "provider_account_id": "1001"}]}

    other_admin = await client.get(
        f"/v1/marketing/integrations/meta/connections/{connection_id}",
        headers=auth("admin-a2", world.a.tenant_id),
    )
    other_tenant = await client.post(
        f"/v1/marketing/integrations/meta/connections/{connection_id}/accounts",
        headers=auth("admin-b", world.b.tenant_id),
        json=pick,
    )
    creator = await client.post(
        f"/v1/marketing/integrations/meta/connections/{connection_id}/accounts",
        headers=auth("creator-a", world.a.tenant_id),
        json=pick,
    )

    assert other_admin.status_code == 404
    assert other_tenant.status_code == 404
    assert creator.status_code == 403


async def test_an_account_the_login_did_not_list_cannot_be_bound(
    client: httpx.AsyncClient, world: World, auth: AuthHeaders
) -> None:
    response = await _connect(
        client,
        auth,
        world,
        [
            {"channel": "facebook", "provider_account_id": "1001"},
            {"channel": "facebook", "provider_account_id": "9999"},
        ],
    )
    listed = await client.get(
        "/v1/marketing/channel-accounts", headers=auth("admin-a", world.a.tenant_id)
    )

    assert response.status_code == 409
    assert response.json()["code"] == "ACCOUNT_NOT_OFFERED"
    assert listed.json()["items"] == []


async def test_a_login_binds_once(
    client: httpx.AsyncClient, world: World, auth: AuthHeaders
) -> None:
    connection_id, state = await _start(client, auth, world.a.tenant_id)
    await _callback(client, state=state, code="good-code")
    url = f"/v1/marketing/integrations/meta/connections/{connection_id}/accounts"
    body = {"accounts": [{"channel": "facebook", "provider_account_id": "1001"}]}

    first = await client.post(url, headers=auth("admin-a", world.a.tenant_id), json=body)
    second = await client.post(url, headers=auth("admin-a", world.a.tenant_id), json=body)

    assert first.status_code == 200
    assert second.status_code == 409
    assert second.json()["code"] == "CONNECT_SESSION_CLOSED"


async def test_missing_permissions_block_publishing_with_a_reason(
    client: httpx.AsyncClient, world: World, auth: AuthHeaders, graph: FakeGraph
) -> None:
    graph.granted = ALL_SCOPES - {"instagram_content_publish"}

    bound = await _connect(client, auth, world)

    facebook, instagram = bound.json()["items"]
    assert facebook["state"] == "TEST_ONLY"
    assert instagram["state"] == "PERMISSION_MISSING"
    assert instagram["can_publish"] is False
    assert instagram["missing_permissions"] == ["instagram_content_publish"]
    assert "instagram_content_publish" in instagram["state_reason"]


@pytest.mark.parametrize(
    ("mode", "state", "can_publish"),
    [("live", "CONNECTED", True), ("in_review", "PENDING_APP_REVIEW", False)],
)
async def test_the_meta_app_mode_decides_whether_a_full_grant_can_publish(
    client: httpx.AsyncClient,
    world: World,
    auth: AuthHeaders,
    config: list[MetaConfig | None],
    mode: str,
    state: str,
    can_publish: bool,
) -> None:
    config[0] = _config(mode)

    bound = await _connect(client, auth, world)

    assert {(a["state"], a["can_publish"]) for a in bound.json()["items"]} == {(state, can_publish)}


async def test_disconnect_stops_publishing_keeps_the_account_listed_and_forgets_the_token(
    client: httpx.AsyncClient, connection: AsyncConnection, world: World, auth: AuthHeaders
) -> None:
    account_id = (await _connect(client, auth, world)).json()["items"][0]["id"]
    url = f"/v1/marketing/channel-accounts/{account_id}/disconnect"

    by_creator = await client.post(url, headers=auth("creator-a", world.a.tenant_id))
    by_other_tenant = await client.post(url, headers=auth("admin-b", world.b.tenant_id))
    first = await client.post(url, headers=auth("admin-a", world.a.tenant_id))
    again = await client.post(url, headers=auth("admin-a", world.a.tenant_id))
    listed = await client.get(
        "/v1/marketing/channel-accounts", headers=auth("admin-a", world.a.tenant_id)
    )

    assert by_creator.status_code == 403
    assert by_other_tenant.status_code == 404
    assert first.json()["state"] == "DISCONNECTED"
    assert first.json()["can_publish"] is False
    assert again.json() == first.json()
    assert account_id in [a["id"] for a in listed.json()["items"]]
    await bind_tenant_on(connection, world.a.tenant_id)
    token = await connection.scalar(
        select(ChannelAccount.token_encrypted).where(ChannelAccount.id == UUID(account_id))
    )
    assert token is None


async def test_reconnect_refreshes_the_same_account(
    client: httpx.AsyncClient, world: World, auth: AuthHeaders
) -> None:
    account_id = (await _connect(client, auth, world)).json()["items"][1]["id"]
    await client.post(
        f"/v1/marketing/channel-accounts/{account_id}/disconnect",
        headers=auth("admin-a", world.a.tenant_id),
    )

    started = await client.post(
        f"/v1/marketing/channel-accounts/{account_id}/reconnect",
        headers=auth("admin-a", world.a.tenant_id),
        json={"return_path": "/accounts"},
    )
    state = parse_qs(urlsplit(started.json()["authorization_url"]).query)["state"][0]
    landing = await _callback(client, state=state, code="good-code")
    listed = await client.get(
        "/v1/marketing/channel-accounts", headers=auth("admin-a", world.a.tenant_id)
    )

    assert landing["meta_connect"] == ["reconnected"]
    account = next(a for a in listed.json()["items"] if a["id"] == account_id)
    assert account["state"] == "TEST_ONLY"
    assert account["disconnected_at"] is None
    assert len(listed.json()["items"]) == 2  # same row, no duplicate


@pytest.mark.parametrize(
    ("answer", "state"),
    [
        (TokenInfo(is_valid=False, expires_at=None, scopes=frozenset()), "REVOKED"),
        (
            TokenInfo(
                is_valid=False,
                expires_at=datetime(2026, 1, 1, tzinfo=UTC),
                scopes=frozenset(),
            ),
            "EXPIRED",
        ),
        (MetaGraphError("Error validating access token", code=190), "REVOKED"),
        (
            TokenInfo(is_valid=True, expires_at=None, scopes=ALL_SCOPES - {"pages_manage_posts"}),
            "PERMISSION_MISSING",
        ),
    ],
    ids=["revoked", "expired", "token-error", "permission-removed"],
)
async def test_check_reports_what_meta_says_now(
    client: httpx.AsyncClient,
    world: World,
    auth: AuthHeaders,
    graph: FakeGraph,
    answer: TokenInfo | Exception,
    state: str,
) -> None:
    account_id = (await _connect(client, auth, world)).json()["items"][0]["id"]
    graph.token_info = answer

    checked = await client.post(
        f"/v1/marketing/channel-accounts/{account_id}/check",
        headers=auth("admin-a", world.a.tenant_id),
    )

    assert checked.status_code == 200, checked.text
    assert checked.json()["state"] == state
    assert checked.json()["can_publish"] is False
    assert checked.json()["state_reason"]


async def test_check_does_not_change_the_account_when_meta_is_down(
    client: httpx.AsyncClient, world: World, auth: AuthHeaders, graph: FakeGraph
) -> None:
    account_id = (await _connect(client, auth, world)).json()["items"][0]["id"]
    graph.token_info = MetaGraphError("Service temporarily unavailable", code=2)

    checked = await client.post(
        f"/v1/marketing/channel-accounts/{account_id}/check",
        headers=auth("admin-a", world.a.tenant_id),
    )
    listed = await client.get(
        "/v1/marketing/channel-accounts", headers=auth("admin-a", world.a.tenant_id)
    )

    assert checked.status_code == 503
    assert listed.json()["items"][0]["state"] == "TEST_ONLY"


async def test_accounts_are_isolated_per_tenant(
    client: httpx.AsyncClient, world: World, auth: AuthHeaders
) -> None:
    await _connect(client, auth, world)

    other = await client.get(
        "/v1/marketing/channel-accounts", headers=auth("admin-b", world.b.tenant_id)
    )

    assert other.json() == {"items": [], "next_cursor": None}


async def test_connecting_answers_501_until_meta_is_configured(
    client: httpx.AsyncClient, world: World, auth: AuthHeaders, config: list[MetaConfig | None]
) -> None:
    config[0] = None

    started = await client.post(
        "/v1/marketing/integrations/meta/connect",
        headers=auth("admin-a", world.a.tenant_id),
        json={},
    )
    listed = await client.get(
        "/v1/marketing/channel-accounts", headers=auth("admin-a", world.a.tenant_id)
    )

    assert started.status_code == 501
    assert listed.status_code == 200


@pytest.mark.parametrize(
    "path", ["https://evil.example/x", "//evil.example", "//evil", "/ok?x=1", "x", "/a//b"]
)
async def test_the_return_path_cannot_point_off_the_web_app(
    client: httpx.AsyncClient, world: World, auth: AuthHeaders, path: str
) -> None:
    response = await client.post(
        "/v1/marketing/integrations/meta/connect",
        headers=auth("admin-a", world.a.tenant_id),
        json={"return_path": path},
    )

    assert response.status_code == 422


# ---- Units: token encryption and the Graph client


def test_a_token_only_decrypts_for_the_record_it_was_sealed_for() -> None:
    cipher = TokenCipher(SecretStr(KEY))
    sealed = cipher.encrypt("secret", context="channel-account:a:1")
    other_key = TokenCipher(SecretStr("k2:" + base64.b64encode(b"1" * 32).decode()))

    assert cipher.decrypt(sealed, context="channel-account:a:1") == "secret"
    with pytest.raises(TokenDecryptError):
        cipher.decrypt(sealed, context="channel-account:b:1")
    with pytest.raises(TokenDecryptError):
        other_key.decrypt(sealed, context="channel-account:a:1")


@pytest.mark.parametrize(
    "spec", ["no-colon", "k1:not base64!", "k1:" + base64.b64encode(b"short").decode()]
)
def test_a_bad_encryption_key_is_refused(spec: str) -> None:
    with pytest.raises(TokenKeyError):
        TokenCipher(SecretStr(spec))


async def test_graph_calls_send_the_token_in_a_header_with_an_app_secret_proof() -> None:
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        if request.url.path.endswith("/me/accounts"):
            return httpx.Response(
                200,
                json={
                    "data": [
                        {
                            "id": "1001",
                            "name": "Page",
                            "access_token": "pt",
                            "tasks": ["CREATE_CONTENT"],
                            "instagram_business_account": {"id": "1789", "username": "u"},
                        }
                    ]
                },
            )
        return httpx.Response(
            400,
            json={"error": {"message": "Invalid OAuth access token", "code": 190}},
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
        graph = HttpMetaGraph(
            http,
            graph_base_url="https://graph.facebook.com",
            version="v26.0",
            app_id="123",
            app_secret=SecretStr("app-secret"),
        )
        pages = await graph.pages(SecretStr("user-tok"))
        with pytest.raises(MetaGraphError) as failure:
            await graph.me(SecretStr("user-tok"))

    assert pages[0].instagram == MetaInstagram(id="1789", username="u", name=None)
    assert seen[0].url.path == "/v26.0/me/accounts"
    assert seen[0].headers["authorization"] == "Bearer user-tok"
    assert "user-tok" not in str(seen[0].url)
    assert len(seen[0].url.params["appsecret_proof"]) == 64
    assert failure.value.token_invalid
    assert "user-tok" not in str(failure.value)


async def test_accounts_page_in_a_stable_order(
    client: httpx.AsyncClient, world: World, auth: AuthHeaders, graph: FakeGraph
) -> None:
    graph.pages_list = [
        MetaPage(
            id=str(3000 + n),
            name=f"Synthetic Page {n}",
            access_token=SecretStr(f"page-token-{n}"),
            tasks=frozenset({"CREATE_CONTENT"}),
            instagram=None,
        )
        for n in range(3)
    ]
    await _connect(
        client,
        auth,
        world,
        [{"channel": "facebook", "provider_account_id": str(3000 + n)} for n in range(3)],
    )
    headers = auth("creator-a", world.a.tenant_id)

    first = (await client.get("/v1/marketing/channel-accounts?limit=2", headers=headers)).json()
    second = (
        await client.get(
            f"/v1/marketing/channel-accounts?limit=2&cursor={first['next_cursor']}",
            headers=headers,
        )
    ).json()
    unknown = await client.get(f"/v1/marketing/channel-accounts?cursor={uuid4()}", headers=headers)

    assert [a["name"] for a in first["items"] + second["items"]] == [
        "Synthetic Page 0",
        "Synthetic Page 1",
        "Synthetic Page 2",
    ]
    assert second["next_cursor"] is None
    assert unknown.status_code == 404


async def test_an_unexpected_meta_answer_is_an_error_without_the_token() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        # A page without its id, and the token echoed back in an odd place.
        return httpx.Response(200, json={"data": [{"name": "x", "access_token": "pt-SECRET"}]})

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
        graph = HttpMetaGraph(
            http,
            graph_base_url="https://graph.facebook.com",
            version="v26.0",
            app_id="123",
            app_secret=SecretStr("app-secret"),
        )
        with pytest.raises(MetaGraphError, match="unexpected") as failure:
            await graph.pages(SecretStr("user-tok"))

    assert "SECRET" not in str(failure.value)
    assert failure.value.__cause__ is None
