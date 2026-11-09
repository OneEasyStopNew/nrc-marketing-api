# Authentication and tenant access.

import httpx
import pytest
from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography.hazmat.primitives.asymmetric.rsa import RSAPrivateKey

from tests.conftest import AuthHeaders, World, make_token


def _bearer(token: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {token}"}


async def test_missing_token_returns_401_error_body(client: httpx.AsyncClient) -> None:
    response = await client.get("/v1/marketing/me")

    assert response.status_code == 401
    assert response.headers["WWW-Authenticate"] == "Bearer"
    body = response.json()
    assert body["code"] == "UNAUTHENTICATED"
    assert body["request_id"] == response.headers["X-Request-Id"]
    assert body["retryable"] is False


@pytest.mark.parametrize(
    "overrides",
    [
        pytest.param({"exp": 1}, id="expired"),
        pytest.param({"iss": "https://evil.example.com"}, id="wrong-issuer"),
        pytest.param({"client_id": "other-app"}, id="wrong-client"),
        pytest.param({"token_use": "id"}, id="id-token"),
    ],
)
async def test_rejects_invalid_access_tokens(
    client: httpx.AsyncClient, signing_key: RSAPrivateKey, overrides: dict[str, object]
) -> None:
    response = await client.get(
        "/v1/marketing/me", headers=_bearer(make_token(signing_key, "creator-a", **overrides))
    )

    assert response.status_code == 401


async def test_rejects_token_signed_by_another_key(client: httpx.AsyncClient) -> None:
    other_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)

    response = await client.get(
        "/v1/marketing/me", headers=_bearer(make_token(other_key, "creator-a"))
    )

    assert response.status_code == 401


async def test_missing_scope_returns_403(
    client: httpx.AsyncClient, signing_key: RSAPrivateKey
) -> None:
    token = make_token(signing_key, "creator-a", scope="openid")

    response = await client.get("/v1/marketing/me", headers=_bearer(token))

    assert response.status_code == 403


async def test_me_lists_only_own_memberships(
    client: httpx.AsyncClient, world: World, auth: AuthHeaders
) -> None:
    response = await client.get("/v1/marketing/me", headers=auth("creator-a"))

    assert response.status_code == 200
    body = response.json()
    assert body["sub"] == "creator-a"
    assert [m["tenant_id"] for m in body["memberships"]] == [str(world.a.tenant_id)]
    assert body["memberships"][0]["roles"] == ["creator"]
    assert "creative:write" in body["memberships"][0]["capabilities"]
    assert response.headers["Cache-Control"] == "no-store"


async def test_context_for_foreign_tenant_is_forbidden(
    client: httpx.AsyncClient, world: World, auth: AuthHeaders
) -> None:
    response = await client.get(
        "/v1/marketing/context", headers=auth("creator-a", world.b.tenant_id)
    )

    assert response.status_code == 403
    assert response.json()["code"] == "FORBIDDEN"


async def test_context_returns_permitted_studies_and_formats(
    client: httpx.AsyncClient, world: World, auth: AuthHeaders
) -> None:
    response = await client.get(
        "/v1/marketing/context", headers=auth("creator-a", world.a.tenant_id)
    )

    assert response.status_code == 200
    body = response.json()
    assert {s["study_id"] for s in body["studies"]} == {
        str(world.a.study_id),
        str(world.a.closed_study_id),
    }
    assert all(s["is_synthetic"] for s in body["studies"])
    assert body["post_formats"][0]["post_format"] == "feed_image"


async def test_reviewer_cannot_create_creatives(
    client: httpx.AsyncClient, world: World, auth: AuthHeaders
) -> None:
    response = await client.post(
        "/v1/marketing/creatives",
        headers=auth("reviewer-a", world.a.tenant_id, **{"Idempotency-Key": "reviewer-key-1"}),
        json={
            "study_id": str(world.a.study_id),
            "prompt": "Invite adults to learn about the study.",
            "language": "en",
            "channel": "facebook",
            "post_format": "feed_image",
        },
    )

    assert response.status_code == 403
