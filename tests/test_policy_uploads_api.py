"""Policy source upload endpoints (NRC-4) against real Postgres/RLS and the local SeaweedFS.

Policy files use the same table as client media but their own bucket, the policy:manage
capability and no study. Tests that only sign URLs or fail before storage is reached run without
SeaweedFS; the ones that PUT or HEAD an object need it.
"""

import hashlib
from collections.abc import Callable
from types import SimpleNamespace
from typing import Any, cast
from uuid import UUID, uuid4

import httpx
import pytest
from sqlalchemy import Row, text
from sqlalchemy.ext.asyncio import AsyncConnection

from app.api.deps import get_optional_storage, get_policy_source_storage, get_storage
from app.core.config import Settings, get_settings
from app.core.errors import PolicySourceStorageUnavailableError
from app.services.storage import ObjectStorage
from tests.conftest import (
    APP_ROLE,
    BUCKET,
    ENDPOINT,
    POLICY_BUCKET,
    AuthHeaders,
    World,
    as_role,
    needs_seaweedfs,
)

POLICY_UPLOADS = "/v1/marketing/policy-sources/uploads"
UPLOADS = "/v1/marketing/assets/uploads"
POLICY_ADMIN = "policy-admin-a"
OTHER_POLICY_ADMIN = "policy-admin-a2"
POLICY_ADMIN_B = "policy-admin-b"
PDF = b"%PDF-1.7\n% [SYNTHETIC] policy source for tests\n"


@pytest.fixture(autouse=True)
def local_credentials(monkeypatch: pytest.MonkeyPatch) -> None:
    # Local-only values that open the SeaweedFS container, not AWS.
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "localdev")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "localdev-password")
    monkeypatch.delenv("AWS_PROFILE", raising=False)


@pytest.fixture
def settings(policy_bucket: str) -> Settings:
    return get_settings().model_copy(
        update={
            "aws_s3_bucket": BUCKET,
            "aws_endpoint_url_s3": ENDPOINT,
            "s3_force_path_style": True,
            "aws_profile": None,
            "policy_source_bucket": policy_bucket,
        }
    )


@pytest.fixture
def media_storage(settings: Settings) -> ObjectStorage:
    return ObjectStorage(settings)


@pytest.fixture
def policy_storage(settings: Settings, policy_bucket: str) -> ObjectStorage:
    return ObjectStorage(settings, bucket=policy_bucket)


@pytest.fixture(autouse=True)
def use_storage(
    app_overrides: dict[Callable[..., object], Callable[..., object]],
    settings: Settings,
    media_storage: ObjectStorage,
    policy_storage: ObjectStorage,
) -> None:
    app_overrides[get_settings] = lambda: settings
    app_overrides[get_storage] = lambda: media_storage
    app_overrides[get_optional_storage] = lambda: media_storage
    app_overrides[get_policy_source_storage] = lambda: policy_storage


# ---- Helpers


async def _owner_sql(
    connection: AsyncConnection,
    tenant_id: UUID,
    statement: str,
    params: dict[str, Any] | None = None,
) -> list[Row[Any]]:
    # As the table owner with the tenant bound, then back to the API's role.
    await connection.execute(text("RESET ROLE"))
    await connection.execute(
        text("SELECT set_config('app.tenant_id', :t, true)"), {"t": str(tenant_id)}
    )
    result = await connection.execute(text(statement), params or {})
    rows = list(result.all()) if result.returns_rows else []
    await as_role(connection, APP_ROLE)
    return rows


async def _add_member(connection: AsyncConnection, tenant_id: UUID, sub: str, role: str) -> None:
    await _owner_sql(
        connection,
        tenant_id,
        "INSERT INTO memberships (tenant_id, user_sub, roles) VALUES (:t, :sub, ARRAY[:role])",
        {"t": tenant_id, "sub": sub, "role": role},
    )


@pytest.fixture(autouse=True)
async def staff(connection: AsyncConnection, world: World) -> None:
    # creator-a and reviewer-a come with the world; these are the other roles under test.
    tenant = world.a.tenant_id
    await _add_member(connection, tenant, POLICY_ADMIN, "policy_admin")
    await _add_member(connection, tenant, OTHER_POLICY_ADMIN, "policy_admin")
    await _add_member(connection, tenant, "policy-approver-a", "policy_approver")
    await _add_member(connection, tenant, "admin-a", "admin")
    await _add_member(connection, world.b.tenant_id, POLICY_ADMIN_B, "policy_admin")


def _body(content: bytes = PDF, **overrides: Any) -> dict[str, Any]:
    body: dict[str, Any] = {
        "name": "[SYNTHETIC] policy.pdf",
        "mime_type": "application/pdf",
        "size_bytes": len(content),
        "checksum_sha256": hashlib.sha256(content).hexdigest(),
    }
    body.update(overrides)
    return body


async def _create(
    client: httpx.AsyncClient,
    auth: AuthHeaders,
    world: World,
    content: bytes = PDF,
    *,
    key: str | None = None,
    sub: str = POLICY_ADMIN,
    **overrides: Any,
) -> httpx.Response:
    headers = auth(sub, world.a.tenant_id, **{"Idempotency-Key": key or uuid4().hex})
    return await client.post(POLICY_UPLOADS, headers=headers, json=_body(content, **overrides))


async def _created(
    client: httpx.AsyncClient, auth: AuthHeaders, world: World, content: bytes = PDF
) -> dict[str, Any]:
    response = await _create(client, auth, world, content)
    assert response.status_code == 201, response.text
    body: dict[str, Any] = response.json()
    return body


async def _put(upload: dict[str, Any], content: bytes) -> httpx.Response:
    # Exactly the returned headers; httpx adds the real Content-Length and Host itself.
    async with httpx.AsyncClient() as http:
        return await http.request(
            upload["method"], upload["url"], content=content, headers=upload["headers"]
        )


def _call(
    client: httpx.AsyncClient,
    auth: AuthHeaders,
    world: World,
    verb: str,
    asset_id: str,
    sub: str = POLICY_ADMIN,
) -> Any:
    suffix = {"get": "", "reissue": "/upload-url", "complete": "/complete"}[verb]
    method = "GET" if verb == "get" else "POST"
    return client.request(
        method, f"{POLICY_UPLOADS}/{asset_id}{suffix}", headers=auth(sub, world.a.tenant_id)
    )


async def _row(connection: AsyncConnection, world: World, asset_id: str) -> Row[Any]:
    rows = await _owner_sql(
        connection,
        world.a.tenant_id,
        "SELECT state, purpose, bucket, study_id, creative_id, identifiable_person, s3_key,"
        " s3_version_id FROM media_uploads WHERE id = :id",
        {"id": UUID(asset_id)},
    )
    assert len(rows) == 1
    return rows[0]


async def _created_and_stored(
    client: httpx.AsyncClient, auth: AuthHeaders, world: World, content: bytes = PDF
) -> dict[str, Any]:
    upload = await _created(client, auth, world, content)
    assert (await _put(upload["upload"], content)).status_code == 200
    return upload


# ---- Create


async def test_create_returns_a_policy_upload_with_a_url_in_the_policy_bucket(
    client: httpx.AsyncClient, auth: AuthHeaders, world: World, connection: AsyncConnection
) -> None:
    response = await _create(client, auth, world)

    assert response.status_code == 201, response.text
    body = response.json()
    assert set(body) == {
        "upload_id",
        "purpose",
        "status",
        "reason",
        "name",
        "mime_type",
        "size_bytes",
        "checksum_sha256",
        "scan",
        "created_by",
        "created_at",
        "completed_at",
        "upload",
    }
    assert body["purpose"] == "policy_source"
    assert body["status"] == "PENDING_UPLOAD"
    assert body["created_by"] == POLICY_ADMIN
    assert body["size_bytes"] == len(PDF)
    assert body["upload"]["method"] == "PUT"
    assert body["upload"]["headers"]["Content-Type"] == "application/pdf"
    assert f"/{POLICY_BUCKET}/" in body["upload"]["url"]
    assert response.headers["Location"] == f"{POLICY_UPLOADS}/{body['upload_id']}"
    row = await _row(connection, world, body["upload_id"])
    assert row.purpose == "policy_source"
    assert row.bucket == POLICY_BUCKET != BUCKET
    assert (row.study_id, row.creative_id) == (None, None)
    assert row.s3_key.endswith(".pdf")


@pytest.mark.parametrize(("mime", "extension"), [("text/plain", ".txt"), ("text/markdown", ".md")])
async def test_create_accepts_text_and_markdown(
    client: httpx.AsyncClient,
    auth: AuthHeaders,
    world: World,
    connection: AsyncConnection,
    mime: str,
    extension: str,
) -> None:
    response = await _create(client, auth, world, b"# Rules\n", mime_type=mime)

    assert response.status_code == 201, response.text
    row = await _row(connection, world, response.json()["upload_id"])
    assert row.s3_key.endswith(extension)


async def test_create_replay_returns_the_same_upload_and_a_new_key_a_new_one(
    client: httpx.AsyncClient, auth: AuthHeaders, world: World
) -> None:
    key = uuid4().hex
    first = await _create(client, auth, world, key=key)
    replay = await _create(client, auth, world, key=key)
    other = await _create(client, auth, world)

    assert first.status_code == replay.status_code == 201
    assert replay.headers["Idempotent-Replayed"] == "true"
    assert "Idempotent-Replayed" not in first.headers
    assert replay.json()["upload_id"] == first.json()["upload_id"]
    assert replay.json()["upload"]["headers"] == first.json()["upload"]["headers"]
    assert other.json()["upload_id"] != first.json()["upload_id"]


async def test_create_same_key_with_another_body_is_409(
    client: httpx.AsyncClient, auth: AuthHeaders, world: World
) -> None:
    key = uuid4().hex
    await _create(client, auth, world, key=key)
    other = await _create(client, auth, world, key=key, name="[SYNTHETIC] other.pdf")

    assert other.status_code == 409
    assert other.json()["code"] == "IDEMPOTENCY_KEY_REUSED"


async def test_create_without_an_idempotency_key_is_422(
    client: httpx.AsyncClient, auth: AuthHeaders, world: World
) -> None:
    response = await client.post(
        POLICY_UPLOADS, headers=auth(POLICY_ADMIN, world.a.tenant_id), json=_body()
    )

    assert response.status_code == 422
    assert response.json()["code"] == "VALIDATION_FAILED"


async def test_create_refuses_a_type_that_is_not_a_policy_source(
    client: httpx.AsyncClient, auth: AuthHeaders, world: World
) -> None:
    response = await _create(client, auth, world, mime_type="image/png")

    assert response.status_code == 422
    body = response.json()
    assert body["code"] == "UPLOAD_NOT_ACCEPTED"
    assert body["details"]["field"] == "mime_type"
    assert body["details"]["accepted"] == ["application/pdf", "text/plain", "text/markdown"]


@pytest.fixture
def small_limit(
    app_overrides: dict[Callable[..., object], Callable[..., object]], settings: Settings
) -> None:
    small = settings.model_copy(update={"policy_source_max_bytes": 2048})
    app_overrides[get_settings] = lambda: small


@pytest.mark.usefixtures("small_limit")
async def test_create_refuses_a_file_over_the_policy_limit(
    client: httpx.AsyncClient, auth: AuthHeaders, world: World
) -> None:
    over = await _create(client, auth, world, size_bytes=2049)
    exact = await _create(client, auth, world, size_bytes=2048)

    assert over.status_code == 422
    assert over.json()["code"] == "UPLOAD_NOT_ACCEPTED"
    assert over.json()["details"] == {"field": "size_bytes", "max_bytes": 2048}
    assert exact.status_code == 201


# ---- Policy storage is never the media bucket


@pytest.fixture
def no_policy_bucket(
    app_overrides: dict[Callable[..., object], Callable[..., object]],
) -> None:
    def unavailable() -> ObjectStorage:
        raise PolicySourceStorageUnavailableError

    app_overrides[get_policy_source_storage] = unavailable


@pytest.mark.usefixtures("no_policy_bucket")
async def test_create_is_503_without_a_policy_bucket(
    client: httpx.AsyncClient, auth: AuthHeaders, world: World
) -> None:
    response = await _create(client, auth, world)

    assert response.status_code == 503
    body = response.json()
    assert body["code"] == "POLICY_STORAGE_UNAVAILABLE"
    assert body["message"] == "Policy file storage is not available."
    assert body["retryable"] is False


async def test_the_policy_storage_dependency_never_falls_back_to_the_media_bucket() -> None:
    media = SimpleNamespace(policy_source_storage=None, storage=object())
    request = SimpleNamespace(app=SimpleNamespace(state=media))

    with pytest.raises(PolicySourceStorageUnavailableError, match="not available"):
        await get_policy_source_storage(cast(Any, request))


# ---- Authorization


@pytest.mark.parametrize("verb", ["create", "get", "reissue", "complete"])
@pytest.mark.parametrize("sub", ["creator-a", "reviewer-a", "policy-approver-a", "admin-a"])
async def test_only_policy_admin_may_use_the_policy_routes(
    client: httpx.AsyncClient, auth: AuthHeaders, world: World, sub: str, verb: str
) -> None:
    if verb == "create":
        response = await _create(client, auth, world, sub=sub)
    else:
        response = await _call(client, auth, world, verb, str(uuid4()), sub=sub)

    assert response.status_code == 403
    assert response.json()["code"] == "FORBIDDEN"
    assert "policy:manage" in response.json()["message"]


@pytest.mark.parametrize("verb", ["create", "get", "reissue", "complete"])
async def test_policy_routes_need_a_token(
    client: httpx.AsyncClient, world: World, verb: str
) -> None:
    headers = {"X-Tenant-Id": str(world.a.tenant_id), "Idempotency-Key": uuid4().hex}
    path = f"{POLICY_UPLOADS}/{uuid4()}"
    response = await {
        "create": lambda: client.post(POLICY_UPLOADS, headers=headers, json=_body()),
        "get": lambda: client.get(path, headers=headers),
        "reissue": lambda: client.post(f"{path}/upload-url", headers=headers),
        "complete": lambda: client.post(f"{path}/complete", headers=headers),
    }[verb]()

    assert response.status_code == 401
    assert response.json()["code"] == "UNAUTHENTICATED"


# ---- Visibility


async def test_another_tenants_upload_is_404_on_every_policy_route(
    client: httpx.AsyncClient, auth: AuthHeaders, world: World
) -> None:
    upload = await _created(client, auth, world)

    for verb in ("get", "reissue", "complete"):
        response = await client.request(
            "GET" if verb == "get" else "POST",
            f"{POLICY_UPLOADS}/{upload['upload_id']}"
            + {"get": "", "reissue": "/upload-url", "complete": "/complete"}[verb],
            headers=auth(POLICY_ADMIN_B, world.b.tenant_id),
        )
        assert response.status_code == 404, verb
        assert response.json()["code"] == "NOT_FOUND"


async def test_a_creative_upload_id_is_404_on_the_policy_routes(
    client: httpx.AsyncClient, auth: AuthHeaders, world: World
) -> None:
    content = b"\x89PNG not decoded here"
    created = await client.post(
        UPLOADS,
        headers=auth("creator-a", world.a.tenant_id, **{"Idempotency-Key": uuid4().hex}),
        json={
            "study_id": str(world.a.study_id),
            "name": "[SYNTHETIC] photo.png",
            "mime_type": "image/png",
            "size_bytes": len(content),
            "checksum_sha256": hashlib.sha256(content).hexdigest(),
            "authorization": {"identifiable_person": "no"},
        },
    )
    assert created.status_code == 201, created.text
    asset_id = created.json()["asset_id"]

    for verb in ("get", "reissue", "complete"):
        response = await _call(client, auth, world, verb, asset_id)
        assert response.status_code == 404, verb


async def test_creative_routes_never_see_a_policy_upload(
    client: httpx.AsyncClient, auth: AuthHeaders, world: World, connection: AsyncConnection
) -> None:
    upload = await _created(client, auth, world)
    asset_id = upload["upload_id"]
    # Even with a study set, which would otherwise make it visible to the study's members: the
    # purpose filter alone must hide it.
    await _owner_sql(
        connection,
        world.a.tenant_id,
        "UPDATE media_uploads SET study_id = :study WHERE id = :id",
        {"study": world.a.study_id, "id": UUID(asset_id)},
    )
    creator = auth("creator-a", world.a.tenant_id, **{"Idempotency-Key": uuid4().hex})

    got = await client.get(f"{UPLOADS}/{asset_id}", headers=creator)
    listed = await client.get(UPLOADS, headers=creator)
    reissued = await client.post(f"/v1/marketing/assets/{asset_id}/upload-url", headers=creator)
    completed = await client.post(f"/v1/marketing/assets/{asset_id}/complete", headers=creator)

    assert got.status_code == 404
    assert listed.status_code == 200
    assert listed.json()["items"] == []
    assert reissued.status_code == 404
    assert completed.status_code == 404


# ---- The upload flow (needs the stored object)


@needs_seaweedfs
async def test_reissue_gives_a_fresh_url_for_the_same_object(
    client: httpx.AsyncClient, auth: AuthHeaders, world: World
) -> None:
    upload = await _created(client, auth, world)

    response = await _call(client, auth, world, "reissue", upload["upload_id"])

    assert response.status_code == 200, response.text
    body = response.json()
    assert body["status"] == "PENDING_UPLOAD"
    assert body["upload"]["headers"] == upload["upload"]["headers"]
    assert body["upload"]["url"].split("?")[0] == upload["upload"]["url"].split("?")[0]


@needs_seaweedfs
async def test_complete_before_the_file_arrives_is_409_and_retryable(
    client: httpx.AsyncClient, auth: AuthHeaders, world: World
) -> None:
    upload = await _created(client, auth, world)

    response = await _call(client, auth, world, "complete", upload["upload_id"])

    assert response.status_code == 409
    assert response.json()["code"] == "UPLOAD_NOT_RECEIVED"
    assert response.json()["retryable"] is True


@needs_seaweedfs
async def test_put_complete_and_get_queue_the_file_for_inspection(
    client: httpx.AsyncClient, auth: AuthHeaders, world: World, connection: AsyncConnection
) -> None:
    upload = await _created_and_stored(client, auth, world)

    completed = await _call(client, auth, world, "complete", upload["upload_id"])
    again = await _call(client, auth, world, "complete", upload["upload_id"])
    got = await _call(client, auth, world, "get", upload["upload_id"], sub=OTHER_POLICY_ADMIN)

    assert completed.status_code == 202, completed.text
    assert completed.json()["status"] == "QUARANTINED"
    assert completed.json()["upload"] is None
    assert again.json()["status"] == "QUARANTINED"
    assert got.status_code == 200
    assert got.json()["upload_id"] == upload["upload_id"]
    row = await _row(connection, world, upload["upload_id"])
    assert row.state == "QUARANTINED"
    assert row.s3_version_id
    # The pinned version is not served anywhere: no policy upload is reachable as an asset.
    asset = await client.get(
        f"/v1/marketing/assets/{upload['upload_id']}", headers=auth(POLICY_ADMIN, world.a.tenant_id)
    )
    assert asset.status_code == 404


@needs_seaweedfs
async def test_reissue_after_completion_is_409(
    client: httpx.AsyncClient, auth: AuthHeaders, world: World
) -> None:
    upload = await _created_and_stored(client, auth, world)
    assert (await _call(client, auth, world, "complete", upload["upload_id"])).status_code == 202

    late = await _call(client, auth, world, "reissue", upload["upload_id"])

    assert late.status_code == 409
    assert late.json()["code"] == "UPLOAD_NOT_PENDING"
    assert late.json()["details"] == {"status": "QUARANTINED"}


@needs_seaweedfs
async def test_only_the_uploader_may_reissue_or_complete(
    client: httpx.AsyncClient, auth: AuthHeaders, world: World
) -> None:
    upload = await _created_and_stored(client, auth, world)

    for verb in ("reissue", "complete"):
        response = await _call(
            client, auth, world, verb, upload["upload_id"], sub=OTHER_POLICY_ADMIN
        )
        assert response.status_code == 403, verb
        assert response.json()["code"] == "FORBIDDEN"


@needs_seaweedfs
async def test_complete_rejects_a_file_that_is_not_the_declared_size(
    client: httpx.AsyncClient,
    auth: AuthHeaders,
    world: World,
    policy_storage: ObjectStorage,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    upload = await _created_and_stored(client, auth, world)
    real_head = policy_storage.head

    async def shorter(key: str, *, version_id: str | None = None) -> Any:
        head = await real_head(key, version_id=version_id)
        assert head is not None
        return type(head)(head.size_bytes + 1, head.version_id, None, head.content_type)

    monkeypatch.setattr(policy_storage, "head", shorter)

    response = await _call(client, auth, world, "complete", upload["upload_id"])

    assert response.status_code == 202
    assert response.json()["status"] == "REJECTED"
    assert response.json()["reason"]["code"] == "SIZE_MISMATCH"
