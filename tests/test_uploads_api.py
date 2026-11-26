"""Client upload endpoints (NRC-3) against real Postgres/RLS and the local SeaweedFS.

Storage is a real ObjectStorage on the test bucket. Tests that only sign URLs or fail before
storage is reached run without SeaweedFS; the ones that PUT or HEAD an object need it.
"""

import asyncio
import hashlib
import io
import logging
from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from typing import Any
from urllib.parse import urlsplit
from uuid import UUID, uuid4

import httpx
import pytest
from PIL import Image
from sqlalchemy import Row, text
from sqlalchemy.ext.asyncio import AsyncConnection

from app.api.deps import get_optional_storage, get_storage
from app.core.config import Settings, get_settings
from app.models import MediaUpload
from app.models.enums import Channel, IdentifiablePerson
from app.schemas.uploads import AuthorizationStatus
from app.services import uploads as uploads_service
from app.services.storage import ObjectHead, ObjectStorage, PresignedUpload, StorageError
from app.services.uploads import media_authorization
from app.workers.connector import run_once
from tests.conftest import (
    APP_ROLE,
    BUCKET,
    ENDPOINT,
    AuthHeaders,
    FakeSink,
    ScriptedProvider,
    Sessions,
    World,
    _creative,
    _post_job,
    as_role,
    needs_seaweedfs,
)

UPLOADS = "/v1/marketing/assets/uploads"
ASSETS = "/v1/marketing/assets"
MEDIA = "/v1/marketing/media"
LIMITS = f"{ASSETS}/upload-limits"


@pytest.fixture(autouse=True)
def local_credentials(monkeypatch: pytest.MonkeyPatch) -> None:
    # Local-only values that open the SeaweedFS container, not AWS.
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "localdev")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "localdev-password")
    monkeypatch.delenv("AWS_PROFILE", raising=False)


@pytest.fixture
def settings() -> Settings:
    return get_settings().model_copy(
        update={
            "aws_s3_bucket": BUCKET,
            "aws_endpoint_url_s3": ENDPOINT,
            "s3_force_path_style": True,
            "aws_profile": None,
        }
    )


@pytest.fixture
def storage(settings: Settings) -> ObjectStorage:
    return ObjectStorage(settings)


@pytest.fixture(autouse=True)
def use_storage(
    app_overrides: dict[Callable[..., object], Callable[..., object]], storage: ObjectStorage
) -> None:
    # Both dependencies, so every route sees the same real storage on the test bucket.
    app_overrides[get_storage] = lambda: storage
    app_overrides[get_optional_storage] = lambda: storage


# ---- Helpers


def _png(shade: int = 255) -> bytes:
    # A small synthetic image; the shade makes the bytes differ between uploads.
    buffer = io.BytesIO()
    Image.new("RGB", (8, 8), (shade, shade, shade)).save(buffer, "PNG")
    return buffer.getvalue()


def _body(world: World, content: bytes, **overrides: Any) -> dict[str, Any]:
    body: dict[str, Any] = {
        "study_id": str(world.a.study_id),
        "name": "[SYNTHETIC] photo.png",
        "mime_type": "image/png",
        "size_bytes": len(content),
        "checksum_sha256": hashlib.sha256(content).hexdigest(),
        "authorization": {
            "identifiable_person": "no",
            "permitted_channels": ["facebook"],
        },
    }
    body.update(overrides)
    return body


async def _create(
    client: httpx.AsyncClient,
    auth: AuthHeaders,
    world: World,
    content: bytes | None = None,
    *,
    key: str | None = None,
    sub: str = "creator-a",
    **overrides: Any,
) -> httpx.Response:
    content = content or _png()
    headers = auth(sub, world.a.tenant_id, **{"Idempotency-Key": key or uuid4().hex})
    return await client.post(UPLOADS, headers=headers, json=_body(world, content, **overrides))


async def _created(
    client: httpx.AsyncClient,
    auth: AuthHeaders,
    world: World,
    content: bytes | None = None,
    **overrides: Any,
) -> dict[str, Any]:
    response = await _create(client, auth, world, content, **overrides)
    assert response.status_code == 201, response.text
    body: dict[str, Any] = response.json()
    return body


async def _put(upload: dict[str, Any], content: bytes) -> httpx.Response:
    # Exactly the returned headers; httpx adds the real Content-Length and Host itself.
    async with httpx.AsyncClient() as http:
        return await http.request(
            upload["method"], upload["url"], content=content, headers=upload["headers"]
        )


def _complete(
    client: httpx.AsyncClient,
    auth: AuthHeaders,
    world: World,
    asset_id: str,
    sub: str = "creator-a",
) -> Any:
    return client.post(f"{ASSETS}/{asset_id}/complete", headers=auth(sub, world.a.tenant_id))


def _reissue(
    client: httpx.AsyncClient,
    auth: AuthHeaders,
    world: World,
    asset_id: str,
    sub: str = "creator-a",
) -> Any:
    return client.post(f"{ASSETS}/{asset_id}/upload-url", headers=auth(sub, world.a.tenant_id))


async def _owner_sql(
    connection: AsyncConnection,
    tenant_id: UUID,
    statement: str,
    params: dict[str, Any] | None = None,
) -> list[Row[Any]]:
    # As the table owner with the tenant bound (FORCE RLS applies to the owner too), then back
    # to the API's role.
    await connection.execute(text("RESET ROLE"))
    await connection.execute(
        text("SELECT set_config('app.tenant_id', :t, true)"), {"t": str(tenant_id)}
    )
    result = await connection.execute(text(statement), params or {})
    rows = list(result.all()) if result.returns_rows else []
    await as_role(connection, APP_ROLE)
    return rows


async def _upload_row(connection: AsyncConnection, world: World, asset_id: str) -> Row[Any]:
    rows = await _owner_sql(
        connection,
        world.a.tenant_id,
        "SELECT state, bucket, s3_key, s3_version_id, available_at, completed_at, completed_by,"
        " upload_url_expires_at, reason_code FROM media_uploads WHERE id = :id",
        {"id": UUID(asset_id)},
    )
    assert len(rows) == 1
    return rows[0]


def _key_path(url: str) -> str:
    # Path-style URL: /<bucket>/<object key>. It identifies the object, unlike the signature.
    return urlsplit(url).path


async def _add_creator(connection: AsyncConnection, world: World, sub: str) -> None:
    # A second creator of tenant A with access to both studies.
    tenant = world.a.tenant_id
    await _owner_sql(
        connection,
        tenant,
        "INSERT INTO memberships (tenant_id, user_sub, roles) VALUES (:t, :sub, ARRAY['creator'])",
        {"t": tenant, "sub": sub},
    )
    for study in (world.a.study_id, world.a.closed_study_id):
        await _owner_sql(
            connection,
            tenant,
            "INSERT INTO membership_studies (tenant_id, user_sub, study_id)"
            " VALUES (:t, :sub, :study)",
            {"t": tenant, "sub": sub, "study": study},
        )


# ---- Limits


@pytest.fixture
def custom_limits(app_overrides: dict[Callable[..., object], Callable[..., object]]) -> Settings:
    # Values that differ from the defaults, so the endpoint cannot pass by echoing constants.
    changed = get_settings().model_copy(
        update={
            "upload_image_types": ("image/png",),
            "upload_image_max_bytes": 2048,
            "upload_url_ttl_seconds": 120,
        }
    )
    app_overrides[get_settings] = lambda: changed
    return changed


async def test_limits_come_from_settings(
    custom_limits: Settings, client: httpx.AsyncClient, auth: AuthHeaders, world: World
) -> None:
    response = await client.get(LIMITS, headers=auth("creator-a", world.a.tenant_id))

    assert response.status_code == 200
    assert response.json() == {
        "image_types": ["image/png"],
        "image_max_bytes": 2048,
        "upload_url_ttl_seconds": 120,
    }


async def test_limits_default_values_and_reviewers_can_read(
    client: httpx.AsyncClient, auth: AuthHeaders, world: World
) -> None:
    configured = get_settings()

    response = await client.get(LIMITS, headers=auth("reviewer-a", world.a.tenant_id))

    assert response.status_code == 200
    assert response.json() == {
        "image_types": list(configured.upload_image_types),
        "image_max_bytes": configured.upload_image_max_bytes,
        "upload_url_ttl_seconds": configured.upload_url_ttl_seconds,
    }


async def test_limits_apply_to_new_uploads(
    custom_limits: Settings, client: httpx.AsyncClient, auth: AuthHeaders, world: World
) -> None:
    # What the limits endpoint reports is what create enforces.
    response = await _create(client, auth, world, size_bytes=2049)
    assert response.status_code == 422
    assert response.json()["details"]["field"] == "size_bytes"
    assert response.json()["details"]["max_bytes"] == 2048

    jpeg = await _create(client, auth, world, mime_type="image/jpeg")
    assert jpeg.status_code == 422
    assert jpeg.json()["details"]["field"] == "mime_type"
    assert jpeg.json()["details"]["accepted"] == ["image/png"]

    assert (await _create(client, auth, world, size_bytes=2048)).status_code == 201


# ---- Create, upload, complete


@needs_seaweedfs
async def test_upload_flow_create_put_complete(
    client: httpx.AsyncClient, auth: AuthHeaders, world: World, connection: AsyncConnection
) -> None:
    content = _png()
    headers = auth("creator-a", world.a.tenant_id, **{"Idempotency-Key": uuid4().hex})
    created = await client.post(UPLOADS, headers=headers, json=_body(world, content))

    assert created.status_code == 201
    upload = created.json()
    asset_id = upload["asset_id"]
    assert created.headers["Location"] == f"/v1/marketing/assets/uploads/{asset_id}"
    assert "Idempotent-Replayed" not in created.headers
    assert upload["status"] == "PENDING_UPLOAD"
    assert upload["purpose"] == "creative_media"
    assert upload["study_id"] == str(world.a.study_id)
    assert upload["creative_id"] is None
    assert upload["name"] == "[SYNTHETIC] photo.png"
    assert upload["mime_type"] == "image/png"
    assert upload["size_bytes"] == len(content)
    assert upload["checksum_sha256"] == hashlib.sha256(content).hexdigest()
    assert upload["created_by"] == "creator-a"
    assert upload["completed_at"] is None
    assert upload["width_px"] is None
    assert upload["reason"] is None
    assert upload["authorization"]["status"] == "NO_PERSON_DECLARED"
    target = upload["upload"]
    assert target["method"] == "PUT"
    assert target["headers"]["Content-Type"] == "image/png"
    assert target["expires_at"] > datetime.now(UTC).isoformat()
    row = await _upload_row(connection, world, asset_id)
    assert row.state == "PENDING_UPLOAD"
    assert row.bucket == BUCKET
    assert _key_path(target["url"]) == f"/{BUCKET}/{row.s3_key}"
    assert row.s3_version_id is None

    stored = await _put(target, content)
    assert stored.status_code == 200, stored.text

    done = await _complete(client, auth, world, asset_id)
    assert done.status_code == 202
    assert done.json()["status"] == "QUARANTINED"
    assert done.json()["asset_id"] == asset_id
    assert done.json()["upload"] is None
    assert done.json()["completed_at"] is not None
    row = await _upload_row(connection, world, asset_id)
    assert row.state == "QUARANTINED"
    assert row.s3_version_id
    assert row.available_at is not None
    assert row.completed_by == "creator-a"
    assert row.completed_at is not None

    again = await _complete(client, auth, world, asset_id)
    assert again.status_code == 202
    assert again.json()["status"] == "QUARANTINED"
    assert again.json()["completed_at"] == done.json()["completed_at"]
    unchanged = await _upload_row(connection, world, asset_id)
    assert unchanged.completed_at == row.completed_at
    assert unchanged.available_at == row.available_at
    assert unchanged.s3_version_id == row.s3_version_id

    polled = await client.get(f"{UPLOADS}/{asset_id}", headers=auth("creator-a", world.a.tenant_id))
    assert polled.status_code == 200
    assert polled.json()["status"] == "QUARANTINED"
    # The URL is only ever handed out while the file is awaited.
    assert polled.json()["upload"] is None


@needs_seaweedfs
async def test_upload_for_a_creative_in_the_same_study(
    client: httpx.AsyncClient, auth: AuthHeaders, world: World, sessions: Sessions
) -> None:
    creative_id = await _creative(client, auth, world.a, sessions)

    upload = await _created(client, auth, world, creative_id=creative_id)

    assert upload["creative_id"] == creative_id
    assert upload["study_id"] == str(world.a.study_id)


@needs_seaweedfs
async def test_complete_before_the_file_arrives_is_a_retryable_409(
    client: httpx.AsyncClient, auth: AuthHeaders, world: World, connection: AsyncConnection
) -> None:
    content = _png()
    upload = await _created(client, auth, world, content)
    asset_id = upload["asset_id"]

    early = await _complete(client, auth, world, asset_id)

    assert early.status_code == 409
    assert early.json()["code"] == "UPLOAD_NOT_RECEIVED"
    assert early.json()["retryable"] is True
    row = await _upload_row(connection, world, asset_id)
    assert row.state == "PENDING_UPLOAD"
    assert row.completed_at is None
    assert row.s3_version_id is None
    polled = await client.get(f"{UPLOADS}/{asset_id}", headers=auth("creator-a", world.a.tenant_id))
    assert polled.json()["status"] == "PENDING_UPLOAD"

    # The retry after the PUT succeeds with the same URL.
    assert (await _put(upload["upload"], content)).status_code == 200
    retried = await _complete(client, auth, world, asset_id)
    assert retried.status_code == 202
    assert retried.json()["status"] == "QUARANTINED"


@pytest.mark.parametrize(
    ("field", "wrong"),
    [("size_bytes", 1), ("checksum_sha256", "f" * 64)],
)
async def test_complete_rejects_a_stored_file_that_differs_from_the_declaration(
    client: httpx.AsyncClient,
    auth: AuthHeaders,
    world: World,
    connection: AsyncConnection,
    monkeypatch: pytest.MonkeyPatch,
    field: str,
    wrong: object,
) -> None:
    # Storage refuses such a PUT itself; complete re-checks what HEAD reports anyway.
    content = _png()
    upload = await _created(client, auth, world, content)
    reported: dict[str, Any] = {
        "size_bytes": len(content),
        "checksum_sha256": hashlib.sha256(content).hexdigest(),
        "version_id": "v1",
        "content_type": "image/png",
    }
    reported[field] = wrong

    async def head(self: ObjectStorage, key: str, *, version_id: str | None = None) -> ObjectHead:
        return ObjectHead(**reported)

    monkeypatch.setattr(ObjectStorage, "head", head)

    response = await _complete(client, auth, world, upload["asset_id"])

    assert response.status_code == 202
    body = response.json()
    assert body["status"] == "REJECTED"
    assert body["reason"]["code"] == f"{field.split('_')[0].upper()}_MISMATCH"
    assert body["reason"]["retryable"] is False
    row = await _upload_row(connection, world, upload["asset_id"])
    assert row.state == "REJECTED"
    assert row.available_at is None


async def test_complete_on_an_unversioned_bucket_warns(
    client: httpx.AsyncClient,
    auth: AuthHeaders,
    world: World,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    # Without a version id the worker reads whatever the key holds then, not a pinned version;
    # say so, as save_generated_image does.
    content = _png()
    upload = await _created(client, auth, world, content)

    async def head(self: ObjectStorage, key: str, *, version_id: str | None = None) -> ObjectHead:
        return ObjectHead(
            size_bytes=len(content),
            checksum_sha256=hashlib.sha256(content).hexdigest(),
            version_id=None,
            content_type="image/png",
        )

    monkeypatch.setattr(ObjectStorage, "head", head)
    with caplog.at_level(logging.WARNING, logger="app.services.uploads"):
        response = await _complete(client, auth, world, upload["asset_id"])

    assert response.status_code == 202
    warnings = [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]
    assert any("no version id" in m and upload["asset_id"] in m for m in warnings)
    assert all(BUCKET not in m and "uploads/" not in m for m in warnings)


async def test_storage_failure_on_complete_is_503_and_leaves_the_upload_pending(
    client: httpx.AsyncClient,
    auth: AuthHeaders,
    world: World,
    connection: AsyncConnection,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    upload = await _created(client, auth, world)

    async def unavailable(*args: object, **kwargs: object) -> ObjectHead:
        raise StorageError

    monkeypatch.setattr(ObjectStorage, "head", unavailable)
    response = await _complete(client, auth, world, upload["asset_id"])

    assert response.status_code == 503
    assert response.json()["code"] == "DEPENDENCY_UNAVAILABLE"
    assert BUCKET not in response.text
    assert (await _upload_row(connection, world, upload["asset_id"])).state == "PENDING_UPLOAD"


# ---- Idempotency


async def test_replay_returns_the_same_upload_with_a_fresh_target(
    client: httpx.AsyncClient, auth: AuthHeaders, world: World
) -> None:
    key = uuid4().hex
    first = await _create(client, auth, world, key=key)
    replay = await _create(client, auth, world, key=key)

    assert first.status_code == 201
    assert replay.status_code == 201
    assert replay.headers["Idempotent-Replayed"] == "true"
    assert "Idempotent-Replayed" not in first.headers
    assert replay.json()["asset_id"] == first.json()["asset_id"]
    assert replay.headers["Location"] == first.headers["Location"]
    assert replay.json()["status"] == "PENDING_UPLOAD"
    old, new = first.json()["upload"], replay.json()["upload"]
    assert new["method"] == "PUT"
    assert new["headers"] == old["headers"]
    assert _key_path(new["url"]) == _key_path(old["url"])
    assert new["expires_at"] >= old["expires_at"]
    listed = await client.get(UPLOADS, headers=auth("creator-a", world.a.tenant_id))
    assert [item["asset_id"] for item in listed.json()["items"]] == [first.json()["asset_id"]]


@needs_seaweedfs
async def test_replay_after_completion_returns_the_current_status_without_a_url(
    client: httpx.AsyncClient, auth: AuthHeaders, world: World
) -> None:
    content = _png()
    key = uuid4().hex
    first = await _create(client, auth, world, content, key=key)
    assert (await _put(first.json()["upload"], content)).status_code == 200
    assert (await _complete(client, auth, world, first.json()["asset_id"])).status_code == 202

    replay = await _create(client, auth, world, content, key=key)

    assert replay.status_code == 201
    assert replay.headers["Idempotent-Replayed"] == "true"
    assert replay.json()["asset_id"] == first.json()["asset_id"]
    assert replay.json()["status"] == "QUARANTINED"
    assert replay.json()["upload"] is None


async def test_same_key_with_another_body_is_409(
    client: httpx.AsyncClient, auth: AuthHeaders, world: World
) -> None:
    key = uuid4().hex
    first = await _create(client, auth, world, key=key)
    other = await _create(client, auth, world, key=key, name="[SYNTHETIC] other.png")

    assert first.status_code == 201
    assert other.status_code == 409
    assert other.json()["code"] == "IDEMPOTENCY_KEY_REUSED"
    listed = await client.get(UPLOADS, headers=auth("creator-a", world.a.tenant_id))
    assert len(listed.json()["items"]) == 1


async def test_a_rejected_request_does_not_burn_its_idempotency_key(
    client: httpx.AsyncClient, auth: AuthHeaders, world: World
) -> None:
    key = uuid4().hex
    refused = await _create(client, auth, world, key=key, mime_type="image/webp")
    assert refused.status_code == 422

    corrected = await _create(client, auth, world, key=key)

    assert corrected.status_code == 201
    assert "Idempotent-Replayed" not in corrected.headers


async def test_storage_failure_on_create_is_503_and_the_key_can_be_retried(
    client: httpx.AsyncClient,
    auth: AuthHeaders,
    world: World,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    key = uuid4().hex
    real_presign = ObjectStorage.presigned_put

    async def unavailable(*args: object, **kwargs: object) -> object:
        raise StorageError

    monkeypatch.setattr(ObjectStorage, "presigned_put", unavailable)
    failed = await _create(client, auth, world, key=key)
    assert failed.status_code == 503
    assert failed.json()["code"] == "DEPENDENCY_UNAVAILABLE"
    listed = await client.get(UPLOADS, headers=auth("creator-a", world.a.tenant_id))
    assert listed.json()["items"] == []

    monkeypatch.setattr(ObjectStorage, "presigned_put", real_presign)
    retried = await _create(client, auth, world, key=key)
    assert retried.status_code == 201
    assert "Idempotent-Replayed" not in retried.headers


@pytest.mark.parametrize("key", [None, "short", "has spaces in it!"])
async def test_idempotency_key_is_required_and_well_formed(
    client: httpx.AsyncClient, auth: AuthHeaders, world: World, key: str | None
) -> None:
    extra = {} if key is None else {"Idempotency-Key": key}
    response = await client.post(
        UPLOADS, headers=auth("creator-a", world.a.tenant_id, **extra), json=_body(world, _png())
    )

    assert response.status_code == 422
    assert response.json()["code"] == "VALIDATION_FAILED"


# ---- Validation


@pytest.mark.parametrize(
    ("overrides", "field"),
    [
        pytest.param(lambda _: {"mime_type": "image/webp"}, "mime_type", id="webp"),
        # Settings are read when the test runs: CI has no env file at collection time.
        pytest.param(
            lambda s: {"size_bytes": s.upload_image_max_bytes + 1}, "size_bytes", id="too-large"
        ),
    ],
)
async def test_type_or_size_outside_the_limits_is_not_accepted(
    client: httpx.AsyncClient,
    auth: AuthHeaders,
    world: World,
    overrides: Callable[[Settings], dict[str, Any]],
    field: str,
) -> None:
    response = await _create(client, auth, world, **overrides(get_settings()))

    assert response.status_code == 422
    assert response.json()["code"] == "UPLOAD_NOT_ACCEPTED"
    assert response.json()["details"]["field"] == field
    listed = await client.get(UPLOADS, headers=auth("creator-a", world.a.tenant_id))
    assert listed.json()["items"] == []


async def test_declared_size_exactly_at_the_limit_is_accepted(
    client: httpx.AsyncClient, auth: AuthHeaders, world: World
) -> None:
    response = await _create(client, auth, world, size_bytes=get_settings().upload_image_max_bytes)

    assert response.status_code == 201
    assert response.json()["size_bytes"] == get_settings().upload_image_max_bytes


# ---- Pending uploads per tenant


@pytest.fixture
def pending_cap_of_two(
    app_overrides: dict[Callable[..., object], Callable[..., object]],
) -> Settings:
    changed = get_settings().model_copy(update={"upload_max_pending_per_tenant": 2})
    app_overrides[get_settings] = lambda: changed
    return changed


async def _set_upload(
    connection: AsyncConnection, world: World, asset_id: str, assignments: str
) -> None:
    # assignments is a literal from the calling test, never input.
    await _owner_sql(
        connection,
        world.a.tenant_id,
        f"UPDATE media_uploads SET {assignments} WHERE id = :id",  # noqa: S608  literal from the test
        {"id": UUID(asset_id)},
    )


async def test_pending_uploads_are_capped_per_tenant(
    pending_cap_of_two: Settings,
    client: httpx.AsyncClient,
    auth: AuthHeaders,
    world: World,
    connection: AsyncConnection,
) -> None:
    # Regression (review, api#23): nothing bounded how many uploads one tenant could start.
    first = await _created(client, auth, world)
    await _created(client, auth, world)

    refused = await _create(client, auth, world)

    assert refused.status_code == 429
    body = refused.json()
    assert body["code"] == "UPLOAD_LIMIT_REACHED"
    assert body["retryable"] is True
    assert BUCKET not in refused.text
    listed = await client.get(UPLOADS, headers=auth("creator-a", world.a.tenant_id))
    assert len(listed.json()["items"]) == 2

    # An upload whose link expired unused stops counting, so abandoned uploads can't lock the
    # tenant out before the cleanup job exists...
    await _set_upload(
        connection, world, first["asset_id"], "upload_url_expires_at = now() - interval '1 second'"
    )
    assert (await _create(client, auth, world)).status_code == 201
    # ...and a new link for it is admitted like a new upload.
    revived = await _reissue(client, auth, world, first["asset_id"])
    assert revived.status_code == 429
    assert revived.json()["code"] == "UPLOAD_LIMIT_REACHED"


@pytest.mark.parametrize(
    ("assignments", "holds_a_slot"),
    [
        pytest.param("state = 'QUARANTINED', available_at = now()", True, id="quarantined"),
        pytest.param("state = 'SCANNING'", True, id="scanning"),
        pytest.param("state = 'REJECTED', reason_code = 'MALWARE_DETECTED'", False, id="rejected"),
    ],
)
async def test_uploads_waiting_for_inspection_hold_a_slot(
    pending_cap_of_two: Settings,
    client: httpx.AsyncClient,
    auth: AuthHeaders,
    world: World,
    connection: AsyncConnection,
    assignments: str,
    holds_a_slot: bool,
) -> None:
    first = await _created(client, auth, world)
    await _created(client, auth, world)
    # A long-expired link: only the state decides whether it still counts.
    await _set_upload(
        connection,
        world,
        first["asset_id"],
        f"{assignments}, upload_url_expires_at = now() - interval '1 day'",
    )

    response = await _create(client, auth, world)

    assert response.status_code == (429 if holds_a_slot else 201)


async def test_the_pending_cap_counts_only_the_callers_tenant(
    pending_cap_of_two: Settings, client: httpx.AsyncClient, auth: AuthHeaders, world: World
) -> None:
    await _created(client, auth, world)
    await _created(client, auth, world)
    headers = auth("creator-b", world.b.tenant_id, **{"Idempotency-Key": uuid4().hex})

    other = await client.post(
        UPLOADS,
        headers=headers,
        json=_body(world, _png(), study_id=str(world.b.study_id)),
    )

    assert other.status_code == 201, other.text


async def test_a_replay_is_not_refused_by_the_pending_cap(
    pending_cap_of_two: Settings, client: httpx.AsyncClient, auth: AuthHeaders, world: World
) -> None:
    # The cap admits new uploads; repeating a request that already made one returns it.
    content = _png()
    first = await _create(client, auth, world, content, key="cap-replay-1")
    await _created(client, auth, world)

    replay = await _create(client, auth, world, content, key="cap-replay-1")

    assert replay.status_code == 201
    assert replay.json()["asset_id"] == first.json()["asset_id"]


@pytest.mark.parametrize(
    "overrides",
    [
        pytest.param({"checksum_sha256": "A" * 64}, id="uppercase-checksum"),
        pytest.param({"checksum_sha256": "a" * 63}, id="short-checksum"),
        pytest.param({"checksum_sha256": "g" * 64}, id="non-hex-checksum"),
        pytest.param({"name": "   "}, id="blank-name"),
        pytest.param({"name": ""}, id="empty-name"),
        pytest.param({"name": "bad\x07name.png"}, id="control-character-in-name"),
        pytest.param({"name": "zero\u200bwidth.png"}, id="format-character-in-name"),
        pytest.param({"name": "n" * 256}, id="name-too-long"),
        # Regression (review, api#23): 255 raw characters passed the length check, then NFC made
        # them 510 and the insert into varchar(255) failed with a 500.
        pytest.param({"name": "क़" * 255}, id="name-too-long-after-nfc"),
        # Regression (review, api#23): a blank reference read as authorization on record.
        *(
            pytest.param(
                {"authorization": {"identifiable_person": "yes", "usage_authorization_ref": ref}},
                id=f"authorization-ref-{case}",
            )
            for case, ref in (
                ("blank", "   "),
                ("newline", "\n"),
                ("control-character", "REL\x00EASE-1"),
                ("inner-space", "RELEASE 1"),
                ("leading-symbol", "-RELEASE-1"),
                ("too-long", "R" * 201),
            )
        ),
        pytest.param({"size_bytes": 0}, id="zero-size"),
        pytest.param({"unexpected": "value"}, id="unknown-field"),
        pytest.param(
            {"authorization": {"identifiable_person": "no", "unexpected": True}},
            id="unknown-authorization-field",
        ),
        pytest.param(
            {"authorization": {"identifiable_person": "maybe"}}, id="unknown-person-value"
        ),
        pytest.param(
            {"authorization": {"identifiable_person": "no", "permitted_channels": ["myspace"]}},
            id="unknown-channel",
        ),
        pytest.param({"authorization": None}, id="authorization-missing"),
        pytest.param({"creative_id": "not-a-uuid"}, id="creative-id-not-a-uuid"),
    ],
)
async def test_invalid_request_bodies_are_422(
    client: httpx.AsyncClient, auth: AuthHeaders, world: World, overrides: dict[str, Any]
) -> None:
    response = await _create(client, auth, world, **overrides)

    assert response.status_code == 422
    assert response.json()["code"] == "VALIDATION_FAILED"
    assert response.json()["details"]["fields"]
    listed = await client.get(UPLOADS, headers=auth("creator-a", world.a.tenant_id))
    assert listed.json()["items"] == []


# ---- Scope: not found


async def test_a_study_of_another_tenant_is_404(
    client: httpx.AsyncClient, auth: AuthHeaders, world: World
) -> None:
    response = await _create(client, auth, world, study_id=str(world.b.study_id))

    assert response.status_code == 404
    assert response.json()["code"] == "NOT_FOUND"


async def test_an_unknown_study_is_404(
    client: httpx.AsyncClient, auth: AuthHeaders, world: World
) -> None:
    response = await _create(client, auth, world, study_id=str(uuid4()))

    assert response.status_code == 404


async def test_a_study_the_caller_lost_access_to_is_404(
    client: httpx.AsyncClient, auth: AuthHeaders, world: World, connection: AsyncConnection
) -> None:
    await _owner_sql(
        connection,
        world.a.tenant_id,
        "DELETE FROM membership_studies WHERE user_sub = 'creator-a' AND study_id = :study",
        {"study": world.a.study_id},
    )

    response = await _create(client, auth, world)

    assert response.status_code == 404


async def test_a_creative_from_another_study_is_404(
    client: httpx.AsyncClient, auth: AuthHeaders, world: World, sessions: Sessions
) -> None:
    # The creative belongs to the open study; the upload names the closed one.
    creative_id = await _creative(client, auth, world.a, sessions)

    response = await _create(
        client, auth, world, study_id=str(world.a.closed_study_id), creative_id=creative_id
    )

    assert response.status_code == 404
    assert response.json()["code"] == "NOT_FOUND"
    listed = await client.get(UPLOADS, headers=auth("creator-a", world.a.tenant_id))
    assert listed.json()["items"] == []


async def test_an_unknown_creative_is_404(
    client: httpx.AsyncClient, auth: AuthHeaders, world: World
) -> None:
    response = await _create(client, auth, world, creative_id=str(uuid4()))

    assert response.status_code == 404


# ---- Roles and ownership


async def test_reviewer_cannot_create_an_upload(
    client: httpx.AsyncClient, auth: AuthHeaders, world: World
) -> None:
    response = await _create(client, auth, world, sub="reviewer-a")

    assert response.status_code == 403
    assert response.json()["code"] == "FORBIDDEN"
    listed = await client.get(UPLOADS, headers=auth("creator-a", world.a.tenant_id))
    assert listed.json()["items"] == []


@pytest.mark.parametrize("action", [_complete, _reissue], ids=["complete", "upload-url"])
async def test_reviewer_cannot_complete_or_reissue(
    client: httpx.AsyncClient,
    auth: AuthHeaders,
    world: World,
    connection: AsyncConnection,
    action: Callable[..., Any],
) -> None:
    upload = await _created(client, auth, world)

    response = await action(client, auth, world, upload["asset_id"], sub="reviewer-a")

    assert response.status_code == 403
    assert response.json()["code"] == "FORBIDDEN"
    assert (await _upload_row(connection, world, upload["asset_id"])).state == "PENDING_UPLOAD"


async def test_reviewer_can_read_uploads(
    client: httpx.AsyncClient, auth: AuthHeaders, world: World
) -> None:
    upload = await _created(client, auth, world)
    headers = auth("reviewer-a", world.a.tenant_id)

    one = await client.get(f"{UPLOADS}/{upload['asset_id']}", headers=headers)
    listed = await client.get(UPLOADS, headers=headers)

    assert one.status_code == 200
    assert one.json()["asset_id"] == upload["asset_id"]
    assert one.json()["upload"] is None
    assert [item["asset_id"] for item in listed.json()["items"]] == [upload["asset_id"]]


@pytest.mark.parametrize("action", [_complete, _reissue], ids=["complete", "upload-url"])
async def test_only_the_uploader_can_complete_or_reissue(
    client: httpx.AsyncClient,
    auth: AuthHeaders,
    world: World,
    connection: AsyncConnection,
    action: Callable[..., Any],
) -> None:
    # Another creator of the same tenant, with the creative:write capability.
    await _add_creator(connection, world, "creator-a2")
    upload = await _created(client, auth, world)

    response = await action(client, auth, world, upload["asset_id"], sub="creator-a2")

    assert response.status_code == 403
    assert response.json()["code"] == "FORBIDDEN"
    row = await _upload_row(connection, world, upload["asset_id"])
    assert row.state == "PENDING_UPLOAD"
    assert row.completed_by is None


# ---- Another tenant


async def test_another_tenant_cannot_see_or_change_an_upload(
    client: httpx.AsyncClient, auth: AuthHeaders, world: World, connection: AsyncConnection
) -> None:
    upload = await _created(client, auth, world)
    asset_id = upload["asset_id"]
    foreign = auth("creator-b", world.b.tenant_id)

    read = await client.get(f"{UPLOADS}/{asset_id}", headers=foreign)
    complete = await client.post(f"{ASSETS}/{asset_id}/complete", headers=foreign)
    reissue = await client.post(f"{ASSETS}/{asset_id}/upload-url", headers=foreign)
    listed = await client.get(UPLOADS, headers=foreign)

    for response in (read, complete, reissue):
        assert response.status_code == 404
        assert response.json()["code"] == "NOT_FOUND"
        assert asset_id not in response.text
    assert listed.status_code == 200
    assert listed.json() == {"items": [], "next_cursor": None}
    assert (await _upload_row(connection, world, asset_id)).state == "PENDING_UPLOAD"
    # Naming tenant A with tenant B's login is refused outright.
    forged = await client.get(UPLOADS, headers=auth("creator-b", world.a.tenant_id))
    assert forged.status_code == 403


async def test_uploads_require_login(client: httpx.AsyncClient, world: World) -> None:
    response = await client.get(UPLOADS, headers={"X-Tenant-Id": str(world.a.tenant_id)})

    assert response.status_code == 401


# ---- New URL for a pending upload


@needs_seaweedfs
async def test_reissue_gives_a_new_url_for_the_same_object_until_completed(
    client: httpx.AsyncClient, auth: AuthHeaders, world: World, connection: AsyncConnection
) -> None:
    content = _png()
    upload = await _created(client, auth, world, content)
    asset_id = upload["asset_id"]
    first = upload["upload"]
    # The signature changes with the signing time, which has one-second resolution.
    await asyncio.sleep(1.1)

    response = await _reissue(client, auth, world, asset_id)

    assert response.status_code == 200
    body = response.json()
    assert body["asset_id"] == asset_id
    assert body["status"] == "PENDING_UPLOAD"
    second = body["upload"]
    assert second["method"] == "PUT"
    assert second["headers"] == first["headers"]
    assert second["url"] != first["url"]
    assert _key_path(second["url"]) == _key_path(first["url"])
    assert second["expires_at"] > first["expires_at"]
    row = await _upload_row(connection, world, asset_id)
    assert (
        row.upload_url_expires_at.isoformat()
        == datetime.fromisoformat(second["expires_at"]).astimezone(UTC).isoformat()
    )

    assert (await _put(second, content)).status_code == 200
    assert (await _complete(client, auth, world, asset_id)).status_code == 202

    late = await _reissue(client, auth, world, asset_id)
    assert late.status_code == 409
    assert late.json()["code"] == "UPLOAD_NOT_PENDING"
    assert late.json()["details"] == {"status": "QUARANTINED"}


@pytest.fixture
def completes_while_signing(
    monkeypatch: pytest.MonkeyPatch, connection: AsyncConnection, world: World
) -> None:
    # A complete that commits between reading the pending upload and storing the new URL.
    real_presign = uploads_service._presign

    async def presign_then_complete(
        storage: ObjectStorage, upload: MediaUpload, settings: Settings
    ) -> PresignedUpload:
        target = await real_presign(storage, upload, settings)
        await _owner_sql(
            connection,
            world.a.tenant_id,
            "UPDATE media_uploads SET state = 'QUARANTINED' WHERE id = :id",
            {"id": upload.id},
        )
        return target

    monkeypatch.setattr(uploads_service, "_presign", presign_then_complete)


async def test_reissue_racing_a_complete_is_409_without_a_url(
    client: httpx.AsyncClient,
    auth: AuthHeaders,
    world: World,
    request: pytest.FixtureRequest,
) -> None:
    upload = await _created(client, auth, world)
    request.getfixturevalue("completes_while_signing")

    response = await _reissue(client, auth, world, upload["asset_id"])

    assert response.status_code == 409
    assert response.json()["code"] == "UPLOAD_NOT_PENDING"
    assert response.json()["details"] == {"status": "QUARANTINED"}
    assert "upload" not in response.json()


async def test_create_replay_racing_a_complete_returns_no_url(
    client: httpx.AsyncClient,
    auth: AuthHeaders,
    world: World,
    request: pytest.FixtureRequest,
) -> None:
    content = _png()
    first = await _create(client, auth, world, content, key="replay-race")
    assert first.status_code == 201
    request.getfixturevalue("completes_while_signing")

    replay = await _create(client, auth, world, content, key="replay-race")

    assert replay.status_code == 201
    assert replay.headers["Idempotent-Replayed"] == "true"
    assert replay.json()["status"] == "QUARANTINED"
    assert replay.json()["upload"] is None


async def test_reissue_of_an_unknown_upload_is_404(
    client: httpx.AsyncClient, auth: AuthHeaders, world: World
) -> None:
    response = await _reissue(client, auth, world, str(uuid4()))

    assert response.status_code == 404


# ---- Listing


async def test_list_is_newest_first_with_status_study_filters_and_paging(
    client: httpx.AsyncClient, auth: AuthHeaders, world: World, connection: AsyncConnection
) -> None:
    oldest = await _created(client, auth, world, _png(10))
    middle = await _created(client, auth, world, _png(20), study_id=str(world.a.closed_study_id))
    newest = await _created(client, auth, world, _png(30))
    base = datetime(2026, 11, 1, tzinfo=UTC)
    stamps = {oldest["asset_id"]: base, middle["asset_id"]: base + timedelta(hours=1)}
    stamps[newest["asset_id"]] = base + timedelta(hours=2)
    for asset_id, stamp in stamps.items():
        await _owner_sql(
            connection,
            world.a.tenant_id,
            "UPDATE media_uploads SET created_at = :at WHERE id = :id",
            {"at": stamp, "id": UUID(asset_id)},
        )
    # One upload the inspection already refused.
    await _owner_sql(
        connection,
        world.a.tenant_id,
        "UPDATE media_uploads SET state = 'REJECTED', reason_code = 'MEDIA_CORRUPT',"
        " reason_message = 'The file is not a valid, complete image.' WHERE id = :id",
        {"id": UUID(middle["asset_id"])},
    )
    headers = auth("creator-a", world.a.tenant_id)

    everything = await client.get(UPLOADS, headers=headers)
    assert everything.status_code == 200
    items = everything.json()["items"]
    assert [item["asset_id"] for item in items] == [
        newest["asset_id"],
        middle["asset_id"],
        oldest["asset_id"],
    ]
    assert everything.json()["next_cursor"] is None
    assert all(item["upload"] is None for item in items)
    rejected = items[1]
    assert rejected["status"] == "REJECTED"
    assert rejected["reason"] == {
        "code": "MEDIA_CORRUPT",
        "message": "The file is not a valid, complete image.",
        "retryable": False,
    }

    only_rejected = await client.get(UPLOADS, headers=headers, params={"status": "REJECTED"})
    assert [item["asset_id"] for item in only_rejected.json()["items"]] == [middle["asset_id"]]
    either = await client.get(
        UPLOADS, headers=headers, params=[("status", "REJECTED"), ("status", "PENDING_UPLOAD")]
    )
    assert [item["asset_id"] for item in either.json()["items"]] == [
        newest["asset_id"],
        middle["asset_id"],
        oldest["asset_id"],
    ]
    none = await client.get(UPLOADS, headers=headers, params={"status": "READY"})
    assert none.json() == {"items": [], "next_cursor": None}

    in_closed = await client.get(
        UPLOADS, headers=headers, params={"study_id": str(world.a.closed_study_id)}
    )
    assert [item["asset_id"] for item in in_closed.json()["items"]] == [middle["asset_id"]]
    in_open = await client.get(UPLOADS, headers=headers, params={"study_id": str(world.a.study_id)})
    assert [item["asset_id"] for item in in_open.json()["items"]] == [
        newest["asset_id"],
        oldest["asset_id"],
    ]

    first = await client.get(UPLOADS, headers=headers, params={"limit": 2})
    assert [item["asset_id"] for item in first.json()["items"]] == [
        newest["asset_id"],
        middle["asset_id"],
    ]
    assert datetime.fromisoformat(first.json()["next_cursor"]) == stamps[middle["asset_id"]]
    second = await client.get(
        UPLOADS, headers=headers, params={"limit": 2, "cursor": first.json()["next_cursor"]}
    )
    assert [item["asset_id"] for item in second.json()["items"]] == [oldest["asset_id"]]
    assert second.json()["next_cursor"] is None


@pytest.mark.parametrize(
    "params",
    [
        {"limit": "0"},
        {"limit": "51"},
        {"cursor": "2026-11-13T12:00:00"},
        {"status": "DONE"},
        {"study_id": "not-a-uuid"},
    ],
)
async def test_list_rejects_invalid_parameters(
    client: httpx.AsyncClient, auth: AuthHeaders, world: World, params: dict[str, str]
) -> None:
    response = await client.get(
        UPLOADS, headers=auth("creator-a", world.a.tenant_id), params=params
    )

    assert response.status_code == 422


async def test_list_hides_uploads_of_studies_the_caller_lost_access_to(
    client: httpx.AsyncClient, auth: AuthHeaders, world: World, connection: AsyncConnection
) -> None:
    upload = await _created(client, auth, world)
    await _owner_sql(
        connection,
        world.a.tenant_id,
        "DELETE FROM membership_studies WHERE user_sub = 'creator-a' AND study_id = :study",
        {"study": world.a.study_id},
    )
    headers = auth("creator-a", world.a.tenant_id)

    listed = await client.get(UPLOADS, headers=headers)
    one = await client.get(f"{UPLOADS}/{upload['asset_id']}", headers=headers)

    assert listed.json() == {"items": [], "next_cursor": None}
    assert one.status_code == 404


# ---- Media authorization


@pytest.mark.parametrize(
    ("authorization", "expected"),
    [
        pytest.param(
            {"identifiable_person": "no", "permitted_channels": ["facebook"]},
            "NO_PERSON_DECLARED",
            id="no-person",
        ),
        pytest.param(
            {"identifiable_person": "no", "usage_authorization_ref": "RELEASE-SYNTH-1"},
            "NO_PERSON_DECLARED",
            id="no-person-with-a-ref",
        ),
        pytest.param({"identifiable_person": "yes"}, "MISSING", id="person-without-ref"),
        pytest.param({"identifiable_person": "unknown"}, "MISSING", id="unknown-without-ref"),
        pytest.param(
            {
                "identifiable_person": "yes",
                "usage_authorization_ref": "RELEASE-SYNTH-2",
                "permitted_channels": ["facebook"],
            },
            "RECORDED",
            id="person-with-ref",
        ),
        pytest.param(
            {
                "identifiable_person": "yes",
                "usage_authorization_ref": "RELEASE-SYNTH-3",
                "expires_at": "2099-01-01T00:00:00Z",
            },
            "RECORDED",
            id="ref-not-yet-expired",
        ),
        pytest.param(
            {
                "identifiable_person": "yes",
                "usage_authorization_ref": "RELEASE-SYNTH-4",
                "expires_at": "2020-01-01T00:00:00Z",
            },
            "EXPIRED",
            id="expired",
        ),
        pytest.param(
            {"identifiable_person": "no", "expires_at": "2020-01-01T00:00:00Z"},
            "EXPIRED",
            id="expiry-beats-no-person",
        ),
    ],
)
async def test_authorization_status_is_derived_from_the_declaration(
    client: httpx.AsyncClient,
    auth: AuthHeaders,
    world: World,
    authorization: dict[str, Any],
    expected: str,
) -> None:
    created = await _create(client, auth, world, authorization=authorization)
    assert created.status_code == 201, created.text
    read = await client.get(
        f"{UPLOADS}/{created.json()['asset_id']}", headers=auth("creator-a", world.a.tenant_id)
    )

    for body in (created.json(), read.json()):
        returned = body["authorization"]
        assert returned["status"] == expected
        assert returned["identifiable_person"] == authorization["identifiable_person"]
        assert returned["usage_authorization_ref"] == authorization.get("usage_authorization_ref")
        assert returned["permitted_channels"] == authorization.get("permitted_channels", [])


async def test_authorization_ref_is_stored_without_surrounding_spaces(
    client: httpx.AsyncClient, auth: AuthHeaders, world: World, connection: AsyncConnection
) -> None:
    authorization = {"identifiable_person": "yes", "usage_authorization_ref": "  RELEASE-SYNTH-7 "}

    created = await _create(client, auth, world, authorization=authorization)

    assert created.status_code == 201, created.text
    returned = created.json()["authorization"]
    assert returned["usage_authorization_ref"] == "RELEASE-SYNTH-7"
    assert returned["status"] == "RECORDED"
    rows = await _owner_sql(
        connection,
        world.a.tenant_id,
        "SELECT usage_authorization_ref FROM media_uploads WHERE id = :id",
        {"id": UUID(created.json()["asset_id"])},
    )
    assert rows[0].usage_authorization_ref == "RELEASE-SYNTH-7"


async def test_more_channels_than_exist_are_refused(
    client: httpx.AsyncClient, auth: AuthHeaders, world: World
) -> None:
    # The list is capped at the number of known channels, so a repeated channel cannot fit.
    response = await _create(
        client,
        auth,
        world,
        authorization={
            "identifiable_person": "no",
            "permitted_channels": ["facebook", "facebook"],
        },
    )

    assert response.status_code == 422
    assert response.json()["code"] == "VALIDATION_FAILED"


async def test_an_authorization_that_expires_later_is_reported_expired_on_read(
    client: httpx.AsyncClient, auth: AuthHeaders, world: World, connection: AsyncConnection
) -> None:
    # Status is derived at read time: the same stored row changes status when its expiry passes.
    created = await _created(
        client,
        auth,
        world,
        authorization={
            "identifiable_person": "yes",
            "usage_authorization_ref": "RELEASE-SYNTH-5",
            "expires_at": "2099-01-01T00:00:00Z",
        },
    )
    assert created["authorization"]["status"] == "RECORDED"
    await _owner_sql(
        connection,
        world.a.tenant_id,
        "UPDATE media_uploads SET authorization_expires_at = now() - interval '1 hour'"
        " WHERE id = :id",
        {"id": UUID(created["asset_id"])},
    )

    read = await client.get(
        f"{UPLOADS}/{created['asset_id']}", headers=auth("creator-a", world.a.tenant_id)
    )

    assert read.json()["authorization"]["status"] == "EXPIRED"


# ---- Uploads in the asset and media reads


@needs_seaweedfs
async def test_only_a_ready_upload_is_an_asset_and_appears_in_the_media_library(
    client: httpx.AsyncClient, auth: AuthHeaders, world: World, connection: AsyncConnection
) -> None:
    content = _png(120)
    upload = await _created(
        client,
        auth,
        world,
        content,
        authorization={
            "identifiable_person": "yes",
            "usage_authorization_ref": "RELEASE-SYNTH-6",
            "permitted_channels": ["facebook"],
        },
    )
    asset_id = upload["asset_id"]
    assert (await _put(upload["upload"], content)).status_code == 200
    assert (await _complete(client, auth, world, asset_id)).status_code == 202
    headers = auth("creator-a", world.a.tenant_id)

    # Quarantined: neither an asset nor a library item.
    hidden = await client.get(f"{ASSETS}/{asset_id}", headers=headers)
    assert hidden.status_code == 404
    assert hidden.json()["code"] == "NOT_FOUND"
    assert (await client.get(MEDIA, headers=headers)).json() == {"items": [], "next_cursor": None}

    # What the media worker does when inspection passes (simulated here).
    tenant = world.a.tenant_id
    params = {"id": UUID(asset_id)}
    await _owner_sql(
        connection,
        tenant,
        "UPDATE media_uploads SET state = 'READY', mime_type = 'image/png', width_px = 8,"
        " height_px = 8 WHERE id = :id",
        params,
    )
    await _owner_sql(
        connection,
        tenant,
        "INSERT INTO assets (id, tenant_id, upload_id, study_id, origin, name, bucket, s3_key,"
        " s3_version_id, mime_type, width_px, height_px, size_bytes, checksum_sha256, created_by)"
        " SELECT id, tenant_id, id, study_id, 'uploaded', name, bucket, s3_key, s3_version_id,"
        " 'image/png', 8, 8, declared_size_bytes, declared_checksum_sha256, created_by"
        " FROM media_uploads WHERE id = :id",
        params,
    )

    ready = await client.get(f"{UPLOADS}/{asset_id}", headers=headers)
    assert ready.json()["status"] == "READY"
    assert ready.json()["width_px"] == 8
    assert ready.json()["height_px"] == 8

    asset = await client.get(f"{ASSETS}/{asset_id}", headers=headers)
    assert asset.status_code == 200
    body = asset.json()
    assert body["asset_id"] == asset_id
    assert body["origin"] == "uploaded"
    assert body["name"] == "[SYNTHETIC] photo.png"
    assert body["study_id"] == str(world.a.study_id)
    assert body["creative_id"] is None
    assert body["mime_type"] == "image/png"
    assert (body["width_px"], body["height_px"]) == (8, 8)
    assert body["size_bytes"] == len(content)
    assert body["checksum_sha256"] == hashlib.sha256(content).hexdigest()
    assert body["provider"] is None
    assert body["model"] is None
    assert body["authorization"] == {
        "identifiable_person": "yes",
        "usage_authorization_ref": "RELEASE-SYNTH-6",
        "permitted_channels": ["facebook"],
        "expires_at": None,
        "status": "RECORDED",
    }
    # The download link reads exactly the stored bytes.
    async with httpx.AsyncClient() as http:
        downloaded = await http.get(body["download_url"])
    assert downloaded.status_code == 200
    assert downloaded.content == content

    media = await client.get(MEDIA, headers=headers)
    assert media.status_code == 200
    assert [item["asset_id"] for item in media.json()["items"]] == [asset_id]
    item = media.json()["items"][0]
    assert item["origin"] == "uploaded"
    assert item["name"] == "[SYNTHETIC] photo.png"
    assert item["study_id"] == str(world.a.study_id)
    assert item["creative_id"] is None
    assert item["creative_version"] is None
    assert item["created_by"] == "creator-a"
    assert item["authorization"] == body["authorization"]
    assert item["download_url"]

    # Another tenant sees neither.
    foreign = auth("creator-b", world.b.tenant_id)
    assert (await client.get(f"{ASSETS}/{asset_id}", headers=foreign)).status_code == 404
    assert (await client.get(MEDIA, headers=foreign)).json() == {"items": [], "next_cursor": None}

    # Access to its study is what makes it visible, as for every asset.
    await _owner_sql(
        connection,
        tenant,
        "DELETE FROM membership_studies WHERE user_sub = 'creator-a' AND study_id = :study",
        {"study": world.a.study_id},
    )
    assert (await client.get(f"{ASSETS}/{asset_id}", headers=headers)).status_code == 404
    assert (await client.get(MEDIA, headers=headers)).json() == {"items": [], "next_cursor": None}


async def test_generated_assets_have_no_upload_fields(
    client: httpx.AsyncClient, auth: AuthHeaders, world: World, sessions: Sessions
) -> None:
    creative_id = await _creative(client, auth, world.a, sessions)
    queued = await _post_job(client, auth, world, creative_id)
    assert queued.status_code == 202
    sink = FakeSink()
    while await run_once(sessions, {"simulator": ScriptedProvider()}, sink, get_settings()):
        pass
    assert len(sink.stored) == 1
    headers = auth("creator-a", world.a.tenant_id)

    asset = await client.get(f"{ASSETS}/{sink.stored[0]}", headers=headers)
    media = await client.get(MEDIA, headers=headers)

    assert asset.status_code == 200
    assert asset.json()["origin"] == "generated"
    assert asset.json()["creative_id"] == creative_id
    assert media.status_code == 200
    items = media.json()["items"]
    assert len(items) == 1
    for body in (asset.json(), items[0]):
        assert body["origin"] == "generated"
        assert body["study_id"] is None
        assert body["name"] is None
        assert body["authorization"] is None


# ---- Logs


@needs_seaweedfs
async def test_no_upload_url_or_signature_is_logged(
    client: httpx.AsyncClient,
    auth: AuthHeaders,
    world: World,
    caplog: pytest.LogCaptureFixture,
) -> None:
    content = _png(77)
    headers = auth("creator-a", world.a.tenant_id)
    with caplog.at_level(logging.INFO):
        upload = await _created(client, auth, world, content)
        url = upload["upload"]["url"]
        asset_id = upload["asset_id"]
        assert (await _complete(client, auth, world, asset_id)).status_code == 409
        replay = await _create(client, auth, world, content, key=uuid4().hex)
        assert replay.status_code == 201
        reissued = await _reissue(client, auth, world, asset_id)
        assert reissued.status_code == 200
        reissued_url = reissued.json()["upload"]["url"]
        assert (await _put(upload["upload"], content)).status_code == 200
        assert (await _complete(client, auth, world, asset_id)).status_code == 202
        assert (await client.get(f"{UPLOADS}/{asset_id}", headers=headers)).status_code == 200
        assert (await client.get(UPLOADS, headers=headers)).status_code == 200

    # The HTTP client libraries log the request URL of this test's own PUT; that is the test
    # talking to storage, not the API, so only the application's loggers are checked.
    app_records = [
        record
        for record in caplog.records
        if not record.name.startswith(("httpx", "httpcore", "asyncio"))
    ]
    assert any("upload created" in record.getMessage() for record in app_records)
    assert any("upload completed" in record.getMessage() for record in app_records)
    for record in app_records:
        rendered = f"{record.getMessage()} {record.exc_text or ''}"
        assert "x-amz-signature" not in rendered.lower()
        assert "x-amz-credential" not in rendered.lower()
        assert url not in rendered
        assert reissued_url not in rendered
        assert _key_path(url).rsplit("/", 1)[-1] not in rendered


def test_an_unknown_stored_channel_is_dropped_not_fatal(caplog: pytest.LogCaptureFixture) -> None:
    # Regression (code review, NRC-3): a channel value later removed from Channel made every
    # read of the upload fail with 500.
    upload = MediaUpload(
        id=uuid4(),
        tenant_id=uuid4(),
        identifiable_person=IdentifiablePerson.YES,
        usage_authorization_ref="RELEASE-1",
        permitted_channels=["facebook", "retired-channel"],
        authorization_expires_at=None,
    )

    with caplog.at_level(logging.WARNING):
        authorization = media_authorization(upload, datetime.now(UTC))

    assert authorization.permitted_channels == [Channel.FACEBOOK]
    assert authorization.status is AuthorizationStatus.RECORDED
    assert any("unknown permitted channels" in record.getMessage() for record in caplog.records)
