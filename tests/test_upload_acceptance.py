"""NRC-3 acceptance criteria for client uploads, end to end through the API and the media worker.

Every rejected file takes the real route: create, PUT the bytes through the presigned URL
(which storage only accepts when they match the declaration), complete, then one worker pass.
A file that fails any check must end REJECTED with a safe reason, and must never exist as an
asset: GET /assets/{id} is 404 and /media does not list it, so nothing reaches validation or
generation. Storage is the local SeaweedFS; the scanner is the labelled simulator.
"""

import asyncio
import hashlib
import io
import logging
import re
import struct
import zlib
from collections.abc import AsyncIterator, Callable
from dataclasses import dataclass, field
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
from app.services.scanning import SimulatedScanner
from app.services.storage import ObjectStorage
from app.workers.media import run_once
from tests.conftest import (
    APP_ROLE,
    BUCKET,
    ENDPOINT,
    AuthHeaders,
    SeededTenant,
    Sessions,
    World,
    as_role,
    needs_seaweedfs,
)

UPLOADS = "/v1/marketing/assets/uploads"
ASSETS = "/v1/marketing/assets"
MEDIA = "/v1/marketing/media"
# The standard EICAR test signature, assembled from pieces on purpose: Windows Defender
# quarantines any file that contains it whole.
EICAR = "".join(
    ("X5O!P%@AP[4\\PZ", "X54(P^)7CC)7}$EICAR-", "STANDARD-ANTIVIRUS-TEST-", "FILE!$H+H*")
).encode("ascii")
SIGNED_URL = re.compile(r"https?://[^\s\"']+")


@pytest.fixture(autouse=True)
def local_credentials(monkeypatch: pytest.MonkeyPatch) -> None:
    # Local-only values that open the SeaweedFS container, not AWS.
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "localdev")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "localdev-password")
    monkeypatch.delenv("AWS_PROFILE", raising=False)


@pytest.fixture
def settings() -> Settings:
    # The worker's settings: local environment with the labelled simulator and the test bucket.
    return get_settings().model_copy(
        update={
            "app_env": "local",
            "malware_scan_mode": "simulator",
            "aws_s3_bucket": BUCKET,
            "aws_endpoint_url_s3": ENDPOINT,
            "s3_force_path_style": True,
            "aws_profile": None,
            "media_inspection_max_attempts": 3,
            "media_inspection_retry_window_seconds": 600,
        }
    )


@pytest.fixture
def storage(settings: Settings) -> ObjectStorage:
    return ObjectStorage(settings)


@pytest.fixture
def scanner(settings: Settings) -> SimulatedScanner:
    return SimulatedScanner(settings)


@pytest.fixture(autouse=True)
def use_storage(
    app_overrides: dict[Callable[..., object], Callable[..., object]], storage: ObjectStorage
) -> None:
    # Both dependencies, so every route sees the same real storage on the test bucket.
    app_overrides[get_storage] = lambda: storage
    app_overrides[get_optional_storage] = lambda: storage


@pytest.fixture
def url_lifetime(
    request: pytest.FixtureRequest,
    app_overrides: dict[Callable[..., object], Callable[..., object]],
) -> None:
    # Parametrize it (indirect) with a lifetime in seconds to shorten upload URLs for one test.
    # The leak guard depends on it, so it is applied before the app is created.
    seconds: int | None = getattr(request, "param", None)
    if seconds is not None:
        short = get_settings().model_copy(update={"upload_url_ttl_seconds": seconds})
        app_overrides[get_settings] = lambda: short


@dataclass
class LeakGuard:
    # What a test's API traffic showed: presigned URLs (which carry the object key) and responses.
    responses: list[httpx.Response] = field(default_factory=list)

    def secrets(self) -> list[str]:
        # Every presigned URL the API handed out in this test, and the object key in each.
        urls = {
            url
            for response in self.responses
            for url in SIGNED_URL.findall(response.text)
            if "x-amz-signature" in url.lower()
        }
        keys = {urlsplit(url).path.removeprefix(f"/{BUCKET}/") for url in urls}
        return sorted(urls | keys)


@pytest.fixture(autouse=True)
async def leak_guard(
    url_lifetime: None, client: httpx.AsyncClient, caplog: pytest.LogCaptureFixture
) -> AsyncIterator[LeakGuard]:
    # Runs for every test of this module, so the whole matrix is covered: no application log
    # record and no error response may hold a presigned URL, a signature, a credential or an
    # object key.
    guard = LeakGuard()

    async def keep(response: httpx.Response) -> None:
        await response.aread()
        guard.responses.append(response)

    client.event_hooks["response"].append(keep)
    caplog.set_level(logging.INFO)
    yield guard

    secrets = guard.secrets()
    # The HTTP client libraries log the request URL of this test's own PUT; that is the test
    # talking to storage, not the application, so only the application's loggers are checked.
    app_records = [
        record
        for record in caplog.get_records("call")
        if not record.name.startswith(("httpx", "httpcore", "asyncio"))
    ]
    if secrets:
        assert app_records, "no application log was captured, so the log check proved nothing"
    forbidden = ["x-amz-signature", "x-amz-credential", *(secret.lower() for secret in secrets)]
    for record in app_records:
        rendered = f"{record.getMessage()} {record.exc_text or ''}".lower()
        assert not [item for item in forbidden if item in rendered], record.getMessage()
    for response in guard.responses:
        if response.status_code >= 400:
            body = response.text.lower()
            assert not [item for item in forbidden if item in body], response.text


# ---- Helpers (no test module imports another)


def _image(fmt: str, shade: int = 255, size: tuple[int, int] = (24, 16)) -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", size, (shade, shade, shade)).save(buffer, fmt)
    return buffer.getvalue()


def _png_header_only(width: int, height: int) -> bytes:
    # A few bytes that declare width x height pixels; nothing large is ever allocated.
    def chunk(kind: bytes, data: bytes) -> bytes:
        return (
            struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data))
        )

    return (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
        + chunk(b"IDAT", zlib.compress(b"\x00"))
        + chunk(b"IEND", b"")
    )


def _truncated_png() -> bytes:
    # Noise-like pixels, so the file is long enough to be cut in the middle of its data.
    pixels = bytes((i * 31 + i // 7) % 256 for i in range(64 * 64 * 3))
    buffer = io.BytesIO()
    Image.frombytes("RGB", (64, 64), pixels).save(buffer, "PNG")
    content = buffer.getvalue()
    return content[: len(content) // 2]


def _body(tenant: SeededTenant, content: bytes, **overrides: Any) -> dict[str, Any]:
    body: dict[str, Any] = {
        "study_id": str(tenant.study_id),
        "name": "[SYNTHETIC] photo.png",
        "mime_type": "image/png",
        "size_bytes": len(content),
        "checksum_sha256": hashlib.sha256(content).hexdigest(),
        "authorization": {"identifiable_person": "no", "permitted_channels": ["facebook"]},
    }
    body.update(overrides)
    return body


async def _create(
    client: httpx.AsyncClient,
    auth: AuthHeaders,
    tenant: SeededTenant,
    content: bytes,
    *,
    sub: str = "creator-a",
    **overrides: Any,
) -> httpx.Response:
    headers = auth(sub, tenant.tenant_id, **{"Idempotency-Key": uuid4().hex})
    return await client.post(UPLOADS, headers=headers, json=_body(tenant, content, **overrides))


async def _created(
    client: httpx.AsyncClient,
    auth: AuthHeaders,
    tenant: SeededTenant,
    content: bytes,
    *,
    sub: str = "creator-a",
    **overrides: Any,
) -> dict[str, Any]:
    response = await _create(client, auth, tenant, content, sub=sub, **overrides)
    assert response.status_code == 201, response.text
    body: dict[str, Any] = response.json()
    return body


async def _put(upload: dict[str, Any], content: bytes, *, url: str | None = None) -> httpx.Response:
    # Exactly the returned headers; httpx adds the real Content-Length and Host itself.
    async with httpx.AsyncClient() as http:
        return await http.request(
            upload["method"], url or upload["url"], content=content, headers=upload["headers"]
        )


def _complete(
    client: httpx.AsyncClient,
    auth: AuthHeaders,
    tenant: SeededTenant,
    asset_id: str,
    sub: str = "creator-a",
) -> Any:
    return client.post(f"{ASSETS}/{asset_id}/complete", headers=auth(sub, tenant.tenant_id))


async def _owner_sql(
    connection: AsyncConnection,
    tenant: SeededTenant,
    statement: str,
    params: dict[str, Any] | None = None,
) -> list[Row[Any]]:
    # As the table owner with the tenant bound (FORCE RLS applies to the owner too), then back
    # to the API's role.
    await connection.execute(text("RESET ROLE"))
    await connection.execute(
        text("SELECT set_config('app.tenant_id', :t, true)"), {"t": str(tenant.tenant_id)}
    )
    result = await connection.execute(text(statement), params or {})
    rows = list(result.all()) if result.returns_rows else []
    await as_role(connection, APP_ROLE)
    return rows


async def _asset_ids(connection: AsyncConnection, tenant: SeededTenant) -> set[str]:
    # Ids of the assets the tenant can see, read as the API's role so row-level security applies.
    await connection.execute(
        text("SELECT set_config('app.tenant_id', :t, true)"), {"t": str(tenant.tenant_id)}
    )
    result = await connection.execute(text("SELECT id FROM assets"))
    return {str(row.id) for row in result.all()}


async def _upload_count(connection: AsyncConnection, tenant: SeededTenant) -> int:
    rows = await _owner_sql(
        connection,
        tenant,
        "SELECT count(*) AS n FROM media_uploads WHERE tenant_id = :t",
        {"t": tenant.tenant_id},
    )
    return int(rows[0].n)


async def _worker(
    sessions: Sessions,
    storage: ObjectStorage,
    scanner: SimulatedScanner,
    settings: Settings,
    owner: str = "w1",
) -> int:
    return await run_once(sessions, storage, scanner, settings, owner=owner)


async def _through_the_worker(
    client: httpx.AsyncClient,
    auth: AuthHeaders,
    world: World,
    sessions: Sessions,
    storage: ObjectStorage,
    scanner: SimulatedScanner,
    settings: Settings,
    content: bytes,
    **overrides: Any,
) -> dict[str, Any]:
    # create -> PUT the exact bytes -> complete -> one worker pass; returns the upload's status.
    upload = await _created(client, auth, world.a, content, **overrides)
    stored = await _put(upload["upload"], content)
    assert stored.status_code == 200, stored.text
    done = await _complete(client, auth, world.a, upload["asset_id"])
    assert done.status_code == 202, done.text
    assert done.json()["status"] == "QUARANTINED"
    assert await _worker(sessions, storage, scanner, settings) == 1
    response = await client.get(
        f"{UPLOADS}/{upload['asset_id']}", headers=auth("creator-a", world.a.tenant_id)
    )
    assert response.status_code == 200, response.text
    body: dict[str, Any] = response.json()
    return body


async def _assert_rejected_and_never_an_asset(
    client: httpx.AsyncClient,
    auth: AuthHeaders,
    world: World,
    connection: AsyncConnection,
    upload: dict[str, Any],
    code: str,
) -> None:
    headers = auth("creator-a", world.a.tenant_id)
    asset_id = upload["asset_id"]
    assert upload["status"] == "REJECTED"
    assert upload["reason"]["code"] == code
    assert upload["reason"]["retryable"] is False
    assert upload["reason"]["message"]
    assert upload["upload"] is None
    # Rejected before validation and generation: there is no asset, in any read.
    assert await _asset_ids(connection, world.a) == set()
    read = await client.get(f"{ASSETS}/{asset_id}", headers=headers)
    assert read.status_code == 404
    assert read.json()["code"] == "NOT_FOUND"
    assert (await client.get(MEDIA, headers=headers)).json() == {"items": [], "next_cursor": None}


# ---- Files that must be rejected


@dataclass(frozen=True, slots=True)
class BadFile:
    content: bytes
    declared_mime: str
    name: str
    code: str
    worker_settings: dict[str, Any] = field(default_factory=dict)


REJECTED_FILES = {
    # A JPEG renamed to .png and declared as PNG.
    "forged-extension": BadFile(_image("JPEG"), "image/png", "photo.png", "TYPE_MISMATCH"),
    # Text declared as a PNG: not decodable at all, so the image is corrupt (not "unsupported").
    "not-an-image": BadFile(
        b"[SYNTHETIC] plain text, not an image\n" * 8, "image/png", "photo.png", "MEDIA_CORRUPT"
    ),
    "truncated-png": BadFile(_truncated_png(), "image/png", "photo.png", "MEDIA_CORRUPT"),
    # About 2.5 billion pixels declared in a file of a few dozen bytes.
    "decompression-bomb": BadFile(
        _png_header_only(50_000, 50_000), "image/png", "photo.png", "MEDIA_TOO_LARGE"
    ),
    # A real image over the configured pixel cap.
    "over-the-pixel-cap": BadFile(
        _image("PNG", size=(200, 100)),
        "image/png",
        "photo.png",
        "MEDIA_TOO_LARGE",
        {"max_image_pixels": 10_000},
    ),
}


@needs_seaweedfs
@pytest.mark.parametrize("case", REJECTED_FILES.values(), ids=REJECTED_FILES.keys())
async def test_a_bad_file_is_rejected_before_validation_or_generation(
    client: httpx.AsyncClient,
    auth: AuthHeaders,
    world: World,
    connection: AsyncConnection,
    sessions: Sessions,
    storage: ObjectStorage,
    scanner: SimulatedScanner,
    settings: Settings,
    case: BadFile,
) -> None:
    upload = await _through_the_worker(
        client,
        auth,
        world,
        sessions,
        storage,
        scanner,
        settings.model_copy(update=case.worker_settings),
        case.content,
        mime_type=case.declared_mime,
        name=case.name,
    )

    await _assert_rejected_and_never_an_asset(client, auth, world, connection, upload, case.code)
    assert upload["scan"] is None


@needs_seaweedfs
async def test_a_file_with_the_malware_signature_is_rejected_as_simulated_evidence(
    client: httpx.AsyncClient,
    auth: AuthHeaders,
    world: World,
    connection: AsyncConnection,
    sessions: Sessions,
    storage: ObjectStorage,
    scanner: SimulatedScanner,
    settings: Settings,
) -> None:
    # A valid PNG with the signature appended: it decodes, so only the scan can stop it.
    content = _image("PNG") + EICAR

    upload = await _through_the_worker(
        client, auth, world, sessions, storage, scanner, settings, content
    )

    await _assert_rejected_and_never_an_asset(
        client, auth, world, connection, upload, "MALWARE_DETECTED"
    )
    assert upload["scan"]["simulated"] is True
    assert upload["scan"]["engine"] == "SIMULATED"
    assert upload["scan"]["scanned_at"]


async def test_an_oversized_declaration_is_refused_at_create_and_leaves_no_row(
    client: httpx.AsyncClient, auth: AuthHeaders, world: World, connection: AsyncConnection
) -> None:
    too_big = get_settings().upload_image_max_bytes + 1

    response = await _create(client, auth, world.a, b"x", size_bytes=too_big)

    assert response.status_code == 422
    assert response.json()["code"] == "UPLOAD_NOT_ACCEPTED"
    assert response.json()["details"]["field"] == "size_bytes"
    assert await _upload_count(connection, world.a) == 0
    listed = await client.get(UPLOADS, headers=auth("creator-a", world.a.tenant_id))
    assert listed.json() == {"items": [], "next_cursor": None}


# ---- Checksums


@needs_seaweedfs
async def test_storage_refuses_bytes_that_differ_from_the_declared_checksum(
    client: httpx.AsyncClient, auth: AuthHeaders, world: World, connection: AsyncConnection
) -> None:
    declared = _image("PNG", 10)
    # Same length, other bytes: only the checksum tells them apart.
    other = bytes(reversed(declared))
    assert len(other) == len(declared)
    upload = await _created(client, auth, world.a, declared)

    refused = await _put(upload["upload"], other)

    assert refused.status_code == 400
    early = await _complete(client, auth, world.a, upload["asset_id"])
    assert early.status_code == 409
    assert early.json()["code"] == "UPLOAD_NOT_RECEIVED"
    assert early.json()["retryable"] is True
    state = await _owner_sql(
        connection,
        world.a,
        "SELECT state FROM media_uploads WHERE id = :id",
        {"id": UUID(upload["asset_id"])},
    )
    assert state[0].state == "PENDING_UPLOAD"
    assert await _asset_ids(connection, world.a) == set()


@needs_seaweedfs
@pytest.mark.parametrize(
    ("swap", "code"),
    [
        pytest.param(lambda content: bytes(reversed(content)), "CHECKSUM_MISMATCH", id="checksum"),
        pytest.param(lambda content: content + b"\x00", "SIZE_MISMATCH", id="size"),
    ],
)
async def test_the_worker_rejects_an_object_that_differs_from_the_declaration(
    client: httpx.AsyncClient,
    auth: AuthHeaders,
    world: World,
    connection: AsyncConnection,
    sessions: Sessions,
    storage: ObjectStorage,
    scanner: SimulatedScanner,
    settings: Settings,
    monkeypatch: pytest.MonkeyPatch,
    swap: Callable[[bytes], bytes],
    code: str,
) -> None:
    # Storage would have refused these bytes, so the object read back is swapped to simulate
    # an object that changed after the PUT.
    async def other_bytes(self: ObjectStorage, key: str, *, version_id: str | None = None) -> bytes:
        return swap(_image("PNG", 20))

    upload = await _created(client, auth, world.a, _image("PNG", 20))
    assert (await _put(upload["upload"], _image("PNG", 20))).status_code == 200
    assert (await _complete(client, auth, world.a, upload["asset_id"])).status_code == 202
    monkeypatch.setattr(ObjectStorage, "read", other_bytes)

    assert await _worker(sessions, storage, scanner, settings) == 1

    status = await client.get(
        f"{UPLOADS}/{upload['asset_id']}", headers=auth("creator-a", world.a.tenant_id)
    )
    await _assert_rejected_and_never_an_asset(client, auth, world, connection, status.json(), code)


# ---- The positive control and authorization


@needs_seaweedfs
async def test_a_valid_image_becomes_ready_and_is_listed_in_the_media_library(
    client: httpx.AsyncClient,
    auth: AuthHeaders,
    world: World,
    sessions: Sessions,
    storage: ObjectStorage,
    scanner: SimulatedScanner,
    settings: Settings,
) -> None:
    content = _image("PNG", 90)

    upload = await _through_the_worker(
        client, auth, world, sessions, storage, scanner, settings, content
    )

    assert upload["status"] == "READY"
    assert upload["reason"] is None
    assert upload["scan"]["simulated"] is True
    headers = auth("creator-a", world.a.tenant_id)
    asset = await client.get(f"{ASSETS}/{upload['asset_id']}", headers=headers)
    assert asset.status_code == 200
    assert asset.json()["origin"] == "uploaded"
    media = await client.get(MEDIA, headers=headers)
    assert [item["asset_id"] for item in media.json()["items"]] == [upload["asset_id"]]
    async with httpx.AsyncClient() as http:
        downloaded = await http.get(asset.json()["download_url"])
    assert downloaded.status_code == 200
    assert downloaded.content == content


@needs_seaweedfs
async def test_an_expired_authorization_is_visible_to_the_gates_on_a_ready_asset(
    client: httpx.AsyncClient,
    auth: AuthHeaders,
    world: World,
    sessions: Sessions,
    storage: ObjectStorage,
    scanner: SimulatedScanner,
    settings: Settings,
) -> None:
    # Inspection does not look at the authorization: the file is READY, and the gates read the
    # authorization status (NRC-12, NRC-14) from the asset.
    upload = await _through_the_worker(
        client,
        auth,
        world,
        sessions,
        storage,
        scanner,
        settings,
        _image("PNG", 100),
        authorization={
            "identifiable_person": "yes",
            "usage_authorization_ref": "RELEASE-SYNTH-ACCEPT",
            "permitted_channels": ["facebook"],
            "expires_at": "2020-01-01T00:00:00Z",
        },
    )

    assert upload["status"] == "READY"
    headers = auth("creator-a", world.a.tenant_id)
    asset = await client.get(f"{ASSETS}/{upload['asset_id']}", headers=headers)
    assert asset.status_code == 200
    assert asset.json()["authorization"]["status"] == "EXPIRED"
    assert asset.json()["authorization"]["usage_authorization_ref"] == "RELEASE-SYNTH-ACCEPT"
    item = (await client.get(MEDIA, headers=headers)).json()["items"][0]
    assert item["authorization"]["status"] == "EXPIRED"


# ---- Another tenant


@needs_seaweedfs
async def test_another_tenant_cannot_use_complete_reissue_read_list_or_download(
    client: httpx.AsyncClient,
    auth: AuthHeaders,
    world: World,
    connection: AsyncConnection,
    sessions: Sessions,
    storage: ObjectStorage,
    scanner: SimulatedScanner,
    settings: Settings,
) -> None:
    content = _image("PNG", 110)
    upload = await _created(client, auth, world.a, content)
    asset_id = upload["asset_id"]
    foreign = auth("creator-b", world.b.tenant_id)

    async def foreign_attempts() -> None:
        for response in (
            await client.post(f"{ASSETS}/{asset_id}/complete", headers=foreign),
            await client.post(f"{ASSETS}/{asset_id}/upload-url", headers=foreign),
            await client.get(f"{UPLOADS}/{asset_id}", headers=foreign),
            await client.get(f"{ASSETS}/{asset_id}", headers=foreign),
        ):
            assert response.status_code == 404
            assert response.json()["code"] == "NOT_FOUND"
            assert asset_id not in response.text
        assert (await client.get(UPLOADS, headers=foreign)).json() == {
            "items": [],
            "next_cursor": None,
        }
        assert (await client.get(MEDIA, headers=foreign)).json() == {
            "items": [],
            "next_cursor": None,
        }

    await foreign_attempts()
    assert (await _put(upload["upload"], content)).status_code == 200
    await foreign_attempts()
    assert (await _complete(client, auth, world.a, asset_id)).status_code == 202
    assert await _worker(sessions, storage, scanner, settings) == 1
    # READY and downloadable by its owner, and still invisible to the other tenant.
    await foreign_attempts()
    owned = await client.get(f"{ASSETS}/{asset_id}", headers=auth("creator-a", world.a.tenant_id))
    assert owned.status_code == 200
    async with httpx.AsyncClient() as http:
        assert (await http.get(owned.json()["download_url"])).content == content
    assert await _asset_ids(connection, world.b) == set()


@needs_seaweedfs
async def test_another_tenant_cannot_write_over_an_object(
    client: httpx.AsyncClient,
    auth: AuthHeaders,
    world: World,
    sessions: Sessions,
    storage: ObjectStorage,
    scanner: SimulatedScanner,
    settings: Settings,
) -> None:
    content = _image("PNG", 120)
    mine = await _created(client, auth, world.a, content)
    theirs = await _created(client, auth, world.b, _image("PNG", 130), sub="creator-b")
    assert (await _put(mine["upload"], content)).status_code == 200
    mine_path = urlsplit(mine["upload"]["url"]).path
    theirs_url = urlsplit(theirs["upload"]["url"])
    # Each upload is signed for its own object.
    assert theirs_url.path != mine_path

    # A's URL with other bytes: the signature covers the checksum and the length.
    other = bytes(reversed(content))
    # Same length: the checksum differs (BadDigest).
    assert (await _put(mine["upload"], other)).status_code == 400
    # Another length: the signed Content-Length differs (SignatureDoesNotMatch).
    assert (await _put(mine["upload"], content + b"\x00")).status_code == 403
    # B's own URL pointed at A's object: the signature was made for another key.
    redirected = theirs_url._replace(path=mine_path).geturl()
    forged = await _put(theirs["upload"], _image("PNG", 130), url=redirected)
    assert forged.status_code == 403

    # A's file is untouched: it passes the checksum and becomes READY with the original bytes.
    assert (await _complete(client, auth, world.a, mine["asset_id"])).status_code == 202
    assert await _worker(sessions, storage, scanner, settings) == 1
    asset = await client.get(
        f"{ASSETS}/{mine['asset_id']}", headers=auth("creator-a", world.a.tenant_id)
    )
    assert asset.status_code == 200
    async with httpx.AsyncClient() as http:
        assert (await http.get(asset.json()["download_url"])).content == content


# ---- Duplicates and retries


@needs_seaweedfs
async def test_duplicate_completes_and_worker_runs_give_one_inspection_and_one_asset(
    client: httpx.AsyncClient,
    auth: AuthHeaders,
    world: World,
    connection: AsyncConnection,
    sessions: Sessions,
    storage: ObjectStorage,
    scanner: SimulatedScanner,
    settings: Settings,
) -> None:
    content = _image("PNG", 140)
    upload = await _created(client, auth, world.a, content)
    assert (await _put(upload["upload"], content)).status_code == 200

    first, second, third = [
        await _complete(client, auth, world.a, upload["asset_id"]) for _ in range(3)
    ]
    assert await _worker(sessions, storage, scanner, settings) == 1
    assert await _worker(sessions, storage, scanner, settings) == 0
    assert await _worker(sessions, storage, scanner, settings, owner="w2") == 0
    late = await _complete(client, auth, world.a, upload["asset_id"])

    assert {first.status_code, second.status_code, third.status_code} == {202}
    assert second.json()["completed_at"] == first.json()["completed_at"]
    assert third.json()["status"] == "QUARANTINED"
    assert late.status_code == 202
    assert late.json()["status"] == "READY"
    assert late.json()["scan"]["simulated"] is True
    counts = await _owner_sql(
        connection,
        world.a,
        "SELECT attempts, failed_attempts, state FROM media_uploads WHERE id = :id",
        {"id": UUID(upload["asset_id"])},
    )
    # Leased and inspected once.
    assert (counts[0].attempts, counts[0].failed_attempts, counts[0].state) == (1, 0, "READY")
    assert await _asset_ids(connection, world.a) == {upload["asset_id"]}


@needs_seaweedfs
# Upload URLs that live one second, so one really expires inside the test.
@pytest.mark.parametrize("url_lifetime", [1], indirect=True)
async def test_an_interrupted_upload_is_retried_safely_with_a_new_url(
    client: httpx.AsyncClient,
    auth: AuthHeaders,
    world: World,
    connection: AsyncConnection,
    sessions: Sessions,
    storage: ObjectStorage,
    scanner: SimulatedScanner,
    settings: Settings,
) -> None:
    content = _image("PNG", 150)
    upload = await _created(client, auth, world.a, content)
    asset_id = upload["asset_id"]
    key_before = urlsplit(upload["upload"]["url"]).path
    # The connection dropped before the PUT and the URL runs out.
    await asyncio.sleep(2.5)
    expired = await _put(upload["upload"], content)
    assert expired.status_code == 403
    assert (await _complete(client, auth, world.a, asset_id)).status_code == 409

    reissued = await client.post(
        f"{ASSETS}/{asset_id}/upload-url", headers=auth("creator-a", world.a.tenant_id)
    )

    assert reissued.status_code == 200
    body = reissued.json()
    assert body["status"] == "PENDING_UPLOAD"
    assert body["created_by"] == "creator-a"
    assert body["asset_id"] == asset_id
    assert urlsplit(body["upload"]["url"]).path == key_before
    assert body["upload"]["url"] != upload["upload"]["url"]
    assert (await _put(body["upload"], content)).status_code == 200
    assert (await _complete(client, auth, world.a, asset_id)).status_code == 202
    assert await _worker(sessions, storage, scanner, settings) == 1
    row = await _owner_sql(
        connection,
        world.a,
        "SELECT state, created_by, completed_by FROM media_uploads WHERE id = :id",
        {"id": UUID(asset_id)},
    )
    assert (row[0].state, row[0].created_by, row[0].completed_by) == (
        "READY",
        "creator-a",
        "creator-a",
    )
    assert await _asset_ids(connection, world.a) == {asset_id}


# ---- Logs and error bodies


@needs_seaweedfs
async def test_the_leak_guard_sees_the_urls_and_logs_of_a_full_flow(
    leak_guard: LeakGuard,
    client: httpx.AsyncClient,
    auth: AuthHeaders,
    world: World,
    sessions: Sessions,
    storage: ObjectStorage,
    scanner: SimulatedScanner,
    settings: Settings,
    caplog: pytest.LogCaptureFixture,
) -> None:
    # The guard checks every test of this module when it finishes; this one shows it is not
    # vacuous: it knows the URLs and keys of this flow, and the flow did log.
    upload = await _through_the_worker(
        client, auth, world, sessions, storage, scanner, settings, _image("PNG", 160)
    )
    assert upload["status"] == "READY"
    # An error response too: the key must not come back in it.
    missing = await client.get(f"{ASSETS}/{uuid4()}", headers=auth("creator-a", world.a.tenant_id))
    assert missing.status_code == 404

    secrets = leak_guard.secrets()
    assert any(secret.startswith("http") and "X-Amz-Signature" in secret for secret in secrets)
    assert any(secret.startswith("uploads/") for secret in secrets)
    messages = [record.getMessage() for record in caplog.records]
    assert any("upload created" in message for message in messages)
    assert any("upload ready" in message for message in messages)
