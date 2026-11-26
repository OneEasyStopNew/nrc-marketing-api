"""Media worker on policy source uploads (NRC-4): same checks and scan as images, no asset.

Real Postgres (RLS, the lease function), the local SeaweedFS for the bytes, the real policy upload
API for create, PUT and complete, and the labelled scanner simulator (or a small fake around it
for a fault). Every test here stores an object, so all need SeaweedFS.
"""

import hashlib
from collections.abc import Callable
from typing import Any
from uuid import UUID, uuid4

import httpx
import pytest
from sqlalchemy import Row, text
from sqlalchemy.ext.asyncio import AsyncConnection

from app.api.deps import get_optional_storage, get_policy_source_storage, get_storage
from app.core.config import Settings, get_settings
from app.services.scanning import (
    EICAR_TEST_SIGNATURE,
    MalwareScanner,
    ScanUnavailableError,
    ScanVerdict,
    SimulatedScanner,
)
from app.services.storage import ObjectStorage
from app.workers.media import run_once
from tests.conftest import (
    APP_ROLE,
    BUCKET,
    ENDPOINT,
    AuthHeaders,
    Sessions,
    World,
    as_role,
    needs_seaweedfs,
)

pytestmark = needs_seaweedfs

POLICY_UPLOADS = "/v1/marketing/policy-sources/uploads"
ADMIN = "policy-admin-a"
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
            "app_env": "local",
            "malware_scan_mode": "simulator",
            "aws_s3_bucket": BUCKET,
            "aws_endpoint_url_s3": ENDPOINT,
            "s3_force_path_style": True,
            "aws_profile": None,
            "policy_source_bucket": policy_bucket,
            "worker_backoff_base_seconds": 30.0,
            "worker_backoff_max_seconds": 60.0,
        }
    )


@pytest.fixture
def media_storage(settings: Settings) -> ObjectStorage:
    return ObjectStorage(settings)


@pytest.fixture
def policy_storage(settings: Settings, policy_bucket: str) -> ObjectStorage:
    return ObjectStorage(
        settings, bucket=policy_bucket, max_read_bytes=settings.policy_source_max_bytes
    )


@pytest.fixture
def simulator(settings: Settings) -> SimulatedScanner:
    return SimulatedScanner(settings)


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


class CountingScanner:
    """Wraps a real scanner; `fault` makes every scan raise ScanUnavailableError."""

    def __init__(self, inner: MalwareScanner, *, fault: bool = False) -> None:
        self._inner = inner
        self._fault = fault
        self.calls = 0

    @property
    def engine(self) -> str:
        return self._inner.engine

    @property
    def simulated(self) -> bool:
        return self._inner.simulated

    async def scan(
        self, content: bytes, *, bucket: str, key: str, version_id: str | None
    ) -> ScanVerdict:
        self.calls += 1
        if self._fault:
            raise ScanUnavailableError
        return await self._inner.scan(content, bucket=bucket, key=key, version_id=version_id)


# ---- Helpers (no test module imports another)


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


@pytest.fixture(autouse=True)
async def staff(connection: AsyncConnection, world: World) -> None:
    await _owner_sql(
        connection,
        world.a.tenant_id,
        "INSERT INTO memberships (tenant_id, user_sub, roles)"
        " VALUES (:t, :sub, ARRAY['policy_admin'])",
        {"t": world.a.tenant_id, "sub": ADMIN},
    )


async def _quarantined(
    client: httpx.AsyncClient,
    auth: AuthHeaders,
    world: World,
    content: bytes,
    mime_type: str = "application/pdf",
) -> str:
    headers = auth(ADMIN, world.a.tenant_id, **{"Idempotency-Key": uuid4().hex})
    created = await client.post(
        POLICY_UPLOADS,
        headers=headers,
        json={
            "name": "[SYNTHETIC] policy source",
            "mime_type": mime_type,
            "size_bytes": len(content),
            "checksum_sha256": hashlib.sha256(content).hexdigest(),
        },
    )
    assert created.status_code == 201, created.text
    target = created.json()["upload"]
    async with httpx.AsyncClient() as http:
        put = await http.request(
            target["method"], target["url"], content=content, headers=target["headers"]
        )
    assert put.status_code == 200, put.text
    asset_id: str = created.json()["upload_id"]
    done = await client.post(
        f"{POLICY_UPLOADS}/{asset_id}/complete", headers=auth(ADMIN, world.a.tenant_id)
    )
    assert done.status_code == 202, done.text
    return asset_id


async def _run(
    sessions: Sessions,
    media_storage: ObjectStorage,
    scanner: MalwareScanner,
    settings: Settings,
    policy_storage: ObjectStorage | None,
) -> int:
    return await run_once(sessions, media_storage, scanner, settings, policy_storage=policy_storage)


async def _row(connection: AsyncConnection, world: World, asset_id: str) -> Row[Any]:
    rows = await _owner_sql(
        connection,
        world.a.tenant_id,
        "SELECT state, mime_type, width_px, height_px, scan_engine, scan_simulated, reason_code,"
        " available_at, leased_until, lease_owner, coalesce(available_at > clock_timestamp(),"
        " false) AS deferred FROM media_uploads WHERE id = :id",
        {"id": UUID(asset_id)},
    )
    assert len(rows) == 1
    return rows[0]


async def _asset_count(connection: AsyncConnection, world: World) -> int:
    rows = await _owner_sql(
        connection,
        world.a.tenant_id,
        "SELECT count(*) AS n FROM assets WHERE tenant_id = :t",
        {"t": world.a.tenant_id},
    )
    return int(rows[0].n)


async def _status(
    client: httpx.AsyncClient, auth: AuthHeaders, world: World, asset_id: str
) -> dict[str, Any]:
    response = await client.get(
        f"{POLICY_UPLOADS}/{asset_id}", headers=auth(ADMIN, world.a.tenant_id)
    )
    assert response.status_code == 200, response.text
    body: dict[str, Any] = response.json()
    return body


# ---- Clean files become READY, and never an asset


@pytest.mark.parametrize(
    ("content", "mime"),
    [
        pytest.param(PDF, "application/pdf", id="pdf"),
        pytest.param(b"Plain policy text.\nSecond line.\n", "text/plain", id="text"),
        pytest.param("# Règles\n\n- “Never” say 100%\n".encode(), "text/markdown", id="markdown"),
    ],
)
async def test_a_clean_policy_file_becomes_ready_without_an_asset(
    client: httpx.AsyncClient,
    auth: AuthHeaders,
    world: World,
    connection: AsyncConnection,
    sessions: Sessions,
    media_storage: ObjectStorage,
    policy_storage: ObjectStorage,
    simulator: SimulatedScanner,
    settings: Settings,
    content: bytes,
    mime: str,
) -> None:
    asset_id = await _quarantined(client, auth, world, content, mime)

    assert await _run(sessions, media_storage, simulator, settings, policy_storage) == 1

    row = await _row(connection, world, asset_id)
    assert row.state == "READY"
    assert row.mime_type == mime
    assert (row.width_px, row.height_px) == (None, None)
    assert (row.scan_engine, row.scan_simulated) == ("SIMULATED", True)
    assert (row.reason_code, row.available_at, row.leased_until, row.lease_owner) == (
        None,
        None,
        None,
        None,
    )
    assert await _asset_count(connection, world) == 0
    body = await _status(client, auth, world, asset_id)
    assert body["status"] == "READY"
    assert body["scan"]["simulated"] is True
    asset = await client.get(
        f"/v1/marketing/assets/{asset_id}", headers=auth(ADMIN, world.a.tenant_id)
    )
    assert asset.status_code == 404


# ---- Rejections: the same reason codes as images


@pytest.mark.parametrize(
    ("content", "mime", "code"),
    [
        pytest.param(
            b"not a pdf at all, just text\n", "application/pdf", "TYPE_MISMATCH", id="text-as-pdf"
        ),
        pytest.param(
            b"\x00\x01binary\xff junk",
            "application/pdf",
            "MEDIA_UNSUPPORTED_TYPE",
            id="binary-as-pdf",
        ),
        pytest.param(
            b" %PDF-1.7\n", "application/pdf", "TYPE_MISMATCH", id="pdf-signature-not-first"
        ),
        pytest.param(PDF, "text/plain", "TYPE_MISMATCH", id="pdf-as-text"),
        pytest.param(PDF, "text/markdown", "TYPE_MISMATCH", id="pdf-as-markdown"),
        pytest.param(
            b"text with a \x00 NUL byte", "text/plain", "MEDIA_UNSUPPORTED_TYPE", id="nul"
        ),
        pytest.param(
            b"caf\xe9 is latin-1, not UTF-8", "text/plain", "MEDIA_UNSUPPORTED_TYPE", id="not-utf8"
        ),
        pytest.param(
            b"# ok\n\xff\xfe", "text/markdown", "MEDIA_UNSUPPORTED_TYPE", id="markdown-not-utf8"
        ),
    ],
)
async def test_a_forged_or_mismatched_file_is_rejected_and_never_scanned_as_clean(
    client: httpx.AsyncClient,
    auth: AuthHeaders,
    world: World,
    connection: AsyncConnection,
    sessions: Sessions,
    media_storage: ObjectStorage,
    policy_storage: ObjectStorage,
    simulator: SimulatedScanner,
    settings: Settings,
    content: bytes,
    mime: str,
    code: str,
) -> None:
    scanner = CountingScanner(simulator)
    asset_id = await _quarantined(client, auth, world, content, mime)

    assert await _run(sessions, media_storage, scanner, settings, policy_storage) == 1

    row = await _row(connection, world, asset_id)
    assert (row.state, row.reason_code) == ("REJECTED", code)
    assert (row.leased_until, row.lease_owner, row.available_at) == (None, None, None)
    assert scanner.calls == 0
    assert await _asset_count(connection, world) == 0
    body = await _status(client, auth, world, asset_id)
    assert body["status"] == "REJECTED"
    assert body["reason"]["code"] == code
    assert body["reason"]["retryable"] is False


async def test_malware_in_a_policy_file_is_rejected(
    client: httpx.AsyncClient,
    auth: AuthHeaders,
    world: World,
    connection: AsyncConnection,
    sessions: Sessions,
    media_storage: ObjectStorage,
    policy_storage: ObjectStorage,
    simulator: SimulatedScanner,
    settings: Settings,
) -> None:
    content = b"# Policy\n" + EICAR_TEST_SIGNATURE
    asset_id = await _quarantined(client, auth, world, content, "text/plain")

    assert await _run(sessions, media_storage, simulator, settings, policy_storage) == 1

    row = await _row(connection, world, asset_id)
    assert (row.state, row.reason_code) == ("REJECTED", "MALWARE_DETECTED")
    assert (row.scan_engine, row.scan_simulated) == ("SIMULATED", True)
    assert await _asset_count(connection, world) == 0


# ---- Waiting instead of passing: scanner and storage faults


async def test_a_scanner_fault_keeps_the_file_quarantined_and_retryable(
    client: httpx.AsyncClient,
    auth: AuthHeaders,
    world: World,
    connection: AsyncConnection,
    sessions: Sessions,
    media_storage: ObjectStorage,
    policy_storage: ObjectStorage,
    simulator: SimulatedScanner,
    settings: Settings,
) -> None:
    scanner = CountingScanner(simulator, fault=True)
    asset_id = await _quarantined(client, auth, world, PDF)

    assert await _run(sessions, media_storage, scanner, settings, policy_storage) == 1

    row = await _row(connection, world, asset_id)
    assert (row.state, row.reason_code, row.deferred) == ("QUARANTINED", "SCAN_UNAVAILABLE", True)
    assert scanner.calls == 1
    assert await _asset_count(connection, world) == 0
    body = await _status(client, auth, world, asset_id)
    assert body["status"] == "QUARANTINED"
    assert body["reason"]["code"] == "SCAN_UNAVAILABLE"
    assert body["reason"]["retryable"] is True


async def test_without_a_policy_bucket_the_worker_defers_and_never_reads_the_media_bucket(
    client: httpx.AsyncClient,
    auth: AuthHeaders,
    world: World,
    connection: AsyncConnection,
    sessions: Sessions,
    media_storage: ObjectStorage,
    policy_storage: ObjectStorage,
    simulator: SimulatedScanner,
    settings: Settings,
) -> None:
    scanner = CountingScanner(simulator)
    asset_id = await _quarantined(client, auth, world, PDF)

    assert await _run(sessions, media_storage, scanner, settings, None) == 1

    row = await _row(connection, world, asset_id)
    assert (row.state, row.reason_code, row.deferred) == (
        "QUARANTINED",
        "STORAGE_UNAVAILABLE",
        True,
    )
    assert scanner.calls == 0
    # Once the bucket is configured the same upload is inspected and passes.
    await _owner_sql(
        connection,
        world.a.tenant_id,
        "UPDATE media_uploads SET available_at = clock_timestamp() - interval '1 second'"
        " WHERE id = :id",
        {"id": UUID(asset_id)},
    )
    assert await _run(sessions, media_storage, scanner, settings, policy_storage) == 1
    assert (await _row(connection, world, asset_id)).state == "READY"


async def test_an_upload_recorded_in_another_bucket_is_never_read_from_the_configured_one(
    client: httpx.AsyncClient,
    auth: AuthHeaders,
    world: World,
    connection: AsyncConnection,
    sessions: Sessions,
    media_storage: ObjectStorage,
    policy_storage: ObjectStorage,
    simulator: SimulatedScanner,
    settings: Settings,
) -> None:
    scanner = CountingScanner(simulator)
    asset_id = await _quarantined(client, auth, world, PDF)
    await _owner_sql(
        connection,
        world.a.tenant_id,
        "UPDATE media_uploads SET bucket = 'nrc-test-other' WHERE id = :id",
        {"id": UUID(asset_id)},
    )

    assert await _run(sessions, media_storage, scanner, settings, policy_storage) == 1

    row = await _row(connection, world, asset_id)
    assert (row.state, row.reason_code) == ("QUARANTINED", "STORAGE_UNAVAILABLE")
    assert scanner.calls == 0


# ---- Memory bound


async def test_a_capped_storage_returns_one_byte_too_many_for_a_longer_object(
    settings: Settings, policy_bucket: str
) -> None:
    content = PDF + b"x" * 64
    capped = ObjectStorage(settings, bucket=policy_bucket, max_read_bytes=len(PDF))
    uncapped = ObjectStorage(settings, bucket=policy_bucket)
    # Stored through a presigned URL, the only way files reach the bucket.
    presigned = await uncapped.presigned_put(
        f"uploads/2026/11/{uuid4()}.pdf",
        content_type="application/pdf",
        size_bytes=len(content),
        checksum_sha256=hashlib.sha256(content).hexdigest(),
        expires_in=60,
    )
    async with httpx.AsyncClient() as http:
        put = await http.put(presigned.url, content=content, headers=dict(presigned.headers))
    assert put.status_code == 200, put.text

    assert len(await capped.read(presigned.key)) == len(PDF) + 1
    assert await uncapped.read(presigned.key) == content
