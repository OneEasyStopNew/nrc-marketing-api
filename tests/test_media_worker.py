"""Media worker (NRC-3): inspects completed uploads and turns clean ones into assets.

Real Postgres (RLS, the lease function), the local SeaweedFS for the bytes, and the real upload
API for create, PUT and complete. The scanner is the labelled simulator, or a small fake wrapped
around it where a test needs a fault or a callback. Tests that PUT an object need SeaweedFS.
"""

import hashlib
import io
import logging
from collections.abc import Awaitable, Callable
from types import SimpleNamespace
from typing import Any, cast
from uuid import UUID, uuid4

import httpx
import pytest
from PIL import Image
from sqlalchemy import Row, text
from sqlalchemy.ext.asyncio import AsyncConnection

from app.api.deps import get_optional_storage, get_storage
from app.core.config import Settings, get_settings
from app.services.scanning import (
    MalwareScanner,
    ScanOutcome,
    ScanUnavailableError,
    ScanVerdict,
    SimulatedScanner,
)
from app.services.storage import ObjectStorage, StorageError
from app.workers import media as media_worker
from app.workers.connector import WorkerConfigError
from app.workers.media import build_dependencies, run_once
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
WORKER_LOGGER = "app.workers.media"


@pytest.fixture(autouse=True)
def local_credentials(monkeypatch: pytest.MonkeyPatch) -> None:
    # Local-only values that open the SeaweedFS container, not AWS.
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "localdev")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "localdev-password")
    monkeypatch.delenv("AWS_PROFILE", raising=False)


@pytest.fixture
def settings() -> Settings:
    # The worker's settings: local environment with the labelled simulator, the test bucket, and
    # a long backoff so a deferred upload is clearly not due again on its own.
    return get_settings().model_copy(
        update={
            "app_env": "local",
            "malware_scan_mode": "simulator",
            "aws_s3_bucket": BUCKET,
            "aws_endpoint_url_s3": ENDPOINT,
            "s3_force_path_style": True,
            "aws_profile": None,
            "worker_backoff_base_seconds": 30.0,
            "worker_backoff_max_seconds": 60.0,
            "media_inspection_max_attempts": 3,
            "media_inspection_retry_window_seconds": 600,
        }
    )


@pytest.fixture
def storage(settings: Settings) -> ObjectStorage:
    return ObjectStorage(settings)


@pytest.fixture
def simulator(settings: Settings) -> SimulatedScanner:
    return SimulatedScanner(settings)


@pytest.fixture(autouse=True)
def use_storage(
    app_overrides: dict[Callable[..., object], Callable[..., object]], storage: ObjectStorage
) -> None:
    # Both dependencies, so every route sees the same real storage on the test bucket.
    app_overrides[get_storage] = lambda: storage
    app_overrides[get_optional_storage] = lambda: storage


class FakeScanner:
    """Wraps a real scanner. `behaviour(call_number)` runs first: it may raise (a scanner fault),
    do something while the scan is "in progress", or return an object to use as the verdict.
    Returning None passes the content to the wrapped scanner.
    """

    def __init__(
        self,
        inner: MalwareScanner,
        behaviour: Callable[[int], Awaitable[object | None]] | None = None,
    ) -> None:
        self._inner = inner
        self._behaviour = behaviour
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
        if self._behaviour is not None:
            override = await self._behaviour(self.calls)
            if override is not None:
                # Deliberately not always a real ScanVerdict: the worker must not trust it.
                return cast(ScanVerdict, override)
        return await self._inner.scan(content, bucket=bucket, key=key, version_id=version_id)


# ---- Helpers (no test module imports another)


def _image(fmt: str, shade: int, size: tuple[int, int]) -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", size, (shade, shade, shade)).save(buffer, fmt)
    return buffer.getvalue()


def _png(shade: int = 255, size: tuple[int, int] = (24, 16)) -> bytes:
    return _image("PNG", shade, size)


def _jpeg(shade: int = 255, size: tuple[int, int] = (24, 16)) -> bytes:
    return _image("JPEG", shade, size)


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


async def _put(upload: dict[str, Any], content: bytes) -> httpx.Response:
    # Exactly the returned headers; httpx adds the real Content-Length and Host itself.
    async with httpx.AsyncClient() as http:
        return await http.request(
            upload["method"], upload["url"], content=content, headers=upload["headers"]
        )


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


async def _quarantined(
    client: httpx.AsyncClient,
    auth: AuthHeaders,
    tenant: SeededTenant,
    content: bytes | None = None,
    *,
    sub: str = "creator-a",
    **overrides: Any,
) -> str:
    # Create, PUT and complete through the API; returns the asset id of the QUARANTINED upload.
    content = _png() if content is None else content
    headers = auth(sub, tenant.tenant_id, **{"Idempotency-Key": _key()})
    created = await client.post(UPLOADS, headers=headers, json=_body(tenant, content, **overrides))
    assert created.status_code == 201, created.text
    upload = created.json()
    stored = await _put(upload["upload"], content)
    assert stored.status_code == 200, stored.text
    done = await client.post(
        f"{ASSETS}/{upload['asset_id']}/complete", headers=auth(sub, tenant.tenant_id)
    )
    assert done.status_code == 202, done.text
    assert done.json()["status"] == "QUARANTINED"
    asset_id: str = upload["asset_id"]
    return asset_id


def _key() -> str:
    return uuid4().hex


async def _run(
    sessions: Sessions,
    storage: ObjectStorage,
    scanner: MalwareScanner,
    settings: Settings,
    owner: str = "w1",
    max_uploads: int | None = None,
) -> int:
    return await run_once(
        sessions, storage, scanner, settings, owner=owner, max_uploads=max_uploads
    )


async def _row(connection: AsyncConnection, tenant: SeededTenant, asset_id: str) -> Row[Any]:
    rows = await _owner_sql(
        connection,
        tenant,
        "SELECT state, mime_type, width_px, height_px, scan_engine, scan_simulated, scanned_at,"
        " reason_code, available_at, leased_until, lease_owner, attempts, failed_attempts,"
        " s3_version_id, completed_at, declared_size_bytes, declared_checksum_sha256, name,"
        " study_id, created_by, coalesce(available_at > clock_timestamp(), false) AS deferred"
        " FROM media_uploads WHERE id = :id",
        {"id": UUID(asset_id)},
    )
    assert len(rows) == 1
    return rows[0]


async def _assets(connection: AsyncConnection, tenant: SeededTenant) -> list[Row[Any]]:
    # Every asset row the tenant can see, read as the API's role so row-level security applies
    # (the test seeds no other assets).
    await connection.execute(
        text("SELECT set_config('app.tenant_id', :t, true)"), {"t": str(tenant.tenant_id)}
    )
    result = await connection.execute(
        text(
            "SELECT id, tenant_id, origin, upload_id, study_id, creative_id, job_id, name, bucket,"
            " s3_key, s3_version_id, mime_type, width_px, height_px, size_bytes, checksum_sha256,"
            " created_by FROM assets ORDER BY created_at, id"
        )
    )
    return list(result.all())


async def _make_due(connection: AsyncConnection, tenant: SeededTenant, asset_id: str) -> None:
    await _owner_sql(
        connection,
        tenant,
        "UPDATE media_uploads SET available_at = clock_timestamp() - interval '1 second'"
        " WHERE id = :id",
        {"id": UUID(asset_id)},
    )


async def _expire_lease(connection: AsyncConnection, tenant: SeededTenant, asset_id: str) -> None:
    await _owner_sql(
        connection,
        tenant,
        "UPDATE media_uploads SET leased_until = clock_timestamp() - interval '1 second'"
        " WHERE id = :id",
        {"id": UUID(asset_id)},
    )


async def _status(
    client: httpx.AsyncClient,
    auth: AuthHeaders,
    tenant: SeededTenant,
    asset_id: str,
    sub: str = "creator-a",
) -> dict[str, Any]:
    response = await client.get(f"{UPLOADS}/{asset_id}", headers=auth(sub, tenant.tenant_id))
    assert response.status_code == 200, response.text
    body: dict[str, Any] = response.json()
    return body


def _messages(
    caplog: pytest.LogCaptureFixture, level: int, logger_name: str = WORKER_LOGGER
) -> list[str]:
    return [
        record.getMessage()
        for record in caplog.records
        if record.name == logger_name and record.levelno == level
    ]


# ---- A clean upload becomes READY with one immutable asset


@needs_seaweedfs
@pytest.mark.parametrize(
    ("make", "mime", "name"),
    [
        pytest.param(_png, "image/png", "[SYNTHETIC] photo.png", id="png"),
        pytest.param(_jpeg, "image/jpeg", "[SYNTHETIC] photo.jpg", id="jpeg"),
    ],
)
async def test_clean_upload_becomes_ready_with_one_asset(
    client: httpx.AsyncClient,
    auth: AuthHeaders,
    world: World,
    connection: AsyncConnection,
    sessions: Sessions,
    storage: ObjectStorage,
    simulator: SimulatedScanner,
    settings: Settings,
    caplog: pytest.LogCaptureFixture,
    make: Callable[..., bytes],
    mime: str,
    name: str,
) -> None:
    content = make()
    asset_id = await _quarantined(client, auth, world.a, content, mime_type=mime, name=name)
    before = await _row(connection, world.a, asset_id)
    assert before.s3_version_id
    assert await _assets(connection, world.a) == []

    with caplog.at_level(logging.INFO):
        assert await _run(sessions, storage, simulator, settings) == 1

    row = await _row(connection, world.a, asset_id)
    assert row.state == "READY"
    assert (row.mime_type, row.width_px, row.height_px) == (mime, 24, 16)
    assert row.scan_engine == "SIMULATED"
    assert row.scan_simulated is True
    assert row.scanned_at is not None
    assert row.leased_until is None
    assert row.lease_owner is None
    assert row.reason_code is None
    assert row.available_at is None
    assert (row.attempts, row.failed_attempts) == (1, 0)
    assets = await _assets(connection, world.a)
    assert len(assets) == 1
    asset = assets[0]
    assert str(asset.id) == asset_id
    assert asset.upload_id == asset.id
    assert asset.origin == "uploaded"
    assert asset.tenant_id == world.a.tenant_id
    assert asset.study_id == world.a.study_id == row.study_id
    assert asset.creative_id is None
    assert asset.job_id is None
    assert asset.name == name == row.name
    assert asset.bucket == BUCKET
    assert asset.s3_version_id == row.s3_version_id == before.s3_version_id
    assert asset.size_bytes == len(content) == row.declared_size_bytes
    assert asset.checksum_sha256 == hashlib.sha256(content).hexdigest()
    assert asset.checksum_sha256 == row.declared_checksum_sha256
    assert (asset.mime_type, asset.width_px, asset.height_px) == (mime, 24, 16)
    assert asset.created_by == row.created_by == "creator-a"

    body = await _status(client, auth, world.a, asset_id)
    assert body["status"] == "READY"
    assert body["reason"] is None
    assert body["scan"]["simulated"] is True
    assert body["scan"]["engine"] == "SIMULATED"
    assert body["scan"]["scanned_at"]
    assert (body["width_px"], body["height_px"]) == (24, 16)
    read = await client.get(f"{ASSETS}/{asset_id}", headers=auth("creator-a", world.a.tenant_id))
    assert read.status_code == 200
    assert read.json()["origin"] == "uploaded"
    simulated_warnings = [m for m in _messages(caplog, logging.WARNING) if "SIMULATED" in m]
    assert len(simulated_warnings) == 1
    assert asset_id in simulated_warnings[0]


@needs_seaweedfs
async def test_duplicate_delivery_changes_nothing(
    client: httpx.AsyncClient,
    auth: AuthHeaders,
    world: World,
    connection: AsyncConnection,
    sessions: Sessions,
    storage: ObjectStorage,
    simulator: SimulatedScanner,
    settings: Settings,
) -> None:
    scanner = FakeScanner(simulator)
    asset_id = await _quarantined(client, auth, world.a)
    assert await _run(sessions, storage, scanner, settings) == 1
    finished = await _row(connection, world.a, asset_id)
    assets = await _assets(connection, world.a)

    assert await _run(sessions, storage, scanner, settings) == 0
    assert await _run(sessions, storage, scanner, settings, owner="w2") == 0

    assert scanner.calls == 1
    assert await _row(connection, world.a, asset_id) == finished
    assert await _assets(connection, world.a) == assets
    assert len(assets) == 1


# ---- Crashes, leases and the attempt cap


@needs_seaweedfs
@pytest.mark.parametrize("crash_in", ["before-the-checks", "during-the-scan"])
async def test_a_crashed_worker_is_replaced_after_its_lease_expires(
    client: httpx.AsyncClient,
    auth: AuthHeaders,
    world: World,
    connection: AsyncConnection,
    sessions: Sessions,
    storage: ObjectStorage,
    simulator: SimulatedScanner,
    settings: Settings,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    crash_in: str,
) -> None:
    async def crash_once(call: int) -> None:
        if call == 1:
            raise RuntimeError("synthetic crash")

    scanner = FakeScanner(simulator, crash_once if crash_in == "during-the-scan" else None)
    if crash_in == "before-the-checks":
        real_inspect = media_worker._inspect
        inspections: list[UUID] = []

        async def flaky(run: Any, upload_id: UUID, tenant_id: UUID) -> None:
            inspections.append(upload_id)
            if len(inspections) == 1:
                raise RuntimeError("synthetic crash")
            await real_inspect(run, upload_id, tenant_id)

        monkeypatch.setattr(media_worker, "_inspect", flaky)
    asset_id = await _quarantined(client, auth, world.a)

    with caplog.at_level(logging.INFO):
        assert await _run(sessions, storage, scanner, settings, owner="w1") == 1

    assert any("upload inspection failed" in m for m in _messages(caplog, logging.ERROR))
    crashed = await _row(connection, world.a, asset_id)
    assert crashed.state == ("QUARANTINED" if crash_in == "before-the-checks" else "SCANNING")
    assert crashed.lease_owner == "w1"
    assert crashed.leased_until is not None
    assert await _assets(connection, world.a) == []
    # While the dead worker's lease holds, nobody else takes the upload.
    assert await _run(sessions, storage, scanner, settings, owner="w2") == 0

    await _expire_lease(connection, world.a, asset_id)
    assert await _run(sessions, storage, scanner, settings, owner="w2") == 1

    row = await _row(connection, world.a, asset_id)
    assert row.state == "READY"
    assert (row.attempts, row.failed_attempts) == (2, 1)
    assert row.lease_owner is None
    assert row.leased_until is None
    assert len(await _assets(connection, world.a)) == 1


@needs_seaweedfs
@pytest.mark.parametrize(
    ("failed_before", "outcome"),
    [
        # The lease counts the expired one as a failure: 1 + 1 stays under the cap of 3 ...
        pytest.param(1, "READY", id="below-the-cap"),
        # ... 2 + 1 reaches it, and so does 3 + 1.
        pytest.param(2, "INSPECTION_FAILED", id="reaches-the-cap"),
        pytest.param(3, "INSPECTION_FAILED", id="at-the-cap"),
    ],
)
async def test_failed_attempts_reaching_the_cap_reject_the_upload(
    client: httpx.AsyncClient,
    auth: AuthHeaders,
    world: World,
    connection: AsyncConnection,
    sessions: Sessions,
    storage: ObjectStorage,
    simulator: SimulatedScanner,
    settings: Settings,
    failed_before: int,
    outcome: str,
) -> None:
    assert settings.media_inspection_max_attempts == 3
    scanner = FakeScanner(simulator)
    asset_id = await _quarantined(client, auth, world.a)
    # A worker that died mid-scan, failed_before times already.
    await _owner_sql(
        connection,
        world.a,
        "UPDATE media_uploads SET state = 'SCANNING', lease_owner = 'w1', attempts = :n,"
        " failed_attempts = :n, leased_until = clock_timestamp() - interval '1 second'"
        " WHERE id = :id",
        {"id": UUID(asset_id), "n": failed_before},
    )

    assert await _run(sessions, storage, scanner, settings, owner="w2") == 1

    row = await _row(connection, world.a, asset_id)
    body = await _status(client, auth, world.a, asset_id)
    if outcome == "READY":
        assert row.state == "READY"
        assert scanner.calls == 1
        assert len(await _assets(connection, world.a)) == 1
        return
    assert row.state == "REJECTED"
    assert row.reason_code == "INSPECTION_FAILED"
    assert row.failed_attempts >= settings.media_inspection_max_attempts
    assert row.lease_owner is None
    assert row.leased_until is None
    assert body["status"] == "REJECTED"
    assert body["reason"]["code"] == "INSPECTION_FAILED"
    assert body["reason"]["retryable"] is False
    # Rejected before any check: nothing was read or scanned, and no asset exists.
    assert scanner.calls == 0
    assert await _assets(connection, world.a) == []
    assert await _run(sessions, storage, scanner, settings, owner="w3") == 0


@needs_seaweedfs
@pytest.mark.parametrize("takeover", [False, True], ids=["lease-expires", "another-worker-wins"])
async def test_a_worker_that_lost_its_lease_changes_nothing(
    client: httpx.AsyncClient,
    auth: AuthHeaders,
    world: World,
    connection: AsyncConnection,
    sessions: Sessions,
    storage: ObjectStorage,
    simulator: SimulatedScanner,
    settings: Settings,
    caplog: pytest.LogCaptureFixture,
    takeover: bool,
) -> None:
    asset_id = await _quarantined(client, auth, world.a)
    successor = FakeScanner(simulator)

    async def lose_the_lease(call: int) -> object:
        # The lease runs out while w1 is scanning; in one case w2 then inspects the upload.
        await _expire_lease(connection, world.a, asset_id)
        if takeover:
            assert await _run(sessions, storage, successor, settings, owner="w2") == 1
        return ScanVerdict(ScanOutcome.CLEAN, engine="SIMULATED", simulated=True)

    scanner = FakeScanner(simulator, lose_the_lease)

    with caplog.at_level(logging.INFO):
        # One lease only: an expired one is leased again within the same pass otherwise.
        assert await _run(sessions, storage, scanner, settings, owner="w1", max_uploads=1) == 1

    assert scanner.calls == 1
    assert any(f"upload lease lost upload={asset_id}" in m for m in _messages(caplog, 30))
    # Only a worker that still held the lease logs the upload as ready.
    ready_logs = [m for m in _messages(caplog, logging.INFO) if "upload ready" in m]
    assert len(ready_logs) == (1 if takeover else 0)
    row = await _row(connection, world.a, asset_id)
    if takeover:
        # w2 finished it; w1's late CLEAN neither re-inserted the asset nor touched the upload.
        assert row.state == "READY"
        assert row.failed_attempts == 1
        assert len(await _assets(connection, world.a)) == 1
        assert successor.calls == 1
        return
    assert row.state == "SCANNING"
    assert row.lease_owner == "w1"
    assert row.scan_engine is None
    assert await _assets(connection, world.a) == []
    # The next lease re-inspects it from scratch and counts w1's unfinished attempt.
    assert await _run(sessions, storage, successor, settings, owner="w2") == 1
    after = await _row(connection, world.a, asset_id)
    assert after.state == "READY"
    assert after.failed_attempts == 1
    assert len(await _assets(connection, world.a)) == 1


# ---- Temporary outages wait in QUARANTINED, within the retry window


@needs_seaweedfs
async def test_scanner_outage_defers_the_upload_then_rejects_it_after_the_window(
    client: httpx.AsyncClient,
    auth: AuthHeaders,
    world: World,
    connection: AsyncConnection,
    sessions: Sessions,
    storage: ObjectStorage,
    simulator: SimulatedScanner,
    settings: Settings,
    caplog: pytest.LogCaptureFixture,
) -> None:
    async def down(call: int) -> None:
        raise ScanUnavailableError

    scanner = FakeScanner(simulator, down)
    asset_id = await _quarantined(client, auth, world.a)

    with caplog.at_level(logging.INFO):
        assert await _run(sessions, storage, scanner, settings) == 1

    row = await _row(connection, world.a, asset_id)
    assert row.state == "QUARANTINED"
    assert row.reason_code == "SCAN_UNAVAILABLE"
    assert row.deferred is True
    assert row.lease_owner is None
    assert row.leased_until is None
    # A scanner outage is not a failed attempt of the worker.
    assert (row.attempts, row.failed_attempts) == (1, 0)
    assert row.scan_engine is None
    assert row.scanned_at is None
    assert await _assets(connection, world.a) == []
    body = await _status(client, auth, world.a, asset_id)
    assert body["status"] == "QUARANTINED"
    assert body["reason"]["code"] == "SCAN_UNAVAILABLE"
    assert body["reason"]["retryable"] is True
    assert any("SCAN_UNAVAILABLE" in m for m in _messages(caplog, logging.WARNING))
    # Not due before its backoff passes.
    assert await _run(sessions, storage, scanner, settings) == 0

    # Due again, still inside the window: waits again.
    await _make_due(connection, world.a, asset_id)
    assert await _run(sessions, storage, scanner, settings) == 1
    again = await _row(connection, world.a, asset_id)
    assert (again.state, again.reason_code, again.attempts) == (
        "QUARANTINED",
        "SCAN_UNAVAILABLE",
        2,
    )

    # Past the window since completion: rejected for good.
    await _owner_sql(
        connection,
        world.a,
        "UPDATE media_uploads SET completed_at = now() - interval '2 days' WHERE id = :id",
        {"id": UUID(asset_id)},
    )
    await _make_due(connection, world.a, asset_id)
    assert await _run(sessions, storage, scanner, settings) == 1

    final = await _row(connection, world.a, asset_id)
    assert (final.state, final.reason_code) == ("REJECTED", "INSPECTION_FAILED")
    assert final.available_at is None
    assert final.lease_owner is None
    body = await _status(client, auth, world.a, asset_id)
    assert body["status"] == "REJECTED"
    assert body["reason"]["retryable"] is False
    assert await _assets(connection, world.a) == []
    assert await _run(sessions, storage, scanner, settings) == 0


@needs_seaweedfs
async def test_storage_read_failure_defers_the_upload_and_recovers(
    client: httpx.AsyncClient,
    auth: AuthHeaders,
    world: World,
    connection: AsyncConnection,
    sessions: Sessions,
    storage: ObjectStorage,
    simulator: SimulatedScanner,
    settings: Settings,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    scanner = FakeScanner(simulator)
    real_read = ObjectStorage.read

    async def unavailable(self: ObjectStorage, key: str, *, version_id: str | None = None) -> bytes:
        raise StorageError

    asset_id = await _quarantined(client, auth, world.a)
    monkeypatch.setattr(ObjectStorage, "read", unavailable)

    assert await _run(sessions, storage, scanner, settings) == 1

    row = await _row(connection, world.a, asset_id)
    assert (row.state, row.reason_code) == ("QUARANTINED", "STORAGE_UNAVAILABLE")
    assert row.deferred is True
    assert row.lease_owner is None
    assert scanner.calls == 0
    assert await _assets(connection, world.a) == []
    body = await _status(client, auth, world.a, asset_id)
    assert body["reason"]["code"] == "STORAGE_UNAVAILABLE"
    assert body["reason"]["retryable"] is True

    # Storage is back and the backoff has passed: the same upload completes normally.
    monkeypatch.setattr(ObjectStorage, "read", real_read)
    await _make_due(connection, world.a, asset_id)
    assert await _run(sessions, storage, scanner, settings) == 1

    ready = await _row(connection, world.a, asset_id)
    assert ready.state == "READY"
    assert ready.reason_code is None
    assert len(await _assets(connection, world.a)) == 1


@needs_seaweedfs
@pytest.mark.parametrize(
    "outcome",
    [
        pytest.param("UNKNOWN", id="unknown-outcome"),
        pytest.param(None, id="no-outcome"),
        pytest.param(True, id="truthy-not-an-outcome"),
    ],
)
async def test_only_an_explicit_clean_verdict_makes_an_upload_ready(
    client: httpx.AsyncClient,
    auth: AuthHeaders,
    world: World,
    connection: AsyncConnection,
    sessions: Sessions,
    storage: ObjectStorage,
    simulator: SimulatedScanner,
    settings: Settings,
    outcome: object,
) -> None:
    async def undecided(call: int) -> object:
        return SimpleNamespace(outcome=outcome, engine="FAKE", simulated=False)

    asset_id = await _quarantined(client, auth, world.a)

    assert await _run(sessions, storage, FakeScanner(simulator, undecided), settings) == 1

    row = await _row(connection, world.a, asset_id)
    assert (row.state, row.reason_code) == ("QUARANTINED", "SCAN_UNAVAILABLE")
    assert row.deferred is True
    # The odd verdict is not recorded as scan evidence.
    assert row.scan_engine is None
    assert row.scanned_at is None
    assert await _assets(connection, world.a) == []
    read = await client.get(f"{ASSETS}/{asset_id}", headers=auth("creator-a", world.a.tenant_id))
    assert read.status_code == 404


# ---- Who gets leased


@needs_seaweedfs
async def test_only_due_uploads_waiting_for_inspection_are_leased(
    client: httpx.AsyncClient,
    auth: AuthHeaders,
    world: World,
    connection: AsyncConnection,
    sessions: Sessions,
    storage: ObjectStorage,
    simulator: SimulatedScanner,
    settings: Settings,
) -> None:
    scanner = FakeScanner(simulator)
    headers = auth("creator-a", world.a.tenant_id, **{"Idempotency-Key": _key()})
    pending = await client.post(UPLOADS, headers=headers, json=_body(world.a, _png(10)))
    assert pending.status_code == 201, pending.text
    ready = await _quarantined(client, auth, world.a, _png(20))
    assert await _run(sessions, storage, scanner, settings) == 1
    assert (await _row(connection, world.a, ready)).state == "READY"
    rejected = await _quarantined(client, auth, world.a, _png(30))
    await _owner_sql(
        connection,
        world.a,
        "UPDATE media_uploads SET state = 'REJECTED', available_at = clock_timestamp()"
        " WHERE id = :id",
        {"id": UUID(rejected)},
    )
    later = await _quarantined(client, auth, world.a, _png(40))
    await _owner_sql(
        connection,
        world.a,
        "UPDATE media_uploads SET available_at = clock_timestamp() + interval '1 hour'"
        " WHERE id = :id",
        {"id": UUID(later)},
    )
    due = await _quarantined(client, auth, world.a, _png(50))
    ids = {
        "pending": pending.json()["asset_id"],
        "ready": ready,
        "rejected": rejected,
        "later": later,
        "due": due,
    }
    before = {name: await _row(connection, world.a, asset_id) for name, asset_id in ids.items()}
    scanner.calls = 0

    # The due upload is the only one leased, so the rest are left alone, not just unfinished.
    assert await _run(sessions, storage, scanner, settings) == 1

    assert scanner.calls == 1
    for name in ("pending", "ready", "rejected", "later"):
        assert await _row(connection, world.a, ids[name]) == before[name], name
    assert (await _row(connection, world.a, ids["due"])).state == "READY"
    assert {str(asset.id) for asset in await _assets(connection, world.a)} == {ids["ready"], due}
    assert await _run(sessions, storage, scanner, settings) == 0


# ---- Startup refusals


def test_worker_refuses_to_start_without_a_bucket(settings: Settings) -> None:
    unconfigured = settings.model_copy(update={"aws_s3_bucket": None})

    with pytest.raises(WorkerConfigError, match="AWS_S3_BUCKET"):
        build_dependencies(unconfigured)


def test_worker_refuses_to_start_in_live_mode_without_a_scanner(settings: Settings) -> None:
    live = settings.model_copy(update={"malware_scan_mode": "live"})

    with pytest.raises(WorkerConfigError, match="MALWARE_SCAN_MODE=live"):
        build_dependencies(live)


@pytest.mark.parametrize("app_env", ["staging", "production"])
def test_worker_refuses_the_simulator_outside_development(settings: Settings, app_env: str) -> None:
    # Even when the settings validator was bypassed: no silent fallback to a fake scanner.
    unsafe = settings.model_copy(update={"app_env": app_env})

    with pytest.raises(WorkerConfigError, match="APP_ENV is local or demo"):
        build_dependencies(unsafe)


def test_worker_starts_with_storage_and_the_labelled_simulator(settings: Settings) -> None:
    built_storage, scanner = build_dependencies(settings)

    assert built_storage.bucket == BUCKET
    assert isinstance(scanner, SimulatedScanner)
    assert (scanner.engine, scanner.simulated) == ("SIMULATED", True)


# ---- Tenants


@needs_seaweedfs
async def test_each_tenants_asset_lands_in_its_own_tenant(
    client: httpx.AsyncClient,
    auth: AuthHeaders,
    world: World,
    connection: AsyncConnection,
    sessions: Sessions,
    storage: ObjectStorage,
    simulator: SimulatedScanner,
    settings: Settings,
) -> None:
    content_a, content_b = _png(60), _jpeg(70)
    id_a = await _quarantined(client, auth, world.a, content_a)
    id_b = await _quarantined(
        client,
        auth,
        world.b,
        content_b,
        sub="creator-b",
        mime_type="image/jpeg",
        name="[SYNTHETIC] tenant-b.jpg",
    )

    assert await _run(sessions, storage, simulator, settings) == 2

    for tenant, asset_id, content, mime in (
        (world.a, id_a, content_a, "image/png"),
        (world.b, id_b, content_b, "image/jpeg"),
    ):
        row = await _row(connection, tenant, asset_id)
        assert row.state == "READY"
        assets = await _assets(connection, tenant)
        assert [str(asset.id) for asset in assets] == [asset_id]
        assert assets[0].tenant_id == tenant.tenant_id
        assert assets[0].study_id == tenant.study_id
        assert assets[0].mime_type == mime
        assert assets[0].checksum_sha256 == hashlib.sha256(content).hexdigest()
    # Each tenant reads only its own, through the API.
    headers_a = auth("creator-a", world.a.tenant_id)
    headers_b = auth("creator-b", world.b.tenant_id)
    assert (await client.get(f"{ASSETS}/{id_a}", headers=headers_a)).status_code == 200
    assert (await client.get(f"{ASSETS}/{id_b}", headers=headers_b)).status_code == 200
    assert (await client.get(f"{ASSETS}/{id_b}", headers=headers_a)).status_code == 404
    assert (await client.get(f"{ASSETS}/{id_a}", headers=headers_b)).status_code == 404
    assert [
        item["asset_id"] for item in (await client.get(MEDIA, headers=headers_a)).json()["items"]
    ] == [id_a]
    assert [
        item["asset_id"] for item in (await client.get(MEDIA, headers=headers_b)).json()["items"]
    ] == [id_b]
