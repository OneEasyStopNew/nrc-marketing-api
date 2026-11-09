"""Generated-image storage end to end (simulator provider, real Postgres and SeaweedFS)."""

import hashlib
import io
from collections.abc import Iterator
from typing import Any
from uuid import UUID

import httpx
import pytest
from PIL import Image
from pydantic import SecretStr
from sqlalchemy import select, text
from sqlalchemy.exc import DBAPIError, IntegrityError
from sqlalchemy.ext.asyncio import AsyncConnection, AsyncSession, async_sessionmaker

from app.core.config import get_settings
from app.db.session import bind_tenant
from app.models import Asset, CreditReservation, Job
from app.models.enums import ExecutionState
from app.models.jobs import ReservationState
from app.services.assets import StoredImageSink
from app.services.providers.base import FetchedImage, ProviderOutputError
from app.services.providers.simulator import SimulatedImageProvider, render_simulated_png
from app.services.storage import ObjectStorage, StorageError
from app.workers.connector import WorkerConfigError, build_providers, run_once
from tests.conftest import AuthHeaders, World
from tests.test_job_worker import ScriptedProvider
from tests.test_jobs_api import _creative, _post_job
from tests.test_storage import needs_seaweedfs

pytestmark = needs_seaweedfs

Sessions = async_sessionmaker[AsyncSession]


@pytest.fixture
def sink() -> StoredImageSink:
    settings = get_settings()
    return StoredImageSink(ObjectStorage(settings), settings)


@pytest.fixture
def without_storage() -> Iterator[None]:
    # Must be requested before `client`, so the app starts without a bucket.
    with pytest.MonkeyPatch.context() as patch:
        patch.setenv("AWS_S3_BUCKET", "")
        get_settings.cache_clear()
        yield
    get_settings.cache_clear()


async def _generate(
    sessions: Sessions,
    sink: StoredImageSink,
    provider: SimulatedImageProvider,
    client: httpx.AsyncClient,
    auth: AuthHeaders,
    world: World,
) -> UUID:
    # Accept a job through the API, then run the worker until nothing is left.
    creative_id = await _creative(client, auth, world.a)
    response = await _post_job(client, auth, world, creative_id)
    assert response.status_code == 202
    while await run_once(sessions, {"simulator": provider}, sink, get_settings()):
        pass
    return UUID(response.json()["job_id"])


async def _job_json(
    client: httpx.AsyncClient, auth: AuthHeaders, world: World, job_id: UUID
) -> dict[str, Any]:
    response = await client.get(
        f"/v1/marketing/jobs/{job_id}", headers=auth("creator-a", world.a.tenant_id)
    )
    assert response.status_code == 200
    body: dict[str, Any] = response.json()
    return body


async def _get_asset(
    client: httpx.AsyncClient, auth: AuthHeaders, sub: str, tenant_id: UUID, asset_id: str
) -> httpx.Response:
    return await client.get(f"/v1/marketing/assets/{asset_id}", headers=auth(sub, tenant_id))


async def _succeeded_asset(
    sessions: Sessions,
    sink: StoredImageSink,
    client: httpx.AsyncClient,
    auth: AuthHeaders,
    world: World,
) -> tuple[UUID, str]:
    job_id = await _generate(sessions, sink, SimulatedImageProvider(), client, auth, world)
    job = await _job_json(client, auth, world, job_id)
    assert job["state"] == "SUCCEEDED"
    return job_id, str(job["result_asset_id"])


async def test_generated_image_is_stored_and_downloadable(
    sessions: Sessions,
    sink: StoredImageSink,
    client: httpx.AsyncClient,
    auth: AuthHeaders,
    world: World,
) -> None:
    job_id = await _generate(sessions, sink, SimulatedImageProvider(), client, auth, world)

    job = await _job_json(client, auth, world, job_id)
    assert job["state"] == "SUCCEEDED"
    assert job["result_asset_id"] is not None
    response = await _get_asset(
        client, auth, "creator-a", world.a.tenant_id, job["result_asset_id"]
    )
    assert response.status_code == 200
    asset = response.json()
    assert asset["asset_id"] == job["result_asset_id"]
    assert asset["creative_id"] == job["creative_id"]
    assert asset["origin"] == "generated"
    assert (asset["mime_type"], asset["width_px"], asset["height_px"]) == ("image/png", 1080, 1080)
    assert asset["provider"] == "simulator"
    assert asset["download_url_expires_at"] is not None
    async with httpx.AsyncClient() as downloader:
        downloaded = await downloader.get(asset["download_url"])
    assert downloaded.status_code == 200
    assert hashlib.sha256(downloaded.content).hexdigest() == asset["checksum_sha256"]
    assert len(downloaded.content) == asset["size_bytes"]
    with Image.open(io.BytesIO(downloaded.content)) as image:
        assert (image.format, image.size) == ("PNG", (1080, 1080))
    async with sessions() as db:
        await bind_tenant(db, world.a.tenant_id)
        row = (await db.execute(select(Asset))).scalar_one()
        reservation = await db.scalar(select(CreditReservation.state))
    assert row.job_id == job_id
    assert row.s3_version_id
    assert row.created_by == "creator-a"
    assert reservation is ReservationState.COMMITTED


async def test_other_tenant_gets_404(
    sessions: Sessions,
    sink: StoredImageSink,
    client: httpx.AsyncClient,
    auth: AuthHeaders,
    world: World,
) -> None:
    _, asset_id = await _succeeded_asset(sessions, sink, client, auth, world)

    response = await _get_asset(client, auth, "creator-b", world.b.tenant_id, asset_id)

    assert response.status_code == 404
    assert response.json()["code"] == "NOT_FOUND"


async def test_asset_of_a_study_the_caller_cannot_see_is_404(
    sessions: Sessions,
    sink: StoredImageSink,
    client: httpx.AsyncClient,
    auth: AuthHeaders,
    world: World,
    connection: AsyncConnection,
) -> None:
    _, asset_id = await _succeeded_asset(sessions, sink, client, auth, world)
    # Take the study away from creator-a, as the owner role (the app role cannot delete).
    await connection.execute(text("RESET ROLE"))
    await connection.execute(
        text("SELECT set_config('app.tenant_id', :t, true)"), {"t": str(world.a.tenant_id)}
    )
    await connection.execute(
        text("DELETE FROM membership_studies WHERE user_sub = 'creator-a' AND study_id = :s"),
        {"s": world.a.study_id},
    )
    await connection.execute(text("SET LOCAL ROLE marketing_app"))

    response = await _get_asset(client, auth, "creator-a", world.a.tenant_id, asset_id)

    assert response.status_code == 404


async def test_storage_not_configured_returns_503(
    without_storage: None, client: httpx.AsyncClient, auth: AuthHeaders, world: World
) -> None:
    response = await _get_asset(
        client, auth, "creator-a", world.a.tenant_id, "6f1d2a3b-4c5d-4e6f-8a7b-9c0d1e2f3a4b"
    )

    assert response.status_code == 503
    assert response.json()["code"] == "DEPENDENCY_UNAVAILABLE"


async def test_storage_failure_while_signing_the_url_returns_503(
    sessions: Sessions,
    sink: StoredImageSink,
    client: httpx.AsyncClient,
    auth: AuthHeaders,
    world: World,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, asset_id = await _succeeded_asset(sessions, sink, client, auth, world)

    async def unavailable(*args: object, **kwargs: object) -> str:
        raise StorageError

    monkeypatch.setattr(ObjectStorage, "presigned_get", unavailable)

    response = await _get_asset(client, auth, "creator-a", world.a.tenant_id, asset_id)

    assert response.status_code == 503
    body = response.json()
    assert (body["code"], body["retryable"]) == ("DEPENDENCY_UNAVAILABLE", True)
    assert "s3" not in body["message"].lower()


async def test_payload_that_is_not_an_image_fails_the_job_without_an_asset(
    sessions: Sessions,
    sink: StoredImageSink,
    client: httpx.AsyncClient,
    auth: AuthHeaders,
    world: World,
) -> None:
    provider = ScriptedProvider()
    junk = FetchedImage(b"<html>not an image</html>", "image/png", None)

    async def not_an_image(*_: object) -> FetchedImage:
        return junk

    provider.fetch_output = not_an_image  # type: ignore[method-assign,assignment]
    job_id = await _generate(sessions, sink, provider, client, auth, world)

    job = await _job_json(client, auth, world, job_id)
    assert (job["state"], job["failure_code"]) == ("FAILED", "OUTPUT_UNAVAILABLE")
    assert job["result_asset_id"] is None
    async with sessions() as db:
        await bind_tenant(db, world.a.tenant_id)
        assert (await db.execute(select(Asset))).first() is None
        reservation = await db.scalar(select(CreditReservation.state))
    # The provider billed a completed generation that could not be kept.
    assert reservation is ReservationState.COMMITTED


async def test_declared_type_must_match_the_decoded_type(
    sessions: Sessions,
    sink: StoredImageSink,
    client: httpx.AsyncClient,
    auth: AuthHeaders,
    world: World,
) -> None:
    creative_id = await _creative(client, auth, world.a)
    await _post_job(client, auth, world, creative_id)
    async with sessions() as db:
        await bind_tenant(db, world.a.tenant_id)
        job = (await db.execute(select(Job))).scalar_one()
        image = FetchedImage(render_simulated_png(), "image/jpeg", None)

        with pytest.raises(ProviderOutputError, match="does not match"):
            await sink.store(db, job, image, "simulator", "m")


async def test_app_role_cannot_update_assets(
    sessions: Sessions,
    sink: StoredImageSink,
    client: httpx.AsyncClient,
    auth: AuthHeaders,
    world: World,
    connection: AsyncConnection,
) -> None:
    await _succeeded_asset(sessions, sink, client, auth, world)
    await connection.execute(
        text("SELECT set_config('app.tenant_id', :t, true)"), {"t": str(world.a.tenant_id)}
    )

    with pytest.raises(DBAPIError, match="permission denied"):
        await connection.execute(text("UPDATE assets SET width_px = 1"))


async def test_second_asset_for_the_same_job_is_blocked(
    sessions: Sessions,
    sink: StoredImageSink,
    client: httpx.AsyncClient,
    auth: AuthHeaders,
    world: World,
) -> None:
    job_id, _ = await _succeeded_asset(sessions, sink, client, auth, world)
    async with sessions() as db:
        await bind_tenant(db, world.a.tenant_id)
        job = await db.get(Job, job_id)
        assert job is not None
        assert job.state is ExecutionState.SUCCEEDED
        image = FetchedImage(render_simulated_png(), "image/png", None)

        with pytest.raises(IntegrityError, match="uq_assets_job_id"):
            await sink.store(db, job, image, "simulator", "m")


def test_live_mode_without_storage_refuses_to_start() -> None:
    settings = get_settings().model_copy(
        update={
            "generation_mode": "live",
            "aws_s3_bucket": None,
            "higgsfield_api_key": SecretStr("test-id:test-secret"),
            "higgsfield_model": "test/model",
            "higgsfield_output_hosts": ("images.example.test",),
        }
    )

    with pytest.raises(WorkerConfigError, match="AWS_S3_BUCKET"):
        build_providers(settings)


async def test_success_queues_one_asset_generated_event_that_this_worker_never_leases(
    sessions: Sessions,
    sink: StoredImageSink,
    client: httpx.AsyncClient,
    auth: AuthHeaders,
    world: World,
    connection: AsyncConnection,
) -> None:
    job_id, asset_id = await _succeeded_asset(sessions, sink, client, auth, world)
    await connection.execute(
        text("SELECT set_config('app.tenant_id', :t, true)"), {"t": str(world.a.tenant_id)}
    )
    # Due now: only the event-type filter keeps the worker away from it.
    await connection.execute(
        text("UPDATE outbox_events SET available_at = clock_timestamp() - interval '1 minute'")
    )

    assert (
        await run_once(sessions, {"simulator": SimulatedImageProvider()}, sink, get_settings()) == 0
    )

    events = (
        await connection.execute(
            text(
                "SELECT payload, attempts, lease_owner, processed_at FROM outbox_events"
                " WHERE event_type = 'asset.generated'"
            )
        )
    ).all()
    async with sessions() as db:
        await bind_tenant(db, world.a.tenant_id)
        asset = (await db.execute(select(Asset))).scalar_one()
    assert len(events) == 1
    event = events[0]
    assert (event.attempts, event.lease_owner, event.processed_at) == (0, None, None)
    assert event.payload == {
        "event_version": 1,
        "job_id": str(job_id),
        "asset_id": asset_id,
        "tenant_id": str(world.a.tenant_id),
        "creative_id": str(asset.creative_id),
        "creative_version": 1,
        "bucket": asset.bucket,
        "s3_key": asset.s3_key,
        "s3_version_id": asset.s3_version_id,
        "checksum_sha256": asset.checksum_sha256,
        "mime_type": "image/png",
        "width_px": 1080,
        "height_px": 1080,
    }
    assert "http" not in str(event.payload)
