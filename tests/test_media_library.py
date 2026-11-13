"""Real media route, Postgres/RLS and worker records; only object storage is replaced."""

from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from typing import Any
from uuid import UUID

import httpx
import pytest
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncConnection

from app.api.deps import get_optional_storage
from app.core.config import get_settings
from app.services.storage import ObjectStorage, StorageError
from app.workers.connector import run_once
from tests.conftest import (
    CREATOR_A_NAME,
    AuthHeaders,
    FakeSink,
    ScriptedProvider,
    Sessions,
    World,
    _creative,
    _post_job,
)

MEDIA = "/v1/marketing/media"


@pytest.fixture
def signed_urls(monkeypatch: pytest.MonkeyPatch) -> list[tuple[str, str | None, int]]:
    calls: list[tuple[str, str | None, int]] = []

    async def sign(
        self: ObjectStorage, key: str, *, version_id: str | None, expires_in: int
    ) -> str:
        calls.append((key, version_id, expires_in))
        return f"https://storage.example.test/image.png?signature={len(calls)}"

    monkeypatch.setattr(ObjectStorage, "presigned_get", sign)
    return calls


async def _generated(
    client: httpx.AsyncClient, auth: AuthHeaders, world: World, sessions: Sessions
) -> UUID:
    creative_id = await _creative(client, auth, world.a)
    response = await _post_job(client, auth, world, creative_id)
    assert response.status_code == 202
    sink = FakeSink()
    provider = ScriptedProvider()
    while await run_once(sessions, {"simulator": provider}, sink, get_settings()):
        pass
    assert len(sink.stored) == 1
    return sink.stored[0]


@pytest.fixture
async def generated(
    client: httpx.AsyncClient, auth: AuthHeaders, world: World, sessions: Sessions
) -> UUID:
    return await _generated(client, auth, world, sessions)


async def _owner_sql(
    connection: AsyncConnection, statement: str, params: dict[str, Any] | None = None
) -> None:
    await connection.execute(text("RESET ROLE"))
    await connection.execute(text(statement), params or {})
    await connection.execute(text("SET LOCAL ROLE marketing_app"))


async def test_lists_real_completed_image_with_fresh_versioned_links(
    generated: UUID,
    signed_urls: list[tuple[str, str | None, int]],
    client: httpx.AsyncClient,
    auth: AuthHeaders,
    world: World,
    connection: AsyncConnection,
) -> None:
    await _owner_sql(
        connection,
        "UPDATE assets SET s3_version_id = 'stored-version' WHERE id = :id",
        {"id": generated},
    )
    headers = auth("creator-a", world.a.tenant_id)
    before = datetime.now(UTC)
    response = await client.get(MEDIA, headers=headers, params={"limit": 50})

    assert response.status_code == 200
    body = response.json()
    assert len(body["items"]) == 1
    item = body["items"][0]
    assert item["asset_id"] == str(generated)
    assert item["creative_version"] == 1
    assert item["origin"] == "generated"
    assert item["mime_type"] == "image/png"
    assert item["created_by"] == "creator-a"
    assert item["created_by_name"] == CREATOR_A_NAME
    assert item["size_bytes"] > 0
    assert not {"bucket", "s3_key", "s3_version_id"} & item.keys()
    assert body["next_cursor"] is None
    ttl = get_settings().asset_url_ttl_seconds
    assert signed_urls[0][1:] == ("stored-version", ttl)
    expires = datetime.fromisoformat(item["download_url_expires_at"])
    assert before + timedelta(seconds=ttl) <= expires <= datetime.now(UTC) + timedelta(seconds=ttl)
    renewed = await client.get(MEDIA, headers=headers)
    assert renewed.json()["items"][0]["download_url"] != item["download_url"]


async def test_paginates_newest_first_without_dropping_timestamp_ties(
    client: httpx.AsyncClient,
    auth: AuthHeaders,
    world: World,
    sessions: Sessions,
    connection: AsyncConnection,
    signed_urls: list[tuple[str, str | None, int]],
) -> None:
    ids = [await _generated(client, auth, world, sessions) for _ in range(3)]
    newest = datetime(2026, 11, 13, tzinfo=UTC)
    for index, asset_id in enumerate(ids):
        await _owner_sql(
            connection,
            "UPDATE assets SET created_at = :at WHERE id = :id",
            {"id": asset_id, "at": newest - timedelta(days=int(index == 2))},
        )
    headers = auth("creator-a", world.a.tenant_id)
    first = await client.get(MEDIA, headers=headers, params={"limit": 1})
    assert first.status_code == 200
    assert [item["asset_id"] for item in first.json()["items"]] == [
        str(asset_id) for asset_id in sorted(ids[:2], reverse=True)
    ]
    second = await client.get(
        MEDIA, headers=headers, params={"limit": 1, "cursor": first.json()["next_cursor"]}
    )
    assert second.status_code == 200
    assert [item["asset_id"] for item in second.json()["items"]] == [str(ids[2])]
    assert second.json()["next_cursor"] is None
    assert len(signed_urls) == 3


async def test_other_tenant_sees_no_media_or_signed_links(
    generated: UUID,
    signed_urls: list[tuple[str, str | None, int]],
    client: httpx.AsyncClient,
    auth: AuthHeaders,
    world: World,
) -> None:
    response = await client.get(MEDIA, headers=auth("creator-b", world.b.tenant_id))
    assert response.status_code == 200
    assert response.json() == {"items": [], "next_cursor": None}
    assert signed_urls == []
    forged = await client.get(MEDIA, headers=auth("creator-b", world.a.tenant_id))
    assert forged.status_code == 403
    assert signed_urls == []


async def test_revoked_study_access_hides_media_before_signing(
    generated: UUID,
    signed_urls: list[tuple[str, str | None, int]],
    client: httpx.AsyncClient,
    auth: AuthHeaders,
    world: World,
    connection: AsyncConnection,
) -> None:
    await _owner_sql(
        connection,
        "DELETE FROM membership_studies WHERE user_sub = 'creator-a' AND study_id = :study",
        {"study": world.a.study_id},
    )
    response = await client.get(MEDIA, headers=auth("creator-a", world.a.tenant_id))
    assert response.status_code == 200
    assert response.json() == {"items": [], "next_cursor": None}
    assert signed_urls == []


@pytest.mark.parametrize(
    "change",
    [
        "UPDATE jobs SET state = 'FAILED'",
        "UPDATE jobs SET result_asset_id = NULL",
        "UPDATE assets SET origin = 'uploaded'",
        "UPDATE assets SET mime_type = 'video/mp4'",
    ],
)
async def test_only_completed_generated_images_are_listed(
    generated: UUID,
    signed_urls: list[tuple[str, str | None, int]],
    client: httpx.AsyncClient,
    auth: AuthHeaders,
    world: World,
    connection: AsyncConnection,
    change: str,
) -> None:
    await _owner_sql(connection, change)
    response = await client.get(MEDIA, headers=auth("creator-a", world.a.tenant_id))
    assert response.status_code == 200
    assert response.json() == {"items": [], "next_cursor": None}
    assert signed_urls == []


async def test_missing_author_name_is_null_and_reviewers_can_read(
    generated: UUID,
    signed_urls: list[tuple[str, str | None, int]],
    client: httpx.AsyncClient,
    auth: AuthHeaders,
    world: World,
    connection: AsyncConnection,
) -> None:
    await _owner_sql(connection, "UPDATE memberships SET display_name = NULL")
    response = await client.get(MEDIA, headers=auth("reviewer-a", world.a.tenant_id))
    assert response.status_code == 200
    assert response.json()["items"][0]["created_by_name"] is None


async def test_signing_failure_is_503_without_storage_details(
    generated: UUID,
    client: httpx.AsyncClient,
    auth: AuthHeaders,
    world: World,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def unavailable(*args: object, **kwargs: object) -> str:
        raise StorageError

    monkeypatch.setattr(ObjectStorage, "presigned_get", unavailable)
    response = await client.get(MEDIA, headers=auth("creator-a", world.a.tenant_id))
    assert response.status_code == 503
    assert response.json()["code"] == "DEPENDENCY_UNAVAILABLE"
    assert "fake/" not in response.text


@pytest.fixture
def without_storage(app_overrides: dict[Callable[..., object], Callable[..., object]]) -> None:
    app_overrides[get_optional_storage] = lambda: None


async def test_storage_configuration_missing_is_503(
    without_storage: None,
    client: httpx.AsyncClient,
    auth: AuthHeaders,
    world: World,
) -> None:
    response = await client.get(MEDIA, headers=auth("creator-a", world.a.tenant_id))
    assert response.status_code == 503
    assert response.json()["code"] == "DEPENDENCY_UNAVAILABLE"


@pytest.mark.parametrize(
    "params",
    [
        {"limit": "0"},
        {"limit": "51"},
        {"cursor": "not-a-date"},
        {"cursor": "2026-11-13T12:00:00"},
    ],
)
async def test_invalid_pagination_is_rejected(
    client: httpx.AsyncClient, auth: AuthHeaders, world: World, params: dict[str, str]
) -> None:
    response = await client.get(MEDIA, headers=auth("creator-a", world.a.tenant_id), params=params)
    assert response.status_code == 422


async def test_media_requires_login(client: httpx.AsyncClient, world: World) -> None:
    response = await client.get(MEDIA, headers={"X-Tenant-Id": str(world.a.tenant_id)})
    assert response.status_code == 401
