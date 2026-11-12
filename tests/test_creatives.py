# Creative drafts: creation, idempotency, versioned edits and isolation.

from datetime import UTC, datetime, timedelta
from typing import Any
from uuid import UUID, uuid4

import httpx
import pytest
from sqlalchemy.ext.asyncio import AsyncConnection

from app.db.session import bind_tenant
from app.models import Asset, Job
from app.models.enums import ExecutionState
from tests.conftest import AuthHeaders, SeededTenant, World, _session

CREATIVES = "/v1/marketing/creatives"


def _draft(study_id: UUID, **overrides: object) -> dict[str, object]:
    return {
        "study_id": str(study_id),
        "prompt": "Invite adults with high blood pressure to learn about the study.",
        "caption": "Learn about a research study near you.",
        "language": "en",
        "channel": "facebook",
        "post_format": "feed_image",
        **overrides,
    }


async def _create(
    client: httpx.AsyncClient, auth: AuthHeaders, world: World, key: str | None = None
) -> dict[str, Any]:
    response = await client.post(
        CREATIVES,
        headers=auth("creator-a", world.a.tenant_id, **{"Idempotency-Key": key or uuid4().hex}),
        json=_draft(world.a.study_id),
    )
    assert response.status_code == 201, response.text
    body: dict[str, Any] = response.json()
    return body


async def test_create_returns_version_1_with_etag(
    client: httpx.AsyncClient, world: World, auth: AuthHeaders
) -> None:
    response = await client.post(
        CREATIVES,
        headers=auth("creator-a", world.a.tenant_id, **{"Idempotency-Key": "create-key-1"}),
        json=_draft(world.a.study_id),
    )

    assert response.status_code == 201
    body = response.json()
    assert body["version"] == 1
    assert body["original_prompt"] == body["current_prompt"]
    assert body["study_snapshot_id"] == str(world.a.snapshot_id)
    assert body["validation_runs"] == []
    assert response.headers["ETag"] == '"1"'
    assert response.headers["Location"] == f"{CREATIVES}/{body['id']}"


async def test_same_key_and_body_replays_the_original(
    client: httpx.AsyncClient, world: World, auth: AuthHeaders
) -> None:
    first = await _create(client, auth, world, key="replay-key-1")

    response = await client.post(
        CREATIVES,
        headers=auth("creator-a", world.a.tenant_id, **{"Idempotency-Key": "replay-key-1"}),
        json=_draft(world.a.study_id),
    )

    assert response.status_code == 201
    assert response.json()["id"] == first["id"]
    assert response.headers["Idempotent-Replayed"] == "true"


async def test_same_key_with_different_body_is_rejected(
    client: httpx.AsyncClient, world: World, auth: AuthHeaders
) -> None:
    await _create(client, auth, world, key="reuse-key-1")

    response = await client.post(
        CREATIVES,
        headers=auth("creator-a", world.a.tenant_id, **{"Idempotency-Key": "reuse-key-1"}),
        json=_draft(world.a.study_id, prompt="A different prompt."),
    )

    assert response.status_code == 409
    assert response.json()["code"] == "IDEMPOTENCY_KEY_REUSED"


async def test_closed_study_rejects_new_creatives(
    client: httpx.AsyncClient, world: World, auth: AuthHeaders
) -> None:
    response = await client.post(
        CREATIVES,
        headers=auth("creator-a", world.a.tenant_id, **{"Idempotency-Key": "closed-key-1"}),
        json=_draft(world.a.closed_study_id),
    )

    assert response.status_code == 409
    assert response.json()["code"] == "STUDY_CLOSED"


async def test_foreign_study_is_not_found(
    client: httpx.AsyncClient, world: World, auth: AuthHeaders
) -> None:
    response = await client.post(
        CREATIVES,
        headers=auth("creator-a", world.a.tenant_id, **{"Idempotency-Key": "foreign-key-1"}),
        json=_draft(world.b.study_id),
    )

    assert response.status_code == 404


@pytest.mark.parametrize(
    "overrides",
    [
        pytest.param({"language": "fr"}, id="unsupported-language"),
        pytest.param({"prompt": "   "}, id="blank-prompt"),
        pytest.param({"prompt": "x" * 4001}, id="prompt-too-long"),
        pytest.param({"is_approved": True}, id="unknown-field"),
    ],
)
async def test_invalid_drafts_return_field_errors(
    client: httpx.AsyncClient, world: World, auth: AuthHeaders, overrides: dict[str, object]
) -> None:
    response = await client.post(
        CREATIVES,
        headers=auth("creator-a", world.a.tenant_id, **{"Idempotency-Key": uuid4().hex}),
        json=_draft(world.a.study_id, **overrides),
    )

    assert response.status_code == 422
    body = response.json()
    assert body["code"] == "VALIDATION_FAILED"
    assert body["details"]["fields"]
    assert "input" not in body["details"]["fields"][0]


async def test_text_is_stored_in_nfc_form(
    client: httpx.AsyncClient, world: World, auth: AuthHeaders
) -> None:
    decomposed = "Café study 🙂"  # e + combining acute accent

    response = await client.post(
        CREATIVES,
        headers=auth("creator-a", world.a.tenant_id, **{"Idempotency-Key": uuid4().hex}),
        json=_draft(world.a.study_id, prompt=decomposed),
    )

    assert response.status_code == 201
    assert response.json()["current_prompt"] == "Café study 🙂"


async def test_missing_idempotency_key_is_rejected(
    client: httpx.AsyncClient, world: World, auth: AuthHeaders
) -> None:
    response = await client.post(
        CREATIVES, headers=auth("creator-a", world.a.tenant_id), json=_draft(world.a.study_id)
    )

    assert response.status_code == 422


async def test_edit_creates_new_version_and_keeps_original_prompt(
    client: httpx.AsyncClient, world: World, auth: AuthHeaders
) -> None:
    created = await _create(client, auth, world)

    response = await client.patch(
        f"{CREATIVES}/{created['id']}",
        headers=auth("creator-a", world.a.tenant_id, **{"If-Match": '"1"'}),
        json={"current_prompt": "Revised prompt.", "caption": None},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["version"] == 2
    assert body["current_prompt"] == "Revised prompt."
    assert body["caption"] is None
    assert body["original_prompt"] == created["original_prompt"]
    assert response.headers["ETag"] == '"2"'


async def test_stale_edit_returns_412_with_current_version(
    client: httpx.AsyncClient, world: World, auth: AuthHeaders
) -> None:
    created = await _create(client, auth, world)
    url = f"{CREATIVES}/{created['id']}"
    headers = auth("creator-a", world.a.tenant_id, **{"If-Match": '"1"'})
    first = await client.patch(url, headers=headers, json={"current_prompt": "Edit one."})

    second = await client.patch(url, headers=headers, json={"current_prompt": "Edit two."})

    assert first.status_code == 200
    assert second.status_code == 412
    assert second.json()["details"] == {"current_version": 2}
    current = await client.get(url, headers=auth("creator-a", world.a.tenant_id))
    assert current.json()["current_prompt"] == "Edit one."


@pytest.mark.parametrize("if_match", [None, "1", 'W/"1"'])
async def test_edit_requires_quoted_if_match(
    client: httpx.AsyncClient, world: World, auth: AuthHeaders, if_match: str | None
) -> None:
    created = await _create(client, auth, world)
    extra = {} if if_match is None else {"If-Match": if_match}

    response = await client.patch(
        f"{CREATIVES}/{created['id']}",
        headers=auth("creator-a", world.a.tenant_id, **extra),
        json={"current_prompt": "Edit."},
    )

    assert response.status_code == 428


async def test_empty_edit_is_rejected(
    client: httpx.AsyncClient, world: World, auth: AuthHeaders
) -> None:
    created = await _create(client, auth, world)

    response = await client.patch(
        f"{CREATIVES}/{created['id']}",
        headers=auth("creator-a", world.a.tenant_id, **{"If-Match": '"1"'}),
        json={},
    )

    assert response.status_code == 422


async def test_other_tenant_cannot_read_or_edit_by_id(
    client: httpx.AsyncClient, world: World, auth: AuthHeaders
) -> None:
    created = await _create(client, auth, world)
    url = f"{CREATIVES}/{created['id']}"

    read = await client.get(url, headers=auth("creator-b", world.b.tenant_id))
    edit = await client.patch(
        url,
        headers=auth("creator-b", world.b.tenant_id, **{"If-Match": '"1"'}),
        json={"current_prompt": "Hijacked."},
    )
    spoofed_tenant = await client.get(url, headers=auth("creator-b", world.a.tenant_id))

    assert read.status_code == 404
    assert edit.status_code == 404
    assert spoofed_tenant.status_code == 403


async def test_version_history_pages_newest_first(
    client: httpx.AsyncClient, world: World, auth: AuthHeaders
) -> None:
    created = await _create(client, auth, world)
    url = f"{CREATIVES}/{created['id']}"
    for version in (1, 2):
        response = await client.patch(
            url,
            headers=auth("creator-a", world.a.tenant_id, **{"If-Match": f'"{version}"'}),
            json={"current_prompt": f"Edit {version}."},
        )
        assert response.status_code == 200

    first_page = await client.get(
        f"{url}/versions", params={"limit": 2}, headers=auth("creator-a", world.a.tenant_id)
    )
    second_page = await client.get(
        f"{url}/versions",
        params={"limit": 2, "cursor": first_page.json()["next_cursor"]},
        headers=auth("creator-a", world.a.tenant_id),
    )

    assert [item["version"] for item in first_page.json()["items"]] == [3, 2]
    assert first_page.json()["next_cursor"] == 2
    assert [item["version"] for item in second_page.json()["items"]] == [1]
    assert second_page.json()["next_cursor"] is None


async def test_unimplemented_contract_endpoints_say_so(
    client: httpx.AsyncClient, world: World, auth: AuthHeaders
) -> None:
    created = await _create(client, auth, world)

    response = await client.post(
        "/v1/marketing/validations",
        headers=auth("creator-a", world.a.tenant_id, **{"Idempotency-Key": uuid4().hex}),
        json={"creative_id": created["id"], "creative_version": 1, "input_kind": "brief"},
    )

    assert response.status_code == 501
    assert response.json()["code"] == "NOT_IMPLEMENTED"


async def _create_for(
    client: httpx.AsyncClient, auth: AuthHeaders, tenant: SeededTenant, sub: str
) -> dict[str, Any]:
    response = await client.post(
        CREATIVES,
        headers=auth(sub, tenant.tenant_id, **{"Idempotency-Key": uuid4().hex}),
        json=_draft(tenant.study_id),
    )
    assert response.status_code == 201, response.text
    body: dict[str, Any] = response.json()
    return body


async def _store_asset(
    connection: AsyncConnection,
    tenant: SeededTenant,
    creative_id: str,
    version: int,
    created_at: datetime,
) -> str:
    # A succeeded job and its asset row, as the worker leaves them (no object storage needed).
    async with _session(connection) as db:
        await bind_tenant(db, tenant.tenant_id)
        job = Job(
            tenant_id=tenant.tenant_id,
            kind="image_generation",
            creative_id=UUID(creative_id),
            creative_version=version,
            study_snapshot_id=tenant.snapshot_id,
            study_snapshot_version=1,
            state=ExecutionState.SUCCEEDED,
            max_attempts=1,
            deadline_at=created_at + timedelta(minutes=10),
            created_by="creator",
        )
        db.add(job)
        await db.flush()
        asset = Asset(
            tenant_id=tenant.tenant_id,
            creative_id=UUID(creative_id),
            creative_version=version,
            job_id=job.id,
            origin="generated",
            bucket="fake",
            s3_key=f"fake/{uuid4()}.png",
            mime_type="image/png",
            width_px=1,
            height_px=1,
            size_bytes=1,
            checksum_sha256="0" * 64,
            created_by="creator",
            created_at=created_at,
        )
        db.add(asset)
        await db.commit()
    return str(asset.id)


async def test_asset_id_is_the_current_versions_image_and_clears_after_an_edit(
    client: httpx.AsyncClient, connection: AsyncConnection, world: World, auth: AuthHeaders
) -> None:
    created = await _create(client, auth, world)
    url = f"{CREATIVES}/{created['id']}"
    assert created["asset_id"] is None
    asset_id = await _store_asset(connection, world.a, created["id"], 1, datetime.now(UTC))

    current = await client.get(url, headers=auth("creator-a", world.a.tenant_id))
    edited = await client.patch(
        url,
        headers=auth("creator-a", world.a.tenant_id, **{"If-Match": '"1"'}),
        json={"current_prompt": "Revised prompt."},
    )
    after_edit = await client.get(url, headers=auth("creator-a", world.a.tenant_id))

    assert current.json()["asset_id"] == asset_id
    assert edited.status_code == 200
    assert edited.json()["asset_id"] is None
    assert after_edit.json()["asset_id"] is None


async def test_asset_id_is_the_newest_image_of_the_version(
    client: httpx.AsyncClient, connection: AsyncConnection, world: World, auth: AuthHeaders
) -> None:
    created = await _create(client, auth, world)
    now = datetime.now(UTC)
    # Stored newest first, so insertion order cannot be what picks the winner.
    newest = await _store_asset(connection, world.a, created["id"], 1, now)
    await _store_asset(connection, world.a, created["id"], 1, now - timedelta(minutes=5))

    response = await client.get(
        f"{CREATIVES}/{created['id']}", headers=auth("creator-a", world.a.tenant_id)
    )

    assert response.json()["asset_id"] == newest


async def test_asset_id_never_shows_another_tenants_image(
    client: httpx.AsyncClient, connection: AsyncConnection, world: World, auth: AuthHeaders
) -> None:
    mine = await _create(client, auth, world)
    theirs = await _create_for(client, auth, world.b, "creator-b")
    their_asset = await _store_asset(connection, world.b, theirs["id"], 1, datetime.now(UTC))

    my_view = await client.get(
        f"{CREATIVES}/{mine['id']}", headers=auth("creator-a", world.a.tenant_id)
    )
    their_view = await client.get(
        f"{CREATIVES}/{theirs['id']}", headers=auth("creator-b", world.b.tenant_id)
    )

    assert my_view.json()["asset_id"] is None
    assert their_view.json()["asset_id"] == their_asset
