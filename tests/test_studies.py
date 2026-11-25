# Studies, approved snapshots and platform capabilities (NRC-2).

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from uuid import UUID, uuid4

import httpx
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncConnection

from app.db.session import bind_tenant
from app.models import MembershipStudy, Study, StudySnapshot
from app.models.enums import Language, RecruitmentStatus
from scripts.seed_demo import SYNTHETIC_FACTS
from tests.conftest import AuthHeaders, World, _session

STUDIES = "/v1/marketing/studies"


@asynccontextmanager
async def _as_owner(connection: AsyncConnection) -> AsyncIterator[None]:
    # Studies and snapshots come from the import, never the API role, so tests add them as owner.
    await connection.execute(text("RESET ROLE"))
    try:
        yield
    finally:
        await connection.execute(text("SET LOCAL ROLE marketing_app"))


async def _add_snapshot(
    connection: AsyncConnection,
    tenant_id: UUID,
    study_id: UUID,
    version: int,
    status: RecruitmentStatus,
) -> None:
    async with _as_owner(connection), _session(connection) as db:
        await bind_tenant(db, tenant_id)
        db.add(
            StudySnapshot(
                tenant_id=tenant_id,
                study_id=study_id,
                version=version,
                recruitment_status=status,
                language=Language.EN,
                facts=SYNTHETIC_FACTS,
                source_ref=f"synthetic:test-fixture:v{version}",
                approved_by="synthetic-fixture",
                approved_at=datetime(2026, 11, 20, tzinfo=UTC),
                is_synthetic=True,
            )
        )
        await db.commit()


async def _add_study_without_snapshot(
    connection: AsyncConnection, tenant_id: UUID, title: str, member: str
) -> UUID:
    study_id = uuid4()
    async with _as_owner(connection), _session(connection) as db:
        await bind_tenant(db, tenant_id)
        db.add(Study(id=study_id, tenant_id=tenant_id, title=title))
        await db.flush()
        db.add(MembershipStudy(tenant_id=tenant_id, user_sub=member, study_id=study_id))
        await db.commit()
    return study_id


async def test_studies_are_listed_by_title_with_availability(
    client: httpx.AsyncClient, world: World, auth: AuthHeaders
) -> None:
    response = await client.get(STUDIES, headers=auth("creator-a", world.a.tenant_id))

    assert response.status_code == 200
    items = response.json()["items"]
    assert [(i["study_id"], i["availability"]) for i in items] == [
        (str(world.a.closed_study_id), "closed"),
        (str(world.a.study_id), "recruiting"),
    ]
    assert items[1]["latest_snapshot"]["snapshot_id"] == str(world.a.snapshot_id)
    assert items[1]["latest_snapshot"]["facts"]["approved_claims"]
    assert response.json()["next_cursor"] is None


async def test_studies_page_by_title(
    client: httpx.AsyncClient, world: World, auth: AuthHeaders
) -> None:
    headers = auth("creator-a", world.a.tenant_id)

    first = await client.get(STUDIES, params={"limit": 1}, headers=headers)
    second = await client.get(
        STUDIES, params={"limit": 1, "cursor": first.json()["next_cursor"]}, headers=headers
    )

    assert [i["study_id"] for i in first.json()["items"]] == [str(world.a.closed_study_id)]
    assert first.json()["next_cursor"] == str(world.a.closed_study_id)
    assert [i["study_id"] for i in second.json()["items"]] == [str(world.a.study_id)]
    assert second.json()["next_cursor"] is None


async def test_study_without_a_snapshot_is_told_apart_from_a_closed_one(
    client: httpx.AsyncClient, connection: AsyncConnection, world: World, auth: AuthHeaders
) -> None:
    study_id = await _add_study_without_snapshot(
        connection, world.a.tenant_id, "[SYNTHETIC] Awaiting approval", "creator-a"
    )

    study = await client.get(f"{STUDIES}/{study_id}", headers=auth("creator-a", world.a.tenant_id))
    create = await client.post(
        "/v1/marketing/creatives",
        headers=auth("creator-a", world.a.tenant_id, **{"Idempotency-Key": uuid4().hex}),
        json={
            "study_id": str(study_id),
            "prompt": "Invite adults to learn about the study.",
            "language": "en",
            "channel": "facebook",
            "post_format": "feed_image",
        },
    )

    assert study.status_code == 200
    assert study.json()["availability"] == "no_approved_snapshot"
    assert study.json()["latest_snapshot"] is None
    assert create.status_code == 409
    assert create.json()["code"] == "STUDY_SNAPSHOT_MISSING"


async def test_snapshot_history_pages_newest_first(
    client: httpx.AsyncClient, connection: AsyncConnection, world: World, auth: AuthHeaders
) -> None:
    for version in (2, 3):
        await _add_snapshot(
            connection, world.a.tenant_id, world.a.study_id, version, RecruitmentStatus.RECRUITING
        )
    url = f"{STUDIES}/{world.a.study_id}/snapshots"
    headers = auth("reviewer-a", world.a.tenant_id)

    first = await client.get(url, params={"limit": 2}, headers=headers)
    second = await client.get(
        url, params={"limit": 2, "cursor": first.json()["next_cursor"]}, headers=headers
    )

    assert [s["version"] for s in first.json()["items"]] == [3, 2]
    assert first.json()["next_cursor"] == 2
    assert [s["version"] for s in second.json()["items"]] == [1]
    assert second.json()["next_cursor"] is None
    assert second.json()["items"][0]["snapshot_id"] == str(world.a.snapshot_id)


async def test_a_new_snapshot_does_not_change_existing_drafts(
    client: httpx.AsyncClient, connection: AsyncConnection, world: World, auth: AuthHeaders
) -> None:
    draft = {
        "study_id": str(world.a.study_id),
        "prompt": "Invite adults with high blood pressure to learn about the study.",
        "language": "en",
        "channel": "facebook",
        "post_format": "feed_image",
    }
    creator = auth("creator-a", world.a.tenant_id, **{"Idempotency-Key": uuid4().hex})
    before = await client.post("/v1/marketing/creatives", headers=creator, json=draft)
    await _add_snapshot(
        connection, world.a.tenant_id, world.a.study_id, 2, RecruitmentStatus.RECRUITING
    )
    url = f"/v1/marketing/creatives/{before.json()['id']}"

    edited = await client.patch(
        url,
        headers=auth("creator-a", world.a.tenant_id, **{"If-Match": '"1"'}),
        json={"current_prompt": "Invite adults to learn about the study."},
    )
    after = await client.post(
        "/v1/marketing/creatives",
        headers=auth("creator-a", world.a.tenant_id, **{"Idempotency-Key": uuid4().hex}),
        json=draft,
    )

    assert before.json()["study_snapshot_version"] == 1
    assert edited.json()["study_snapshot_version"] == 1
    assert edited.json()["study_snapshot_id"] == str(world.a.snapshot_id)
    assert after.json()["study_snapshot_version"] == 2


async def test_closing_a_study_blocks_new_drafts_but_keeps_existing_ones(
    client: httpx.AsyncClient, connection: AsyncConnection, world: World, auth: AuthHeaders
) -> None:
    draft = {
        "study_id": str(world.a.study_id),
        "prompt": "Invite adults with high blood pressure to learn about the study.",
        "language": "en",
        "channel": "facebook",
        "post_format": "feed_image",
    }
    existing = await client.post(
        "/v1/marketing/creatives",
        headers=auth("creator-a", world.a.tenant_id, **{"Idempotency-Key": uuid4().hex}),
        json=draft,
    )
    await _add_snapshot(
        connection, world.a.tenant_id, world.a.study_id, 2, RecruitmentStatus.CLOSED
    )

    study = await client.get(
        f"{STUDIES}/{world.a.study_id}", headers=auth("creator-a", world.a.tenant_id)
    )
    read = await client.get(
        f"/v1/marketing/creatives/{existing.json()['id']}",
        headers=auth("creator-a", world.a.tenant_id),
    )
    history = await client.get(
        f"{STUDIES}/{world.a.study_id}/snapshots", headers=auth("creator-a", world.a.tenant_id)
    )
    new = await client.post(
        "/v1/marketing/creatives",
        headers=auth("creator-a", world.a.tenant_id, **{"Idempotency-Key": uuid4().hex}),
        json=draft,
    )

    assert study.json()["availability"] == "closed"
    assert read.status_code == 200
    assert [s["recruitment_status"] for s in history.json()["items"]] == ["closed", "recruiting"]
    assert new.status_code == 409
    assert new.json()["code"] == "STUDY_CLOSED"
    assert new.json()["message"]


async def test_other_tenants_study_is_not_found_or_listed(
    client: httpx.AsyncClient, world: World, auth: AuthHeaders
) -> None:
    headers = auth("creator-b", world.b.tenant_id)

    listed = await client.get(STUDIES, headers=headers)
    one = await client.get(f"{STUDIES}/{world.a.study_id}", headers=headers)
    snapshots = await client.get(f"{STUDIES}/{world.a.study_id}/snapshots", headers=headers)
    foreign_cursor = await client.get(
        STUDIES, params={"cursor": str(world.a.study_id)}, headers=headers
    )

    assert {i["study_id"] for i in listed.json()["items"]} == {
        str(world.b.study_id),
        str(world.b.closed_study_id),
    }
    assert one.status_code == 404
    assert snapshots.status_code == 404
    assert foreign_cursor.status_code == 200
    assert foreign_cursor.json() == {"items": [], "next_cursor": None}


async def test_study_without_a_grant_is_not_found(
    client: httpx.AsyncClient, connection: AsyncConnection, world: World, auth: AuthHeaders
) -> None:
    # Granted to reviewer-a only: creator-a in the same tenant must not see it.
    study_id = await _add_study_without_snapshot(
        connection, world.a.tenant_id, "[SYNTHETIC] Reviewer only", "reviewer-a"
    )

    listed = await client.get(STUDIES, headers=auth("creator-a", world.a.tenant_id))
    one = await client.get(f"{STUDIES}/{study_id}", headers=auth("creator-a", world.a.tenant_id))

    assert str(study_id) not in {i["study_id"] for i in listed.json()["items"]}
    assert one.status_code == 404


async def test_capabilities_list_formats_and_text_limits(
    client: httpx.AsyncClient, world: World, auth: AuthHeaders
) -> None:
    response = await client.get(
        "/v1/marketing/capabilities", headers=auth("reviewer-a", world.a.tenant_id)
    )

    assert response.status_code == 200
    body = response.json()
    assert body["languages"] == ["en"]
    assert [f["post_format"] for f in body["post_formats"]] == ["feed_image", "story_image"]
    assert body["text_limits"] == {
        "prompt_max_chars": 4000,
        "caption_max_chars": 2200,
        "image_text_max_chars": 300,
        "image_text_max_lines": 6,
        "headline_max_chars": 150,
        "audience_max_chars": 500,
        "visual_instructions_max_chars": 2000,
        "destination_url_max_chars": 2048,
        "generation_text_max_chars": 1448,
    }


async def test_capabilities_need_membership_of_the_tenant(
    client: httpx.AsyncClient, world: World, auth: AuthHeaders
) -> None:
    response = await client.get(
        "/v1/marketing/capabilities", headers=auth("creator-b", world.a.tenant_id)
    )

    assert response.status_code == 403
