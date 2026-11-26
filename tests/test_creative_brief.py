# Brief fields on creative drafts: headline, CTA, link, site, audience, visual instructions (NRC-2).

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from typing import Any
from uuid import UUID, uuid4

import httpx
import pytest
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncConnection

from app.db.session import bind_tenant
from app.models import StudySnapshot
from app.models.enums import Language, RecruitmentStatus
from scripts.seed_demo import SYNTHETIC_FACTS
from tests.conftest import AuthHeaders, World, _session

CREATIVES = "/v1/marketing/creatives"
SITE = "Synthetic Staging Site"
LINK = "https://example.org/synthetic-staging-study"
FULL_BRIEF: dict[str, Any] = {
    "headline": "See if you qualify for a blood pressure study",
    "call_to_action": "learn_more",
    "destination_url": LINK,
    "site": SITE,
    "audience": "Adults 18 to 75 told they have high blood pressure",
    "visual_instructions": "Warm, natural light.\nA middle-aged adult reading at home.",
}
NO_BRIEF = dict.fromkeys(FULL_BRIEF)


def _draft(world: World, **brief: Any) -> dict[str, Any]:
    return {
        "study_id": str(world.a.study_id),
        "prompt": "Invite adults with high blood pressure to learn about the study.",
        "language": "en",
        "channel": "facebook",
        "post_format": "feed_image",
        **brief,
    }


async def _create(
    client: httpx.AsyncClient, auth: AuthHeaders, world: World, **body: Any
) -> httpx.Response:
    return await client.post(
        CREATIVES,
        headers=auth("creator-a", world.a.tenant_id, **{"Idempotency-Key": uuid4().hex}),
        json=_draft(world, **body),
    )


async def _patch(
    client: httpx.AsyncClient,
    auth: AuthHeaders,
    world: World,
    creative_id: str,
    version: int,
    body: dict[str, Any],
) -> httpx.Response:
    return await client.patch(
        f"{CREATIVES}/{creative_id}",
        headers=auth("creator-a", world.a.tenant_id, **{"If-Match": f'"{version}"'}),
        json=body,
    )


def _brief(body: dict[str, Any]) -> dict[str, Any]:
    return {name: body[name] for name in FULL_BRIEF}


def _field_errors(response: httpx.Response) -> list[str]:
    # The form maps the last loc entry to its field.
    return [error["loc"][-1] for error in response.json()["details"]["fields"]]


async def test_a_full_brief_is_saved_and_returned_with_no_issues(
    client: httpx.AsyncClient, world: World, auth: AuthHeaders
) -> None:
    created = await _create(client, auth, world, **FULL_BRIEF)

    versions = await client.get(
        f"{CREATIVES}/{created.json()['id']}/versions", headers=auth("creator-a", world.a.tenant_id)
    )

    assert created.status_code == 201, created.text
    assert _brief(created.json()) == FULL_BRIEF
    assert created.json()["brief_issues"] == []
    assert _brief(versions.json()["items"][0]) == FULL_BRIEF


async def test_a_draft_without_brief_fields_has_them_null_and_no_issues(
    client: httpx.AsyncClient, world: World, auth: AuthHeaders
) -> None:
    created = await _create(client, auth, world)

    assert created.status_code == 201
    assert _brief(created.json()) == NO_BRIEF
    assert created.json()["brief_issues"] == []


@pytest.mark.parametrize(
    ("post_format", "filled", "missing"),
    [
        ("feed_image", {"call_to_action": "sign_up"}, ["headline", "destination_url"]),
        ("feed_image", {"destination_url": LINK}, ["headline", "call_to_action"]),
        ("feed_image", {"headline": "Learn more"}, ["call_to_action", "destination_url"]),
        ("story_image", {"call_to_action": "apply_now"}, ["destination_url"]),
        ("story_image", {"headline": "Shown nowhere in a Story"}, []),
    ],
    ids=[
        "feed-cta-only",
        "feed-link-only",
        "feed-headline-only",
        "story-cta-only",
        "story-headline",
    ],
)
async def test_a_half_filled_link_still_saves_and_lists_what_is_missing(
    client: httpx.AsyncClient,
    world: World,
    auth: AuthHeaders,
    post_format: str,
    filled: dict[str, Any],
    missing: list[str],
) -> None:
    # Drafts autosave while the user types, so the link rule never blocks a save.
    created = await _create(client, auth, world, post_format=post_format, **filled)

    assert created.status_code == 201, created.text
    assert created.json()["brief_issues"] == [
        {"field": field, "code": "REQUIRED_WITH_LINK"} for field in missing
    ]


async def test_edits_set_clear_and_keep_brief_fields_in_new_versions(
    client: httpx.AsyncClient, world: World, auth: AuthHeaders
) -> None:
    created = await _create(client, auth, world, **FULL_BRIEF)
    creative_id = created.json()["id"]

    cleared = await _patch(
        client,
        auth,
        world,
        creative_id,
        1,
        {"destination_url": None, "call_to_action": None, "audience": "Adults 40 and over"},
    )
    caption_only = await _patch(client, auth, world, creative_id, 2, {"caption": "Learn more."})
    versions = await client.get(
        f"{CREATIVES}/{creative_id}/versions", headers=auth("creator-a", world.a.tenant_id)
    )

    assert cleared.status_code == 200, cleared.text
    assert _brief(cleared.json()) == {
        **FULL_BRIEF,
        "destination_url": None,
        "call_to_action": None,
        "audience": "Adults 40 and over",
    }
    # The headline alone is a link field without its link, so it is reported, not refused.
    assert cleared.json()["brief_issues"] == [
        {"field": "call_to_action", "code": "REQUIRED_WITH_LINK"},
        {"field": "destination_url", "code": "REQUIRED_WITH_LINK"},
    ]
    assert _brief(caption_only.json()) == _brief(cleared.json())
    assert [_brief(v) for v in versions.json()["items"]][2] == FULL_BRIEF


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("destination_url", "http://example.org/study"),
        ("destination_url", "https://user:secret@example.org/study"),
        ("destination_url", "not a link"),
        ("destination_url", "https://example.org/" + "a" * 2048),
        ("call_to_action", "buy_now"),
        ("headline", "Two\nlines"),
        ("headline", "   "),
        ("headline", "h" * 151),
        # Within max_length as sent, but NFC turns it into 300 characters.
        ("headline", "क़" * 150),
        ("audience", "a" * 501),
        ("visual_instructions", "Bell\u0007 character"),
        ("site", "x" * 301),
    ],
    ids=[
        "plain-http",
        "credentials",
        "not-a-url",
        "url-too-long",
        "unknown-cta",
        "two-line-headline",
        "blank-headline",
        "headline-too-long",
        "headline-longer-after-nfc",
        "audience-too-long",
        "control-character",
        "site-too-long",
    ],
)
async def test_each_field_is_checked_on_save_and_names_itself(
    client: httpx.AsyncClient, world: World, auth: AuthHeaders, field: str, value: str
) -> None:
    created = await _create(client, auth, world, **{field: value})
    edited_base = await _create(client, auth, world)
    edited = await _patch(client, auth, world, edited_base.json()["id"], 1, {field: value})

    assert created.status_code == 422
    assert created.json()["code"] == "VALIDATION_FAILED"
    assert _field_errors(created) == [field]
    assert edited.status_code == 422
    assert _field_errors(edited) == [field]


async def test_a_site_must_be_one_of_the_study_sites(
    client: httpx.AsyncClient, world: World, auth: AuthHeaders
) -> None:
    created = await _create(client, auth, world, site="Another Clinic")
    base = await _create(client, auth, world)
    edited = await _patch(client, auth, world, base.json()["id"], 1, {"site": "another clinic"})
    after = await client.get(
        f"{CREATIVES}/{base.json()['id']}", headers=auth("creator-a", world.a.tenant_id)
    )

    assert created.status_code == 422
    assert created.json()["details"]["fields"] == [
        {
            "loc": ["body", "site"],
            "msg": "must be one of the study's approved sites",
            "type": "site_not_approved",
        }
    ]
    assert edited.status_code == 422
    # The refused edit made no version.
    assert after.json()["version"] == 1


@asynccontextmanager
async def _as_owner(connection: AsyncConnection) -> AsyncIterator[None]:
    await connection.execute(text("RESET ROLE"))
    try:
        yield
    finally:
        await connection.execute(text("SET LOCAL ROLE marketing_app"))


async def _new_snapshot_with_sites(
    connection: AsyncConnection, tenant_id: UUID, study_id: UUID, sites: list[str]
) -> None:
    async with _as_owner(connection), _session(connection) as db:
        await bind_tenant(db, tenant_id)
        db.add(
            StudySnapshot(
                tenant_id=tenant_id,
                study_id=study_id,
                version=2,
                recruitment_status=RecruitmentStatus.RECRUITING,
                language=Language.EN,
                facts={**SYNTHETIC_FACTS, "sites": sites},
                source_ref="synthetic:test-fixture:v2",
                approved_by="synthetic-fixture",
                approved_at=datetime(2026, 11, 22, tzinfo=UTC),
                is_synthetic=True,
            )
        )
        await db.commit()


async def test_the_site_is_checked_against_the_snapshot_the_draft_is_pinned_to(
    client: httpx.AsyncClient, connection: AsyncConnection, world: World, auth: AuthHeaders
) -> None:
    pinned = await _create(client, auth, world)
    await _new_snapshot_with_sites(
        connection, world.a.tenant_id, world.a.study_id, ["Synthetic Site North"]
    )
    creative_id = pinned.json()["id"]

    newer_site = await _patch(client, auth, world, creative_id, 1, {"site": "Synthetic Site North"})
    pinned_site = await _patch(client, auth, world, creative_id, 1, {"site": SITE})
    new_draft = await _create(client, auth, world, site="Synthetic Site North")

    assert newer_site.status_code == 422
    assert pinned_site.status_code == 200, pinned_site.text
    assert pinned_site.json()["study_snapshot_version"] == 1
    assert new_draft.status_code == 201
    assert new_draft.json()["study_snapshot_version"] == 2


async def test_a_study_without_sites_takes_drafts_without_a_site_only(
    client: httpx.AsyncClient, connection: AsyncConnection, world: World, auth: AuthHeaders
) -> None:
    await _new_snapshot_with_sites(connection, world.a.tenant_id, world.a.study_id, [])

    without = await _create(client, auth, world)
    with_site = await _create(client, auth, world, site=SITE)

    assert without.status_code == 201
    assert with_site.status_code == 422


async def test_context_and_capabilities_list_ctas_and_link_fields(
    client: httpx.AsyncClient, world: World, auth: AuthHeaders
) -> None:
    headers = auth("creator-a", world.a.tenant_id)

    context = (await client.get("/v1/marketing/context", headers=headers)).json()
    capabilities = (await client.get("/v1/marketing/capabilities", headers=headers)).json()

    for body in (context, capabilities):
        assert body["call_to_actions"] == ["learn_more", "sign_up", "apply_now", "contact_us"]
        assert {f["post_format"]: f["link_fields"] for f in body["post_formats"]} == {
            "feed_image": ["headline", "call_to_action", "destination_url"],
            "story_image": ["call_to_action", "destination_url"],
        }
    assert capabilities["text_limits"]["headline_max_chars"] == 150
