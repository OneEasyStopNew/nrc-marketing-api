"""Simulated validation, generation, assets and export (DEMO_STUBS=true)."""

from collections.abc import Callable
from typing import Any
from uuid import UUID, uuid4

import httpx
import pytest

from app.api.deps import get_demo_stubs
from app.services.demo_stubs import DemoStubs, StubTimings
from tests.conftest import AuthHeaders, World

API = "/v1/marketing"
CLEAN_PROMPT = "Invite adults with high blood pressure to learn about our research study."
RISKY_PROMPT = "Guaranteed results! Try this new drug for free treatment."
CAPTION = "This is a research study. Taking part is voluntary."
BASE_URL = "http://192.0.2.10:8001"


class FakeClock:
    # Monotonic time the test moves forward by hand.

    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now


@pytest.fixture
def clock() -> FakeClock:
    return FakeClock()


@pytest.fixture
def stubs(clock: FakeClock) -> DemoStubs:
    return DemoStubs(
        BASE_URL, timings=StubTimings(validation_seconds=2, generation_seconds=6), clock=clock
    )


@pytest.fixture
def app_overrides(stubs: DemoStubs) -> dict[Callable[..., object], Callable[..., object]]:
    return {get_demo_stubs: lambda: stubs}


def _key() -> dict[str, str]:
    return {"Idempotency-Key": uuid4().hex}


async def _creative(
    client: httpx.AsyncClient, auth: AuthHeaders, world: World, prompt: str = CLEAN_PROMPT
) -> dict[str, Any]:
    response = await client.post(
        f"{API}/creatives",
        headers=auth("creator-a", world.a.tenant_id, **_key()),
        json={
            "study_id": str(world.a.study_id),
            "prompt": prompt,
            "caption": CAPTION,
            "language": "en",
            "channel": "facebook",
            "post_format": "feed_image",
        },
    )
    assert response.status_code == 201, response.text
    body: dict[str, Any] = response.json()
    return body


async def _validate(
    client: httpx.AsyncClient,
    auth: AuthHeaders,
    world: World,
    creative: dict[str, Any],
    kind: str = "brief",
    **extra: str,
) -> httpx.Response:
    return await client.post(
        f"{API}/validations",
        headers=auth("creator-a", world.a.tenant_id, **_key()),
        json={
            "creative_id": creative["id"],
            "creative_version": creative["version"],
            "input_kind": kind,
            **extra,
        },
    )


async def _finished_run(
    client: httpx.AsyncClient,
    auth: AuthHeaders,
    world: World,
    clock: FakeClock,
    creative: dict[str, Any],
    kind: str = "brief",
    **extra: str,
) -> dict[str, Any]:
    started = await _validate(client, auth, world, creative, kind, **extra)
    assert started.status_code == 202, started.text
    clock.now += 2
    run = await client.get(
        f"{API}/validations/{started.json()['validation_run_id']}",
        headers=auth("creator-a", world.a.tenant_id),
    )
    assert run.status_code == 200, run.text
    body: dict[str, Any] = run.json()
    return body


async def _generate(
    client: httpx.AsyncClient, auth: AuthHeaders, world: World, creative: dict[str, Any]
) -> httpx.Response:
    return await client.post(
        f"{API}/creative-jobs",
        headers=auth("creator-a", world.a.tenant_id, **_key()),
        json={
            "creative_id": creative["id"],
            "creative_version": creative["version"],
            "kind": "image_generation",
        },
    )


async def test_validation_runs_then_reports_findings_with_spans(
    client: httpx.AsyncClient, world: World, auth: AuthHeaders, clock: FakeClock
) -> None:
    creative = await _creative(client, auth, world, RISKY_PROMPT)

    started = await _validate(client, auth, world, creative)
    run_url = f"{API}/validations/{started.json()['validation_run_id']}"
    running = await client.get(run_url, headers=auth("creator-a", world.a.tenant_id))
    clock.now += 2
    finished = await client.get(run_url, headers=auth("creator-a", world.a.tenant_id))

    assert started.status_code == 202
    assert running.json()["state"] == "RUNNING"
    assert running.json()["outcome"] is None
    body = finished.json()
    assert body["state"] == "SUCCEEDED"
    assert body["outcome"] == "FAIL"
    assert body["policy_bundle_version"] == 1
    excerpts = {f["rule_id"]: f["offending_excerpt"] for f in body["findings"]}
    assert excerpts == {"REC-03": "new drug", "REC-04": "Guaranteed", "REC-05": "free treatment"}
    span = next(f["text_span"] for f in body["findings"] if f["rule_id"] == "REC-04")
    assert RISKY_PROMPT[span["start"] : span["end"]] == "Guaranteed"
    assert body["coverage_gaps"][0]["reason"].startswith("Demo stub")


async def test_clean_brief_passes_and_makes_the_version_eligible(
    client: httpx.AsyncClient, world: World, auth: AuthHeaders, clock: FakeClock
) -> None:
    creative = await _creative(client, auth, world)

    run = await _finished_run(client, auth, world, clock, creative)
    eligibility = await client.get(
        f"{API}/creatives/{creative['id']}/generation-eligibility",
        headers=auth("creator-a", world.a.tenant_id),
    )

    assert run["outcome"] == "PASS"
    assert run["findings"] == []
    assert eligibility.json()["eligible"] is True
    assert eligibility.json()["reason"] == "ELIGIBLE"
    assert eligibility.json()["validation_run_id"] == run["validation_run_id"]


async def test_failed_brief_blocks_generation(
    client: httpx.AsyncClient, world: World, auth: AuthHeaders, clock: FakeClock
) -> None:
    creative = await _creative(client, auth, world, RISKY_PROMPT)
    await _finished_run(client, auth, world, clock, creative)

    response = await _generate(client, auth, world, creative)

    assert response.status_code == 409
    assert response.json()["code"] == "GENERATION_NOT_ELIGIBLE"
    assert response.json()["details"] == {"reason": "VALIDATION_FAILED"}


async def test_generation_progresses_to_a_downloadable_image_and_export(
    client: httpx.AsyncClient, world: World, auth: AuthHeaders, clock: FakeClock
) -> None:
    creative = await _creative(client, auth, world)
    await _finished_run(client, auth, world, clock, creative)
    headers = auth("creator-a", world.a.tenant_id)

    accepted = await _generate(client, auth, world, creative)
    job_url = f"{API}/jobs/{accepted.json()['job_id']}"
    states = []
    for _ in range(3):
        states.append((await client.get(job_url, headers=headers)).json()["state"])
        clock.now += 3
    job = (await client.get(job_url, headers=headers)).json()
    asset = await client.get(f"{API}/assets/{job['result_asset_id']}", headers=headers)
    image = await client.get(asset.json()["download_url"].removeprefix(BASE_URL))
    export = await client.get(f"{API}/creatives/{creative['id']}/export", headers=headers)

    assert accepted.status_code == 202
    assert states == ["QUEUED", "WAITING_PROVIDER", "SUCCEEDED"]
    assert job["state"] == "SUCCEEDED"
    assert asset.status_code == 200
    assert asset.json()["download_url"].startswith(f"{BASE_URL}/demo-files/")
    assert (asset.json()["width_px"], asset.json()["height_px"]) == (1080, 1080)
    assert image.status_code == 200
    assert image.headers["content-type"] == "image/svg+xml"
    assert len(image.content) == asset.json()["size_bytes"]
    assert export.status_code == 200
    assert export.json()["status"] == "DRAFT"
    assert export.json()["image"]["asset_id"] == job["result_asset_id"]
    assert export.json()["latest_validation"]["outcome"] == "PASS"


async def test_image_validation_flags_payment_in_a_region(
    client: httpx.AsyncClient, world: World, auth: AuthHeaders, clock: FakeClock
) -> None:
    creative = await _creative(client, auth, world, f"{CLEAN_PROMPT} Get paid $50.")
    brief = await _finished_run(client, auth, world, clock, creative)
    job_id = (await _generate(client, auth, world, creative)).json()["job_id"]
    clock.now += 6
    job = (
        await client.get(f"{API}/jobs/{job_id}", headers=auth("creator-a", world.a.tenant_id))
    ).json()

    image = await _finished_run(
        client, auth, world, clock, creative, "image", asset_id=job["result_asset_id"]
    )

    # REC-06 (payment emphasis) applies to text and images, not briefs. It is optional, so the
    # image needs a human decision but does not fail.
    assert brief["outcome"] == "PASS"
    assert image["outcome"] == "NEEDS_REVIEW"
    assert image["findings"][0]["rule_id"] == "REC-06"
    assert image["findings"][0]["image_region"] is not None
    assert image["findings"][0]["needs_human_decision"] is True


async def test_same_key_replays_and_different_body_is_rejected(
    client: httpx.AsyncClient, world: World, auth: AuthHeaders
) -> None:
    creative = await _creative(client, auth, world)
    headers = auth("creator-a", world.a.tenant_id, **{"Idempotency-Key": "stub-replay-1"})
    body = {"creative_id": creative["id"], "creative_version": 1, "input_kind": "brief"}

    first = await client.post(f"{API}/validations", headers=headers, json=body)
    again = await client.post(f"{API}/validations", headers=headers, json=body)
    other = await client.post(
        f"{API}/validations", headers=headers, json={**body, "input_kind": "text"}
    )

    assert again.status_code == 202
    assert again.json()["validation_run_id"] == first.json()["validation_run_id"]
    assert again.headers["Idempotent-Replayed"] == "true"
    assert other.status_code == 409
    assert other.json()["code"] == "IDEMPOTENCY_KEY_REUSED"


async def test_other_tenant_cannot_see_runs_jobs_or_assets(
    client: httpx.AsyncClient, world: World, auth: AuthHeaders, clock: FakeClock
) -> None:
    creative = await _creative(client, auth, world)
    run = await _finished_run(client, auth, world, clock, creative)
    job_id = (await _generate(client, auth, world, creative)).json()["job_id"]
    clock.now += 6
    other = auth("creator-b", world.b.tenant_id)

    responses = [
        await client.get(f"{API}/validations/{run['validation_run_id']}", headers=other),
        await client.get(f"{API}/jobs/{job_id}", headers=other),
    ]

    assert [r.status_code for r in responses] == [404, 404]


async def test_export_needs_an_image_for_the_current_version(
    client: httpx.AsyncClient, world: World, auth: AuthHeaders
) -> None:
    creative = await _creative(client, auth, world)

    response = await client.get(
        f"{API}/creatives/{creative['id']}/export", headers=auth("creator-a", world.a.tenant_id)
    )

    assert response.status_code == 409
    assert response.json()["code"] == "EXPORT_NOT_READY"


async def test_image_validation_needs_a_generated_asset(
    client: httpx.AsyncClient, world: World, auth: AuthHeaders
) -> None:
    creative = await _creative(client, auth, world)

    response = await _validate(client, auth, world, creative, "image", asset_id=str(uuid4()))

    assert response.status_code == 404


async def test_unknown_or_unfinished_file_is_not_found(client: httpx.AsyncClient) -> None:
    response = await client.get(f"/demo-files/{UUID(int=1)}")

    assert response.status_code == 404
