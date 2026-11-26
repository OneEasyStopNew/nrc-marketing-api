# Stored brief and caption validation (POST/GET /validations) and the generation gate it feeds.

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from typing import Any
from uuid import UUID, uuid4

import httpx
import pytest
from sqlalchemy import insert, text
from sqlalchemy.exc import DBAPIError
from sqlalchemy.ext.asyncio import AsyncConnection

from app.models import PolicyBundle
from app.models.enums import BundleLifecycleState
from scripts.import_policy_graph import load_synthetic_graph
from scripts.seed_demo import demo_policy_rules
from tests.conftest import AuthHeaders, SeededTenant, Sessions, World, store_validations

API = "/v1/marketing"
RISKY_PROMPT = "Guaranteed results! Try this new drug for free treatment."
CLEAN_PROMPT = "Invite adults with high blood pressure to learn about our research study."
CAPTION = "Learn about a research study near you. Taking part is voluntary."


@asynccontextmanager
async def as_owner(connection: AsyncConnection) -> AsyncIterator[None]:
    # Policies are loaded by the owner; the API role can only read them.
    await connection.execute(text("RESET ROLE"))
    try:
        yield
    finally:
        await connection.execute(text("SET LOCAL ROLE marketing_app"))


async def _creative(
    client: httpx.AsyncClient,
    auth: AuthHeaders,
    tenant: SeededTenant,
    sub: str = "creator-a",
    prompt: str = CLEAN_PROMPT,
    caption: str | None = CAPTION,
) -> dict[str, Any]:
    response = await client.post(
        f"{API}/creatives",
        headers=auth(sub, tenant.tenant_id, **{"Idempotency-Key": uuid4().hex}),
        json={
            "study_id": str(tenant.study_id),
            "prompt": prompt,
            "caption": caption,
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
    tenant_id: UUID,
    creative: dict[str, Any],
    kind: str = "brief",
    *,
    sub: str = "creator-a",
    key: str | None = None,
    worker: Sessions | None = None,
) -> httpx.Response:
    # With worker, one worker pass then stores the queued run (validation is a job).
    response = await client.post(
        f"{API}/validations",
        headers=auth(sub, tenant_id, **{"Idempotency-Key": key or uuid4().hex}),
        json={
            "creative_id": creative["id"],
            "creative_version": creative["version"],
            "input_kind": kind,
        },
    )
    if worker is not None and response.status_code == 202:
        await store_validations(worker)
    return response


async def _run(
    client: httpx.AsyncClient, auth: AuthHeaders, tenant_id: UUID, run_id: str
) -> httpx.Response:
    return await client.get(f"{API}/validations/{run_id}", headers=auth("creator-a", tenant_id))


async def _eligibility(
    client: httpx.AsyncClient, auth: AuthHeaders, tenant_id: UUID, creative_id: str
) -> dict[str, Any]:
    response = await client.get(
        f"{API}/creatives/{creative_id}/generation-eligibility",
        headers=auth("creator-a", tenant_id),
    )
    assert response.status_code == 200, response.text
    body: dict[str, Any] = response.json()
    return body


async def test_brief_is_queued_then_checked_and_stored_by_the_worker(
    sessions: Sessions, client: httpx.AsyncClient, world: World, auth: AuthHeaders
) -> None:
    # The model check is off (the test default): validation is still a job.
    creative = await _creative(client, auth, world.a, prompt=RISKY_PROMPT)

    accepted = await _validate(client, auth, world.a.tenant_id, creative)
    run_id = accepted.json()["validation_run_id"]
    queued = (await _run(client, auth, world.a.tenant_id, run_id)).json()
    ran = await store_validations(sessions)
    run = await _run(client, auth, world.a.tenant_id, run_id)

    assert accepted.status_code == 202
    assert accepted.json()["state"] == "QUEUED"
    assert accepted.json()["job_id"] == run_id
    assert (queued["state"], queued["outcome"], ran) == ("QUEUED", None, 1)
    body = run.json()
    assert body["state"] == "SUCCEEDED"
    assert body["outcome"] == "FAIL"
    assert body["is_current"] is True
    assert (body["policy_bundle_version"], body["study_snapshot_version"]) == (1, 1)
    assert len(body["input_hash"]) == 64
    assert {f["rule_id"] for f in body["findings"]} == {"REC-03", "REC-04", "REC-05"}
    span = next(f["text_span"] for f in body["findings"] if f["rule_id"] == "REC-04")
    assert RISKY_PROMPT[span["start"] : span["end"]] == "Guaranteed"


async def test_caption_is_checked_as_text(
    sessions: Sessions, client: httpx.AsyncClient, world: World, auth: AuthHeaders
) -> None:
    creative = await _creative(client, auth, world.a, caption="Learn about a study near you.")

    accepted = await _validate(client, auth, world.a.tenant_id, creative, "text", worker=sessions)
    body = (
        await _run(client, auth, world.a.tenant_id, accepted.json()["validation_run_id"])
    ).json()

    assert body["input_kind"] == "text"
    assert body["outcome"] == "FAIL"
    assert [f["rule_id"] for f in body["findings"]] == ["REC-07"]


async def test_reviewer_can_validate(
    sessions: Sessions, client: httpx.AsyncClient, world: World, auth: AuthHeaders
) -> None:
    creative = await _creative(client, auth, world.a)

    response = await _validate(
        client, auth, world.a.tenant_id, creative, sub="reviewer-a", worker=sessions
    )

    assert response.status_code == 202


async def test_same_key_replays_and_a_different_body_is_rejected(
    sessions: Sessions, client: httpx.AsyncClient, world: World, auth: AuthHeaders
) -> None:
    creative = await _creative(client, auth, world.a)

    first = await _validate(
        client, auth, world.a.tenant_id, creative, key="validate-key-1", worker=sessions
    )
    again = await _validate(
        client, auth, world.a.tenant_id, creative, key="validate-key-1", worker=sessions
    )
    other = await _validate(
        client, auth, world.a.tenant_id, creative, "text", key="validate-key-1", worker=sessions
    )

    assert again.status_code == 202
    assert again.json()["validation_run_id"] == first.json()["validation_run_id"]
    assert again.headers["Idempotent-Replayed"] == "true"
    assert other.status_code == 409
    assert other.json()["code"] == "IDEMPOTENCY_KEY_REUSED"


async def test_other_tenant_cannot_see_or_validate(
    sessions: Sessions, client: httpx.AsyncClient, world: World, auth: AuthHeaders
) -> None:
    creative = await _creative(client, auth, world.a)
    run_id = (await _validate(client, auth, world.a.tenant_id, creative, worker=sessions)).json()[
        "validation_run_id"
    ]
    other = auth("creator-b", world.b.tenant_id)

    read = await client.get(f"{API}/validations/{run_id}", headers=other)
    write = await _validate(
        client, auth, world.b.tenant_id, creative, sub="creator-b", worker=sessions
    )

    assert read.status_code == 404
    assert write.status_code == 404


async def test_unknown_version_is_not_found(
    sessions: Sessions, client: httpx.AsyncClient, world: World, auth: AuthHeaders
) -> None:
    creative = await _creative(client, auth, world.a)

    response = await _validate(
        client, auth, world.a.tenant_id, {**creative, "version": 2}, worker=sessions
    )

    assert response.status_code == 404


async def test_tenant_without_policy_gets_409(
    sessions: Sessions, client: httpx.AsyncClient, world: World, auth: AuthHeaders
) -> None:
    creative = await _creative(client, auth, world.b, sub="creator-b")

    response = await _validate(
        client, auth, world.b.tenant_id, creative, sub="creator-b", worker=sessions
    )

    assert response.status_code == 409
    assert response.json()["code"] == "NO_ACTIVE_POLICY"


async def test_image_input_is_not_implemented_without_demo_stubs(
    client: httpx.AsyncClient, world: World, auth: AuthHeaders
) -> None:
    creative = await _creative(client, auth, world.a)

    response = await client.post(
        f"{API}/validations",
        headers=auth("creator-a", world.a.tenant_id, **{"Idempotency-Key": uuid4().hex}),
        json={
            "creative_id": creative["id"],
            "creative_version": 1,
            "input_kind": "image",
            "asset_id": str(uuid4()),
        },
    )

    assert response.status_code == 501


async def test_passing_brief_opens_the_gate_and_an_edit_closes_it(
    sessions: Sessions, client: httpx.AsyncClient, world: World, auth: AuthHeaders
) -> None:
    creative = await _creative(client, auth, world.a)
    run_id = (await _validate(client, auth, world.a.tenant_id, creative, worker=sessions)).json()[
        "validation_run_id"
    ]

    passed = await _eligibility(client, auth, world.a.tenant_id, creative["id"])
    edited = await client.patch(
        f"{API}/creatives/{creative['id']}",
        headers=auth("creator-a", world.a.tenant_id, **{"If-Match": '"1"'}),
        json={"current_prompt": f"{CLEAN_PROMPT} Sites near you."},
    )
    after_edit = await _eligibility(client, auth, world.a.tenant_id, creative["id"])
    old_run = await _run(client, auth, world.a.tenant_id, run_id)

    assert (passed["reason"], passed["eligible"]) == ("ELIGIBLE", True)
    assert passed["validation_run_id"] == run_id
    assert edited.status_code == 200
    assert (after_edit["creative_version"], after_edit["reason"]) == (2, "NOT_VALIDATED")
    assert old_run.json()["is_current"] is False


async def test_failing_brief_keeps_the_gate_closed(
    sessions: Sessions, client: httpx.AsyncClient, world: World, auth: AuthHeaders
) -> None:
    creative = await _creative(client, auth, world.a, prompt=RISKY_PROMPT)
    await _validate(client, auth, world.a.tenant_id, creative, worker=sessions)

    result = await _eligibility(client, auth, world.a.tenant_id, creative["id"])

    assert (result["reason"], result["validation_run_id"]) == ("VALIDATION_FAILED", None)


async def test_a_new_policy_version_requires_validating_again(
    sessions: Sessions,
    client: httpx.AsyncClient,
    world: World,
    auth: AuthHeaders,
    connection: AsyncConnection,
) -> None:
    creative = await _creative(client, auth, world.a)
    run_id = (await _validate(client, auth, world.a.tenant_id, creative, worker=sessions)).json()[
        "validation_run_id"
    ]
    async with as_owner(connection):
        # One ACTIVE bundle per tenant: the new version replaces v1, as activation does.
        await connection.execute(
            text("UPDATE policy_bundles SET lifecycle_state = 'RETIRED' WHERE tenant_id = :t"),
            {"t": world.a.tenant_id},
        )
        await connection.execute(
            insert(PolicyBundle).values(
                id=uuid4(),
                tenant_id=world.a.tenant_id,
                version=2,
                name="[SYNTHETIC] Policy v2",
                source_ref="synthetic:test-fixture",
                rules=demo_policy_rules(),
                is_synthetic=True,
                approved_by="synthetic-fixture",
                approved_at=datetime(2026, 11, 2, tzinfo=UTC),
                lifecycle_state=BundleLifecycleState.ACTIVE,
            )
        )

    result = await _eligibility(client, auth, world.a.tenant_id, creative["id"])
    old_run = await _run(client, auth, world.a.tenant_id, run_id)

    assert (result["reason"], result["policy_bundle_version"]) == ("NOT_VALIDATED", 2)
    assert old_run.json()["is_current"] is False


async def test_the_api_role_cannot_change_a_stored_run(
    sessions: Sessions,
    client: httpx.AsyncClient,
    world: World,
    auth: AuthHeaders,
    connection: AsyncConnection,
) -> None:
    creative = await _creative(client, auth, world.a, prompt=RISKY_PROMPT)
    await _validate(client, auth, world.a.tenant_id, creative, worker=sessions)

    with pytest.raises(DBAPIError, match="permission denied"):
        async with connection.begin_nested():
            await connection.execute(text("UPDATE validation_runs SET outcome = 'PASS'"))


async def _run_of(
    sessions: Sessions, client: httpx.AsyncClient, auth: AuthHeaders, world: World, prompt: str
) -> dict[str, Any]:
    creative = await _creative(client, auth, world.a, prompt=prompt)
    accepted = await _validate(client, auth, world.a.tenant_id, creative, worker=sessions)
    run: dict[str, Any] = (
        await _run(client, auth, world.a.tenant_id, accepted.json()["validation_run_id"])
    ).json()
    return run


# The real synthetic graph: REC-01 to REC-06 are semantic rules, UNABLE_TO_ASSESS until the model
# check exists.
@pytest.mark.parametrize("graph", [load_synthetic_graph()])
async def test_clean_brief_needs_review_while_semantic_rules_are_unassessed(
    sessions: Sessions, client: httpx.AsyncClient, world: World, auth: AuthHeaders
) -> None:
    run = await _run_of(sessions, client, auth, world, CLEAN_PROMPT)
    context = (
        await client.get(
            f"{API}/validations/{run['validation_run_id']}/context",
            headers=auth("creator-a", world.a.tenant_id),
        )
    ).json()

    # Not PASS: REC-01 to REC-05 are semantic rules, and the model check has not run.
    assert run["outcome"] == "NEEDS_REVIEW"
    assert run["findings"] == []
    results = {r["rule_id"]: r["result"] for r in run["rule_results"]}
    # The run lists exactly the rules the stored manifest selected (all mandatory for a brief).
    assert set(results) == {r["rule_id"] for r in context["required_rules"]}
    assert context["optional_rules"] == []
    assert results == {
        **dict.fromkeys(["REC-01", "REC-02", "REC-03", "REC-04", "REC-05"], "UNABLE_TO_ASSESS"),
        "REC-08": "PASS",
    }
    gap_rules = {g["rule_id"] for g in run["coverage_gaps"] if g["rule_id"]}
    assert gap_rules == {"REC-01", "REC-02", "REC-03", "REC-04", "REC-05"}
    reasons = {r["reason"] for r in run["rule_results"] if r["result"] == "UNABLE_TO_ASSESS"}
    assert reasons == {"Needs a review of meaning; the model check has not run."}


# The real synthetic graph: REC-01 to REC-06 are semantic rules, UNABLE_TO_ASSESS until the model
# check exists.
@pytest.mark.parametrize("graph", [load_synthetic_graph()])
async def test_guaranteed_fails_the_graph_rule_with_its_span_and_source(
    sessions: Sessions, client: httpx.AsyncClient, world: World, auth: AuthHeaders
) -> None:
    prompt = f"🙂 Guaranteed relief for adults. {CLEAN_PROMPT}"

    run = await _run_of(sessions, client, auth, world, prompt)

    assert run["outcome"] == "FAIL"
    (finding,) = run["findings"]
    assert finding["rule_id"] == "REC-04"
    assert prompt[finding["text_span"]["start"] : finding["text_span"]["end"]] == "Guaranteed"
    assert finding["text_span"]["start"] == 2
    # The source is the clause the rule quotes in the policy graph.
    assert finding["source"]["clause"] == "FDA-4"
    assert finding["source"]["source_id"].startswith("[SYNTHETIC]")


# The real synthetic graph: REC-01 to REC-06 are semantic rules, UNABLE_TO_ASSESS until the model
# check exists.
@pytest.mark.parametrize("graph", [load_synthetic_graph()])
async def test_wrong_age_fails_the_study_facts_rule(
    sessions: Sessions, client: httpx.AsyncClient, world: World, auth: AuthHeaders
) -> None:
    run = await _run_of(
        sessions, client, auth, world, f"Adults aged 30 to 40 can join. {CLEAN_PROMPT}"
    )

    assert run["outcome"] == "FAIL"
    assert [f["rule_id"] for f in run["findings"]] == ["REC-08"]
    gates = {g["gate"]: g["outcome"] for g in run["gates"]}
    assert gates["study_facts"] == "FAIL"
