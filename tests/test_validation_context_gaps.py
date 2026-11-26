# What the policy context couldn't settle is part of the saved validation (NRC-5; policy bundle
# contract, "Retrieval gaps"): a mandatory rule with a missing source, unresolved applicability
# or an incomplete graph walk is saved UNABLE_TO_ASSESS, one in a conflict NEEDS_REVIEW, FAIL
# stays FAIL, and a gap on an optional rule is only a warning. The generation gate reads the
# saved run, so each case also checks admission. Validation is a worker job (NRC-7): each case
# queues the run and lets one worker pass store it, as production does.

from typing import Any
from uuid import UUID, uuid4

import httpx
from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncConnection, AsyncSession

from app.core.config import get_settings
from app.models import PolicyEdge, PolicyNode, PolicyRuleRecord
from app.models.enums import PolicyEdgeType
from app.services.policy_context import AuroraPolicyContextProvider, PolicyContextProvider
from app.services.validations import INPUT_BY_JOB_KIND, VALIDATION_RUN_EVENT
from app.workers.connector import JobRunHandler, ValidationJobHandler, run_once
from tests.conftest import JOBS, AuthHeaders, Sessions, World, owner_session

API = "/v1/marketing"
CLEAN_PROMPT = "Invite adults with high blood pressure to learn about our research study."
RISKY_PROMPT = "Guaranteed results! Try this new drug for free treatment."
CAPTION = "Learn about a research study near you. Taking part is voluntary."
# Gets the same rule results as CAPTION, but isn't equal text (no reuse of CAPTION's result).
OTHER_CAPTION = "Learn about a research study in your area. Taking part is voluntary."
# A caption every seeded caption rule passes (the demo's passing caption).
PASSING_CAPTION = (
    "Adults aged 18 to 75 with high blood pressure may be able to join a research study of an "
    "investigational treatment. Taking part is voluntary, and there is no cost to take part."
)
# Mandatory rules the synthetic rulebook applies to briefs.
BRIEF_MANDATORY = ["REC-01", "REC-02", "REC-03", "REC-04", "REC-05", "REC-08"]


async def _node_id(db: AsyncSession, world: World, node_key: str) -> UUID:
    node_id = await db.scalar(
        select(PolicyNode.id).where(
            PolicyNode.policy_bundle_id == world.a.policy_bundle_id,
            PolicyNode.node_key == node_key,
        )
    )
    assert node_id is not None
    return node_id


async def _remove_source(connection: AsyncConnection, world: World, *node_keys: str) -> None:
    # The rule no longer quotes a clause and its supported_by edges are unapproved: SOURCE_MISSING.
    async with owner_session(connection, world.a.tenant_id) as db:
        for node_key in node_keys:
            node_id = await _node_id(db, world, node_key)
            await db.execute(
                update(PolicyNode).where(PolicyNode.id == node_id).values(source_clause_id=None)
            )
            await db.execute(
                update(PolicyEdge)
                .where(
                    PolicyEdge.from_node_id == node_id,
                    PolicyEdge.edge_type == PolicyEdgeType.SUPPORTED_BY,
                )
                .values(approved=False)
            )


async def _validate(
    client: httpx.AsyncClient,
    auth: AuthHeaders,
    world: World,
    sessions: Sessions,
    *,
    prompt: str = CLEAN_PROMPT,
    caption: str = CAPTION,
    kind: str = "brief",
    provider: PolicyContextProvider | None = None,
) -> tuple[str, dict[str, Any]]:
    headers = auth("creator-a", world.a.tenant_id, **{"Idempotency-Key": uuid4().hex})
    creative = await client.post(
        f"{API}/creatives",
        headers=headers,
        json={
            "study_id": str(world.a.study_id),
            "prompt": prompt,
            "caption": caption,
            "language": "en",
            "channel": "facebook",
            "post_format": "feed_image",
        },
    )
    assert creative.status_code == 201, creative.text
    creative_id: str = creative.json()["id"]
    accepted = await client.post(
        f"{API}/validations",
        headers=auth("creator-a", world.a.tenant_id, **{"Idempotency-Key": uuid4().hex}),
        json={"creative_id": creative_id, "creative_version": 1, "input_kind": kind},
    )
    assert accepted.status_code == 202, accepted.text
    # The worker's own handler (from settings), or one with this context provider.
    handler = ValidationJobHandler(context_provider=provider)
    ran = await run_once(
        sessions,
        {},
        None,
        get_settings(),
        event_handlers={
            VALIDATION_RUN_EVENT: JobRunHandler(dict.fromkeys(INPUT_BY_JOB_KIND, handler))
        },
    )
    assert ran == 1
    run = await client.get(
        f"{API}/validations/{accepted.json()['validation_run_id']}",
        headers=auth("creator-a", world.a.tenant_id),
    )
    body: dict[str, Any] = run.json()
    return creative_id, body


async def _eligibility(
    client: httpx.AsyncClient, auth: AuthHeaders, world: World, creative_id: str
) -> dict[str, Any]:
    response = await client.get(
        f"{API}/creatives/{creative_id}/generation-eligibility",
        headers=auth("creator-a", world.a.tenant_id),
    )
    body: dict[str, Any] = response.json()
    return body


def _results(run: dict[str, Any]) -> dict[str, str]:
    return {item["rule_id"]: item["result"] for item in run["rule_results"]}


def _gap_rules(run: dict[str, Any], code: str) -> set[str]:
    return {gap["rule_id"] for gap in run["coverage_gaps"] if f"({code}" in gap["reason"]}


async def test_clean_brief_with_a_complete_context_still_passes(
    client: httpx.AsyncClient, auth: AuthHeaders, world: World, sessions: Sessions
) -> None:
    # The baseline the cases below change: the seeded rulebook settles every mandatory rule.
    creative_id, run = await _validate(client, auth, world, sessions)

    assert run["outcome"] == "PASS"
    assert (await _eligibility(client, auth, world, creative_id))["eligible"] is True


async def test_missing_source_on_a_passing_mandatory_rule_is_saved_and_blocks_generation(
    client: httpx.AsyncClient,
    auth: AuthHeaders,
    world: World,
    sessions: Sessions,
    connection: AsyncConnection,
) -> None:
    # REC-02's phrase check passes on this brief, but its evidence was never retrieved.
    await _remove_source(connection, world, "REC-02")

    creative_id, run = await _validate(client, auth, world, sessions)

    assert _results(run)["REC-02"] == "UNABLE_TO_ASSESS"
    assert run["outcome"] == "NEEDS_REVIEW"
    assert _gap_rules(run, "SOURCE_MISSING") == {"REC-02"}
    eligibility = await _eligibility(client, auth, world, creative_id)
    assert (eligibility["eligible"], eligibility["reason"]) == (False, "NEEDS_REVIEW")
    job = await client.post(
        JOBS,
        headers=auth("creator-a", world.a.tenant_id, **{"Idempotency-Key": uuid4().hex}),
        json={"creative_id": creative_id, "creative_version": 1, "kind": "image_generation"},
    )
    assert (job.status_code, job.json()["code"]) == (409, "VALIDATION_NOT_PASSED")


async def test_unresolved_applicability_on_a_mandatory_rule_needs_review(
    client: httpx.AsyncClient,
    auth: AuthHeaders,
    world: World,
    sessions: Sessions,
    connection: AsyncConnection,
) -> None:
    # A validation never knows the audience, so an audience-restricted rule stays unresolved.
    async with owner_session(connection, world.a.tenant_id) as db:
        node_id = await _node_id(db, world, "REC-03")
        await db.execute(
            update(PolicyNode).where(PolicyNode.id == node_id).values(audiences=["adults"])
        )

    creative_id, run = await _validate(client, auth, world, sessions)

    assert _results(run)["REC-03"] == "UNABLE_TO_ASSESS"
    assert run["outcome"] == "NEEDS_REVIEW"
    assert _gap_rules(run, "APPLICABILITY_UNRESOLVED") == {"REC-03"}
    assert (await _eligibility(client, auth, world, creative_id))["eligible"] is False


def _one_hop() -> AuroraPolicyContextProvider:
    # Read when the test runs, never while pytest collects (CI has no env file then).
    settings = get_settings().model_copy(update={"policy_retrieval_max_hops": 1})
    return AuroraPolicyContextProvider(settings)


async def test_incomplete_walk_leaves_every_mandatory_rule_unassessed(
    client: httpx.AsyncClient,
    auth: AuthHeaders,
    world: World,
    sessions: Sessions,
    connection: AsyncConnection,
) -> None:
    # REC-01 -> EXC-01 -> EXC-02 needs two hops; with one, the walk is cut short. The cut names no
    # rule, so no mandatory rule can be trusted.
    async with owner_session(connection, world.a.tenant_id) as db:
        assert world.a.policy_bundle_id is not None
        db.add(
            PolicyEdge(
                tenant_id=world.a.tenant_id,
                policy_bundle_id=world.a.policy_bundle_id,
                edge_type=PolicyEdgeType.EXCEPT_WHEN,
                from_node_id=await _node_id(db, world, "REC-01"),
                to_node_id=await _node_id(db, world, "EXC-01"),
                approved=True,
                provenance_note="[SYNTHETIC] exception chain for a test",
            )
        )

    creative_id, run = await _validate(client, auth, world, sessions, provider=_one_hop())

    results = _results(run)
    assert {rule_id: results[rule_id] for rule_id in BRIEF_MANDATORY} == dict.fromkeys(
        BRIEF_MANDATORY, "UNABLE_TO_ASSESS"
    )
    assert run["outcome"] == "NEEDS_REVIEW"
    assert _gap_rules(run, "TRAVERSAL_TRUNCATED") == set(BRIEF_MANDATORY)
    assert (await _eligibility(client, auth, world, creative_id))["eligible"] is False


async def test_conflicting_mandatory_rules_need_review_not_unassessed(
    client: httpx.AsyncClient,
    auth: AuthHeaders,
    world: World,
    sessions: Sessions,
    connection: AsyncConnection,
) -> None:
    # REC-02 and REC-03 supersede each other: a person must decide, and may (NRC-12).
    async with owner_session(connection, world.a.tenant_id) as db:
        rec_02, rec_03 = await _node_id(db, world, "REC-02"), await _node_id(db, world, "REC-03")
        assert world.a.policy_bundle_id is not None
        db.add_all(
            PolicyEdge(
                tenant_id=world.a.tenant_id,
                policy_bundle_id=world.a.policy_bundle_id,
                edge_type=PolicyEdgeType.SUPERSEDES,
                from_node_id=newer,
                to_node_id=older,
                approved=True,
                provenance_note="[SYNTHETIC] conflicting supersedes for a test",
            )
            for newer, older in ((rec_02, rec_03), (rec_03, rec_02))
        )

    creative_id, run = await _validate(client, auth, world, sessions)

    conflicted = _gap_rules(run, "CONFLICT")
    assert conflicted
    assert conflicted <= {"REC-02", "REC-03"}
    assert {_results(run)[rule_id] for rule_id in conflicted} == {"NEEDS_REVIEW"}
    assert run["outcome"] == "NEEDS_REVIEW"
    assert (await _eligibility(client, auth, world, creative_id))["eligible"] is False


async def test_failed_rule_stays_failed_when_its_context_is_incomplete(
    client: httpx.AsyncClient,
    auth: AuthHeaders,
    world: World,
    sessions: Sessions,
    connection: AsyncConnection,
) -> None:
    await _remove_source(connection, world, *BRIEF_MANDATORY)

    creative_id, run = await _validate(client, auth, world, sessions, prompt=RISKY_PROMPT)

    results = _results(run)
    failed = {rule_id for rule_id, result in results.items() if result == "FAIL"}
    assert failed, "the risky brief should fail at least one rule"
    assert run["outcome"] == "FAIL"
    # The other mandatory rules are still marked: the saved result shows every gap.
    assert {results[rule_id] for rule_id in set(BRIEF_MANDATORY) - failed} == {"UNABLE_TO_ASSESS"}
    eligibility = await _eligibility(client, auth, world, creative_id)
    assert (eligibility["eligible"], eligibility["reason"]) == (False, "VALIDATION_FAILED")


async def test_gap_on_an_optional_rule_is_only_a_warning(
    client: httpx.AsyncClient,
    auth: AuthHeaders,
    world: World,
    sessions: Sessions,
    connection: AsyncConnection,
) -> None:
    # REC-06 is optional and applies to captions.
    _, before = await _validate(client, auth, world, sessions, kind="text")
    await _remove_source(connection, world, "REC-06")

    # Other text, so the earlier finished result isn't reused (tests edit the graph in place,
    # which an imported bundle never allows).
    _, after = await _validate(client, auth, world, sessions, caption=OTHER_CAPTION, kind="text")

    assert (after["outcome"], _results(after)) == (before["outcome"], _results(before))
    [warning] = [gap for gap in after["coverage_gaps"] if "(SOURCE_MISSING" in gap["reason"]]
    assert (warning["rule_id"], warning["reason"].endswith("(optional rule)")) == ("REC-06", True)


async def test_rule_mandatory_only_in_the_graph_still_blocks_the_run(
    client: httpx.AsyncClient,
    auth: AuthHeaders,
    world: World,
    sessions: Sessions,
    connection: AsyncConnection,
) -> None:
    # The import matches rule ids, not mandatory flags: here the graph makes REC-06 mandatory
    # while the bundle's rules keep it optional, and its source is missing.
    _, before = await _validate(client, auth, world, sessions, caption=PASSING_CAPTION, kind="text")
    assert before["outcome"] == "PASS"
    async with owner_session(connection, world.a.tenant_id) as db:
        await db.execute(
            update(PolicyRuleRecord)
            .where(
                PolicyRuleRecord.policy_bundle_id == world.a.policy_bundle_id,
                PolicyRuleRecord.rule_id == "REC-06",
            )
            .values(mandatory=True)
        )
    await _remove_source(connection, world, "REC-06")

    # Other passing text, so the earlier PASS isn't reused (see the test above).
    _, run = await _validate(
        client, auth, world, sessions, caption=f"{PASSING_CAPTION} Learn more today.", kind="text"
    )

    assert _results(run)["REC-06"] == "UNABLE_TO_ASSESS"
    assert run["outcome"] == "NEEDS_REVIEW"
    assert {gate["gate"]: gate["outcome"] for gate in run["gates"]}["policy"] == "NEEDS_REVIEW"


async def test_run_without_any_context_marks_every_mandatory_rule(
    client: httpx.AsyncClient,
    auth: AuthHeaders,
    world: World,
    sessions: Sessions,
    connection: AsyncConnection,
) -> None:
    # A bundle without a graph: the whole context is missing, not just part of it.
    async with owner_session(connection, world.a.tenant_id) as db:
        await db.execute(
            update(PolicyNode)
            .where(PolicyNode.policy_bundle_id == world.a.policy_bundle_id)
            .values(approved=False)
        )

    creative_id, run = await _validate(client, auth, world, sessions)

    results = _results(run)
    assert {results[rule_id] for rule_id in BRIEF_MANDATORY} == {"UNABLE_TO_ASSESS"}
    assert run["outcome"] == "NEEDS_REVIEW"
    assert any("(GRAPH_MISSING)" in gap["reason"] for gap in run["coverage_gaps"])
    assert (await _eligibility(client, auth, world, creative_id))["eligible"] is False
