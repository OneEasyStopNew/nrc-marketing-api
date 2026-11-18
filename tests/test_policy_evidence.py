# Policy evidence (NRC-5): the context stored by each validation run, and the evidence API.

import re
from typing import Any
from uuid import UUID, uuid4

import httpx
from sqlalchemy import Row, select, text
from sqlalchemy.ext.asyncio import AsyncConnection

from app.models import ContextManifestRecord, ValidationRun
from app.models.enums import BundleLifecycleState
from scripts.import_policy_graph import import_policy_graph, load_synthetic_graph
from scripts.seed_demo import demo_policy_rules
from tests.conftest import (
    AuthHeaders,
    World,
    add_policy_bundle,
    as_tenant_owner,
    count_rows,
    owner_session,
)

API = "/v1/marketing"
CLEAN_PROMPT = "Invite adults with high blood pressure to learn about our research study."
RISKY_PROMPT = "Guaranteed results! This proven cure is free treatment."
CAPTION = "Learn about a research study near you. Taking part is voluntary."
UUID_TEXT = re.compile(r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}")
BRIEF_RULES = ["REC-01", "REC-02", "REC-03", "REC-04", "REC-05", "REC-08"]


async def _with_graph(connection: AsyncConnection, world: World) -> UUID:
    # Tenant A's active bundle with the synthetic rulebook imported into its graph.
    assert world.a.policy_bundle_id is not None
    async with owner_session(connection, world.a.tenant_id) as db:
        await import_policy_graph(
            db, world.a.tenant_id, world.a.policy_bundle_id, load_synthetic_graph()
        )
    return world.a.policy_bundle_id


async def _creative(
    client: httpx.AsyncClient,
    auth: AuthHeaders,
    tenant_id: UUID,
    study_id: UUID,
    prompt: str = CLEAN_PROMPT,
    caption: str | None = CAPTION,
) -> dict[str, Any]:
    response = await client.post(
        f"{API}/creatives",
        headers=auth("creator-a", tenant_id, **{"Idempotency-Key": uuid4().hex}),
        json={
            "study_id": str(study_id),
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
    world: World,
    creative: dict[str, Any],
    kind: str = "brief",
    key: str | None = None,
) -> str:
    response = await client.post(
        f"{API}/validations",
        headers=auth("creator-a", world.a.tenant_id, **{"Idempotency-Key": key or uuid4().hex}),
        json={
            "creative_id": creative["id"],
            "creative_version": creative["version"],
            "input_kind": kind,
        },
    )
    assert response.status_code == 202, response.text
    run_id: str = response.json()["validation_run_id"]
    return run_id


async def _get(
    client: httpx.AsyncClient, auth: AuthHeaders, path: str, tenant_id: UUID, sub: str = "creator-a"
) -> httpx.Response:
    return await client.get(f"{API}{path}", headers=auth(sub, tenant_id))


async def _run(connection: AsyncConnection, run_id: str) -> Row[UUID, UUID | None]:
    # The stored run's id and context link (a Core query: the connection has no ORM session).
    return (
        await connection.execute(
            select(ValidationRun.id, ValidationRun.context_manifest_id).where(
                ValidationRun.id == UUID(run_id)
            )
        )
    ).one()


# ---- The context stored by POST /validations


async def test_brief_run_stores_lexical_context(
    client: httpx.AsyncClient, auth: AuthHeaders, world: World, connection: AsyncConnection
) -> None:
    await _with_graph(connection, world)
    creative = await _creative(client, auth, world.a.tenant_id, world.a.study_id)

    run_id = await _validate(client, auth, world, creative)

    response = await _get(client, auth, f"/validations/{run_id}/context", world.a.tenant_id)
    assert response.status_code == 200, response.text
    context = response.json()
    assert context["validation_run_id"] == run_id
    assert context["policy_bundle_id"] == str(world.a.policy_bundle_id)
    assert context["study_snapshot_id"] == str(world.a.snapshot_id)
    assert (context["input_kind"], context["channel"], context["language"]) == (
        "brief",
        "facebook",
        "en",
    )
    # Inside a request there's no embedder: full-text only, and the manifest says so.
    assert context["retrieval_mode"] == "lexical_only"
    assert context["embedding_model_id"] is None
    assert {gap["code"] for gap in context["missing_context"]} >= {"SEMANTIC_SEARCH_SKIPPED"}
    assert context["statements"] == [{"span": None}]
    assert [rule["rule_id"] for rule in context["required_rules"]] == BRIEF_RULES
    # The outcome is still the rule checks' only: no gap for the context.
    run = (await _get(client, auth, f"/validations/{run_id}", world.a.tenant_id)).json()
    assert not any("CONTEXT_UNAVAILABLE" in gap["reason"] for gap in run["coverage_gaps"])


async def test_context_shows_no_internal_ids_or_statement_text(
    client: httpx.AsyncClient, auth: AuthHeaders, world: World, connection: AsyncConnection
) -> None:
    await _with_graph(connection, world)
    creative = await _creative(client, auth, world.a.tenant_id, world.a.study_id)
    run_id = await _validate(client, auth, world, creative)

    body = (await _get(client, auth, f"/validations/{run_id}/context", world.a.tenant_id)).text

    # The only ids are the run's, the bundle's and the snapshot's (all public elsewhere).
    public = {run_id, str(world.a.policy_bundle_id), str(world.a.snapshot_id)}
    assert set(UUID_TEXT.findall(body)) == public
    assert "sha256" not in body
    assert CLEAN_PROMPT not in body


async def test_context_finds_evidence_for_the_text(
    client: httpx.AsyncClient, auth: AuthHeaders, world: World, connection: AsyncConnection
) -> None:
    await _with_graph(connection, world)
    creative = await _creative(client, auth, world.a.tenant_id, world.a.study_id, RISKY_PROMPT)
    run_id = await _validate(client, auth, world, creative)

    context = (await _get(client, auth, f"/validations/{run_id}/context", world.a.tenant_id)).json()

    found = {item["node_key"]: item for item in context["evidence"] if item["matches"]}
    assert found, "full-text search found no clause for the risky brief"
    assert all(
        match["found_by"] == ["lexical"] for item in found.values() for match in item["matches"]
    )


async def test_caption_run_uses_the_caption_scope(
    client: httpx.AsyncClient, auth: AuthHeaders, world: World, connection: AsyncConnection
) -> None:
    await _with_graph(connection, world)
    creative = await _creative(client, auth, world.a.tenant_id, world.a.study_id)

    run_id = await _validate(client, auth, world, creative, kind="text")

    context = (await _get(client, auth, f"/validations/{run_id}/context", world.a.tenant_id)).json()
    assert context["input_kind"] == "text"
    required = [rule["rule_id"] for rule in context["required_rules"]]
    assert "REC-07" in required  # The disclosure rule applies to captions, not briefs.
    disclosures = {item["node_key"] for item in context["evidence"]}
    assert {"DISC-01", "DISC-01-ES"} <= disclosures


async def test_empty_caption_stores_graph_only_context(
    client: httpx.AsyncClient, auth: AuthHeaders, world: World, connection: AsyncConnection
) -> None:
    await _with_graph(connection, world)
    creative = await _creative(client, auth, world.a.tenant_id, world.a.study_id, caption=None)

    run_id = await _validate(client, auth, world, creative, kind="text")

    context = (await _get(client, auth, f"/validations/{run_id}/context", world.a.tenant_id)).json()
    assert context["retrieval_mode"] == "graph_only"
    assert context["statements"] == []


async def test_context_does_not_change_the_outcome(
    client: httpx.AsyncClient, auth: AuthHeaders, world: World, connection: AsyncConnection
) -> None:
    # The same risky brief, before and after the graph exists: same outcome, findings and rules.
    before = await _creative(client, auth, world.a.tenant_id, world.a.study_id, RISKY_PROMPT)
    before_id = await _validate(client, auth, world, before)
    await _with_graph(connection, world)
    after = await _creative(client, auth, world.a.tenant_id, world.a.study_id, RISKY_PROMPT)
    after_id = await _validate(client, auth, world, after)

    first = (await _get(client, auth, f"/validations/{before_id}", world.a.tenant_id)).json()
    second = (await _get(client, auth, f"/validations/{after_id}", world.a.tenant_id)).json()

    for field in ("outcome", "gates", "rule_results", "findings"):
        assert first[field] == second[field], field
    assert first["outcome"] == "FAIL"


async def test_bundle_without_graph_stores_run_with_gap(
    client: httpx.AsyncClient, auth: AuthHeaders, world: World, connection: AsyncConnection
) -> None:
    # Tenant A's seeded bundle has JSONB rules only (an older bundle): no graph to retrieve from.
    creative = await _creative(client, auth, world.a.tenant_id, world.a.study_id)

    run_id = await _validate(client, auth, world, creative)

    assert (await _run(connection, run_id)).context_manifest_id is None
    run = (await _get(client, auth, f"/validations/{run_id}", world.a.tenant_id)).json()
    # Unchanged: the context is evidence only, so its absence doesn't change the outcome yet.
    assert run["outcome"] == "PASS"
    gaps = [gap["reason"] for gap in run["coverage_gaps"]]
    assert any(
        reason.startswith("CONTEXT_UNAVAILABLE") and "GRAPH_MISSING" in reason for reason in gaps
    )
    response = await _get(client, auth, f"/validations/{run_id}/context", world.a.tenant_id)
    assert response.status_code == 404
    assert response.json()["code"] == "CONTEXT_NOT_AVAILABLE"


async def test_replay_returns_the_same_run_and_manifest(
    client: httpx.AsyncClient, auth: AuthHeaders, world: World, connection: AsyncConnection
) -> None:
    await _with_graph(connection, world)
    creative = await _creative(client, auth, world.a.tenant_id, world.a.study_id)
    key = uuid4().hex

    first = await _validate(client, auth, world, creative, key=key)
    replay = await _validate(client, auth, world, creative, key=key)

    assert replay == first
    async with as_tenant_owner(connection, world.a.tenant_id):
        assert await count_rows(connection, ContextManifestRecord) == 1
        assert await count_rows(connection, ValidationRun) == 1


async def test_identical_runs_share_one_manifest(
    client: httpx.AsyncClient, auth: AuthHeaders, world: World, connection: AsyncConnection
) -> None:
    # A new key is a new run, but the same inputs reuse the stored manifest.
    await _with_graph(connection, world)
    creative = await _creative(client, auth, world.a.tenant_id, world.a.study_id)

    first = await _run(connection, await _validate(client, auth, world, creative))
    second = await _run(connection, await _validate(client, auth, world, creative))

    assert first.id != second.id
    assert first.context_manifest_id is not None
    assert first.context_manifest_id == second.context_manifest_id


async def test_context_of_another_tenants_run_is_not_found(
    client: httpx.AsyncClient, auth: AuthHeaders, world: World, connection: AsyncConnection
) -> None:
    await _with_graph(connection, world)
    creative = await _creative(client, auth, world.a.tenant_id, world.a.study_id)
    run_id = await _validate(client, auth, world, creative)

    response = await _get(
        client, auth, f"/validations/{run_id}/context", world.b.tenant_id, sub="creator-b"
    )

    assert response.status_code == 404
    assert response.json()["code"] == "NOT_FOUND"


async def test_context_of_unknown_run_is_not_found(
    client: httpx.AsyncClient, auth: AuthHeaders, world: World
) -> None:
    response = await _get(client, auth, f"/validations/{uuid4()}/context", world.a.tenant_id)

    assert response.status_code == 404
    assert response.json()["code"] == "NOT_FOUND"


async def test_context_needs_membership(
    client: httpx.AsyncClient, auth: AuthHeaders, world: World, connection: AsyncConnection
) -> None:
    await _with_graph(connection, world)
    creative = await _creative(client, auth, world.a.tenant_id, world.a.study_id)
    run_id = await _validate(client, auth, world, creative)

    response = await _get(
        client, auth, f"/validations/{run_id}/context", world.a.tenant_id, sub="creator-b"
    )

    assert response.status_code == 403


async def test_reviewer_can_read_context(
    client: httpx.AsyncClient, auth: AuthHeaders, world: World, connection: AsyncConnection
) -> None:
    await _with_graph(connection, world)
    creative = await _creative(client, auth, world.a.tenant_id, world.a.study_id)
    run_id = await _validate(client, auth, world, creative)

    response = await _get(
        client, auth, f"/validations/{run_id}/context", world.a.tenant_id, sub="reviewer-a"
    )

    assert response.status_code == 200


# ---- GET /policy-bundles/{id}/rules


async def test_bundle_rules_show_sources_disclosures_and_exceptions(
    client: httpx.AsyncClient, auth: AuthHeaders, world: World, connection: AsyncConnection
) -> None:
    bundle_id = await _with_graph(connection, world)

    response = await _get(client, auth, f"/policy-bundles/{bundle_id}/rules", world.a.tenant_id)

    assert response.status_code == 200, response.text
    body = response.json()
    assert (body["policy_bundle_id"], body["lifecycle_state"], body["is_synthetic"]) == (
        str(bundle_id),
        "ACTIVE",
        True,
    )
    rules = {rule["rule_id"]: rule for rule in body["rules"]}
    assert list(rules) == sorted(rules)
    assert len(rules) == 9
    rec01 = rules["REC-01"]
    assert [(s["clause_ref"], s["language"]) for s in rec01["sources"]] == [
        ("FDA-1", "en"),
        ("FDA-1-ES", "es"),
    ]
    assert rec01["prohibited_phrases"] == ["safe and effective", "proven", "cure", "cures"]
    assert rec01["applicability"]["input_kinds"] == ["brief", "text", "image"]
    assert rec01["applicability"]["study_specific"] is False
    assert [(d["node_key"], d["language"]) for d in rules["REC-07"]["disclosures"]] == [
        ("DISC-01", "en"),
        ("DISC-01-ES", "es"),
    ]
    # Exceptions are listed one hop from the rule; EXC-02 is an exception to EXC-01.
    assert [(e["node_key"], e["requires_study_fact"]) for e in rules["REC-06"]["exceptions"]] == [
        ("EXC-01", "compensation_statement")
    ]
    # REC-08 is supported by an interpretation, which quotes no clause.
    assert rules["REC-08"]["sources"] == []


async def test_bundle_rules_show_no_internal_ids(
    client: httpx.AsyncClient, auth: AuthHeaders, world: World, connection: AsyncConnection
) -> None:
    bundle_id = await _with_graph(connection, world)

    body = (await _get(client, auth, f"/policy-bundles/{bundle_id}/rules", world.a.tenant_id)).text

    assert set(UUID_TEXT.findall(body)) == {str(bundle_id)}


def test_graph_rule_ids_match_the_stored_rules() -> None:
    # Until NRC-7 judges with the context, the outcome comes from the JSONB rules and the
    # evidence from the graph: the synthetic rulebook must describe the same rules.
    graph = load_synthetic_graph()

    assert sorted((r.rule_id, r.rule_version) for r in graph.rules) == sorted(
        (str(rule["rule_id"]), int(str(rule["rule_version"]))) for rule in demo_policy_rules()
    )


async def test_rules_of_a_bundle_without_graph_are_unavailable(
    client: httpx.AsyncClient, auth: AuthHeaders, world: World
) -> None:
    response = await _get(
        client, auth, f"/policy-bundles/{world.a.policy_bundle_id}/rules", world.a.tenant_id
    )

    assert response.status_code == 409
    assert response.json()["code"] == "POLICY_CONTEXT_UNAVAILABLE"
    assert response.json()["details"] == {"reason": "GRAPH_MISSING"}


async def test_rules_of_a_draft_bundle_are_not_found(
    client: httpx.AsyncClient, auth: AuthHeaders, world: World, connection: AsyncConnection
) -> None:
    async with as_tenant_owner(connection, world.a.tenant_id):
        draft = await add_policy_bundle(
            connection, world.a.tenant_id, 2, BundleLifecycleState.DRAFT
        )

    response = await _get(client, auth, f"/policy-bundles/{draft}/rules", world.a.tenant_id)

    assert response.status_code == 404
    assert response.json()["code"] == "NOT_FOUND"


async def test_rules_of_another_tenants_bundle_are_not_found(
    client: httpx.AsyncClient, auth: AuthHeaders, world: World, connection: AsyncConnection
) -> None:
    bundle_id = await _with_graph(connection, world)

    response = await _get(
        client, auth, f"/policy-bundles/{bundle_id}/rules", world.b.tenant_id, sub="creator-b"
    )

    assert response.status_code == 404


async def test_rules_need_membership(
    client: httpx.AsyncClient, auth: AuthHeaders, world: World, connection: AsyncConnection
) -> None:
    bundle_id = await _with_graph(connection, world)

    response = await _get(
        client, auth, f"/policy-bundles/{bundle_id}/rules", world.a.tenant_id, sub="creator-b"
    )

    assert response.status_code == 403


async def test_unapproved_rules_are_hidden(
    client: httpx.AsyncClient, auth: AuthHeaders, world: World, connection: AsyncConnection
) -> None:
    bundle_id = await _with_graph(connection, world)
    async with as_tenant_owner(connection, world.a.tenant_id):
        await connection.execute(
            text(
                "UPDATE policy_nodes SET approved = false"
                " WHERE policy_bundle_id = :b AND node_key IN ('REC-09', 'DISC-01-ES')"
            ),
            {"b": bundle_id},
        )

    body = (
        await _get(client, auth, f"/policy-bundles/{bundle_id}/rules", world.a.tenant_id)
    ).json()

    rules = {rule["rule_id"]: rule for rule in body["rules"]}
    assert "REC-09" not in rules
    assert [d["node_key"] for d in rules["REC-07"]["disclosures"]] == ["DISC-01"]
