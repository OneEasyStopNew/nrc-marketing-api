"""Policy source and bundle endpoints (NRC-4, step 9) through the HTTP client with real tokens.

The services are tested in test_policy_sources.py and test_policy_lifecycle.py; these tests check
what the routes add: status codes, ETags, bodies, capabilities and tenant isolation.
"""

from collections.abc import Awaitable, Callable
from datetime import UTC, datetime, timedelta
from typing import Any
from uuid import UUID, uuid4

import httpx
import pytest
from sqlalchemy.ext.asyncio import AsyncConnection

from app.core.config import Settings, get_settings
from app.models.enums import (
    BundleLifecycleState,
    CheckType,
    Gate,
    PolicyEdgeType,
    Severity,
    UploadState,
    WordingKind,
)
from app.models.policy_graph import PROHIBITED_PHRASE_MAX_CHARS
from tests.conftest import (
    AuthHeaders,
    World,
    add_policy_bundle,
    add_policy_upload,
    owner_sql,
)

BASE = "/v1/marketing"
ADMIN, APPROVER, B_ADMIN = "policy-admin-a", "policy-approver-a", "policy-admin-b"
CLAUSE_TEXT = "[SYNTHETIC] Every post must say: This is a research study. Taking part is voluntary."
WORDING = "This is a research study. Taking part is voluntary."


@pytest.fixture
def seed_graph() -> bool:
    # These tests count or write their own sources, graph rows and wordings in tenant A, so its
    # bundle starts without NRC-5's synthetic graph.
    return False


@pytest.fixture(autouse=True)
async def staff(connection: AsyncConnection, world: World) -> None:
    members = [
        (world.a.tenant_id, ADMIN, "policy_admin"),
        (world.a.tenant_id, APPROVER, "policy_approver"),
        (world.b.tenant_id, B_ADMIN, "policy_admin"),
    ]
    for tenant_id, sub, role in members:
        await owner_sql(
            connection,
            tenant_id,
            "INSERT INTO memberships (tenant_id, user_sub, roles) VALUES (:t, :sub, ARRAY[:role])",
            {"t": tenant_id, "sub": sub, "role": role},
        )


class Api:
    """The caller's view: one tenant, any user."""

    def __init__(self, client: httpx.AsyncClient, auth: AuthHeaders, tenant_id: UUID) -> None:
        self.client = client
        self.auth = auth
        self.tenant_id = tenant_id

    async def call(
        self,
        method: str,
        path: str,
        sub: str = ADMIN,
        *,
        json: Any = None,
        if_match: int | str | None = None,
        params: dict[str, Any] | None = None,
    ) -> httpx.Response:
        headers = self.auth(sub, self.tenant_id)
        if if_match is not None:
            headers["If-Match"] = if_match if isinstance(if_match, str) else f'"{if_match}"'
        return await self.client.request(
            method, BASE + path, headers=headers, json=json, params=params
        )

    async def ok(
        self, response_call: Awaitable[httpx.Response], status: int = 200
    ) -> httpx.Response:
        response = await response_call
        assert response.status_code == status, response.text
        return response


@pytest.fixture
def api(client: httpx.AsyncClient, auth: AuthHeaders, world: World) -> Api:
    return Api(client, auth, world.a.tenant_id)


async def new_bundle(api: Api, **changes: Any) -> dict[str, Any]:
    body = {"name": "[SYNTHETIC] Policy", "source_ref": "synthetic:nrc-4-api", "is_synthetic": True}
    response = await api.ok(api.call("POST", "/policy-bundles", json=body | changes), 201)
    created: dict[str, Any] = response.json()
    return created


def node_body(clause_id: UUID | None = None, key: str = "CL-1") -> dict[str, Any]:
    return {
        "node_key": key,
        "node_type": "CLAUSE",
        "text": CLAUSE_TEXT,
        "source_clause_id": str(clause_id) if clause_id else None,
    }


RULE_BODY: dict[str, Any] = {
    "rule_id": "REC-07",
    "title": "Research and voluntary disclosure",
    "requirement": "Say that this is a research study and that taking part is voluntary.",
    "gate": Gate.COMPLETENESS.value,
    "severity": Severity.CRITICAL.value,
    "mandatory": True,
    "check_type": CheckType.DETERMINISTIC.value,
    "text": "[SYNTHETIC] Say that this is a research study.",
}
WORDING_BODY: dict[str, Any] = {"kind": WordingKind.EXACT_WORDING.value, "text": WORDING}


async def add_clause(connection: AsyncConnection, tenant_id: UUID, source_id: str) -> UUID:
    # What the policy worker does after extraction (tested in test_policy_extraction.py).
    clause_id = uuid4()
    await owner_sql(
        connection,
        tenant_id,
        "INSERT INTO policy_source_clauses (id, tenant_id, source_id, clause_ref, section, page,"
        " text, checksum_sha256, language) VALUES (:id, :t, :s, 'p1-c1', 'Disclosures', 1, :text,"
        " :sum, 'en')",
        {
            "id": clause_id,
            "t": tenant_id,
            "s": UUID(source_id),
            "text": CLAUSE_TEXT,
            "sum": uuid4().hex * 2,
        },
    )
    return clause_id


async def draft_graph(
    api: Api, bundle_id: str, clause_id: UUID, revision: int = 1
) -> dict[str, Any]:
    """Clause node, rule, edge and wording; returns their ids and the next revision."""
    path = f"/policy-bundles/{bundle_id}"
    node = (
        await api.ok(
            api.call("POST", f"{path}/nodes", json=node_body(clause_id), if_match=revision), 201
        )
    ).json()
    rule = (
        await api.ok(api.call("POST", f"{path}/rules", json=RULE_BODY, if_match=revision + 1), 201)
    ).json()
    edge_body = {
        "edge_type": PolicyEdgeType.SUPPORTED_BY.value,
        "from_node_id": rule["node"]["id"],
        "to_node_id": node["id"],
        "provenance_clause_id": str(clause_id),
    }
    edge = (
        await api.ok(api.call("POST", f"{path}/edges", json=edge_body, if_match=revision + 2), 201)
    ).json()
    wording = (
        await api.ok(
            api.call(
                "POST",
                f"{path}/rules/{rule['id']}/wordings",
                json=WORDING_BODY,
                if_match=revision + 3,
            ),
            201,
        )
    ).json()
    return {
        "clause_node": node["id"],
        "rule": rule["id"],
        "rule_node": rule["node"]["id"],
        "edge": edge["id"],
        "wording": wording["id"],
        "revision": revision + 4,
    }


async def review_all(api: Api, bundle_id: str, graph: dict[str, Any]) -> None:
    items = [
        ("node", graph["clause_node"]),
        ("node", graph["rule_node"]),
        ("edge", graph["edge"]),
        ("wording", graph["wording"]),
    ]
    for kind, item_id in items:
        body = {"item_kind": kind, "item_id": item_id, "decision": "APPROVED"}
        response = await api.ok(
            api.call(
                "POST",
                f"/policy-bundles/{bundle_id}/review-decisions",
                json=body,
                if_match=graph["revision"],
            )
        )
        graph["revision"] += 1
        assert response.headers["ETag"] == f'"{graph["revision"]}"'


async def register_source(api: Api, connection: AsyncConnection, **changes: Any) -> httpx.Response:
    upload_id = await add_policy_upload(connection, api.tenant_id)
    body = {
        "upload_id": str(upload_id),
        "title": "[SYNTHETIC] Policy source",
        "source_ref": "synthetic:nrc-4-api",
        "language": "en",
        "is_synthetic": True,
    }
    return await api.call("POST", "/policy-sources", json=body | changes)


async def spread_audit(
    connection: AsyncConnection, tenant_id: UUID, bundle_id: str, base: datetime
) -> None:
    # One test transaction gives every row the same created_at: spread them by revision, as
    # separate requests would.
    await owner_sql(
        connection,
        tenant_id,
        "UPDATE policy_audit_events SET created_at = CAST(:base AS timestamptz)"
        " + (details->>'revision')::int * interval '1 minute' WHERE policy_bundle_id = :id",
        {"base": base, "id": UUID(bundle_id)},
    )


# ---- Happy path


async def test_a_bundle_goes_from_upload_to_active(
    api: Api, connection: AsyncConnection, world: World
) -> None:
    created = await register_source(api, connection)
    assert created.status_code == 201, created.text
    source = created.json()
    assert created.headers["Location"] == f"{BASE}/policy-sources/{source['id']}"
    assert source["extraction_state"] == "PENDING"
    assert source["is_synthetic"] is True
    assert source["created_by"] == ADMIN
    # Registering the same upload again returns the same source (AC3).
    again = await api.call(
        "POST",
        "/policy-sources",
        json={
            "upload_id": source["upload_id"],
            "title": "[SYNTHETIC] Other title",
            "source_ref": "synthetic:other",
            "is_synthetic": True,
        },
    )
    assert again.status_code == 200, again.text
    assert again.json()["id"] == source["id"]
    assert again.headers["Location"] == created.headers["Location"]
    listed = (await api.ok(api.call("GET", "/policy-sources", APPROVER))).json()
    assert [item["id"] for item in listed["items"]] == [source["id"]]
    fetched = (await api.ok(api.call("GET", f"/policy-sources/{source['id']}"))).json()
    assert fetched == source

    clause_id = await add_clause(connection, world.a.tenant_id, source["id"])
    clauses = (await api.ok(api.call("GET", f"/policy-sources/{source['id']}/clauses"))).json()[
        "items"
    ]
    assert [(c["id"], c["clause_ref"], c["text"]) for c in clauses] == [
        (str(clause_id), "p1-c1", CLAUSE_TEXT)
    ]

    bundle = await new_bundle(api)
    bundle_id = bundle["id"]
    assert bundle["lifecycle_state"] == "DRAFT"
    assert bundle["revision"] == 1
    assert bundle["rule_count"] == 0
    # The world's tenant already has version 1 (its demo bundle).
    assert bundle["version"] == 2
    path = f"/policy-bundles/{bundle_id}"
    got = await api.ok(api.call("GET", path))
    assert got.headers["ETag"] == '"1"'

    graph = await draft_graph(api, bundle_id, clause_id)
    assert graph["revision"] == 5
    await review_all(api, bundle_id, graph)

    checks = (await api.ok(api.call("GET", f"{path}/checks", APPROVER))).json()
    assert checks == {"issues": []}
    nodes = (await api.ok(api.call("GET", f"{path}/nodes"))).json()["items"]
    assert sorted(n["node_key"] for n in nodes) == ["CL-1", "REC-07"]
    assert {n["review_state"] for n in nodes} == {"APPROVED"}
    assert {n["origin"] for n in nodes} == {"HUMAN"}
    edges = (await api.ok(api.call("GET", f"{path}/edges"))).json()["items"]
    assert [e["edge_type"] for e in edges] == ["supported_by"]
    rules = (await api.ok(api.call("GET", f"{path}/rule-records"))).json()["items"]
    assert [r["rule_id"] for r in rules] == ["REC-07"]
    assert [w["text"] for w in rules[0]["wordings"]] == [WORDING]

    transitions = f"{path}/transitions"
    submitted = await api.ok(
        api.call("POST", transitions, json={"command": "submit"}, if_match=graph["revision"])
    )
    assert submitted.json()["lifecycle_state"] == "REVIEW"
    assert submitted.json()["rule_count"] == 1
    revision = submitted.json()["revision"]
    assert submitted.headers["ETag"] == f'"{revision}"'
    approved = await api.ok(
        api.call("POST", transitions, APPROVER, json={"command": "approve"}, if_match=revision)
    )
    assert approved.json()["lifecycle_state"] == "APPROVED"
    assert approved.json()["approved_by"] == APPROVER

    # What the build worker does (test_policy_build_worker.py), as the owner.
    await owner_sql(
        connection,
        world.a.tenant_id,
        "UPDATE policy_bundles SET lifecycle_state = 'READY', available_at = NULL WHERE id = :id",
        {"id": UUID(bundle_id)},
    )
    ready = await api.ok(api.call("GET", path, APPROVER))
    assert ready.json()["lifecycle_state"] == "READY"
    activated = await api.ok(
        api.call(
            "POST",
            transitions,
            ADMIN,
            json={"command": "activate", "reason": "[SYNTHETIC] Reviewed sample."},
            if_match=ready.headers["ETag"],
        )
    )
    assert activated.json()["lifecycle_state"] == "ACTIVE"
    assert activated.json()["activated_by"] == ADMIN

    active = (await api.ok(api.call("GET", "/policy-bundles", params={"state": "ACTIVE"}))).json()
    assert [item["id"] for item in active["items"]] == [bundle_id]
    assert active["next_cursor"] is None
    everything = (await api.ok(api.call("GET", "/policy-bundles"))).json()["items"]
    assert [item["version"] for item in everything] == [2, 1]
    assert everything[1]["lifecycle_state"] == "RETIRED"

    await spread_audit(connection, world.a.tenant_id, bundle_id, datetime(2026, 11, 20, tzinfo=UTC))
    audit = (await api.ok(api.call("GET", f"{path}/audit"))).json()["items"]
    actions = {item["action"] for item in audit}
    assert {"CREATE", "EDIT", "REVIEW", "SUBMIT", "APPROVE", "ACTIVATE"} <= actions
    assert audit[0]["action"] == "ACTIVATE"
    assert audit[0]["actor"] == ADMIN
    assert audit[0]["reason"] == "[SYNTHETIC] Reviewed sample."
    assert audit[0]["from_state"] == "READY"
    assert audit[0]["to_state"] == "ACTIVE"
    assert audit[0]["request_id"]


async def test_a_source_needs_a_ready_upload(api: Api, connection: AsyncConnection) -> None:
    upload_id = await add_policy_upload(connection, api.tenant_id, state=UploadState.QUARANTINED)

    response = await api.call(
        "POST",
        "/policy-sources",
        json={
            "upload_id": str(upload_id),
            "title": "[SYNTHETIC] x",
            "source_ref": "synthetic:x",
            "is_synthetic": True,
        },
    )

    assert response.status_code == 409
    assert response.json()["code"] == "POLICY_SOURCE_UPLOAD_NOT_READY"


# ---- Authentication and capabilities

SOME_ID = "6f1c2b7e-3d49-4a85-9c0e-1b2d3c4e5f60"
WRITE_BODIES: dict[str, Any] = {
    "node": node_body(),
    "rule": RULE_BODY,
    "edge": {
        "edge_type": "supported_by",
        "from_node_id": SOME_ID,
        "to_node_id": str(uuid4()),
        "provenance_note": "[SYNTHETIC] note",
    },
    "wording": WORDING_BODY,
    "review": {"item_kind": "node", "item_id": SOME_ID, "decision": "APPROVED"},
    "submit": {"command": "submit"},
}
# (method, path, body) for every policy route; ids are placeholders.
ROUTES: list[tuple[str, str, Any]] = [
    ("GET", "/policy-sources", None),
    (
        "POST",
        "/policy-sources",
        {"upload_id": SOME_ID, "title": "x", "source_ref": "x", "is_synthetic": True},
    ),
    ("GET", f"/policy-sources/{SOME_ID}", None),
    ("GET", f"/policy-sources/{SOME_ID}/clauses", None),
    ("POST", "/policy-bundles", {"name": "x", "source_ref": "x", "is_synthetic": True}),
    ("GET", "/policy-bundles", None),
    ("GET", f"/policy-bundles/{SOME_ID}", None),
    ("GET", f"/policy-bundles/{SOME_ID}/rule-records", None),
    ("GET", f"/policy-bundles/{SOME_ID}/nodes", None),
    ("GET", f"/policy-bundles/{SOME_ID}/edges", None),
    ("GET", f"/policy-bundles/{SOME_ID}/checks", None),
    ("GET", f"/policy-bundles/{SOME_ID}/audit", None),
    ("POST", f"/policy-bundles/{SOME_ID}/nodes", WRITE_BODIES["node"]),
    ("PATCH", f"/policy-bundles/{SOME_ID}/nodes/{SOME_ID}", {}),
    ("POST", f"/policy-bundles/{SOME_ID}/rules", WRITE_BODIES["rule"]),
    ("PATCH", f"/policy-bundles/{SOME_ID}/rules/{SOME_ID}", {}),
    ("POST", f"/policy-bundles/{SOME_ID}/edges", WRITE_BODIES["edge"]),
    ("POST", f"/policy-bundles/{SOME_ID}/rules/{SOME_ID}/wordings", WRITE_BODIES["wording"]),
    ("POST", f"/policy-bundles/{SOME_ID}/review-decisions", WRITE_BODIES["review"]),
    ("POST", f"/policy-bundles/{SOME_ID}/transitions", WRITE_BODIES["submit"]),
]


@pytest.mark.parametrize(("method", "path", "body"), ROUTES)
async def test_every_policy_route_needs_a_token(
    client: httpx.AsyncClient, method: str, path: str, body: Any
) -> None:
    response = await client.request(method, BASE + path, json=body)

    assert response.status_code == 401, response.text
    assert response.json()["code"] == "UNAUTHENTICATED"


@pytest.mark.parametrize(("method", "path", "body"), ROUTES)
@pytest.mark.parametrize("sub", ["creator-a", "reviewer-a"])
async def test_roles_without_a_policy_capability_are_refused(
    api: Api, method: str, path: str, body: Any, sub: str
) -> None:
    response = await api.call(method, path, sub, json=body, if_match=1)

    assert response.status_code == 403, response.text
    assert response.json()["code"] == "FORBIDDEN"


# D-16: approval and activation are separate permissions, each refused to the other role.
@pytest.mark.parametrize("command", ["approve", "reject"])
async def test_a_policy_admin_cannot_approve(api: Api, command: str) -> None:
    bundle = await new_bundle(api)

    response = await api.call(
        "POST",
        f"/policy-bundles/{bundle['id']}/transitions",
        json={"command": command, "reason": "[SYNTHETIC] reason"},
        if_match=1,
    )

    assert response.status_code == 403, response.text
    assert "cannot do this" in response.json()["message"]


@pytest.mark.parametrize("command", ["build", "activate", "rollback", "retire"])
async def test_an_approver_cannot_activate(api: Api, command: str) -> None:
    bundle = await new_bundle(api)

    response = await api.call(
        "POST",
        f"/policy-bundles/{bundle['id']}/transitions",
        APPROVER,
        json={"command": command, "reason": "[SYNTHETIC] reason"},
        if_match=1,
    )

    assert response.status_code == 403, response.text
    assert "cannot do this" in response.json()["message"]


async def test_an_approver_cannot_edit_or_register(api: Api, connection: AsyncConnection) -> None:
    bundle = await new_bundle(api)
    path = f"/policy-bundles/{bundle['id']}"

    attempts = [
        api.call("POST", "/policy-bundles", APPROVER, json=ROUTES[4][2]),
        api.call("POST", f"{path}/nodes", APPROVER, json=node_body(), if_match=1),
        api.call("POST", f"{path}/rules", APPROVER, json=RULE_BODY, if_match=1),
        api.call("POST", f"{path}/transitions", APPROVER, json={"command": "submit"}, if_match=1),
    ]
    # Which items an approver may review depends on the item (D-16): see the review tests in
    # test_policy_lifecycle.py, which use real items.
    for attempt in attempts:
        response = await attempt
        assert response.status_code == 403, response.text
    registered = await register_source(api, connection)
    assert registered.status_code == 201
    denied = await api.call("POST", "/policy-sources", APPROVER, json=ROUTES[1][2])
    assert denied.status_code == 403


async def test_an_approver_can_return_a_bundle_in_review_to_draft(api: Api) -> None:
    bundle = await new_bundle(api)
    path = f"/policy-bundles/{bundle['id']}/transitions"
    await api.ok(api.call("POST", path, json={"command": "submit"}, if_match=1))

    returned = await api.ok(
        api.call("POST", path, APPROVER, json={"command": "return_to_draft"}, if_match=2)
    )

    assert returned.json()["lifecycle_state"] == "DRAFT"
    assert returned.headers["ETag"] == '"3"'


# ---- Tenant isolation


@pytest.fixture
async def other_tenant(connection: AsyncConnection, world: World) -> dict[str, str]:
    bundle_id = await add_policy_bundle(
        connection, world.b.tenant_id, version=2, state=BundleLifecycleState.DRAFT
    )
    upload_id = await add_policy_upload(connection, world.b.tenant_id)
    source_id = uuid4()
    await owner_sql(
        connection,
        world.b.tenant_id,
        "INSERT INTO policy_sources (id, tenant_id, title, source_ref, checksum_sha256, language,"
        " is_synthetic, created_by) VALUES (:id, :t, 'x', 'x', :sum, 'en', true, 'x')",
        {"id": source_id, "t": world.b.tenant_id, "sum": uuid4().hex * 2},
    )
    return {"bundle": str(bundle_id), "source": str(source_id), "upload": str(upload_id)}


@pytest.mark.parametrize(
    "template",
    [
        "/policy-sources/{source}",
        "/policy-sources/{source}/clauses",
        "/policy-bundles/{bundle}",
        "/policy-bundles/{bundle}/rule-records",
        "/policy-bundles/{bundle}/nodes",
        "/policy-bundles/{bundle}/edges",
        "/policy-bundles/{bundle}/checks",
        "/policy-bundles/{bundle}/audit",
    ],
)
async def test_another_tenants_reads_are_not_found(
    api: Api, other_tenant: dict[str, str], template: str
) -> None:
    response = await api.call("GET", template.format(**other_tenant))

    assert response.status_code == 404, response.text
    assert response.json()["code"] == "NOT_FOUND"


@pytest.mark.parametrize(
    ("method", "template", "body"),
    [
        ("POST", "/policy-bundles/{bundle}/nodes", WRITE_BODIES["node"]),
        ("PATCH", "/policy-bundles/{bundle}/nodes/" + SOME_ID, {}),
        ("POST", "/policy-bundles/{bundle}/rules", WRITE_BODIES["rule"]),
        ("PATCH", "/policy-bundles/{bundle}/rules/" + SOME_ID, {}),
        ("POST", "/policy-bundles/{bundle}/edges", WRITE_BODIES["edge"]),
        (
            "POST",
            "/policy-bundles/{bundle}/rules/" + SOME_ID + "/wordings",
            WRITE_BODIES["wording"],
        ),
        ("POST", "/policy-bundles/{bundle}/review-decisions", WRITE_BODIES["review"]),
        ("POST", "/policy-bundles/{bundle}/transitions", WRITE_BODIES["submit"]),
    ],
)
async def test_another_tenants_writes_are_not_found(
    api: Api, other_tenant: dict[str, str], method: str, template: str, body: Any
) -> None:
    response = await api.call(method, template.format(**other_tenant), json=body, if_match=1)

    assert response.status_code == 404, response.text
    assert response.json()["code"] == "NOT_FOUND"


async def test_another_tenants_upload_and_base_bundle_are_not_found(
    api: Api, other_tenant: dict[str, str]
) -> None:
    source = await api.call(
        "POST",
        "/policy-sources",
        json={
            "upload_id": other_tenant["upload"],
            "title": "x",
            "source_ref": "x",
            "is_synthetic": True,
        },
    )
    copy = await api.call(
        "POST",
        "/policy-bundles",
        json={
            "name": "x",
            "source_ref": "x",
            "is_synthetic": True,
            "based_on_bundle_id": other_tenant["bundle"],
        },
    )

    assert source.status_code == 404, source.text
    assert copy.status_code == 404, copy.text


# ---- If-Match


@pytest.mark.parametrize("header", [None, "3", 'W/"3"', '"0"', '"1","2"'])
async def test_edits_need_one_quoted_version(api: Api, header: str | None) -> None:
    bundle = await new_bundle(api)
    headers = api.auth(ADMIN, api.tenant_id)
    if header is not None:
        headers["If-Match"] = header

    response = await api.client.post(
        f"{BASE}/policy-bundles/{bundle['id']}/nodes", headers=headers, json=node_body()
    )

    assert response.status_code == 428, response.text
    assert response.json()["code"] == "PRECONDITION_REQUIRED"
    assert response.headers.get("ETag") is None


async def test_a_stale_version_is_refused_with_the_current_one(api: Api) -> None:
    bundle = await new_bundle(api)
    path = f"/policy-bundles/{bundle['id']}"
    await api.ok(api.call("POST", f"{path}/nodes", json=node_body(), if_match=1), 201)

    stale = await api.call("POST", f"{path}/nodes", json=node_body(key="CL-2"), if_match=1)

    assert stale.status_code == 412, stale.text
    assert stale.json()["code"] == "VERSION_CONFLICT"
    assert stale.json()["details"] == {"current_version": 2}
    transition = await api.call(
        "POST", f"{path}/transitions", json={"command": "submit"}, if_match=1
    )
    assert transition.status_code == 412
    assert transition.json()["details"] == {"current_version": 2}


async def test_transitions_need_if_match(api: Api) -> None:
    bundle = await new_bundle(api)

    response = await api.call(
        "POST", f"/policy-bundles/{bundle['id']}/transitions", json={"command": "submit"}
    )

    assert response.status_code == 428, response.text


# ---- Conflicts


async def test_a_submitted_bundle_cannot_be_edited(api: Api) -> None:
    bundle = await new_bundle(api)
    path = f"/policy-bundles/{bundle['id']}"
    await api.ok(api.call("POST", f"{path}/transitions", json={"command": "submit"}, if_match=1))

    response = await api.call("POST", f"{path}/nodes", json=node_body(), if_match=2)

    assert response.status_code == 409, response.text
    assert response.json()["code"] == "POLICY_BUNDLE_STATE"
    assert response.json()["details"] == {"state": "REVIEW", "allowed": ["DRAFT"]}


async def test_activating_a_bundle_that_is_not_ready_is_a_state_conflict(api: Api) -> None:
    bundle = await new_bundle(api)

    response = await api.call(
        "POST",
        f"/policy-bundles/{bundle['id']}/transitions",
        ADMIN,
        json={"command": "activate", "reason": "[SYNTHETIC] reason"},
        if_match=1,
    )

    assert response.status_code == 409, response.text
    assert response.json()["code"] == "POLICY_BUNDLE_STATE"
    assert response.json()["details"] == {"state": "DRAFT", "allowed": ["READY"]}


async def test_approving_a_bundle_with_issues_lists_them(api: Api) -> None:
    bundle = await new_bundle(api)
    path = f"/policy-bundles/{bundle['id']}"
    await api.ok(api.call("POST", f"{path}/transitions", json={"command": "submit"}, if_match=1))

    response = await api.call(
        "POST", f"{path}/transitions", APPROVER, json={"command": "approve"}, if_match=2
    )

    assert response.status_code == 409, response.text
    body = response.json()
    assert body["code"] == "POLICY_BUNDLE_HAS_ISSUES"
    assert [issue["code"] for issue in body["details"]["issues"]] == ["EMPTY_BUNDLE"]
    assert set(body["details"]["issues"][0]) == {"code", "subject", "message"}
    checks = (await api.ok(api.call("GET", f"{path}/checks"))).json()
    assert checks == {"issues": body["details"]["issues"]}


async def test_a_reason_is_required_to_reject(api: Api) -> None:
    bundle = await new_bundle(api)
    path = f"/policy-bundles/{bundle['id']}/transitions"
    await api.ok(api.call("POST", path, json={"command": "submit"}, if_match=1))

    response = await api.call("POST", path, APPROVER, json={"command": "reject"}, if_match=2)

    assert response.status_code == 422, response.text
    assert response.json()["code"] == "VALIDATION_FAILED"
    (field,) = response.json()["details"]["fields"]
    assert "reject needs a reason" in field["msg"]


@pytest.mark.parametrize("command", ["activate", "rollback", "retire"])
async def test_a_reason_is_required_for_every_command_that_changes_the_active_bundle(
    api: Api, command: str
) -> None:
    # Refused from the body alone, before the bundle's state is looked at.
    bundle = await new_bundle(api)

    response = await api.call(
        "POST",
        f"/policy-bundles/{bundle['id']}/transitions",
        ADMIN,
        json={"command": command},
        if_match=1,
    )

    assert response.status_code == 422, response.text
    assert f"{command} needs a reason" in response.json()["details"]["fields"][0]["msg"]


@pytest.fixture
def adjustable_settings(
    app_overrides: dict[Callable[..., object], Callable[..., object]],
) -> dict[str, Settings]:
    # The client copies app_overrides when it starts, so a test changes settings through this.
    current = {"settings": get_settings()}
    app_overrides[get_settings] = lambda: current["settings"]
    return current


@pytest.fixture
def tight_policy_caps(adjustable_settings: dict[str, Settings]) -> None:
    adjustable_settings["settings"] = adjustable_settings["settings"].model_copy(
        update={"policy_max_open_bundles_per_tenant": 2, "policy_max_items_per_bundle": 10}
    )


async def fill_with_nodes(
    api: Api, connection: AsyncConnection, world: World, count: int
) -> tuple[str, UUID, list[dict[str, Any]]]:
    # A draft holding `count` clause nodes; returns its path, the cited clause and the nodes.
    # The bundle's revision is then count + 1.
    source = (await api.ok(register_source(api, connection), 201)).json()
    clause_id = await add_clause(connection, world.a.tenant_id, source["id"])
    bundle = await new_bundle(api)
    path = f"/policy-bundles/{bundle['id']}"
    nodes = []
    for index in range(count):
        body = node_body(clause_id, key=f"CL-{index}")
        created = await api.ok(
            api.call("POST", f"{path}/nodes", json=body, if_match=index + 1), 201
        )
        nodes.append(created.json())
    return path, clause_id, nodes


def assert_cap_reached(response: httpx.Response) -> None:
    assert response.status_code == 409, response.text
    assert response.json()["code"] == "POLICY_LIMIT_REACHED"
    assert response.json()["details"] == {"limit": 10}


@pytest.mark.usefixtures("tight_policy_caps")
async def test_open_bundles_are_capped_per_tenant(api: Api) -> None:
    # Regression (review lesson from api#23): nothing bounded how many drafts a tenant could make.
    await new_bundle(api)
    at_the_cap = await new_bundle(api)

    over = await api.call(
        "POST",
        "/policy-bundles",
        json={"name": "[SYNTHETIC] x", "source_ref": "synthetic:x", "is_synthetic": True},
    )

    assert at_the_cap["lifecycle_state"] == "DRAFT"
    assert over.status_code == 409, over.text
    assert over.json()["code"] == "POLICY_LIMIT_REACHED"
    assert over.json()["details"] == {"limit": 2}


@pytest.mark.usefixtures("tight_policy_caps")
async def test_items_per_bundle_are_capped(
    api: Api, connection: AsyncConnection, world: World
) -> None:
    path, clause_id, _ = await fill_with_nodes(api, connection, world, 10)

    over = await api.call(
        "POST", f"{path}/nodes", json=node_body(clause_id, key="CL-X"), if_match=11
    )

    assert_cap_reached(over)


def _decision(node: dict[str, Any], decision: str) -> dict[str, Any]:
    return {"item_kind": "node", "item_id": node["id"], "decision": decision}


@pytest.mark.usefixtures("tight_policy_caps")
async def test_rejected_items_do_not_use_up_the_item_cap(
    api: Api, connection: AsyncConnection, world: World
) -> None:
    # Regression (code review): rejected rows can't be deleted, so counting them left a full
    # draft unable to take any correction.
    path, clause_id, nodes = await fill_with_nodes(api, connection, world, 10)
    rejection = _decision(nodes[0], "REJECTED")
    await api.ok(api.call("POST", f"{path}/review-decisions", json=rejection, if_match=11))

    replacement = await api.call(
        "POST", f"{path}/nodes", json=node_body(clause_id, key="CL-NEW"), if_match=12
    )

    assert replacement.status_code == 201, replacement.text


@pytest.mark.usefixtures("tight_policy_caps")
async def test_a_rejected_item_cannot_come_back_over_the_cap(
    api: Api, connection: AsyncConnection, world: World
) -> None:
    # Regression (altitude review): approving or editing a REJECTED item makes it count again,
    # so add-reject-add-revive would otherwise pass the cap without limit.
    path, clause_id, nodes = await fill_with_nodes(api, connection, world, 10)
    await api.ok(
        api.call(
            "POST", f"{path}/review-decisions", json=_decision(nodes[0], "REJECTED"), if_match=11
        )
    )
    await api.ok(
        api.call("POST", f"{path}/nodes", json=node_body(clause_id, key="CL-NEW"), if_match=12),
        201,
    )

    approved = await api.call(
        "POST", f"{path}/review-decisions", json=_decision(nodes[0], "APPROVED"), if_match=13
    )
    edited = await api.call(
        "PATCH", f"{path}/nodes/{nodes[0]['id']}", json={"channels": ["facebook"]}, if_match=13
    )
    rejected_again = await api.call(
        "POST", f"{path}/review-decisions", json=_decision(nodes[1], "REJECTED"), if_match=13
    )

    assert_cap_reached(approved)
    assert_cap_reached(edited)
    # Rejecting never needs a place.
    assert rejected_again.status_code == 200, rejected_again.text


async def test_a_copy_over_the_item_cap_is_refused(
    adjustable_settings: dict[str, Settings],
    api: Api,
    connection: AsyncConnection,
    world: World,
) -> None:
    # Regression (code review): copying skipped the cap, so a copy could start over it.
    path, _, _ = await fill_with_nodes(api, connection, world, 11)
    adjustable_settings["settings"] = adjustable_settings["settings"].model_copy(
        update={"policy_max_items_per_bundle": 10}
    )

    copy = await api.call(
        "POST",
        "/policy-bundles",
        json={
            "name": "[SYNTHETIC] copy",
            "source_ref": "synthetic:copy",
            "is_synthetic": True,
            "based_on_bundle_id": path.rsplit("/", 1)[1],
        },
    )

    assert_cap_reached(copy)


@pytest.mark.parametrize("field", ["name", "source_ref"])
@pytest.mark.parametrize("blank", [" ", "\n\t"])
async def test_a_bundle_name_or_reference_cannot_be_blank(api: Api, field: str, blank: str) -> None:
    # Regression (review lesson from api#23): min_length=1 alone let " " through.
    body = {"name": "[SYNTHETIC] Policy", "source_ref": "synthetic:nrc-4-api", "is_synthetic": True}

    response = await api.call("POST", "/policy-bundles", json=body | {field: blank})

    assert response.status_code == 422, response.text
    assert response.json()["code"] == "VALIDATION_FAILED"


@pytest.mark.parametrize("field", ["title", "source_ref"])
async def test_a_source_title_or_reference_cannot_be_blank(
    api: Api, connection: AsyncConnection, field: str
) -> None:
    response = await register_source(api, connection, **{field: "   "})

    assert response.status_code == 422, response.text
    assert response.json()["code"] == "VALIDATION_FAILED"


@pytest.mark.parametrize("key", [" ", "-CL-1", "CL 1", "CL/1", "A" * 81])
async def test_a_node_key_must_be_a_plain_key(api: Api, key: str) -> None:
    bundle = await new_bundle(api)

    response = await api.call(
        "POST", f"/policy-bundles/{bundle['id']}/nodes", json=node_body(key=key), if_match=1
    )

    assert response.status_code == 422, response.text
    assert response.json()["code"] == "VALIDATION_FAILED"


async def test_a_node_key_at_the_length_limit_is_accepted(
    api: Api, connection: AsyncConnection, world: World
) -> None:
    source = (await api.ok(register_source(api, connection), 201)).json()
    clause_id = await add_clause(connection, world.a.tenant_id, source["id"])
    bundle = await new_bundle(api)

    response = await api.call(
        "POST",
        f"/policy-bundles/{bundle['id']}/nodes",
        json=node_body(clause_id, key="A" * 80),
        if_match=1,
    )

    assert response.status_code == 201, response.text


async def test_study_approved_wording_is_a_pointer_without_text_or_clause(
    api: Api, connection: AsyncConnection, world: World
) -> None:
    # v0.3, option A: the bundle names where the study's approved wording comes from; the text and
    # its approval record live in the study snapshot (NRC-2). It passes the checks with no clause.
    source = (await api.ok(register_source(api, connection), 201)).json()
    clause_id = await add_clause(connection, world.a.tenant_id, source["id"])
    bundle = await new_bundle(api)
    graph = await draft_graph(api, bundle["id"], clause_id)
    wordings = f"/policy-bundles/{bundle['id']}/rules/{graph['rule']}/wordings"
    pointer = {"kind": "STUDY_APPROVED_WORDING", "text": "study_snapshot.required_disclosures"}

    created = await api.call("POST", wordings, json=pointer, if_match=graph["revision"])
    checks = await api.ok(api.call("GET", f"/policy-bundles/{bundle['id']}/checks"))

    assert created.status_code == 201, created.text
    assert (created.json()["kind"], created.json()["source_clause_id"]) == (
        "STUDY_APPROVED_WORDING",
        None,
    )
    codes = {issue["code"] for issue in checks.json()["issues"]}
    assert "EXACT_WORDING_NOT_IN_SOURCE" not in codes


@pytest.mark.parametrize(
    "changes",
    [
        {"text": "This is a research study."},
        {"text": "study_snapshot.facts"},
        {"source_clause_id": SOME_ID},
        {"translation_of_id": SOME_ID},
    ],
)
async def test_study_approved_wording_must_be_a_bare_pointer(
    api: Api, changes: dict[str, Any]
) -> None:
    bundle = await new_bundle(api)
    body = {"kind": "STUDY_APPROVED_WORDING", "text": "study_snapshot.approved_phrases"} | changes

    response = await api.call(
        "POST", f"/policy-bundles/{bundle['id']}/rules/{SOME_ID}/wordings", json=body, if_match=1
    )

    assert response.status_code == 422, response.text
    assert response.json()["code"] == "VALIDATION_FAILED"


# Conditions use the same enums as the import script: a typo is refused, not a rule that never
# applies to any post. One wrong value per field, on create and on update.
@pytest.mark.parametrize(
    ("field", "wrong"),
    [
        ("input_kinds", "caption"),
        ("channels", "Facebook"),
        ("post_formats", "feed-image"),
        ("languages", "english"),
    ],
)
async def test_a_condition_outside_its_enum_is_refused(api: Api, field: str, wrong: str) -> None:
    bundle = await new_bundle(api)
    path = f"/policy-bundles/{bundle['id']}/nodes"

    created = await api.call("POST", path, json=node_body() | {field: [wrong]}, if_match=1)
    updated = await api.call("PATCH", f"{path}/{SOME_ID}", json={field: [wrong]}, if_match=1)

    for response in (created, updated):
        assert response.status_code == 422, response.text
        assert response.json()["code"] == "VALIDATION_FAILED"


async def test_a_node_keeps_its_typed_conditions(
    api: Api, connection: AsyncConnection, world: World
) -> None:
    source = (await api.ok(register_source(api, connection), 201)).json()
    clause_id = await add_clause(connection, world.a.tenant_id, source["id"])
    bundle = await new_bundle(api)
    conditions = {
        "input_kinds": ["text"],
        "channels": ["facebook"],
        "post_formats": ["feed_image"],
        "languages": ["en"],
    }

    response = await api.call(
        "POST",
        f"/policy-bundles/{bundle['id']}/nodes",
        json=node_body(clause_id) | conditions,
        if_match=1,
    )

    assert response.status_code == 201, response.text
    assert {field: response.json()[field] for field in conditions} == conditions


async def test_an_unknown_command_is_a_validation_error(api: Api) -> None:
    bundle = await new_bundle(api)

    response = await api.call(
        "POST",
        f"/policy-bundles/{bundle['id']}/transitions",
        json={"command": "explode"},
        if_match=1,
    )

    assert response.status_code == 422, response.text
    assert response.json()["code"] == "VALIDATION_FAILED"


# ---- Proposals


async def test_proposed_rules_are_hidden_unless_a_manager_asks(
    api: Api, connection: AsyncConnection, world: World
) -> None:
    source = (await api.ok(register_source(api, connection), 201)).json()
    clause_id = await add_clause(connection, world.a.tenant_id, source["id"])
    bundle = await new_bundle(api)
    path = f"/policy-bundles/{bundle['id']}"
    graph = await draft_graph(api, bundle["id"], clause_id)
    listing = f"{path}/rule-records"

    hidden = (await api.ok(api.call("GET", listing, APPROVER))).json()
    shown = (await api.ok(api.call("GET", listing, params={"include_proposals": "true"}))).json()
    refused = await api.call("GET", listing, APPROVER, params={"include_proposals": "true"})

    assert hidden == {"items": [], "next_cursor": None}
    assert [r["rule_id"] for r in shown["items"]] == ["REC-07"]
    assert shown["items"][0]["node"]["review_state"] == "PROPOSED"
    assert shown["items"][0]["wordings"][0]["review_state"] == "PROPOSED"
    assert refused.status_code == 403, refused.text
    assert "policy managers" in refused.json()["message"]

    # Approving the rule's node shows the rule, but its proposed wording stays out.
    decision = {"item_kind": "node", "item_id": graph["rule_node"], "decision": "APPROVED"}
    await api.ok(
        api.call(
            "POST",
            f"{path}/review-decisions",
            json=decision,
            if_match=graph["revision"],
        )
    )
    partly = (await api.ok(api.call("GET", listing, APPROVER))).json()["items"]
    assert [r["rule_id"] for r in partly] == ["REC-07"]
    assert partly[0]["wordings"] == []


# ---- Edits return the item and the new ETag


async def test_edits_return_the_item_and_the_next_revision(
    api: Api, connection: AsyncConnection, world: World
) -> None:
    source = (await api.ok(register_source(api, connection), 201)).json()
    clause_id = await add_clause(connection, world.a.tenant_id, source["id"])
    bundle = await new_bundle(api)
    path = f"/policy-bundles/{bundle['id']}"

    node = await api.ok(
        api.call("POST", f"{path}/nodes", json=node_body(clause_id), if_match=1), 201
    )
    assert node.headers["ETag"] == '"2"'
    assert node.json()["review_state"] == "PROPOSED"
    assert node.json()["source_clause_id"] == str(clause_id)

    patched = await api.ok(
        api.call(
            "PATCH",
            f"{path}/nodes/{node.json()['id']}",
            json={"channels": ["facebook"]},
            if_match=2,
        )
    )
    assert patched.headers["ETag"] == '"3"'
    assert patched.json()["channels"] == ["facebook"]

    rule = await api.ok(api.call("POST", f"{path}/rules", json=RULE_BODY, if_match=3), 201)
    assert rule.headers["ETag"] == '"4"'
    assert rule.json()["node"]["node_key"] == "REC-07"
    rule_patch = await api.ok(
        api.call(
            "PATCH",
            f"{path}/rules/{rule.json()['id']}",
            json={"title": "[SYNTHETIC] New title"},
            if_match=4,
        )
    )
    assert rule_patch.headers["ETag"] == '"5"'
    assert rule_patch.json()["title"] == "[SYNTHETIC] New title"
    assert rule_patch.json()["id"] == rule.json()["id"]

    invalid = await api.call(
        "POST",
        f"{path}/edges",
        json={
            "edge_type": "supported_by",
            "from_node_id": node.json()["id"],
            "to_node_id": node.json()["id"],
        },
        if_match=5,
    )
    assert invalid.status_code == 422
    assert invalid.json()["code"] == "VALIDATION_FAILED"
    assert "two different nodes" in invalid.json()["details"]["fields"][0]["msg"]
    assert (await api.ok(api.call("GET", path))).headers["ETag"] == '"5"'


async def test_a_node_window_needs_a_time_zone(
    api: Api, connection: AsyncConnection, world: World
) -> None:
    # A naive time cannot be compared with the stored, aware window: 422, never a 500.
    source = (await api.ok(register_source(api, connection), 201)).json()
    clause_id = await add_clause(connection, world.a.tenant_id, source["id"])
    bundle = await new_bundle(api)
    path = f"/policy-bundles/{bundle['id']}/nodes"
    body = {**node_body(clause_id), "effective_to": "2027-01-01T00:00:00Z"}
    node = await api.ok(api.call("POST", path, json=body, if_match=1), 201)

    naive = await api.call(
        "PATCH",
        f"{path}/{node.json()['id']}",
        json={"effective_from": "2026-06-01T00:00:00"},
        if_match=2,
    )

    assert naive.status_code == 422
    assert (await api.ok(api.call("GET", f"/policy-bundles/{bundle['id']}"))).headers[
        "ETag"
    ] == '"2"'


WINDOW_BACKWARDS = {
    "effective_from": "2027-01-02T00:00:00Z",
    "effective_to": "2027-01-01T00:00:00Z",
}


async def test_a_window_that_ends_before_it_starts_is_a_validation_error(api: Api) -> None:
    bundle = await new_bundle(api)
    path = f"/policy-bundles/{bundle['id']}"

    responses = [
        await api.call("POST", f"{path}/nodes", json=node_body() | WINDOW_BACKWARDS, if_match=1),
        await api.call("PATCH", f"{path}/nodes/{SOME_ID}", json=WINDOW_BACKWARDS, if_match=1),
        await api.call("POST", f"{path}/rules", json=RULE_BODY | WINDOW_BACKWARDS, if_match=1),
    ]

    for response in responses:
        assert response.status_code == 422, response.text
        assert "must not be after" in response.json()["details"]["fields"][0]["msg"]


async def test_one_date_against_the_stored_other_is_a_conflict(
    api: Api, connection: AsyncConnection, world: World
) -> None:
    # Only the stored node shows the window is backwards, so this one stays a 409.
    source = (await api.ok(register_source(api, connection), 201)).json()
    clause_id = await add_clause(connection, world.a.tenant_id, source["id"])
    bundle = await new_bundle(api)
    path = f"/policy-bundles/{bundle['id']}/nodes"
    body = {**node_body(clause_id), "effective_to": "2027-01-01T00:00:00Z"}
    node = await api.ok(api.call("POST", path, json=body, if_match=1), 201)

    response = await api.call(
        "PATCH",
        f"{path}/{node.json()['id']}",
        json={"effective_from": "2027-01-02T00:00:00Z"},
        if_match=2,
    )

    assert response.status_code == 409, response.text
    assert response.json()["code"] == "POLICY_ITEM_INVALID"


@pytest.mark.parametrize(
    ("path", "body"),
    [("nodes", {"text": None}), ("nodes", {"channels": None}), ("rules", {"title": None})],
)
async def test_a_required_field_cannot_be_cleared(
    api: Api, path: str, body: dict[str, Any]
) -> None:
    bundle = await new_bundle(api)

    response = await api.call(
        "PATCH", f"/policy-bundles/{bundle['id']}/{path}/{SOME_ID}", json=body, if_match=1
    )

    assert response.status_code == 422, response.text
    assert "cannot be empty" in response.json()["details"]["fields"][0]["msg"]


async def test_a_prohibited_phrase_fits_its_column(
    api: Api, connection: AsyncConnection, world: World
) -> None:
    # Approval copies each phrase into policy_rules.prohibited_phrases (varchar 200).
    source = (await api.ok(register_source(api, connection), 201)).json()
    clause_id = await add_clause(connection, world.a.tenant_id, source["id"])
    bundle = await new_bundle(api)
    graph = await draft_graph(api, bundle["id"], clause_id)
    wordings = f"/policy-bundles/{bundle['id']}/rules/{graph['rule']}/wordings"

    over = await api.call(
        "POST",
        wordings,
        json={"kind": "PROHIBITED_PHRASE", "text": "x" * (PROHIBITED_PHRASE_MAX_CHARS + 1)},
        if_match=graph["revision"],
    )
    at_limit = await api.call(
        "POST",
        wordings,
        json={"kind": "PROHIBITED_PHRASE", "text": "x" * PROHIBITED_PHRASE_MAX_CHARS},
        if_match=graph["revision"],
    )

    assert over.status_code == 422, over.text
    assert "at most 200 characters" in over.json()["details"]["fields"][0]["msg"]
    assert at_limit.status_code == 201, at_limit.text


# ---- Paging


async def walk(api: Api, path: str, **params: Any) -> list[dict[str, Any]]:
    # Every item of a cursor-paged list, one page of one item at a time.
    items: list[dict[str, Any]] = []
    cursor: str | None = None
    for _ in range(20):
        query = {"limit": 1, **params} | ({"after": cursor} if cursor else {})
        page = (await api.ok(api.call("GET", path, params=query))).json()
        assert len(page["items"]) <= 1
        items += page["items"]
        cursor = page["next_cursor"]
        if cursor is None:
            return items
    raise AssertionError("the list never ended")


async def test_bundle_items_page_in_a_fixed_order(
    api: Api, connection: AsyncConnection, world: World
) -> None:
    source = (await api.ok(register_source(api, connection), 201)).json()
    clause_id = await add_clause(connection, world.a.tenant_id, source["id"])
    bundle = await new_bundle(api)
    path = f"/policy-bundles/{bundle['id']}"
    graph = await draft_graph(api, bundle["id"], clause_id)
    await api.ok(
        api.call(
            "POST",
            f"{path}/nodes",
            json=node_body(clause_id, key="CL-0"),
            if_match=graph["revision"],
        ),
        201,
    )

    nodes = await walk(api, f"{path}/nodes")
    edges = await walk(api, f"{path}/edges")
    rules = await walk(api, f"{path}/rule-records", include_proposals="true")
    clauses = await walk(api, f"/policy-sources/{source['id']}/clauses")

    assert [n["node_key"] for n in nodes] == ["CL-0", "CL-1", "REC-07"]
    assert [e["id"] for e in edges] == [graph["edge"]]
    assert [r["rule_id"] for r in rules] == ["REC-07"]
    assert [c["id"] for c in clauses] == [str(clause_id)]


async def test_a_cursor_from_another_list_is_not_found(
    api: Api, connection: AsyncConnection, world: World
) -> None:
    source = (await api.ok(register_source(api, connection), 201)).json()
    clause_id = await add_clause(connection, world.a.tenant_id, source["id"])
    bundle = await new_bundle(api)
    graph = await draft_graph(api, bundle["id"], clause_id)

    response = await api.call(
        "GET", f"/policy-bundles/{bundle['id']}/nodes", params={"after": graph["edge"]}
    )

    assert response.status_code == 404, response.text
    assert "cursor names no item" in response.json()["message"]


@pytest.mark.parametrize("limit", [0, 501])
async def test_a_page_size_outside_its_bounds_is_refused(api: Api, limit: int) -> None:
    bundle = await new_bundle(api)

    response = await api.call(
        "GET", f"/policy-bundles/{bundle['id']}/nodes", params={"limit": limit}
    )

    assert response.status_code == 422, response.text


async def test_the_audit_trail_pages_newest_first(
    api: Api, world: World, connection: AsyncConnection
) -> None:
    bundle = await new_bundle(api)
    path = f"/policy-bundles/{bundle['id']}"
    for index in range(3):
        await api.ok(
            api.call(
                "POST", f"{path}/nodes", json=node_body(key=f"CL-{index}"), if_match=index + 1
            ),
            201,
        )
    base = datetime(2026, 11, 20, tzinfo=UTC)
    await spread_audit(connection, world.a.tenant_id, bundle["id"], base)

    first = (await api.ok(api.call("GET", f"{path}/audit", params={"limit": 2}))).json()
    second = (
        await api.ok(
            api.call("GET", f"{path}/audit", params={"limit": 2, "before": first["next_cursor"]})
        )
    ).json()

    assert [i["details"]["revision"] for i in first["items"]] == [4, 3]
    assert first["next_cursor"] == (base + timedelta(minutes=3)).isoformat().replace("+00:00", "Z")
    assert [i["details"]["revision"] for i in second["items"]] == [2, 1]
    assert second["next_cursor"] is None
    assert second["items"][-1]["action"] == "CREATE"


async def test_bundles_page_by_version(api: Api) -> None:
    for _ in range(2):
        await new_bundle(api)

    first = (await api.ok(api.call("GET", "/policy-bundles", params={"limit": 2}))).json()
    second = (
        await api.ok(
            api.call(
                "GET",
                "/policy-bundles",
                params={"limit": 2, "before_version": first["next_cursor"]},
            )
        )
    ).json()

    assert [b["version"] for b in first["items"]] == [3, 2]
    assert first["next_cursor"] == 2
    assert [b["version"] for b in second["items"]] == [1]
    assert second["next_cursor"] is None
