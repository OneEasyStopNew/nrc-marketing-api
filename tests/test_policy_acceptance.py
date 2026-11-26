"""NRC-4 acceptance criteria (AC1 to AC8) for policy ingestion, end to end through the API.

Every test drives the frontend's view: a policy file goes up through the presigned URL, the media
worker inspects it, the policy worker extracts clauses and builds the index, and people with the
policy_admin and policy_approver roles author, review, approve and activate a bundle over HTTP.
Workers run in-process, one pass at a time. Direct SQL (as the table owner, tenant bound) is used
only where the API cannot observe something: outbox rows, index builds, lease expiry. Storage is
the local SeaweedFS, the scanner the labelled simulator and the embedder the labelled simulator.
Each test name or comment names the acceptance criterion it covers.
"""

import hashlib
import json
from collections import Counter
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any
from uuid import UUID, uuid4

import httpx
import pytest
from sqlalchemy import Row, func, select
from sqlalchemy.ext.asyncio import AsyncConnection, AsyncSession, async_sessionmaker

from app.api.deps import get_optional_storage, get_policy_source_storage, get_storage
from app.core.config import Settings, get_settings
from app.models import PolicySourceClause
from app.models.enums import CheckType, Gate, PolicyEdgeType, Severity, WordingKind
from app.services.policy_bundles import POLICY_BUNDLE_EVENT
from app.services.providers.embeddings import SimulatedEmbeddingProvider
from app.services.scanning import SimulatedScanner
from app.services.storage import ObjectStorage
from app.workers import policy as policy_worker
from app.workers.media import run_once as media_run_once
from tests.conftest import (
    BUCKET,
    ENDPOINT,
    AuthHeaders,
    World,
    make_pdf,
    needs_seaweedfs,
    owner_sql,
)

pytestmark = needs_seaweedfs

BASE = "/v1/marketing"
ADMIN, APPROVER = "policy-admin-a", "policy-approver-a"
B_ADMIN, B_APPROVER = "policy-admin-b", "policy-approver-b"
# creator-a, reviewer-a and admin-a hold no policy capability; every role that reads the active
# policy (validation does) has creative:read, which creator-a stands for.
READER = "creator-a"
SOURCE_REF = "synthetic:nrc-4-acceptance"
REASON = "[SYNTHETIC] Reviewed and signed off."

CLAUSE_VOLUNTARY = (
    "[SYNTHETIC] Every post must say: This is a research study. Taking part is voluntary."
)
WORDING = "This is a research study. Taking part is voluntary."
CLAUSE_CLAIMS = "[SYNTHETIC] Posts must never promise a cure or guaranteed results."
REQUIREMENT_VOLUNTARY = "Say that this is a research study and that taking part is voluntary."
REQUIREMENT_CLAIMS = "Never promise a cure or guaranteed results."
POLICY_PDF = make_pdf([["1. Disclosures", None, CLAUSE_VOLUNTARY, None, CLAUSE_CLAIMS]])
POLICY_MD = f"# [SYNTHETIC] Disclosures\n\n{CLAUSE_VOLUNTARY}\n\n{CLAUSE_CLAIMS}\n".encode()

ROOT = Path(__file__).resolve().parents[1]
CONTRACT: dict[str, Any] = json.loads((ROOT / "contracts" / "openapi.json").read_text("utf-8"))
EXAMPLES: dict[str, Any] = json.loads(
    (ROOT / "contracts" / "examples" / "nrc4.json").read_text("utf-8")
)

_EXPIRE_LEASE = {
    "policy_sources": "UPDATE policy_sources SET leased_until = clock_timestamp()"
    " - interval '1 second' WHERE id = :id",
    "policy_bundles": "UPDATE policy_bundles SET leased_until = clock_timestamp()"
    " - interval '1 second' WHERE id = :id",
}


# ---- Fixtures (no test module imports another)


@pytest.fixture
def seed_graph() -> bool:
    # These tests count or write their own sources, graph rows and wordings in tenant A, so its
    # bundle starts without NRC-5's synthetic graph.
    return False


@pytest.fixture(autouse=True)
def local_credentials(monkeypatch: pytest.MonkeyPatch) -> None:
    # Local-only values that open the SeaweedFS container, not AWS.
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "localdev")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "localdev-password")
    monkeypatch.delenv("AWS_PROFILE", raising=False)


@pytest.fixture
def settings(policy_bucket: str) -> Settings:
    return get_settings().model_copy(
        update={
            "app_env": "local",
            "malware_scan_mode": "simulator",
            "aws_s3_bucket": BUCKET,
            "aws_endpoint_url_s3": ENDPOINT,
            "s3_force_path_style": True,
            "aws_profile": None,
            "policy_source_bucket": policy_bucket,
            "worker_backoff_base_seconds": 30.0,
            "worker_backoff_max_seconds": 60.0,
            "policy_worker_max_attempts": 3,
            "policy_extraction_timeout_seconds": 5.0,
        }
    )


@pytest.fixture
def media_storage(settings: Settings) -> ObjectStorage:
    return ObjectStorage(settings)


@pytest.fixture
def policy_storage(settings: Settings, policy_bucket: str) -> ObjectStorage:
    return ObjectStorage(
        settings, bucket=policy_bucket, max_read_bytes=settings.policy_source_max_bytes
    )


@pytest.fixture(autouse=True)
def use_storage(
    app_overrides: dict[Callable[..., object], Callable[..., object]],
    settings: Settings,
    media_storage: ObjectStorage,
    policy_storage: ObjectStorage,
) -> None:
    app_overrides[get_settings] = lambda: settings
    app_overrides[get_storage] = lambda: media_storage
    app_overrides[get_optional_storage] = lambda: media_storage
    app_overrides[get_policy_source_storage] = lambda: policy_storage


@pytest.fixture(autouse=True)
async def staff(connection: AsyncConnection, world: World) -> None:
    # creator-a and reviewer-a come with the world; these are the other roles under test.
    members = [
        (world.a.tenant_id, ADMIN, "policy_admin"),
        (world.a.tenant_id, APPROVER, "policy_approver"),
        (world.a.tenant_id, "admin-a", "admin"),
        (world.b.tenant_id, B_ADMIN, "policy_admin"),
        (world.b.tenant_id, B_APPROVER, "policy_approver"),
    ]
    for tenant_id, sub, role in members:
        await owner_sql(
            connection,
            tenant_id,
            "INSERT INTO memberships (tenant_id, user_sub, roles) VALUES (:t, :sub, ARRAY[:role])",
            {"t": tenant_id, "sub": sub, "role": role},
        )


# ---- The caller's view


def data(response: httpx.Response) -> dict[str, Any]:
    body: dict[str, Any] = response.json()
    return body


def items(response: httpx.Response) -> list[dict[str, Any]]:
    listed: list[dict[str, Any]] = response.json()["items"]
    return listed


class Api:
    """One tenant, any user."""

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
        if_match: int | None = None,
        params: dict[str, Any] | None = None,
        headers: dict[str, str] | None = None,
    ) -> httpx.Response:
        sent = self.auth(sub, self.tenant_id, **(headers or {}))
        if if_match is not None:
            sent["If-Match"] = f'"{if_match}"'
        return await self.client.request(
            method, BASE + path, headers=sent, json=json, params=params
        )

    async def ok(
        self, method: str, path: str, sub: str = ADMIN, *, status: int = 200, **kwargs: Any
    ) -> httpx.Response:
        response = await self.call(method, path, sub, **kwargs)
        assert response.status_code == status, response.text
        return response

    async def refused(
        self, method: str, path: str, sub: str, status: int, code: str, **kwargs: Any
    ) -> dict[str, Any]:
        response = await self.call(method, path, sub, **kwargs)
        assert response.status_code == status, response.text
        assert response.json()["code"] == code, response.text
        return data(response)


@pytest.fixture
def api(client: httpx.AsyncClient, auth: AuthHeaders, world: World) -> Api:
    return Api(client, auth, world.a.tenant_id)


@dataclass
class Stack:
    """The API plus the workers and storage behind it, as one test sees them."""

    api: Api
    connection: AsyncConnection
    world: World
    sessions: async_sessionmaker[AsyncSession]
    settings: Settings
    media_storage: ObjectStorage
    policy_storage: ObjectStorage
    scanner: SimulatedScanner

    async def work(self, *, lexical: bool = False, owner: str = "w1") -> int:
        # One policy worker pass (extraction, then index builds); lexical builds no vectors.
        embedder = None if lexical else SimulatedEmbeddingProvider()
        return await policy_worker.run_once(
            self.sessions, self.policy_storage, self.settings, embedder=embedder, owner=owner
        )

    async def sql(self, statement: str, params: dict[str, Any] | None = None) -> list[Row[Any]]:
        return await owner_sql(self.connection, self.world.a.tenant_id, statement, params)

    async def events(self) -> list[dict[str, Any]]:
        # What invalidation consumers (approval, dispatch) are told, from the outbox.
        rows = await self.sql(
            "SELECT payload FROM outbox_events WHERE event_type = :t", {"t": POLICY_BUNDLE_EVENT}
        )
        return [row.payload for row in rows]

    async def expire_lease(self, table: str, item_id: str) -> None:
        await self.sql(_EXPIRE_LEASE[table], {"id": UUID(item_id)})


@pytest.fixture
def stack(
    api: Api,
    connection: AsyncConnection,
    world: World,
    sessions: async_sessionmaker[AsyncSession],
    settings: Settings,
    media_storage: ObjectStorage,
    policy_storage: ObjectStorage,
) -> Stack:
    return Stack(
        api,
        connection,
        world,
        sessions,
        settings,
        media_storage,
        policy_storage,
        SimulatedScanner(settings),
    )


# ---- Sources: upload, register, extract


@dataclass(frozen=True)
class Ingested:
    source: dict[str, Any]
    clauses: list[dict[str, Any]]

    def clause(self, text: str) -> dict[str, Any]:
        found = [clause for clause in self.clauses if clause["text"] == text]
        assert len(found) == 1, [clause["text"] for clause in self.clauses]
        return found[0]


async def upload_ready(
    stack: Stack, content: bytes = POLICY_MD, mime_type: str = "text/markdown"
) -> str:
    # create -> PUT the exact bytes -> complete -> one media worker pass -> READY.
    created = await stack.api.ok(
        "POST",
        "/policy-sources/uploads",
        status=201,
        json={
            "name": "[SYNTHETIC] policy source",
            "mime_type": mime_type,
            "size_bytes": len(content),
            "checksum_sha256": hashlib.sha256(content).hexdigest(),
        },
        headers={"Idempotency-Key": uuid4().hex},
    )
    target = created.json()["upload"]
    async with httpx.AsyncClient() as http:
        stored = await http.request(
            target["method"], target["url"], content=content, headers=target["headers"]
        )
    assert stored.status_code == 200, stored.text
    upload_id: str = created.json()["upload_id"]
    done = await stack.api.ok("POST", f"/policy-sources/uploads/{upload_id}/complete", status=202)
    assert done.json()["status"] == "QUARANTINED"
    leased = await media_run_once(
        stack.sessions,
        stack.media_storage,
        stack.scanner,
        stack.settings,
        policy_storage=stack.policy_storage,
    )
    assert leased == 1
    status = await stack.api.ok("GET", f"/policy-sources/uploads/{upload_id}")
    assert status.json()["status"] == "READY"
    return upload_id


async def register(stack: Stack, upload_id: str, **changes: Any) -> httpx.Response:
    body = {
        "upload_id": upload_id,
        "title": "[SYNTHETIC] Disclosure policy",
        "source_ref": SOURCE_REF,
        "language": "en",
        "is_synthetic": True,
    }
    return await stack.api.call("POST", "/policy-sources", json=body | changes)


async def read_ingested(stack: Stack, source_id: str) -> Ingested:
    source = data(await stack.api.ok("GET", f"/policy-sources/{source_id}"))
    clauses = items(await stack.api.ok("GET", f"/policy-sources/{source_id}/clauses"))
    return Ingested(source, clauses)


async def ingest(
    stack: Stack, content: bytes = POLICY_MD, mime_type: str = "text/markdown"
) -> Ingested:
    upload_id = await upload_ready(stack, content, mime_type)
    registered = await register(stack, upload_id)
    assert registered.status_code == 201, registered.text
    assert await stack.work() == 1
    ingested = await read_ingested(stack, registered.json()["id"])
    assert ingested.source["extraction_state"] == "EXTRACTED"
    return ingested


# ---- Bundles: draft, review, move


class Draft:
    """A bundle being edited: sends If-Match with the revision the last response returned."""

    def __init__(self, api: Api, detail: dict[str, Any]) -> None:
        self.api = api
        self.id: str = detail["id"]
        self.version: int = detail["version"]
        self.revision: int = detail["revision"]
        self.path = f"/policy-bundles/{self.id}"

    def follow(self, response: httpx.Response) -> None:
        etag = response.headers.get("ETag")
        assert etag is not None, "every change returns the new ETag"
        self.revision = int(etag.strip('"'))

    async def edit(
        self,
        suffix: str,
        body: dict[str, Any],
        *,
        method: str = "POST",
        status: int = 201,
        sub: str = ADMIN,
    ) -> dict[str, Any]:
        response = await self.api.ok(
            method, self.path + suffix, sub, status=status, json=body, if_match=self.revision
        )
        self.follow(response)
        return data(response)

    async def review(self, reviewed: Iterable[tuple[str, str]], decision: str = "APPROVED") -> None:
        for kind, item_id in reviewed:
            body = {"item_kind": kind, "item_id": item_id, "decision": decision}
            self.follow(
                await self.api.ok(
                    "POST", f"{self.path}/review-decisions", json=body, if_match=self.revision
                )
            )

    async def move(
        self, command: str, sub: str, reason: str | None = None, *, status: int = 200
    ) -> httpx.Response:
        body = {"command": command} | ({"reason": reason} if reason else {})
        response = await self.api.ok(
            "POST",
            f"{self.path}/transitions",
            sub,
            status=status,
            json=body,
            if_match=self.revision,
        )
        if status == 200:
            self.follow(response)
        return response

    async def reload(self) -> dict[str, Any]:
        response = await self.api.ok("GET", self.path, APPROVER)
        self.revision = int(response.headers["ETag"].strip('"'))
        return data(response)

    async def audit(self) -> list[dict[str, Any]]:
        return items(await self.api.ok("GET", f"{self.path}/audit", APPROVER))

    async def state(self) -> str:
        state: str = (await self.reload())["lifecycle_state"]
        return state


async def new_draft(api: Api, **changes: Any) -> Draft:
    body = {"name": "[SYNTHETIC] Disclosure rules", "source_ref": SOURCE_REF, "is_synthetic": True}
    response = await api.ok("POST", "/policy-bundles", status=201, json=body | changes)
    return Draft(api, data(response))


@dataclass(frozen=True)
class Rule:
    record_id: str
    node_id: str
    clause_node_id: str
    edge_id: str
    wording_id: str | None

    def items(self) -> list[tuple[str, str]]:
        # Everything a reviewer approves for this rule.
        reviewed = [("node", self.clause_node_id), ("node", self.node_id), ("edge", self.edge_id)]
        return reviewed + ([("wording", self.wording_id)] if self.wording_id else [])


def rule_body(rule_id: str, requirement: str) -> dict[str, Any]:
    return {
        "rule_id": rule_id,
        "title": f"[SYNTHETIC] Rule {rule_id}",
        "requirement": requirement,
        "gate": Gate.COMPLETENESS.value,
        "severity": Severity.CRITICAL.value,
        "mandatory": True,
        "check_type": CheckType.DETERMINISTIC.value,
        "text": f"[SYNTHETIC] {requirement}",
    }


async def authored(
    draft: Draft,
    clause: dict[str, Any],
    *,
    rule_id: str,
    requirement: str,
    wording: tuple[WordingKind, str] | None = None,
    review: bool = True,
) -> Rule:
    # A clause node citing the extracted clause, a rule supported by it (the edge cites the same
    # clause) and optionally its wording; all PROPOSED until reviewed.
    node = await draft.edit(
        "/nodes",
        {
            "node_key": f"CL-{rule_id}",
            "node_type": "CLAUSE",
            "text": clause["text"],
            "source_clause_id": clause["id"],
        },
    )
    rule = await draft.edit("/rules", rule_body(rule_id, requirement))
    edge = await draft.edit(
        "/edges",
        {
            "edge_type": PolicyEdgeType.SUPPORTED_BY.value,
            "from_node_id": rule["node"]["id"],
            "to_node_id": node["id"],
            "provenance_clause_id": clause["id"],
        },
    )
    wording_id = None
    if wording is not None:
        kind, text = wording
        added = await draft.edit(
            f"/rules/{rule['id']}/wordings", {"kind": kind.value, "text": text}
        )
        wording_id = added["id"]
    result = Rule(rule["id"], rule["node"]["id"], node["id"], edge["id"], wording_id)
    if review:
        await draft.review(result.items())
    return result


async def build(stack: Stack, draft: Draft, *, lexical: bool = False) -> None:
    # submit -> approve -> one worker pass builds the index -> READY.
    await draft.move("submit", ADMIN)
    await draft.move("approve", APPROVER)
    assert await stack.work(lexical=lexical) == 1
    assert await draft.state() == "READY"


async def ready_bundle(
    stack: Stack,
    ingested: Ingested,
    *,
    rule_id: str = "REC-07",
    wording: str | None = WORDING,
) -> tuple[Draft, Rule]:
    draft = await new_draft(stack.api)
    exact = (WordingKind.EXACT_WORDING, wording) if wording else None
    rule = await authored(
        draft,
        ingested.clause(CLAUSE_VOLUNTARY),
        rule_id=rule_id,
        requirement=REQUIREMENT_VOLUNTARY,
        wording=exact,
    )
    await build(stack, draft)
    return draft, rule


async def activate(draft: Draft, reason: str = REASON) -> httpx.Response:
    await draft.reload()
    return await draft.move("activate", ADMIN, reason)


async def active_policy(api: Api, *, status: int = 200) -> dict[str, Any]:
    return data(await api.ok("GET", "/policies/active", READER, status=status))


async def bundle_rules(api: Api, draft: Draft) -> dict[str, Any]:
    # The evidence view of an active or retired bundle: exact wording, sources, interpretations.
    return data(await api.ok("GET", f"{draft.path}/rules", READER))


def cited(rule: dict[str, Any]) -> tuple[str, str]:
    # Which source and clause (or interpretation node) an active rule rests on.
    return rule["source"]["source_id"], rule["source"]["clause"]


def rule_by_id(read: dict[str, Any], rule_id: str) -> dict[str, Any]:
    rules: list[dict[str, Any]] = read["rules"]
    found = [rule for rule in rules if rule["rule_id"] == rule_id]
    assert len(found) == 1, [rule["rule_id"] for rule in rules]
    return found[0]


# ---- The journey (AC1, AC3, AC6, AC7, AC8)


async def test_the_whole_journey_from_upload_to_rollback(stack: Stack) -> None:
    api = stack.api
    ingested = await ingest(stack, POLICY_PDF, "application/pdf")

    # Upload, register and extract: the clauses are the file's own text (AC7).
    assert ingested.source["is_synthetic"] is True
    assert ingested.source["created_by"] == ADMIN
    assert ingested.source["page_count"] == 1
    assert [(c["clause_ref"], c["page"], c["section"], c["text"]) for c in ingested.clauses] == [
        ("p1-c1", 1, "1. Disclosures", CLAUSE_VOLUNTARY),
        ("p1-c2", 1, "1. Disclosures", CLAUSE_CLAIMS),
    ]
    listed = items(await api.ok("GET", "/policy-sources", APPROVER))
    assert [source["id"] for source in listed] == [ingested.source["id"]]

    # Bundle 2 (the world's tenant already has version 1): rules, edges, exact wording (AC1, AC7).
    v2 = await new_draft(api)
    assert v2.version == 2
    await authored(
        v2,
        ingested.clause(CLAUSE_VOLUNTARY),
        rule_id="REC-07",
        requirement=REQUIREMENT_VOLUNTARY,
        wording=(WordingKind.EXACT_WORDING, WORDING),
    )
    await authored(
        v2,
        ingested.clause(CLAUSE_CLAIMS),
        rule_id="REC-08",
        requirement=REQUIREMENT_CLAIMS,
        wording=(WordingKind.PROHIBITED_PHRASE, "guaranteed results"),
    )
    assert (await api.ok("GET", f"{v2.path}/checks", APPROVER)).json() == {"issues": []}
    assert await v2.state() == "DRAFT"
    submitted = await v2.move("submit", ADMIN)
    assert submitted.json()["lifecycle_state"] == "REVIEW"
    assert submitted.json()["rule_count"] == 2
    approved = await v2.move("approve", APPROVER)
    assert approved.json()["lifecycle_state"] == "APPROVED"
    assert approved.json()["approved_by"] == APPROVER

    # The build worker makes it READY; only then can it be activated (AC4, AC6).
    assert await stack.work() == 1
    assert await v2.state() == "READY"
    activated = await v2.move("activate", ADMIN, REASON)
    assert activated.json()["lifecycle_state"] == "ACTIVE"
    assert activated.json()["activated_by"] == ADMIN
    demo = Draft(api, {"id": world_a_bundle(stack), "version": 1, "revision": 1})
    assert await demo.state() == "RETIRED"

    # What validation reads now: the new rules, each with its source clause, and the wording
    # byte for byte (AC1, AC7).
    policy = await active_policy(api)
    assert (policy["policy_bundle_id"], policy["version"]) == (v2.id, 2)
    assert [rule["rule_id"] for rule in policy["rules"]] == ["REC-07", "REC-08"]
    assert [cited(rule) for rule in policy["rules"]] == [
        (SOURCE_REF, "p1-c1"),
        (SOURCE_REF, "p1-c2"),
    ]
    evidence = await bundle_rules(api, v2)
    assert rule_by_id(evidence, "REC-07")["exact_wording"].encode() == WORDING.encode()
    assert rule_by_id(evidence, "REC-08")["prohibited_phrases"] == ["guaranteed results"]
    assert rule_by_id(evidence, "REC-07")["sources"][0]["excerpt"] == CLAUSE_VOLUNTARY

    # Bundle 3 copies bundle 2, changes one rule's title and is reviewed again; activating it
    # retires bundle 2 (AC3: one active bundle, so no duplicate active obligations).
    v3 = await new_draft(api, based_on_bundle_id=v2.id)
    assert (await v3.reload())["rule_count"] == 2
    records = items(await api.ok("GET", f"{v3.path}/rule-records", APPROVER))
    claims = next(record for record in records if record["rule_id"] == "REC-08")
    changed = await v3.edit(
        f"/rules/{claims['id']}", {"title": "[SYNTHETIC] Claims, v3"}, method="PATCH", status=200
    )
    assert changed["node"]["review_state"] == "PROPOSED"
    await v3.review([("node", claims["node"]["id"])])
    await build(stack, v3)
    await activate(v3, "[SYNTHETIC] v3 replaces v2")
    assert await v2.state() == "RETIRED"
    policy = await active_policy(api)
    assert policy["policy_bundle_id"] == v3.id
    assert [rule["title"] for rule in policy["rules"]] == [
        "[SYNTHETIC] Rule REC-07",
        "[SYNTHETIC] Claims, v3",
    ]
    active = items(await api.ok("GET", "/policy-bundles", params={"state": "ACTIVE"}))
    assert [bundle["id"] for bundle in active] == [v3.id]
    assert rule_by_id(await bundle_rules(api, v3), "REC-07")["exact_wording"] == WORDING

    # Retire leaves no active policy; rollback brings bundle 2 back with its own content (AC6).
    await v3.move("retire", ADMIN, "[SYNTHETIC] Withdrawn.")
    missing = await api.refused("GET", "/policies/active", READER, 404, "NOT_FOUND")
    assert "no active policy" in missing["message"]
    await v2.reload()
    rolled = await v2.move("rollback", ADMIN, "[SYNTHETIC] v3 was wrong.")
    assert rolled.json()["lifecycle_state"] == "ACTIVE"
    policy = await active_policy(api)
    assert policy["policy_bundle_id"] == v2.id
    assert [rule["title"] for rule in policy["rules"]] == [
        "[SYNTHETIC] Rule REC-07",
        "[SYNTHETIC] Rule REC-08",
    ]
    assert await v3.state() == "RETIRED"


def world_a_bundle(stack: Stack) -> str:
    # The demo bundle the world's tenant starts with (version 1, active until bundle 2).
    return str(stack.world.a.policy_bundle_id)


# ---- AC1: every active rule and edge links to an approved source clause or interpretation


async def test_ac1_every_active_rule_and_edge_traces_to_an_extracted_clause(stack: Stack) -> None:
    api = stack.api
    ingested = await ingest(stack)
    draft, _ = await ready_bundle(stack, ingested)
    await activate(draft)

    clause_ids = {clause["id"] for clause in ingested.clauses}
    nodes = items(await api.ok("GET", f"{draft.path}/nodes", APPROVER))
    edges = items(await api.ok("GET", f"{draft.path}/edges", APPROVER))
    assert {node["review_state"] for node in nodes} == {"APPROVED"}
    assert {edge["review_state"] for edge in edges} == {"APPROVED"}
    clause_nodes = [node for node in nodes if node["node_type"] == "CLAUSE"]
    assert clause_nodes
    assert all(node["source_clause_id"] in clause_ids for node in clause_nodes)
    assert edges
    assert all(
        edge["provenance_clause_id"] in clause_ids or edge["provenance_note"] for edge in edges
    )
    policy = await active_policy(api)
    clause_refs = {clause["clause_ref"] for clause in ingested.clauses}
    assert policy["rules"]
    for rule in policy["rules"]:
        assert rule["source"]["source_id"] == SOURCE_REF
        assert rule["source"]["clause"] in clause_refs
    sources = rule_by_id(await bundle_rules(api, draft), "REC-07")["sources"]
    assert [(s["clause_ref"], s["source_title"]) for s in sources] == [
        ("c1", ingested.source["title"])
    ]


async def test_ac1_an_approved_interpretation_stands_in_for_a_source_clause(stack: Stack) -> None:
    api = stack.api
    draft = await new_draft(api)
    interpretation = await draft.edit(
        "/nodes",
        {
            "node_key": "INT-1",
            "node_type": "INTERPRETATION",
            "text": "[SYNTHETIC] Counsel reads the policy as requiring a study notice.",
        },
    )
    rule = await draft.edit("/rules", rule_body("REC-30", "Carry a study notice."))
    edge = await draft.edit(
        "/edges",
        {
            "edge_type": PolicyEdgeType.SUPPORTED_BY.value,
            "from_node_id": rule["node"]["id"],
            "to_node_id": interpretation["id"],
            "provenance_note": "[SYNTHETIC] Approved interpretation, no clause states it.",
        },
    )
    await draft.review(
        [("node", interpretation["id"]), ("node", rule["node"]["id"]), ("edge", edge["id"])]
    )

    await build(stack, draft)
    await activate(draft)

    policy = await active_policy(api)
    assert cited(policy["rules"][0]) == ("interpretation", "INT-1")
    evidence = rule_by_id(await bundle_rules(api, draft), "REC-30")
    assert [text["node_key"] for text in evidence["interpretations"]] == ["INT-1"]
    assert evidence["sources"] == []


async def test_ac1_an_edge_needs_a_source_clause_or_a_note_to_be_approved(stack: Stack) -> None:
    ingested = await ingest(stack)
    draft = await new_draft(stack.api)
    rule = await authored(
        draft,
        ingested.clause(CLAUSE_VOLUNTARY),
        rule_id="REC-07",
        requirement=REQUIREMENT_VOLUNTARY,
        review=False,
    )
    bare = await draft.edit(
        "/edges",
        {
            "edge_type": PolicyEdgeType.APPLIES_TO.value,
            "from_node_id": rule.node_id,
            "to_node_id": rule.clause_node_id,
        },
    )

    refused = await stack.api.refused(
        "POST",
        f"{draft.path}/review-decisions",
        ADMIN,
        409,
        "POLICY_ITEM_INVALID",
        json={"item_kind": "edge", "item_id": bare["id"], "decision": "APPROVED"},
        if_match=draft.revision,
    )

    assert "needs a source clause or a note" in refused["message"]
    edges = items(await stack.api.ok("GET", f"{draft.path}/edges", ADMIN))
    assert {edge["id"]: edge["review_state"] for edge in edges}[bare["id"]] == "PROPOSED"


# ---- AC2: partial, unreviewed, contradictory or provenance-free bundles cannot activate


async def _unreviewed(draft: Draft, ingested: Ingested) -> None:
    await authored(
        draft,
        ingested.clause(CLAUSE_VOLUNTARY),
        rule_id="REC-07",
        requirement=REQUIREMENT_VOLUNTARY,
        review=False,
    )


async def _without_provenance(draft: Draft, ingested: Ingested) -> None:
    # An approved rule with nothing behind it: no clause node, no edge, no interpretation.
    rule = await authored(
        draft,
        ingested.clause(CLAUSE_VOLUNTARY),
        rule_id="REC-07",
        requirement=REQUIREMENT_VOLUNTARY,
        review=False,
    )
    await draft.review([("node", rule.clause_node_id), ("node", rule.node_id)])


async def _contradictory(draft: Draft, ingested: Ingested) -> None:
    await authored(
        draft,
        ingested.clause(CLAUSE_VOLUNTARY),
        rule_id="REC-07",
        requirement=REQUIREMENT_VOLUNTARY,
        wording=(WordingKind.EXACT_WORDING, WORDING),
    )
    # Another rule bans a word the first rule's required wording contains.
    await authored(
        draft,
        ingested.clause(CLAUSE_CLAIMS),
        rule_id="REC-08",
        requirement=REQUIREMENT_CLAIMS,
        wording=(WordingKind.PROHIBITED_PHRASE, "voluntary"),
    )


DEFECTS = {
    "unreviewed": (_unreviewed, "UNREVIEWED_ITEM"),
    "no-provenance": (_without_provenance, "MISSING_PROVENANCE"),
    "contradictory": (_contradictory, "CONTRADICTORY_WORDING"),
}


@pytest.mark.parametrize("defect", DEFECTS.values(), ids=DEFECTS.keys())
async def test_ac2_a_bundle_with_a_defect_cannot_be_approved_built_or_activated(
    stack: Stack, defect: tuple[Callable[[Draft, Ingested], Any], str]
) -> None:
    api = stack.api
    make, code = defect
    ingested = await ingest(stack)
    draft = await new_draft(api)
    await make(draft, ingested)
    await draft.move("submit", ADMIN)

    refused = await draft.move("approve", APPROVER, status=409)

    body = refused.json()
    assert body["code"] == "POLICY_BUNDLE_HAS_ISSUES"
    assert code in {issue["code"] for issue in body["details"]["issues"]}
    # The editor shows the same list before anyone tries to approve.
    checks = await api.ok("GET", f"{draft.path}/checks", APPROVER)
    assert checks.json() == {"issues": body["details"]["issues"]}
    detail = await draft.reload()
    assert (detail["lifecycle_state"], detail["approved_by"]) == ("REVIEW", None)
    # Neither a worker pass nor an admin's activate or build gets it any further.
    assert await stack.work() == 0
    for command, allowed in (("activate", "READY"), ("build", "APPROVED")):
        stuck = await draft.move(command, ADMIN, REASON, status=409)
        assert stuck.json()["code"] == "POLICY_BUNDLE_STATE"
        assert stuck.json()["details"] == {"state": "REVIEW", "allowed": [allowed]}
    assert (await active_policy(api))["policy_bundle_id"] == world_a_bundle(stack)
    assert await stack.events() == []


async def test_ac2_a_bundle_that_is_not_fully_built_cannot_activate(stack: Stack) -> None:
    api = stack.api
    ingested = await ingest(stack)
    draft = await new_draft(api)
    await authored(
        draft,
        ingested.clause(CLAUSE_VOLUNTARY),
        rule_id="REC-07",
        requirement=REQUIREMENT_VOLUNTARY,
    )
    await draft.move("submit", ADMIN)
    await draft.move("approve", APPROVER)

    # Approved but not built yet.
    early = await draft.move("activate", ADMIN, REASON, status=409)
    assert early.json()["details"] == {"state": "APPROVED", "allowed": ["READY"]}

    # Built, but an index of other content is on record: it no longer matches the approved nodes.
    assert await stack.work(lexical=True) == 1
    await stack.sql(
        "INSERT INTO policy_index_builds (id, tenant_id, policy_bundle_id, model_id, dimensions,"
        " state, node_count, content_checksum, completed_at) VALUES (:id, :t, :b,"
        " 'synthetic-stale', 1024, 'READY', 1, :sum, now())",
        {"id": uuid4(), "t": stack.world.a.tenant_id, "b": UUID(draft.id), "sum": "0" * 64},
    )
    assert await draft.state() == "READY"
    stale = await draft.move("activate", ADMIN, REASON, status=409)

    assert stale.json()["code"] == "POLICY_BUNDLE_HAS_ISSUES"
    assert [issue["code"] for issue in stale.json()["details"]["issues"]] == ["INDEX_STALE"]
    assert await draft.state() == "READY"
    assert (await active_policy(api))["policy_bundle_id"] == world_a_bundle(stack)
    assert await stack.events() == []


# ---- AC3: duplicate imports keep provenance and add no duplicate active obligation


async def test_ac3_registering_the_same_file_again_keeps_the_first_provenance(
    stack: Stack,
) -> None:
    api = stack.api
    first = await ingest(stack)
    source_id = first.source["id"]
    # The same bytes through a second upload: same checksum, another upload id.
    second_upload = await upload_ready(stack)
    assert second_upload != first.source["upload_id"]

    again = await register(
        stack, second_upload, title="[SYNTHETIC] Renamed", source_ref="synthetic:other"
    )
    same_upload = await register(stack, first.source["upload_id"])

    for response in (again, same_upload):
        assert response.status_code == 200, response.text
        assert response.json() == first.source
        assert response.headers["Location"] == f"{BASE}/policy-sources/{source_id}"
    assert [s["id"] for s in items(await api.ok("GET", "/policy-sources", APPROVER))] == [source_id]
    # Nothing is extracted twice: the clause set is the same rows.
    assert await stack.work() == 0
    assert (await read_ingested(stack, source_id)).clauses == first.clauses


async def test_ac3_the_same_obligation_imported_twice_does_not_become_two_active_rules(
    stack: Stack,
) -> None:
    api = stack.api
    ingested = await ingest(stack)
    clause = ingested.clause(CLAUSE_VOLUNTARY)
    draft = await new_draft(api)
    await authored(draft, clause, rule_id="REC-07", requirement=REQUIREMENT_VOLUNTARY)
    duplicate = await authored(draft, clause, rule_id="REC-07B", requirement=REQUIREMENT_VOLUNTARY)
    await draft.move("submit", ADMIN)

    refused = await draft.move("approve", APPROVER, status=409)

    issues = refused.json()["details"]["issues"]
    assert {(issue["code"], issue["subject"]) for issue in issues} == {
        ("DUPLICATE_OBLIGATION", "rule:REC-07"),
        ("DUPLICATE_OBLIGATION", "rule:REC-07B"),
    }
    # The reviewer rejects the second import (its nodes and edge are inert once rejected).
    await draft.review(duplicate.items(), "REJECTED")
    await draft.move("approve", APPROVER)
    assert await stack.work() == 1
    await activate(draft)

    assert [rule["rule_id"] for rule in (await active_policy(api))["rules"]] == ["REC-07"]
    active = items(await api.ok("GET", "/policy-bundles", params={"state": "ACTIVE"}))
    assert [bundle["id"] for bundle in active] == [draft.id]


# ---- AC4: interrupted work never exposes a mixed or half-built version


async def test_ac4_a_crash_while_building_never_exposes_a_half_built_bundle(
    stack: Stack, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    api = stack.api
    ingested = await ingest(stack)
    old, _ = await ready_bundle(stack, ingested)
    await activate(old)
    new = await new_draft(api)
    await authored(
        new,
        ingested.clause(CLAUSE_CLAIMS),
        rule_id="REC-08",
        requirement=REQUIREMENT_CLAIMS,
        wording=(WordingKind.PROHIBITED_PHRASE, "guaranteed results"),
    )
    await new.move("submit", ADMIN)
    await new.move("approve", APPROVER)

    class Boom(Exception):
        pass

    async def crash(*args: object) -> None:
        raise Boom

    # The index rows and READY are written by now; only the audit row and the commit are left.
    monkeypatch.setattr(policy_worker, "_audit", crash)
    assert await stack.work() == 1

    assert "policy work failed" in caplog.text
    assert await new.state() == "BUILDING"
    stuck = await new.move("activate", ADMIN, REASON, status=409)
    assert stuck.json()["details"] == {"state": "BUILDING", "allowed": ["READY"]}
    # Validation still reads the old bundle only, and no read shows the new one.
    policy = await active_policy(api)
    assert (policy["policy_bundle_id"], [r["rule_id"] for r in policy["rules"]]) == (
        old.id,
        ["REC-07"],
    )
    hidden = await api.refused("GET", f"{new.path}/rules", READER, 404, "NOT_FOUND")
    assert hidden["code"] == "NOT_FOUND"
    assert {a["action"] for a in await new.audit()}.isdisjoint({"BUILD_READY"})
    builds = await stack.sql(
        "SELECT count(*) AS n FROM policy_index_builds WHERE policy_bundle_id = :id",
        {"id": UUID(new.id)},
    )
    embeddings = await stack.sql(
        "SELECT count(*) AS n FROM policy_node_embeddings WHERE policy_bundle_id = :id",
        {"id": UUID(new.id)},
    )
    assert (builds[0].n, embeddings[0].n) == (0, 0)

    # While the lease holds nobody else takes it; once it expires the next worker redoes it all.
    assert await stack.work(owner="w2") == 0
    monkeypatch.undo()
    await stack.expire_lease("policy_bundles", new.id)
    assert await stack.work(owner="w2") == 1
    assert await new.state() == "READY"
    ready = await stack.sql(
        "SELECT state, node_count FROM policy_index_builds WHERE policy_bundle_id = :id",
        {"id": UUID(new.id)},
    )
    assert [(row.state, row.node_count) for row in ready] == [("READY", 1)]
    await activate(new)
    policy = await active_policy(api)
    assert (policy["policy_bundle_id"], [r["rule_id"] for r in policy["rules"]]) == (
        new.id,
        ["REC-08"],
    )
    assert await old.state() == "RETIRED"


async def test_ac4_a_crash_while_extracting_shows_no_clauses_until_the_retry_commits(
    stack: Stack, monkeypatch: pytest.MonkeyPatch
) -> None:
    api = stack.api
    upload_id = await upload_ready(stack)
    registered = await register(stack, upload_id)
    source_id: str = registered.json()["id"]
    real_commit = AsyncSession.commit

    class Boom(Exception):
        pass

    async def commit_unless_clauses_pending(self: AsyncSession) -> None:
        # The worker's own session sees its uncommitted inserts: fail exactly after them.
        if await self.scalar(select(func.count()).select_from(PolicySourceClause)):
            raise Boom
        await real_commit(self)

    monkeypatch.setattr(AsyncSession, "commit", commit_unless_clauses_pending)
    await stack.work()
    monkeypatch.setattr(AsyncSession, "commit", real_commit)

    crashed = await read_ingested(stack, source_id)
    assert crashed.source["extraction_state"] == "EXTRACTING"
    assert crashed.clauses == []
    assert await stack.work(owner="w2") == 0
    await stack.expire_lease("policy_sources", source_id)
    assert await stack.work(owner="w2") == 1

    done = await read_ingested(stack, source_id)
    assert done.source["extraction_state"] == "EXTRACTED"
    assert [clause["text"] for clause in done.clauses] == [CLAUSE_VOLUNTARY, CLAUSE_CLAIMS]
    assert (await api.ok("GET", f"/policy-sources/{source_id}")).json()["page_count"] is None


# ---- AC5: unauthorized users cannot alter rules, approve or activate


@dataclass(frozen=True)
class Targets:
    ready: Draft
    draft: Draft
    rule: Rule


async def targets(stack: Stack) -> Targets:
    # A bundle that could be activated right now and one that could be edited right now.
    ingested = await ingest(stack)
    ready, _ = await ready_bundle(stack, ingested)
    draft = await new_draft(stack.api)
    rule = await authored(
        draft,
        ingested.clause(CLAUSE_CLAIMS),
        rule_id="REC-08",
        requirement=REQUIREMENT_CLAIMS,
        review=False,
    )
    return Targets(ready, draft, rule)


async def observed(stack: Stack, drafts: Iterable[Draft]) -> dict[str, Any]:
    # Everything an unauthorized call must leave as it was.
    seen: dict[str, Any] = {"events": len(await stack.events())}
    for draft in drafts:
        detail = await draft.reload()
        seen[draft.id] = (
            detail["lifecycle_state"],
            detail["revision"],
            [(a["action"], a["actor"]) for a in await draft.audit()],
        )
    seen["active"] = (await active_policy(stack.api))["policy_bundle_id"]
    return seen


def alterations(found: Targets) -> list[tuple[str, str, Draft, dict[str, Any]]]:
    # Every way to edit a rule, review it or move a bundle: (method, path suffix, bundle, body).
    rule_path = f"/rules/{found.rule.record_id}"
    return [
        ("POST", "/rules", found.draft, rule_body("REC-99", REQUIREMENT_CLAIMS)),
        ("PATCH", rule_path, found.draft, {"title": "[SYNTHETIC] hijacked"}),
        (
            "POST",
            f"{rule_path}/wordings",
            found.draft,
            {"kind": "EXACT_WORDING", "text": "Anything."},
        ),
        ("POST", "/nodes", found.draft, {"node_key": "X", "node_type": "CLAUSE", "text": "x"}),
        (
            "POST",
            "/review-decisions",
            found.draft,
            {"item_kind": "node", "item_id": found.rule.node_id, "decision": "APPROVED"},
        ),
        ("POST", "/transitions", found.draft, {"command": "submit"}),
        ("POST", "/transitions", found.ready, {"command": "activate", "reason": REASON}),
        ("POST", "/transitions", found.ready, {"command": "retire", "reason": REASON}),
        ("POST", "/transitions", found.ready, {"command": "approve"}),
    ]


@pytest.mark.parametrize("sub", ["creator-a", "reviewer-a", "admin-a"])
async def test_ac5_roles_without_policy_rights_cannot_alter_review_approve_or_activate(
    stack: Stack, sub: str
) -> None:
    found = await targets(stack)
    drafts = [found.ready, found.draft]
    before = await observed(stack, drafts)

    for method, suffix, draft, body in alterations(found):
        await stack.api.refused(
            method,
            draft.path + suffix,
            sub,
            403,
            "FORBIDDEN",
            json=body,
            if_match=draft.revision,
        )

    assert await observed(stack, drafts) == before


async def test_ac5_editing_and_approving_are_kept_apart(stack: Stack) -> None:
    found = await targets(stack)
    drafts = [found.ready, found.draft]
    before = await observed(stack, drafts)
    # What an approver may not do (D-16): everything except approve.
    not_approving = [
        call
        for call in alterations(found)
        if call[1] != "/transitions" or call[3]["command"] != "approve"
    ]

    # A policy admin edits, submits and activates but cannot approve or reject (D-16).
    for command in ("approve", "reject"):
        refused = await stack.api.refused(
            "POST",
            f"{found.ready.path}/transitions",
            ADMIN,
            403,
            "FORBIDDEN",
            json={"command": command, "reason": REASON},
            if_match=found.ready.revision,
        )
        assert "cannot do this" in refused["message"]
    # An approver approves but cannot edit, review, submit, create, activate or retire.
    for method, suffix, draft, body in not_approving:
        await stack.api.refused(
            method,
            draft.path + suffix,
            APPROVER,
            403,
            "FORBIDDEN",
            json=body,
            if_match=draft.revision,
        )
    await stack.api.refused(
        "POST",
        "/policy-bundles",
        APPROVER,
        403,
        "FORBIDDEN",
        json={"name": "x", "source_ref": "x", "is_synthetic": True},
    )

    assert await observed(stack, drafts) == before


async def test_ac5_another_tenants_admin_cannot_alter_review_approve_or_activate(
    stack: Stack, client: httpx.AsyncClient, auth: AuthHeaders
) -> None:
    found = await targets(stack)
    drafts = [found.ready, found.draft]
    upload_id = await upload_ready(stack)
    before = await observed(stack, drafts)
    foreign = Api(client, auth, stack.world.b.tenant_id)

    for method, suffix, draft, body in alterations(found):
        # Capabilities are checked before the bundle is looked up, so each call comes from the
        # foreign role that holds its capability: only then is the 404 the tenant boundary.
        # D-16: only approve and reject need policy:approve; activation is the admin's.
        approving = suffix == "/transitions" and body["command"] in ("approve", "reject")
        await foreign.refused(
            method,
            draft.path + suffix,
            B_APPROVER if approving else B_ADMIN,
            404,
            "NOT_FOUND",
            json=body,
            if_match=draft.revision,
        )
    for suffix in ("", "/rules", "/rule-records", "/audit", "/checks"):
        await foreign.refused("GET", found.ready.path + suffix, B_ADMIN, 404, "NOT_FOUND")
    # Nor can they build on this tenant's bundle or register this tenant's upload.
    await foreign.refused(
        "POST",
        "/policy-bundles",
        B_ADMIN,
        404,
        "NOT_FOUND",
        json={
            "name": "x",
            "source_ref": "x",
            "is_synthetic": True,
            "based_on_bundle_id": found.ready.id,
        },
    )
    await foreign.refused(
        "POST",
        "/policy-sources",
        B_ADMIN,
        404,
        "NOT_FOUND",
        json={"upload_id": upload_id, "title": "x", "source_ref": "x", "is_synthetic": True},
    )

    assert await observed(stack, drafts) == before


# ---- AC6: activation, retirement and rollback name actor, reason and time, and notify


async def test_ac6_every_change_of_the_active_bundle_names_who_why_when_and_is_announced(
    stack: Stack,
) -> None:
    ingested = await ingest(stack)
    v2, _ = await ready_bundle(stack, ingested)
    v3, _ = await ready_bundle(stack, ingested, rule_id="REC-08")
    steps = [
        (v2, "activate", "[SYNTHETIC] v2 goes live.", "ACTIVATE", "READY", "ACTIVE"),
        (v3, "activate", "[SYNTHETIC] v3 replaces v2.", "ACTIVATE", "READY", "ACTIVE"),
        (v2, "rollback", "[SYNTHETIC] v3 was wrong.", "ROLLBACK", "RETIRED", "ACTIVE"),
        (v2, "retire", "[SYNTHETIC] Policy withdrawn.", "RETIRE", "ACTIVE", "RETIRED"),
    ]
    for draft, command, reason, *_ in steps:
        await draft.reload()
        await draft.move(command, ADMIN, reason)

    events = {event["audit_event_id"]: event for event in await stack.events()}
    assert Counter(event["action"] for event in events.values()) == {
        "ACTIVATE": 2,
        "ROLLBACK": 1,
        "RETIRE": 1,
    }
    for draft, _, reason, action, from_state, to_state in steps:
        (entry,) = [
            item
            for item in await draft.audit()
            if item["action"] == action
            and item["reason"] == reason
            and item["from_state"] == from_state
        ]
        assert (entry["actor"], entry["to_state"]) == (ADMIN, to_state)
        assert entry["request_id"]
        assert datetime.fromisoformat(entry["created_at"]).utcoffset() is not None
        # The outbox row points at that audit entry, so a consumer can tell who and why.
        event = events[entry["id"]]
        assert event["event_version"] == 1
        assert event["tenant_id"] == str(stack.world.a.tenant_id)
        assert (event["policy_bundle_id"], event["version"], event["action"]) == (
            draft.id,
            draft.version,
            action,
        )
    # Replacing the active bundle is announced with the bundle it replaced, and recorded on it.
    announced = {e["previous_bundle_id"] for e in events.values() if e["action"] == "ACTIVATE"}
    assert announced == {str(stack.world.a.policy_bundle_id), v2.id}
    replaced = [
        item for item in await v2.audit() if item["details"].get("replaced_by_bundle_id") == v3.id
    ]
    assert [(item["action"], item["actor"], item["reason"]) for item in replaced] == [
        ("RETIRE", ADMIN, "[SYNTHETIC] v3 replaces v2.")
    ]


async def test_ac6_a_change_without_a_reason_is_refused_and_announces_nothing(
    stack: Stack,
) -> None:
    ingested = await ingest(stack)
    draft, _ = await ready_bundle(stack, ingested)
    before = await draft.reload()

    missing = await draft.move("activate", ADMIN, status=409)
    blank = await stack.api.refused(
        "POST",
        f"{draft.path}/transitions",
        APPROVER,
        422,
        "VALIDATION_FAILED",
        json={"command": "activate", "reason": "   "},
        if_match=draft.revision,
    )

    assert missing.json()["code"] == "POLICY_ITEM_INVALID"
    assert missing.json()["details"] == {"field": "reason"}
    assert blank["code"] == "VALIDATION_FAILED"
    assert await draft.reload() == before
    assert await stack.events() == []
    assert {a["action"] for a in await draft.audit()}.isdisjoint({"ACTIVATE"})


# ---- AC7: exact required language is preserved, never paraphrased


async def test_ac7_exact_wording_survives_every_step_byte_for_byte(stack: Stack) -> None:
    api = stack.api
    # Typographic quotes, an em dash, accents and a doubled space: nothing a model might tidy.
    wording = "“Taking part is  voluntary — you may withdraw at any time, sans pénalité.”"
    clause_text = f"[SYNTHETIC] Posts must say exactly: {wording} Nothing else counts."
    ingested = await ingest(
        stack, f"# [SYNTHETIC] Wording\n\n{clause_text}\n".encode(), "text/markdown"
    )
    clause = ingested.clause(clause_text)
    assert wording in clause["text"]
    draft = await new_draft(api)
    await authored(
        draft,
        clause,
        rule_id="REC-07",
        requirement=REQUIREMENT_VOLUNTARY,
        wording=(WordingKind.EXACT_WORDING, wording),
    )
    await build(stack, draft)
    await activate(draft)

    records = items(await api.ok("GET", f"{draft.path}/rule-records", APPROVER))
    assert [w["text"].encode() for w in records[0]["wordings"]] == [wording.encode()]
    evidence = rule_by_id(await bundle_rules(api, draft), "REC-07")
    assert evidence["exact_wording"].encode() == wording.encode()
    stored = await stack.sql("SELECT exact_wording FROM policy_rules WHERE rule_id = 'REC-07'")
    assert [row.exact_wording.encode() for row in stored] == [wording.encode()]


PARAPHRASES = {
    "reworded": "Taking part in this research study is voluntary.",
    "recased": "this is a research study. taking part is voluntary.",
}


@pytest.mark.parametrize("paraphrase", PARAPHRASES.values(), ids=PARAPHRASES.keys())
async def test_ac7_a_paraphrase_cannot_be_approved_and_only_the_verbatim_wording_can(
    stack: Stack, paraphrase: str
) -> None:
    api = stack.api
    ingested = await ingest(stack)
    draft = await new_draft(api)
    rule = await authored(
        draft,
        ingested.clause(CLAUSE_VOLUNTARY),
        rule_id="REC-07",
        requirement=REQUIREMENT_VOLUNTARY,
        wording=(WordingKind.EXACT_WORDING, paraphrase),
    )
    await draft.move("submit", ADMIN)

    refused = await draft.move("approve", APPROVER, status=409)

    issues = refused.json()["details"]["issues"]
    assert [(i["code"], i["subject"]) for i in issues] == [
        ("EXACT_WORDING_NOT_IN_SOURCE", "rule:REC-07")
    ]
    assert "verbatim" in issues[0]["message"]
    # The fix: back to draft, reject the paraphrase, add the source's own words, review again.
    await draft.move("return_to_draft", ADMIN)
    await draft.review([("wording", rule.wording_id or "")], "REJECTED")
    verbatim = await draft.edit(
        f"/rules/{rule.record_id}/wordings", {"kind": "EXACT_WORDING", "text": WORDING}
    )
    await draft.review([("wording", verbatim["id"])])
    await build(stack, draft)
    await activate(draft)

    evidence = rule_by_id(await bundle_rules(api, draft), "REC-07")
    assert evidence["exact_wording"].encode() == WORDING.encode()


# ---- AC8: the frontend can list sources, view rule details and render review/activation state


GUIDE_ROUTES = [
    ("post", "/policy-sources/uploads"),
    ("post", "/policy-sources/uploads/{upload_id}/complete"),
    ("get", "/policy-sources/uploads/{upload_id}"),
    ("post", "/policy-sources"),
    ("get", "/policy-sources"),
    ("get", "/policy-sources/{source_id}"),
    ("get", "/policy-sources/{source_id}/clauses"),
    ("post", "/policy-bundles"),
    ("get", "/policy-bundles"),
    ("get", "/policy-bundles/{policy_bundle_id}"),
    ("post", "/policy-bundles/{policy_bundle_id}/nodes"),
    ("patch", "/policy-bundles/{policy_bundle_id}/nodes/{node_id}"),
    ("post", "/policy-bundles/{policy_bundle_id}/rules"),
    ("patch", "/policy-bundles/{policy_bundle_id}/rules/{rule_id}"),
    ("post", "/policy-bundles/{policy_bundle_id}/edges"),
    ("post", "/policy-bundles/{policy_bundle_id}/rules/{rule_id}/wordings"),
    ("post", "/policy-bundles/{policy_bundle_id}/review-decisions"),
    ("get", "/policy-bundles/{policy_bundle_id}/checks"),
    ("post", "/policy-bundles/{policy_bundle_id}/transitions"),
    ("get", "/policy-bundles/{policy_bundle_id}/audit"),
    ("get", "/policy-bundles/{policy_bundle_id}/rule-records"),
    ("get", "/policy-bundles/{policy_bundle_id}/nodes"),
    ("get", "/policy-bundles/{policy_bundle_id}/edges"),
    ("get", "/policy-bundles/{policy_bundle_id}/rules"),
    ("get", "/policies/active"),
]


def test_ac8_every_route_of_the_guide_is_in_the_published_contract() -> None:
    paths = CONTRACT["paths"]

    missing = [(m, p) for m, p in GUIDE_ROUTES if m not in paths.get(f"{BASE}{p}", {})]

    assert missing == []


def shape(example: str) -> set[str]:
    return set(EXAMPLES[example]["body"])


def item_shape(example: str) -> set[str]:
    return set(EXAMPLES[example]["body"]["items"][0])


async def test_ac8_the_guide_flow_returns_the_documented_shapes_and_states(stack: Stack) -> None:
    api = stack.api
    ingested = await ingest(stack)

    # Steps 1 and 2: 201 then 200 with Location, and the source and clause shapes of the examples.
    again = await register(stack, ingested.source["upload_id"])
    assert again.status_code == 200
    assert again.headers["Location"] == f"{BASE}/policy-sources/{ingested.source['id']}"
    assert set(ingested.source) == shape("policy_source")
    assert set(ingested.clauses[0]) == item_shape("policy_clauses")
    listing = items(await api.ok("GET", "/policy-sources", APPROVER))
    assert set(listing[0]) == shape("policy_source")

    # Steps 3 to 6: a draft with an unreviewed rule shows its issue before approval.
    draft = await new_draft(api)
    assert set(await draft.reload()) == shape("policy_bundle_detail")
    states = [await draft.state()]
    rule = await authored(
        draft,
        ingested.clause(CLAUSE_VOLUNTARY),
        rule_id="REC-07",
        requirement=REQUIREMENT_VOLUNTARY,
        wording=(WordingKind.EXACT_WORDING, WORDING),
        review=False,
    )
    unreviewed = await api.ok("GET", f"{draft.path}/checks", APPROVER)
    assert unreviewed.json()["issues"]
    assert set(unreviewed.json()["issues"][0]) == set(
        EXAMPLES["policy_checks_with_issue"]["body"]["issues"][0]
    )
    proposed = items(
        await api.ok("GET", f"{draft.path}/rule-records", params={"include_proposals": "true"})
    )
    assert set(proposed[0]) == item_shape("policy_rules")
    assert proposed[0]["node"]["review_state"] == "PROPOSED"
    assert set(proposed[0]["node"]) == set(EXAMPLES["policy_rules"]["body"]["items"][0]["node"])
    assert set(proposed[0]["wordings"][0]) == set(
        EXAMPLES["policy_rules"]["body"]["items"][0]["wordings"][0]
    )
    await draft.review(rule.items())
    reviewed = items(await api.ok("GET", f"{draft.path}/nodes", APPROVER))
    assert {(n["review_state"], n["origin"]) for n in reviewed} == {("APPROVED", "HUMAN")}
    assert (await api.ok("GET", f"{draft.path}/checks", APPROVER)).json() == {"issues": []}

    # Step 7: each state is what the guide says; the error body has the documented shape.
    await draft.move("submit", ADMIN)
    states.append(await draft.state())
    approved = await draft.move("approve", APPROVER)
    assert set(approved.json()) == shape("policy_bundle_detail")
    states.append(approved.json()["lifecycle_state"])
    assert await stack.work() == 1
    states.append(await draft.state())
    ready_etag = (await api.ok("GET", draft.path, APPROVER)).headers["ETag"]
    assert ready_etag == f'"{draft.revision}"'
    activated = await draft.move("activate", ADMIN, REASON)
    assert set(activated.json()) == shape("policy_bundle_active")
    states.append(activated.json()["lifecycle_state"])
    assert states == ["DRAFT", "REVIEW", "APPROVED", "READY", "ACTIVE"]

    issue_probe = await new_draft(api)
    await authored(
        issue_probe,
        ingested.clause(CLAUSE_CLAIMS),
        rule_id="REC-08",
        requirement=REQUIREMENT_CLAIMS,
        review=False,
    )
    await issue_probe.move("submit", ADMIN)
    error = await issue_probe.move("approve", APPROVER, status=409)
    documented = EXAMPLES["error_policy_bundle_has_issues"]["body"]
    assert set(error.json()) == set(documented)
    assert set(error.json()["details"]["issues"][0]) == set(documented["details"]["issues"][0])

    # Step 8 and paging: the audit trail and the bundle list as documented.
    audit = await api.ok("GET", f"{draft.path}/audit", APPROVER)
    assert set(audit.json()) == shape("policy_audit_page")
    assert set(audit.json()["items"][0]) == item_shape("policy_audit_page")
    first = await api.ok("GET", "/policy-bundles", params={"limit": 1})
    assert [b["version"] for b in items(first)] == [3]
    older = await api.ok(
        "GET", "/policy-bundles", params={"limit": 1, "before_version": first.json()["next_cursor"]}
    )
    assert [b["version"] for b in items(older)] == [2]
    assert (await bundle_rules(api, draft))["lifecycle_state"] == "ACTIVE"
