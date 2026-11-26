# Importing the policy deliverable's bundle document (v0.3) into a DRAFT bundle: lossless storage,
# the runtime graph it maps to, agreement with the reference contract (tests/nrc_policy_ref), and
# the guards (checksums, DRAFT only, one document per bundle). Real Postgres; synthetic tenants.

import copy
import json
from pathlib import Path
from typing import Any
from uuid import UUID

import pytest
from sqlalchemy import func, select, update
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncConnection, AsyncSession

from app.models import (
    PolicyBundle,
    PolicyEdge,
    PolicyNode,
    PolicyRuleRecord,
    PolicyRuleWording,
    PolicySourceClause,
)
from app.models.enums import (
    BundleLifecycleState,
    CheckCode,
    PolicyEdgeType,
    PolicyLanguage,
    PolicyNodeType,
    ProposalOrigin,
    ReviewState,
    WordingKind,
)
from app.services.policy_checks import load_snapshot, run_checks
from scripts.import_policy_bundle import (
    BUNDLE_V03_PATH,
    V03_STUDY_SOURCES,
    content_sha256,
    export_policy_bundle,
    import_policy_bundle,
    load_document,
    runtime_rules,
    verify_document,
)
from scripts.import_policy_graph import PolicyGraphImportError
from tests.conftest import World, add_policy_bundle, as_tenant_owner, owner_session
from tests.nrc_policy_ref.contract import build_graph, load_bundle
from tests.nrc_policy_ref.contract import content_sha256 as reference_sha256

DOCUMENT = load_document(BUNDLE_V03_PATH)
RUNTIME = {rule.rule_id for rule in runtime_rules(verify_document(DOCUMENT))}
UNQUOTED = {
    rule["rule_id"]
    for rule in DOCUMENT["rules"]
    if rule["rule_id"] in RUNTIME and not rule["source_quotations"]
}


@pytest.fixture
def seed_graph() -> bool:
    # Tenant A's bundles start without NRC-5's synthetic graph.
    return False


@pytest.fixture
async def draft(connection: AsyncConnection, world: World) -> UUID:
    async with as_tenant_owner(connection, world.a.tenant_id):
        return await add_policy_bundle(
            connection, world.a.tenant_id, 50, BundleLifecycleState.DRAFT
        )


def canonical(document: dict[str, Any]) -> str:
    return json.dumps(document, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def rehashed(document: dict[str, Any]) -> dict[str, Any]:
    # An edited document with a fresh content_sha256, so the edit itself is what gets refused.
    document["bundle"]["content_sha256"] = content_sha256(document)
    return document


async def run_import(
    connection: AsyncConnection, world: World, bundle_id: UUID, document: dict[str, Any]
) -> None:
    async with owner_session(connection, world.a.tenant_id) as db:
        await import_policy_bundle(db, world.a.tenant_id, bundle_id, document)


async def count(db: AsyncSession, model: Any, bundle_id: UUID) -> int:
    return int(
        await db.scalar(
            select(func.count()).select_from(model).where(model.policy_bundle_id == bundle_id)
        )
        or 0
    )


async def test_the_runtime_rules_become_a_proposed_graph(
    connection: AsyncConnection, world: World, draft: UUID
) -> None:
    await run_import(connection, world, draft, copy.deepcopy(DOCUMENT))

    async with owner_session(connection, world.a.tenant_id) as db:
        rules = (
            await db.scalars(
                select(PolicyRuleRecord).where(PolicyRuleRecord.policy_bundle_id == draft)
            )
        ).all()
        nodes = (
            await db.scalars(select(PolicyNode).where(PolicyNode.policy_bundle_id == draft))
        ).all()
        edges = (
            await db.scalars(select(PolicyEdge).where(PolicyEdge.policy_bundle_id == draft))
        ).all()

    records = {rule["rule_id"]: rule for rule in DOCUMENT["rules"]}
    assert {rule.rule_id for rule in rules} == RUNTIME
    assert len(RUNTIME) == 15
    # Each rule row keeps its own record (outcome policy, stage, study predicates) as imported.
    assert all(rule.contract == records[rule.rule_id] for rule in rules)
    assert {rule.rule_id: rule.mandatory for rule in rules} == {
        rid: records[rid]["outcome_policy"]["mandatory"] for rid in RUNTIME
    }
    # Nothing in v0.3 is approved yet: every row waits for a reviewer.
    reviews = {(node.review_state, node.origin) for node in nodes}
    reviews |= {(edge.review_state, edge.origin) for edge in edges}
    assert reviews == {(ReviewState.PROPOSED, ProposalOrigin.IMPORT)}
    image_only = next(node for node in nodes if node.node_key == "IMG-01")
    assert image_only.input_kinds == ["image"]
    assert not {"CH-01", "FED-01", "FED-10", "ORG-01"} & {node.node_key for node in nodes}


async def test_export_gives_back_the_document_it_imported(
    connection: AsyncConnection, world: World, draft: UUID, tmp_path: Path
) -> None:
    await run_import(connection, world, draft, copy.deepcopy(DOCUMENT))

    async with owner_session(connection, world.a.tenant_id) as db:
        exported = await export_policy_bundle(db, world.a.tenant_id, draft)
        stored = await db.scalar(
            select(PolicyBundle.contract_sha256).where(PolicyBundle.id == draft)
        )

    assert canonical(exported) == canonical(DOCUMENT)
    assert stored == DOCUMENT["bundle"]["content_sha256"] == reference_sha256(exported)
    # The reference loader accepts the export: schema, content and quotation checksums.
    path = tmp_path / "exported.json"
    path.write_text(json.dumps(exported, ensure_ascii=False), encoding="utf-8")
    assert load_bundle(path)["bundle"]["bundle_version"] == "0.3.0"


async def test_the_graph_agrees_with_the_reference_mapping(
    connection: AsyncConnection, world: World, draft: UUID
) -> None:
    await run_import(connection, world, draft, copy.deepcopy(DOCUMENT))
    ref_nodes, ref_edges = build_graph(DOCUMENT, include_proposed_exceptions=True)

    async with owner_session(connection, world.a.tenant_id) as db:
        nodes = {
            node.id: node
            for node in await db.scalars(
                select(PolicyNode).where(PolicyNode.policy_bundle_id == draft)
            )
        }
        clause_sha = dict(
            (
                await db.execute(
                    select(PolicyNode.node_key, PolicySourceClause.checksum_sha256)
                    .join(PolicySourceClause, PolicySourceClause.id == PolicyNode.source_clause_id)
                    .where(PolicyNode.policy_bundle_id == draft)
                )
            ).all()
        )
        edges = (
            await db.scalars(select(PolicyEdge).where(PolicyEdge.policy_bundle_id == draft))
        ).all()
        pointers = (
            await db.execute(
                select(PolicyRuleRecord.rule_id, PolicyRuleWording.text)
                .join(PolicyRuleRecord, PolicyRuleRecord.id == PolicyRuleWording.policy_rule_id)
                .where(
                    PolicyRuleWording.policy_bundle_id == draft,
                    PolicyRuleWording.kind == WordingKind.STUDY_APPROVED_WORDING,
                )
            )
        ).all()

    def runtime(rule_id: str) -> bool:
        return rule_id in RUNTIME

    # Quotations: same ids and checksums as the reference's SourceQuotation nodes.
    assert clause_sha == {
        node["id"]: node["sha256"]
        for node in ref_nodes
        if node["type"] == "SourceQuotation" and runtime(node["id"].rsplit("-Q", 1)[0])
    }
    links = {
        (nodes[edge.from_node_id].node_key, edge.edge_type, nodes[edge.to_node_id].node_key)
        for edge in edges
    }
    assert {link for link in links if link[1] is PolicyEdgeType.SUPPORTED_BY} == {
        (edge["from"], PolicyEdgeType.SUPPORTED_BY, edge["to"])
        for edge in ref_edges
        if edge["type"] == "QUOTES" and runtime(edge["from"])
    }
    assert {link for link in links if link[1] is PolicyEdgeType.EXCEPT_WHEN} == {
        (edge["from"], PolicyEdgeType.EXCEPT_WHEN, edge["to"])
        for edge in ref_edges
        if edge["type"] == "HAS_EXCEPTION" and runtime(edge["from"])
    }
    # Required post wording: a pointer to the study snapshot, never text from a source. The
    # v0.3 names become the StudyFacts fields that hold the approved texts.
    assert set(pointers) == {
        (edge["from"], V03_STUDY_SOURCES[edge["to"]].value)
        for edge in ref_edges
        if edge["type"] == "REQUIRES_DISCLOSURE" and runtime(edge["from"])
    }
    assert {nodes[e.to_node_id].node_type for e in edges} <= {
        PolicyNodeType.CLAUSE,
        PolicyNodeType.EXCEPTION,
    }


async def test_once_reviewed_only_rules_without_a_quotation_lack_provenance(
    connection: AsyncConnection, world: World, draft: UUID
) -> None:
    # AC1 on the real bundle: approving every imported item still leaves the rules that quote no
    # source unsupported, so the bundle cannot be approved until someone adds their provenance.
    await run_import(connection, world, draft, copy.deepcopy(DOCUMENT))
    async with owner_session(connection, world.a.tenant_id) as db:
        reviewed = {
            "review_state": ReviewState.APPROVED,
            "reviewed_by": "synthetic-reviewer",
            "reviewed_at": func.now(),
        }
        for model in (PolicyNode, PolicyEdge):
            await db.execute(
                update(model)
                .where(model.policy_bundle_id == draft)
                .values(approved=True, **reviewed)
            )
        await db.execute(
            update(PolicyRuleWording)
            .where(PolicyRuleWording.policy_bundle_id == draft)
            .values(**reviewed)
        )
        issues = run_checks(await load_snapshot(db, world.a.tenant_id, draft), (PolicyLanguage.EN,))

    assert UNQUOTED
    assert {(issue.code, issue.subject) for issue in issues} == {
        (CheckCode.MISSING_PROVENANCE, f"rule:{rule_id}") for rule_id in UNQUOTED
    }


@pytest.mark.parametrize(
    ("edit", "message"),
    [
        (lambda d: d["rules"][1]["outcome_policy"].update(mandatory=False), "content_sha256"),
        (
            lambda d: (
                rehashed(d)
                if d["rules"][1]["source_quotations"][0].update(text="(edited)") is None
                else d
            ),
            "text_sha256",
        ),
    ],
    ids=["content", "quotation"],
)
async def test_an_edited_document_is_refused_and_nothing_is_written(
    connection: AsyncConnection, world: World, draft: UUID, edit: Any, message: str
) -> None:
    document = copy.deepcopy(DOCUMENT)
    edit(document)

    with pytest.raises(PolicyGraphImportError, match=message):
        await run_import(connection, world, draft, document)

    async with owner_session(connection, world.a.tenant_id) as db:
        assert await count(db, PolicyNode, draft) == 0
        assert (
            await db.scalar(select(PolicyBundle.contract_sha256).where(PolicyBundle.id == draft))
            is None
        )


@pytest.mark.parametrize(
    ("field", "edit", "message"),
    [
        ("language", lambda d: d["scope"].update(enabled_languages=["es"]), "enabled_languages"),
        # rules[2] is FED-02, a validation-stage rule the import maps.
        ("check method", lambda d: d["rules"][2]["check"].update(method="telepathy"), "telepathy"),
        ("gate", lambda d: d["rules"][1].update(gate="vibes"), "gate"),
        (
            "post wording",
            lambda d: d["rules"][4]["required_post_wording"].update(source="study_snapshot.x"),
            "source",
        ),
        ("video only", lambda d: d["rules"][2]["check"].update(inputs=["video"]), "not supported"),
    ],
)
async def test_one_wrong_value_per_mapped_field_is_refused(
    connection: AsyncConnection, world: World, draft: UUID, field: str, edit: Any, message: str
) -> None:
    document = copy.deepcopy(DOCUMENT)
    edit(document)

    with pytest.raises(PolicyGraphImportError, match=message):
        await run_import(connection, world, draft, rehashed(document))

    async with owner_session(connection, world.a.tenant_id) as db:
        assert await count(db, PolicyNode, draft) == 0, field


async def test_the_same_document_again_changes_nothing_and_another_is_refused(
    connection: AsyncConnection, world: World, draft: UUID
) -> None:
    await run_import(connection, world, draft, copy.deepcopy(DOCUMENT))
    await run_import(connection, world, draft, copy.deepcopy(DOCUMENT))
    other = copy.deepcopy(DOCUMENT)
    other["rules"][1]["title"] += " (renamed)"

    with pytest.raises(PolicyGraphImportError, match="holds another document"):
        await run_import(connection, world, draft, rehashed(other))

    async with owner_session(connection, world.a.tenant_id) as db:
        assert await count(db, PolicyRuleRecord, draft) == len(RUNTIME)


async def test_only_a_draft_takes_an_import(connection: AsyncConnection, world: World) -> None:
    assert world.a.policy_bundle_id is not None

    with pytest.raises(PolicyGraphImportError, match="import into a DRAFT"):
        await run_import(connection, world, world.a.policy_bundle_id, copy.deepcopy(DOCUMENT))


async def test_a_bundle_document_needs_its_checksum(
    connection: AsyncConnection, world: World, draft: UUID
) -> None:
    async with as_tenant_owner(connection, world.a.tenant_id):
        with pytest.raises(IntegrityError, match="contract_has_checksum"):
            async with connection.begin_nested():
                await connection.execute(
                    update(PolicyBundle)
                    .where(PolicyBundle.id == draft)
                    .values(contract_document={"rules": []})
                )


# ---- Reading the file


def test_a_repeated_key_in_the_file_is_refused(tmp_path: Path) -> None:
    # A second "approval" would otherwise win, and the checksum of the parsed document matches.
    path = tmp_path / "bundle.json"
    path.write_text('{"bundle": {"approval": "a", "approval": "b"}}', encoding="utf-8")

    with pytest.raises(PolicyGraphImportError, match="Repeated keys in one object: approval"):
        load_document(path)


@pytest.mark.parametrize("constant", ["NaN", "Infinity", "-Infinity"])
def test_a_non_number_constant_is_refused(tmp_path: Path, constant: str) -> None:
    path = tmp_path / "bundle.json"
    path.write_text(f'{{"bundle": {{"weight": {constant}}}}}', encoding="utf-8")

    with pytest.raises(PolicyGraphImportError, match=f"{constant} is not a JSON number"):
        load_document(path)


def test_a_file_over_the_size_cap_is_refused_before_it_is_read(tmp_path: Path) -> None:
    path = tmp_path / "bundle.json"
    path.write_text('{"bundle": "0123456789"}', encoding="utf-8")
    size = path.stat().st_size

    assert load_document(path, max_bytes=size) == {"bundle": "0123456789"}
    with pytest.raises(PolicyGraphImportError, match=f"at most {size - 1} bytes"):
        load_document(path, max_bytes=size - 1)


def test_a_file_that_is_not_utf8_json_is_refused(tmp_path: Path) -> None:
    path = tmp_path / "bundle.json"
    path.write_bytes(b'{"bundle": "caf\xe9"}')

    with pytest.raises(PolicyGraphImportError, match="UTF-8 JSON"):
        load_document(path)
