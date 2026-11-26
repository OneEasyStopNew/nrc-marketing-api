"""Restricted import of approved study snapshots (scripts.import_study_snapshot, NRC-2)."""

import hashlib
import importlib.util
import json
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any
from uuid import UUID, uuid4

import httpx
import pytest
from pydantic import ValidationError
from sqlalchemy import select, text
from sqlalchemy.exc import DBAPIError, IntegrityError
from sqlalchemy.ext.asyncio import AsyncConnection, AsyncSession

from app.db.session import TenantSession, bind_tenant
from app.models import MembershipStudy, Study, StudySnapshot
from app.models.enums import Language
from app.schemas.tenancy import ApprovedWording, StudyFacts
from scripts.grant_access import UnknownTenantError
from scripts.import_study_snapshot import (
    MAX_FILE_BYTES,
    ImportResult,
    SnapshotDocument,
    SnapshotFileError,
    SnapshotImportRefusedError,
    import_snapshot,
    load_document,
)
from tests.conftest import AuthHeaders, SeededTenant, World, _session, seed_tenant

SOURCE_ID = "SM-2026-0142"


def _file(**changes: Any) -> dict[str, Any]:
    # A labelled synthetic snapshot file; tests change one thing at a time.
    body: dict[str, Any] = {
        "external_ref": SOURCE_ID,
        "source_ref": f"study-management:{SOURCE_ID}:v7",
        "title": "[SYNTHETIC] Blood pressure study",
        "recruitment_status": "recruiting",
        "language": "en",
        "facts": {
            "condition": "high blood pressure",
            "eligibility_summary": "Adults aged 18 to 75 who have been told they have high "
            "blood pressure.",
            "min_age": 18,
            "max_age": 75,
            "sites": ["Synthetic Site North"],
            "approved_claims": [
                {"key": "treatment", "text": "The study is testing an investigational treatment."}
            ],
            "required_disclosures": [
                {"key": "voluntary", "text": "This is a research study. Taking part is voluntary."}
            ],
            "compensation_statement": None,
            "contact_url": "https://example.org/synthetic-bp-study",
        },
        "approved_by": "reviewer.synthetic@example.org",
        "approved_at": "2026-11-20T09:30:00+00:00",
        "is_synthetic": True,
    }
    body.update(changes)
    return body


def _doc(**changes: Any) -> SnapshotDocument:
    return load_document(json.dumps(_file(**changes)).encode())


def _owner_session(conn: AsyncConnection) -> AsyncSession:
    return AsyncSession(
        bind=conn,
        join_transaction_mode="create_savepoint",
        expire_on_commit=False,
        sync_session_class=TenantSession,
    )


async def _tenant(db: AsyncSession) -> SeededTenant:
    return await seed_tenant(db, name="Import", members={}, with_policy=False)


async def _snapshots(db: AsyncSession, study_id: UUID) -> list[StudySnapshot]:
    return list(
        await db.scalars(
            select(StudySnapshot)
            .where(StudySnapshot.study_id == study_id)
            .order_by(StudySnapshot.version)
        )
    )


async def test_a_new_source_study_is_created_with_snapshot_version_1(
    connection: AsyncConnection,
) -> None:
    async with _owner_session(connection) as db:
        tenant = await _tenant(db)

        outcome = await import_snapshot(db, tenant.tenant_id, _doc())

        study = await db.get(Study, outcome.study_id)
        [snapshot] = await _snapshots(db, outcome.study_id)
    assert (outcome.result, outcome.version) == (ImportResult.CREATED, 1)
    assert study is not None
    assert (study.external_ref, study.title) == (SOURCE_ID, "[SYNTHETIC] Blood pressure study")
    assert snapshot.id == outcome.snapshot_id
    assert snapshot.source_ref == f"study-management:{SOURCE_ID}:v7"
    assert snapshot.approved_by == "reviewer.synthetic@example.org"
    assert snapshot.approved_at == datetime(2026, 11, 20, 9, 30, tzinfo=UTC)
    assert snapshot.is_synthetic is True
    assert snapshot.facts["approved_claims"] == [
        "The study is testing an investigational treatment."
    ]


async def test_importing_the_same_source_version_again_changes_nothing(
    connection: AsyncConnection,
) -> None:
    async with _owner_session(connection) as db:
        tenant = await _tenant(db)
        first = await import_snapshot(db, tenant.tenant_id, _doc())

        again = await import_snapshot(db, tenant.tenant_id, _doc())

        snapshots = await _snapshots(db, first.study_id)
    assert again.result is ImportResult.UNCHANGED
    assert (again.study_id, again.snapshot_id, again.version) == (
        first.study_id,
        first.snapshot_id,
        1,
    )
    assert len(snapshots) == 1


async def test_a_new_source_version_adds_the_next_snapshot_and_keeps_the_old_one(
    connection: AsyncConnection,
) -> None:
    async with _owner_session(connection) as db:
        tenant = await _tenant(db)
        first = await import_snapshot(db, tenant.tenant_id, _doc())

        closed = await import_snapshot(
            db,
            tenant.tenant_id,
            _doc(
                source_ref=f"study-management:{SOURCE_ID}:v8",
                recruitment_status="closed",
                title="[SYNTHETIC] Blood pressure study (closed)",
            ),
        )

        study = await db.get(Study, first.study_id)
        snapshots = await _snapshots(db, first.study_id)
    assert (closed.result, closed.study_id, closed.version) == (
        ImportResult.NEW_VERSION,
        first.study_id,
        2,
    )
    assert [(s.version, s.recruitment_status.value) for s in snapshots] == [
        (1, "recruiting"),
        (2, "closed"),
    ]
    assert study is not None
    assert study.title == "[SYNTHETIC] Blood pressure study (closed)"


@pytest.mark.parametrize(
    "changes",
    [
        {"recruitment_status": "closed"},
        {"approved_by": "someone.else@example.org"},
        {"title": "[SYNTHETIC] Renamed study"},
    ],
    ids=["status", "approver", "title"],
)
async def test_a_source_version_cannot_change_its_content(
    connection: AsyncConnection, changes: dict[str, Any]
) -> None:
    async with _owner_session(connection) as db:
        tenant = await _tenant(db)
        first = await import_snapshot(db, tenant.tenant_id, _doc())

        with pytest.raises(SnapshotImportRefusedError, match="imported with different content"):
            await import_snapshot(db, tenant.tenant_id, _doc(**changes))

        [snapshot] = await _snapshots(db, first.study_id)
    assert snapshot.recruitment_status.value == "recruiting"


async def test_an_older_source_version_cannot_roll_the_study_back(
    connection: AsyncConnection,
) -> None:
    async with _owner_session(connection) as db:
        tenant = await _tenant(db)
        await import_snapshot(db, tenant.tenant_id, _doc())
        await import_snapshot(
            db,
            tenant.tenant_id,
            _doc(source_ref=f"study-management:{SOURCE_ID}:v8", recruitment_status="closed"),
        )

        with pytest.raises(SnapshotImportRefusedError, match="older than the latest"):
            await import_snapshot(db, tenant.tenant_id, _doc())


async def test_synthetic_and_real_snapshots_never_share_a_study(
    connection: AsyncConnection,
) -> None:
    async with _owner_session(connection) as db:
        tenant = await _tenant(db)
        await import_snapshot(db, tenant.tenant_id, _doc())

        with pytest.raises(SnapshotImportRefusedError, match="never share a study"):
            await import_snapshot(
                db,
                tenant.tenant_id,
                _doc(
                    source_ref=f"study-management:{SOURCE_ID}:v8",
                    title="Blood pressure study",
                    is_synthetic=False,
                ),
            )


async def test_the_same_source_id_in_another_tenant_is_a_separate_study(
    connection: AsyncConnection,
) -> None:
    async with _owner_session(connection) as db:
        one = await _tenant(db)
        other = await seed_tenant(db, name="Other", members={}, with_policy=False)

        first = await import_snapshot(db, one.tenant_id, _doc())
        second = await import_snapshot(db, other.tenant_id, _doc())

    assert second.result is ImportResult.CREATED
    assert second.study_id != first.study_id


async def test_one_source_id_names_one_study_per_tenant(connection: AsyncConnection) -> None:
    # The import looks studies up by source id, so the database keeps it unique per tenant.
    async with _owner_session(connection) as db:
        tenant = await _tenant(db)
        await bind_tenant(db, tenant.tenant_id)
        db.add(Study(tenant_id=tenant.tenant_id, title="[SYNTHETIC] A", external_ref=SOURCE_ID))
        await db.flush()
        db.add(Study(tenant_id=tenant.tenant_id, title="[SYNTHETIC] B", external_ref=SOURCE_ID))

        with pytest.raises(IntegrityError, match="uq_studies_tenant_id_external_ref"):
            await db.flush()


async def test_an_unknown_tenant_is_refused(connection: AsyncConnection) -> None:
    async with _owner_session(connection) as db:
        with pytest.raises(UnknownTenantError, match="no tenant with id"):
            await import_snapshot(db, uuid4(), _doc())


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"facts": {**_file()["facts"], "patient_name": "x"}}, "extra_forbidden"),
        ({"facts": {**_file()["facts"], "min_age": 80}}, "min_age must not be greater"),
        ({"facts": {**_file()["facts"], "approved_claims": []}}, "at least 1 item"),
        (
            {"facts": {**_file()["facts"], "approved_claims": [{"key": "a", "text": "   "}]}},
            "must not be blank",
        ),
        (
            {"facts": {**_file()["facts"], "approved_claims": ["The study is testing it."]}},
            "model_type",
        ),
        (
            {"facts": {**_file()["facts"], "approved_claims": [{"key": "Claim 1", "text": "x"}]}},
            "string_pattern_mismatch",
        ),
        (
            {
                "facts": {
                    **_file()["facts"],
                    "required_disclosures": [
                        {"key": "same", "text": "One."},
                        {"key": "same", "text": "Two."},
                    ],
                }
            },
            "keys must be unique",
        ),
        ({"facts": {**_file()["facts"], "sites": ["North\u200bSite"]}}, "control characters"),
        ({"facts": {**_file()["facts"], "contact_url": "http://example.org"}}, "https URL"),
        ({"approved_at": "2026-11-20T09:30:00"}, "timezone"),
        (
            {"approved_at": (datetime.now(UTC) + timedelta(days=1)).isoformat()},
            "must not be in the future",
        ),
        ({"title": "Blood pressure study"}, "exactly when is_synthetic is true"),
        ({"title": "[SYNTHETIC] Real study", "is_synthetic": False}, "exactly when"),
        ({"is_synthetic": "yes"}, "valid boolean"),
        ({"external_ref": "SM 0142"}, "string_pattern_mismatch"),
        ({"approved_by": ""}, "at least 1 character"),
        ({"source_ref": None}, "string_type"),
        ({"language": "fr"}, "enum"),
    ],
    ids=[
        "unknown-fact",
        "ages-reversed",
        "no-claims",
        "blank-claim",
        "claim-without-key",
        "bad-key",
        "duplicate-key",
        "control-char",
        "plain-http",
        "naive-approval",
        "future-approval",
        "unlabelled-synthetic",
        "labelled-real",
        "non-bool-flag",
        "space-in-source-id",
        "no-approver",
        "no-source-version",
        "unsupported-language",
    ],
)
def test_an_invalid_file_is_refused(changes: dict[str, Any], message: str) -> None:
    with pytest.raises(ValidationError, match=message):
        _doc(**changes)


@pytest.mark.parametrize(
    ("raw", "message"),
    [
        (b'{"external_ref": "A", "external_ref": "B"}', "duplicate keys"),
        (b'{"facts": {"min_age": NaN}}', "NaN is not valid JSON"),
        (b"\xff\xfe", "not valid UTF-8 JSON"),
        (b"{" + b" " * MAX_FILE_BYTES + b"}", "larger than"),
    ],
    ids=["duplicate-key", "nan", "not-utf8", "too-large"],
)
def test_a_file_that_is_not_strict_json_is_refused(raw: bytes, message: str) -> None:
    with pytest.raises(SnapshotFileError, match=message):
        load_document(raw)


@asynccontextmanager
async def _as_owner(connection: AsyncConnection) -> AsyncIterator[None]:
    # The client fixture runs requests as marketing_app; the import runs as the owner.
    await connection.execute(text("RESET ROLE"))
    try:
        yield
    finally:
        await connection.execute(text("SET LOCAL ROLE marketing_app"))


async def test_an_imported_study_is_listed_once_granted_and_takes_drafts(
    client: httpx.AsyncClient, connection: AsyncConnection, world: World, auth: AuthHeaders
) -> None:
    async with _as_owner(connection), _session(connection) as db:
        outcome = await import_snapshot(db, world.a.tenant_id, _doc())
        db.add(
            MembershipStudy(
                tenant_id=world.a.tenant_id, user_sub="creator-a", study_id=outcome.study_id
            )
        )
        await db.commit()

    listed = await client.get("/v1/marketing/studies", headers=auth("creator-a", world.a.tenant_id))
    draft = await client.post(
        "/v1/marketing/creatives",
        headers=auth("creator-a", world.a.tenant_id, **{"Idempotency-Key": uuid4().hex}),
        json={
            "study_id": str(outcome.study_id),
            "prompt": "Invite adults with high blood pressure to learn about the study.",
            "language": "en",
            "channel": "facebook",
            "post_format": "feed_image",
        },
    )

    [item] = [i for i in listed.json()["items"] if i["study_id"] == str(outcome.study_id)]
    assert item["availability"] == "recruiting"
    assert item["external_ref"] == SOURCE_ID
    assert item["latest_snapshot"]["source_ref"] == f"study-management:{SOURCE_ID}:v7"
    assert draft.status_code == 201, draft.text
    assert draft.json()["study_snapshot_id"] == str(outcome.snapshot_id)


async def test_the_api_role_cannot_add_studies(
    client: httpx.AsyncClient, connection: AsyncConnection, world: World
) -> None:
    # Only the owner (this import, the seed) writes studies; the API role has no INSERT.
    async with _session(connection) as db:
        await bind_tenant(db, world.a.tenant_id)
        with pytest.raises(DBAPIError, match="permission denied"):
            await db.execute(
                text("INSERT INTO studies (id, tenant_id, title) VALUES (:id, :t, 'x')"),
                {"id": uuid4(), "t": world.a.tenant_id},
            )


# Approval records for claims and disclosures (Firoz's request for the policy checks).

_VOLUNTARY = "This is a research study. Taking part is voluntary."


def _facts(**changes: Any) -> dict[str, Any]:
    return {**_file()["facts"], **changes}


def _records(snapshot: StudySnapshot) -> dict[str, ApprovedWording]:
    facts = StudyFacts.model_validate(snapshot.facts)
    return {
        record.key: record
        for record in [*facts.approved_claim_records, *facts.required_disclosure_records]
    }


async def test_each_claim_and_disclosure_is_stored_with_its_approval(
    connection: AsyncConnection,
) -> None:
    async with _owner_session(connection) as db:
        tenant = await _tenant(db)
        outcome = await import_snapshot(db, tenant.tenant_id, _doc())
        [snapshot] = await _snapshots(db, outcome.study_id)

    records = _records(snapshot)
    voluntary = records["voluntary"]
    assert (voluntary.text, voluntary.language, voluntary.version) == (_VOLUNTARY, "en", 1)
    assert voluntary.text_sha256 == hashlib.sha256(_VOLUNTARY.encode()).hexdigest()
    assert voluntary.approved_by == "reviewer.synthetic@example.org"
    assert voluntary.approved_at == datetime(2026, 11, 20, 9, 30, tzinfo=UTC)
    assert records["treatment"].version == 1
    # The plain texts stay, for readers that only need the wording.
    assert snapshot.facts["required_disclosures"] == [_VOLUNTARY]


async def test_unchanged_wording_keeps_its_approval_and_changed_wording_gets_the_next_version(
    connection: AsyncConnection,
) -> None:
    async with _owner_session(connection) as db:
        tenant = await _tenant(db)
        first = await import_snapshot(db, tenant.tenant_id, _doc())
        await import_snapshot(
            db,
            tenant.tenant_id,
            _doc(
                source_ref=f"study-management:{SOURCE_ID}:v8",
                approved_by="legal.synthetic@example.org",
                approved_at="2026-11-25T08:00:00+00:00",
                facts=_facts(
                    required_disclosures=[
                        {"key": "voluntary", "text": "This is research. Joining is voluntary."},
                        {"key": "no-cost", "text": "There is no cost to take part."},
                    ]
                ),
            ),
        )
        old, new = await _snapshots(db, first.study_id)

    before, after = _records(old), _records(new)
    # Same text: the record is carried over as it was, approval included.
    assert after["treatment"] == before["treatment"]
    # Changed text: next version, approved by this file's approver.
    assert after["voluntary"].version == 2
    assert after["voluntary"].approved_by == "legal.synthetic@example.org"
    assert after["voluntary"].approved_at == datetime(2026, 11, 25, 8, tzinfo=UTC)
    assert after["no-cost"].version == 1
    # The earlier snapshot never changes.
    assert before["voluntary"].version == 1
    assert before["voluntary"].text == _VOLUNTARY


async def test_a_wording_dropped_and_brought_back_continues_its_versions(
    connection: AsyncConnection,
) -> None:
    async with _owner_session(connection) as db:
        tenant = await _tenant(db)
        first = await import_snapshot(db, tenant.tenant_id, _doc())
        for version, disclosures in (
            ("v8", []),
            ("v9", [{"key": "voluntary", "text": _VOLUNTARY}]),
            ("v10", [{"key": "voluntary", "text": "Taking part is voluntary."}]),
        ):
            await import_snapshot(
                db,
                tenant.tenant_id,
                _doc(
                    source_ref=f"study-management:{SOURCE_ID}:{version}",
                    facts=_facts(required_disclosures=disclosures),
                ),
            )
        v7, v8, v9, v10 = await _snapshots(db, first.study_id)

    assert "voluntary" not in _records(v8)
    assert _records(v9)["voluntary"] == _records(v7)["voluntary"]
    assert _records(v10)["voluntary"].version == 2


async def test_reimporting_a_source_version_with_new_wording_is_refused(
    connection: AsyncConnection,
) -> None:
    async with _owner_session(connection) as db:
        tenant = await _tenant(db)
        await import_snapshot(db, tenant.tenant_id, _doc())

        with pytest.raises(SnapshotImportRefusedError, match="different content"):
            await import_snapshot(
                db,
                tenant.tenant_id,
                _doc(facts=_facts(required_disclosures=[{"key": "other", "text": _VOLUNTARY}])),
            )


def _stored(**changes: Any) -> dict[str, Any]:
    record = ApprovedWording.approve(
        "voluntary",
        _VOLUNTARY,
        Language.EN,
        approved_by="reviewer.synthetic@example.org",
        approved_at=datetime(2026, 11, 20, tzinfo=UTC),
    ).model_dump(mode="json")
    facts: dict[str, Any] = {
        **_file()["facts"],
        "approved_claims": [],
        "approved_claim_records": [],
        "required_disclosures": [_VOLUNTARY],
        "required_disclosure_records": [record],
    }
    facts.update(changes)
    return facts


def test_stored_facts_with_matching_records_are_read() -> None:
    facts = StudyFacts.model_validate(_stored())

    assert facts.required_disclosure_records[0].key == "voluntary"


@pytest.mark.parametrize(
    "changes",
    [
        {"required_disclosures": ["Something else."]},
        {"required_disclosure_records": []},
        {"required_disclosures": [_VOLUNTARY, _VOLUNTARY]},
    ],
    ids=["other-text", "missing-record", "extra-text"],
)
def test_stored_facts_whose_records_disagree_with_the_texts_are_refused(
    changes: dict[str, Any],
) -> None:
    with pytest.raises(ValidationError, match="same texts, in the same order"):
        StudyFacts.model_validate(_stored(**changes))


def test_a_record_whose_hash_does_not_match_its_text_is_refused() -> None:
    record = {**_stored()["required_disclosure_records"][0], "text_sha256": "0" * 64}

    with pytest.raises(ValidationError, match="text_sha256 does not match"):
        StudyFacts.model_validate(_stored(required_disclosure_records=[record]))


def test_a_key_used_twice_in_stored_records_is_refused() -> None:
    record = _stored()["required_disclosure_records"][0]

    with pytest.raises(ValidationError, match="has a key more than once"):
        StudyFacts.model_validate(
            _stored(
                required_disclosures=[_VOLUNTARY, _VOLUNTARY],
                required_disclosure_records=[record, record],
            )
        )


def _backfill_module() -> Any:
    # The migration's own copy of the logic (a migration never imports app code that changes).
    versions = Path(__file__).parents[1] / "migrations" / "versions"
    path = next(versions.glob("*_study_wording_approvals.py"))
    spec = importlib.util.spec_from_file_location("wording_backfill", path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_the_backfill_keys_existing_wording_by_position_and_versions_changes() -> None:
    backfill = _backfill_module()
    old = {"approved_claims": ["Claim."], "required_disclosures": [_VOLUNTARY]}
    new = {"approved_claims": ["Claim."], "required_disclosures": ["Changed.", "Added."]}

    first, latest = backfill.with_records(
        old,
        language="en",
        approved_by="first@example.org",
        approved_at=datetime(2026, 11, 1, tzinfo=UTC),
        previous={},
    )
    second, _ = backfill.with_records(
        new,
        language="en",
        approved_by="second@example.org",
        approved_at=datetime(2026, 11, 2, tzinfo=UTC),
        previous=latest,
    )

    StudyFacts.model_validate({**_file()["facts"], **first})
    facts = StudyFacts.model_validate({**_file()["facts"], **second})
    assert [(c.key, c.version, c.approved_by) for c in facts.approved_claim_records] == [
        ("claim-1", 1, "first@example.org")
    ]
    assert [(d.key, d.version, d.approved_by) for d in facts.required_disclosure_records] == [
        ("disclosure-1", 2, "second@example.org"),
        ("disclosure-2", 1, "second@example.org"),
    ]
