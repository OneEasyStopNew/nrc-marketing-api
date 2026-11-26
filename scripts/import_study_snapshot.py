"""Import one approved study snapshot from a reviewed file, until Study Management exists (NRC-2).

Usage (runs as the database owner, MIGRATION_DATABASE_URL):
    uv run python -m scripts.import_study_snapshot --tenant-id <tenant-uuid> --file snapshot.json

The file names the study in the source system (external_ref), the source version of these facts
(source_ref), and who approved them and when. Each approved claim and required disclosure has a
key that stays the same across source versions; the import stores it as an approval record
(text, language, version, text hash, approved by and when). A wording whose text is unchanged
since an earlier snapshot keeps that record; a changed text gets the next version and this
file's approval. Marketing only stores what the file says: it never
fills in or changes a study fact. The file holds marketing-safe study facts only, never patient
data, and unknown fields are refused.

Results:
    created      a new study, with snapshot version 1
    new_version  the next snapshot version of an existing study
    unchanged    the same source version with the same content, imported again (safe to re-run)

Refused (exit code 2, nothing written):
    - a source version already imported with different content (a source version never changes)
    - a source version older than the latest snapshot (no rollback; import a new source version)
    - synthetic and real snapshots on the same study
    - an invalid file

Snapshots are immutable: existing drafts, validation runs and jobs keep the snapshot they were
created with. A new study is visible to nobody until scripts.grant_access runs for its members.
approved_by is the approver named in the file; whoever can use MIGRATION_DATABASE_URL can run
this, so access to that secret is the authorization.
"""

import argparse
import asyncio
import json
import unicodedata
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import Annotated, Final, Self
from uuid import UUID, uuid4

from pydantic import (
    AfterValidator,
    AwareDatetime,
    BaseModel,
    ConfigDict,
    Field,
    HttpUrl,
    StrictBool,
    StringConstraints,
    ValidationError,
    model_validator,
)
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.pool import NullPool

from app.core.config import OwnerDatabaseSettings
from app.db.session import TenantSession, bind_tenant
from app.models import Study, StudySnapshot, Tenant
from app.models.enums import Language, RecruitmentStatus
from app.schemas.tenancy import WORDING_KEY_PATTERN, ApprovedWording, StudyFacts, wording_sha256
from scripts.grant_access import UnknownTenantError

MAX_FILE_BYTES: Final = 256 * 1024
SYNTHETIC_LABEL: Final = "[SYNTHETIC]"
# Source ids and approvers: ASCII record ids or e-mail addresses, no spaces or control characters.
_REF_PATTERN: Final = r"^[A-Za-z0-9][A-Za-z0-9._:/@-]*$"


def _text(max_chars: int) -> Callable[[str], str]:
    # NFC for display text (PY-120), then the length, so normalizing can't push it past the limit.
    def clean(value: str) -> str:
        value = unicodedata.normalize("NFC", value).strip()
        if not value:
            raise ValueError("must not be blank")
        if len(value) > max_chars:
            raise ValueError(f"must be at most {max_chars} characters")
        if any(unicodedata.category(char) in ("Cc", "Cf") for char in value):
            raise ValueError("must not contain control characters")
        return value

    return clean


def _ref(max_chars: int) -> StringConstraints:
    return StringConstraints(
        strip_whitespace=True, min_length=1, max_length=max_chars, pattern=_REF_PATTERN
    )


ShortText = Annotated[str, AfterValidator(_text(200))]
LongText = Annotated[str, AfterValidator(_text(1000))]
Statement = Annotated[str, AfterValidator(_text(500))]


class Wording(BaseModel):
    """An approved claim or required disclosure in the file."""

    model_config = ConfigDict(extra="forbid")

    key: Annotated[str, StringConstraints(pattern=WORDING_KEY_PATTERN)] = Field(
        description="Keep the same key for this wording in every source version."
    )
    text: Statement


def _unique_keys(items: list[Wording]) -> list[Wording]:
    keys = [item.key for item in items]
    if len(set(keys)) != len(keys):
        raise ValueError(f"keys must be unique: {sorted({k for k in keys if keys.count(k) > 1})}")
    return items


Wordings = Annotated[list[Wording], AfterValidator(_unique_keys)]


def _https(value: HttpUrl) -> HttpUrl:
    if value.scheme != "https":
        raise ValueError("must be an https URL")
    return value


class SnapshotFacts(BaseModel):
    """The approved, marketing-safe facts. Same shape as StudyFacts, with stricter checks."""

    model_config = ConfigDict(extra="forbid")

    condition: ShortText
    eligibility_summary: LongText
    min_age: int | None = Field(default=None, ge=0, le=120)
    max_age: int | None = Field(default=None, ge=0, le=120)
    sites: list[ShortText] = Field(min_length=1, max_length=50)
    approved_claims: Wordings = Field(min_length=1, max_length=50)
    required_disclosures: Wordings = Field(max_length=20)
    compensation_statement: Statement | None = None
    contact_url: Annotated[HttpUrl, AfterValidator(_https)] | None = None

    @model_validator(mode="after")
    def _ages_in_order(self) -> Self:
        if self.min_age is not None and self.max_age is not None and self.min_age > self.max_age:
            raise ValueError("min_age must not be greater than max_age")
        return self


class SnapshotDocument(BaseModel):
    """One reviewed snapshot file."""

    model_config = ConfigDict(extra="forbid")

    external_ref: Annotated[str, _ref(100)] = Field(description="The study's id at the source.")
    source_ref: Annotated[str, _ref(300)] = Field(description="The source version of these facts.")
    title: Annotated[str, AfterValidator(_text(300))]
    recruitment_status: RecruitmentStatus
    language: Language
    facts: SnapshotFacts
    approved_by: Annotated[str, _ref(200)]
    approved_at: AwareDatetime
    is_synthetic: StrictBool

    @model_validator(mode="after")
    def _checked(self) -> Self:
        if self.approved_at > datetime.now(UTC):
            raise ValueError("approved_at must not be in the future")
        # Synthetic fixtures are labelled where users see them, and real studies never are.
        if self.is_synthetic != self.title.startswith(SYNTHETIC_LABEL):
            raise ValueError(
                f"title must start with {SYNTHETIC_LABEL!r} exactly when is_synthetic is true"
            )
        return self


class SnapshotFileError(ValueError):
    pass


class SnapshotImportRefusedError(ValueError):
    pass


class ImportResult(StrEnum):
    CREATED = "created"
    NEW_VERSION = "new_version"
    UNCHANGED = "unchanged"


@dataclass(frozen=True, slots=True)
class ImportOutcome:
    result: ImportResult
    study_id: UUID
    snapshot_id: UUID
    version: int


def _no_duplicate_keys(pairs: list[tuple[str, object]]) -> dict[str, object]:
    keys = [key for key, _ in pairs]
    if len(keys) != len(set(keys)):
        raise SnapshotFileError(f"duplicate keys: {sorted({k for k in keys if keys.count(k) > 1})}")
    return dict(pairs)


def _no_constant(value: str) -> object:
    raise SnapshotFileError(f"{value} is not valid JSON")


def load_document(raw: bytes) -> SnapshotDocument:
    """Parse a snapshot file strictly (PY-125): size-limited, UTF-8, no duplicate keys or NaN.

    Raises:
        SnapshotFileError: the file is too large or not strict JSON.
        ValidationError: the content breaks the snapshot schema.
    """
    if len(raw) > MAX_FILE_BYTES:
        raise SnapshotFileError(f"file is larger than {MAX_FILE_BYTES} bytes")
    try:
        parsed = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=_no_duplicate_keys,
            parse_constant=_no_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as err:
        raise SnapshotFileError(f"not valid UTF-8 JSON: {err}") from err
    return SnapshotDocument.model_validate(parsed)


def _content(
    status: RecruitmentStatus,
    language: Language,
    facts: dict[str, object],
    approved_by: str,
    approved_at: datetime,
) -> tuple[object, ...]:
    return (status, language, facts, approved_by, approved_at.astimezone(UTC))


async def import_snapshot(
    db: AsyncSession, tenant_id: UUID, doc: SnapshotDocument
) -> ImportOutcome:
    """Create the study or add its next snapshot version. The caller commits.

    Raises:
        UnknownTenantError: no tenant has this ID.
        SnapshotImportRefusedError: the import would change, roll back or mix approved snapshots.
    """
    if await db.get(Tenant, tenant_id) is None:
        raise UnknownTenantError(f"no tenant with id {tenant_id}")
    await bind_tenant(db, tenant_id)
    # Two imports of the same study can't both take the next version.
    await db.execute(
        select(
            func.pg_advisory_xact_lock(
                func.hashtextextended(f"study-import:{tenant_id}:{doc.external_ref}", 0)
            )
        )
    )
    study = await db.scalar(
        select(Study).where(Study.tenant_id == tenant_id, Study.external_ref == doc.external_ref)
    )
    if study is None:
        study = Study(
            id=uuid4(), tenant_id=tenant_id, title=doc.title, external_ref=doc.external_ref
        )
        db.add(study)
        await db.flush()
        facts = stored_facts(doc, earlier=[])
        return await _add_snapshot(db, study, doc, facts, version=1, result=ImportResult.CREATED)

    snapshots = list(
        await db.scalars(
            select(StudySnapshot)
            .where(StudySnapshot.study_id == study.id)
            .order_by(StudySnapshot.version.desc())
        )
    )
    latest = snapshots[0] if snapshots else None
    if latest is not None and latest.is_synthetic != doc.is_synthetic:
        raise SnapshotImportRefusedError(
            f"study {doc.external_ref!r} has {'synthetic' if latest.is_synthetic else 'real'} "
            "snapshots; synthetic and real snapshots never share a study"
        )
    same_source = next((s for s in snapshots if s.source_ref == doc.source_ref), None)
    if same_source is not None:
        if same_source is not latest:
            raise SnapshotImportRefusedError(
                f"source version {doc.source_ref!r} is snapshot {same_source.version}, older than "
                f"the latest ({snapshots[0].version}); import a new source version instead"
            )
        # Rebuilt against the snapshots before it, as the first import did.
        facts = stored_facts(doc, earlier=snapshots[1:])
        same = _content(
            same_source.recruitment_status,
            same_source.language,
            same_source.facts,
            same_source.approved_by,
            same_source.approved_at,
        ) == _content(doc.recruitment_status, doc.language, facts, doc.approved_by, doc.approved_at)
        if not same or study.title != doc.title:
            raise SnapshotImportRefusedError(
                f"source version {doc.source_ref!r} was imported with different content; a "
                "source version never changes, so give the new facts a new source version"
            )
        return ImportOutcome(ImportResult.UNCHANGED, study.id, same_source.id, same_source.version)

    study.title = doc.title
    facts = stored_facts(doc, earlier=snapshots)
    version = (latest.version if latest is not None else 0) + 1
    return await _add_snapshot(
        db, study, doc, facts, version=version, result=ImportResult.NEW_VERSION
    )


def stored_facts(doc: SnapshotDocument, *, earlier: Iterable[StudySnapshot]) -> dict[str, object]:
    """The facts as stored: the plain texts plus an approval record for each wording.

    `earlier` are the study's snapshots before this one, newest first. For each key, the newest
    earlier record decides: the same text (and language) keeps that record as it is; a changed
    text gets the next version with this file's approval; a new key starts at version 1.
    """
    history = [StudyFacts.model_validate(snapshot.facts) for snapshot in earlier]
    facts = doc.facts.model_dump(mode="json", exclude={"approved_claims", "required_disclosures"})
    for texts_field, records_field, wordings in (
        ("approved_claims", "approved_claim_records", doc.facts.approved_claims),
        ("required_disclosures", "required_disclosure_records", doc.facts.required_disclosures),
    ):
        latest: dict[str, ApprovedWording] = {}
        for old in reversed(history):  # oldest first, so the newest record per key wins
            latest.update({record.key: record for record in getattr(old, records_field)})
        records = [_record(wording, doc, latest.get(wording.key)) for wording in wordings]
        facts[texts_field] = [record.text for record in records]
        facts[records_field] = [record.model_dump(mode="json") for record in records]
    # The stored shape is what the API reads back.
    return StudyFacts.model_validate(facts).model_dump(mode="json")


def _record(
    wording: Wording, doc: SnapshotDocument, previous: ApprovedWording | None
) -> ApprovedWording:
    if (
        previous is not None
        and previous.text_sha256 == wording_sha256(wording.text)
        and previous.language == doc.language
    ):
        return previous
    return ApprovedWording.approve(
        wording.key,
        wording.text,
        doc.language,
        approved_by=doc.approved_by,
        approved_at=doc.approved_at,
        version=previous.version + 1 if previous is not None else 1,
    )


async def _add_snapshot(
    db: AsyncSession,
    study: Study,
    doc: SnapshotDocument,
    facts: dict[str, object],
    *,
    version: int,
    result: ImportResult,
) -> ImportOutcome:
    snapshot = StudySnapshot(
        id=uuid4(),
        tenant_id=study.tenant_id,
        study_id=study.id,
        version=version,
        recruitment_status=doc.recruitment_status,
        language=doc.language,
        facts=facts,
        source_ref=doc.source_ref,
        approved_by=doc.approved_by,
        approved_at=doc.approved_at,
        is_synthetic=doc.is_synthetic,
    )
    db.add(snapshot)
    await db.flush()
    return ImportOutcome(result, study.id, snapshot.id, version)


async def _run(tenant_id: UUID, doc: SnapshotDocument) -> ImportOutcome:
    engine = create_async_engine(
        OwnerDatabaseSettings().migration_database_url.get_secret_value(), poolclass=NullPool
    )
    try:
        async with AsyncSession(engine, sync_session_class=TenantSession) as db:
            outcome = await import_snapshot(db, tenant_id, doc)
            await db.commit()
    finally:
        await engine.dispose()
    return outcome


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0] if __doc__ else None)
    parser.add_argument("--tenant-id", type=UUID, required=True)
    parser.add_argument("--file", type=Path, required=True, help="the reviewed snapshot JSON")
    args = parser.parse_args()
    try:
        doc = load_document(args.file.read_bytes())
    except OSError as err:
        parser.error(f"cannot read {args.file}: {err.strerror}")
    except (SnapshotFileError, ValidationError) as err:
        parser.error(f"{args.file} is not a valid snapshot file:\n{err}")
    try:
        outcome = asyncio.run(_run(args.tenant_id, doc))
    except (UnknownTenantError, SnapshotImportRefusedError) as err:
        parser.error(str(err))
    result = {
        "result": outcome.result.value,
        "tenant_id": str(args.tenant_id),
        "study_id": str(outcome.study_id),
        "snapshot_id": str(outcome.snapshot_id),
        "version": outcome.version,
        "is_synthetic": doc.is_synthetic,
    }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
