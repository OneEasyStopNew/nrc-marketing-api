"""Approval records for each study's approved claims and required disclosures (NRC-2).

Study facts (study_snapshots.facts, JSONB) gain approved_claim_records and
required_disclosure_records: per wording its key, text, language, version, SHA-256 of the text,
and who approved it and when. The plain text lists stay as they are, so readers of the texts
don't change.

Existing snapshots get records built from what they already hold: keys claim-1, claim-2, ... and
disclosure-1, ... by position; per study and key, the version goes up when the text changes from
one snapshot to the next, and the approval is the snapshot's own (approved_by, approved_at).
Snapshots that already have records are left alone, so a re-run changes nothing.

study_snapshots forces row level security, so even the owner only sees the current tenant's
rows. The backfill switches FORCE off for the length of this transaction to reach every tenant,
then back on; the DDL is transactional, so a failure leaves FORCE on. The table is small.

Downgrade removes the two record lists; the texts stay.

The logic is copied here, not imported from the app, so this migration never changes.

Revision ID: 9b3e5d71c2a4
Revises: 5d8f1c2a9e47
Create Date: 2026-11-26 12:00:00.000000
"""

import hashlib
import json
import unicodedata
from collections.abc import Mapping, Sequence
from datetime import datetime
from typing import Any

import sqlalchemy as sa
from alembic import op

revision: str = "9b3e5d71c2a4"
down_revision: str | None = "5d8f1c2a9e47"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

# (text list, record list, key prefix)
LISTS = (
    ("approved_claims", "approved_claim_records", "claim"),
    ("required_disclosures", "required_disclosure_records", "disclosure"),
)


def _sha256(text: str) -> str:
    return hashlib.sha256(unicodedata.normalize("NFC", text).encode()).hexdigest()


def with_records(
    facts: Mapping[str, Any],
    *,
    language: str,
    approved_by: str,
    approved_at: datetime,
    previous: Mapping[str, Mapping[str, Any]],
) -> tuple[dict[str, Any], dict[str, Mapping[str, Any]]]:
    """One snapshot's facts with records added, and the latest record per key afterwards.

    `previous` maps "<record list>/<key>" to the study's latest record for that key so far.
    """
    updated = dict(facts)
    latest: dict[str, Mapping[str, Any]] = dict(previous)
    for texts_field, records_field, prefix in LISTS:
        records: list[Mapping[str, Any]] = []
        for position, text in enumerate(facts.get(texts_field) or [], start=1):
            key = f"{prefix}-{position}"
            earlier = latest.get(f"{records_field}/{key}")
            digest = _sha256(text)
            record: Mapping[str, Any]
            if earlier is not None and earlier["text_sha256"] == digest:
                record = earlier
            else:
                record = {
                    "key": key,
                    "text": text,
                    "language": language,
                    "version": earlier["version"] + 1 if earlier is not None else 1,
                    "text_sha256": digest,
                    "approved_by": approved_by,
                    "approved_at": approved_at.isoformat(),
                }
            latest[f"{records_field}/{key}"] = record
            records.append(record)
        updated[records_field] = records
    return updated, latest


def upgrade() -> None:
    bind = op.get_bind()
    op.execute("ALTER TABLE study_snapshots NO FORCE ROW LEVEL SECURITY")
    rows = bind.execute(
        sa.text(
            "SELECT id, study_id, language, facts, approved_by, approved_at "
            "FROM study_snapshots ORDER BY study_id, version"
        )
    ).mappings()
    latest: dict[object, dict[str, Mapping[str, Any]]] = {}
    for row in list(rows):
        study = row["study_id"]
        if all(records_field in row["facts"] for _, records_field, _ in LISTS):
            # Already has records: keep them, and let later snapshots build on them.
            latest[study] = latest.get(study, {}) | {
                f"{records_field}/{record['key']}": record
                for _, records_field, _ in LISTS
                for record in row["facts"][records_field]
            }
            continue
        facts, latest[study] = with_records(
            row["facts"],
            language=row["language"],
            approved_by=row["approved_by"],
            approved_at=row["approved_at"],
            previous=latest.get(study, {}),
        )
        bind.execute(
            sa.text("UPDATE study_snapshots SET facts = CAST(:facts AS jsonb) WHERE id = :id"),
            {"facts": json.dumps(facts, allow_nan=False), "id": row["id"]},
        )
    op.execute("ALTER TABLE study_snapshots FORCE ROW LEVEL SECURITY")


def downgrade() -> None:
    op.execute("ALTER TABLE study_snapshots NO FORCE ROW LEVEL SECURITY")
    op.execute(
        "UPDATE study_snapshots "
        "SET facts = facts - 'approved_claim_records' - 'required_disclosure_records'"
    )
    op.execute("ALTER TABLE study_snapshots FORCE ROW LEVEL SECURITY")
