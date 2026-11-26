"""Import a policy bundle document (policy deliverable, schema nrc.policy-bundle/0.2, v0.3) into
one DRAFT policy bundle.

Usage (runs as the database owner, MIGRATION_DATABASE_URL):
    uv run python -m scripts.import_policy_bundle --tenant-id <uuid> --bundle-id <uuid> \
        [--file scripts/data/nrc-ipb-candidate-bundle-v0.3.json]

Create the DRAFT bundle first (POST /policy-bundles). The import is lossless: the whole document
is stored on the bundle with its content_sha256, and each rule's own record on its rule row, so
export_policy_bundle() gives the document back. It refuses a document whose content_sha256 or any
quotation's text_sha256 does not match (edited after hashing).

What becomes runtime graph (the rest stays in the stored document):
- Rules of the stages validation runs (context, validation, correction) that are enabled in NRC:
  a RULE node and rule row each, PROPOSED unless the rule's approval is APPROVED.
- Each source quotation: a clause of one source for the document (its sha256 is the quotation's)
  and a CLAUSE node the rule is supported_by. A rule without quotations has no source support and
  stays MISSING_PROVENANCE until someone adds one: no provenance is made up here.
- Exceptions (except REJECTED): EXCEPTION nodes behind except_when, reviewed as their status.
- required_post_wording: a STUDY_APPROVED_WORDING pointer to the study snapshot (no text).
Reviewer-gate and publish-readiness rules (approvals, channel publishing) are enforced outside
validation, so they are not graph rules. Study predicates, outcome policy and channel readiness
are read from the stored records (NRC-5, NRC-7).
"""

import argparse
import asyncio
import hashlib
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any, Final, Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, ValidationError
from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.pool import NullPool

from app.core.config import OwnerDatabaseSettings
from app.db.session import TenantSession, bind_tenant
from app.models import PolicyBundle, PolicyRuleRecord, PolicyRuleWording
from app.models.enums import (
    BundleLifecycleState,
    CheckType,
    Gate,
    PolicyEdgeType,
    PolicyLanguage,
    PolicyNodeType,
    ProposalOrigin,
    ReviewState,
    Severity,
    StudyWordingSource,
    ValidationInputKind,
    WordingKind,
)
from app.models.policy_graph import NODE_KEY_PATTERN, RULE_ID_PATTERN
from scripts.import_policy_graph import (
    ClauseImport,
    EdgeImport,
    ImportResult,
    NodeImport,
    PolicyGraphImport,
    PolicyGraphImportError,
    RuleImport,
    SourceImport,
    import_policy_graph,
)

BUNDLE_V03_PATH: Final = (
    Path(__file__).resolve().parent / "data" / "nrc-ipb-candidate-bundle-v0.3.json"
)
SCHEMA_VERSION: Final = "nrc.policy-bundle/0.2"
# The stages that run inside validation and correction; reviewer_gate and publish_readiness run at
# approval and publishing time, outside the validation graph.
RUNTIME_STAGES: Final = frozenset({"context", "validation", "correction"})
# check.inputs → what the validation request carries. Inputs that are context (study_snapshot,
# post.language, correction) don't restrict the input kind.
_INPUT_KINDS: Final[dict[str, ValidationInputKind]] = {
    "brief": ValidationInputKind.BRIEF,
    "text": ValidationInputKind.TEXT,
    "image": ValidationInputKind.IMAGE,
    "image_ocr": ValidationInputKind.IMAGE,
}
# Media validation can't take yet (NRC-9): a rule only for these must not turn into "every input".
_UNSUPPORTED_INPUTS: Final = frozenset({"video", "video_captions"})
_SECTION_MAX: Final = 200


class _Doc(BaseModel):
    # Only what the mapping reads is validated here; the whole document is stored as given.
    model_config = ConfigDict(extra="ignore", frozen=True)


class _Quotation(_Doc):
    source_ref: str = Field(min_length=1)
    locator: str = Field(min_length=1)
    text: str = Field(min_length=1)
    text_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class _Exception(_Doc):
    exception_id: str = Field(max_length=80, pattern=NODE_KEY_PATTERN)
    text: str = Field(min_length=1)
    status: Literal["PROPOSED", "APPROVED", "REJECTED"]


class _PostWording(_Doc):
    source: StudyWordingSource


class _Check(_Doc):
    method: str = Field(min_length=1)
    inputs: list[str] = Field(default_factory=list)


class _OutcomePolicy(_Doc):
    mandatory: bool


class _Approval(_Doc):
    status: Literal["PENDING", "APPROVED", "REJECTED"]


class _Rule(_Doc):
    rule_id: str = Field(pattern=RULE_ID_PATTERN)
    rule_version: int = Field(ge=1)
    stage: Literal["context", "validation", "correction", "reviewer_gate", "publish_readiness"]
    gate: Gate
    severity: Severity
    title: str = Field(min_length=1, max_length=200)
    requirement: str = Field(min_length=1, max_length=1000)
    check: _Check
    outcome_policy: _OutcomePolicy
    source_quotations: list[_Quotation]
    required_post_wording: _PostWording | None
    exceptions: list[_Exception]
    approval: _Approval
    enabled_in_nrc: bool


class _Meta(_Doc):
    bundle_id: str = Field(min_length=1)
    bundle_version: str = Field(pattern=r"^\d+\.\d+\.\d+$")
    is_synthetic: bool
    content_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class _Scope(_Doc):
    # Quotations carry no language of their own; until a bundle enables another language (with a
    # language per quotation), English is the only one the import can label correctly.
    enabled_languages: list[Literal["en"]] = Field(min_length=1)


class PolicyBundleDocument(_Doc):
    schema_version: Literal["nrc.policy-bundle/0.2"]
    bundle: _Meta
    scope: _Scope
    rules: list[_Rule] = Field(min_length=1)


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def content_sha256(document: dict[str, Any]) -> str:
    # The deliverable's definition: canonical JSON of every top-level key except "bundle".
    body = {key: value for key, value in document.items() if key != "bundle"}
    return _sha256(json.dumps(body, sort_keys=True, ensure_ascii=False, separators=(",", ":")))


def verify_document(document: dict[str, Any]) -> PolicyBundleDocument:
    """Validate what the import reads and every checksum; raises PolicyGraphImportError."""
    try:
        parsed = PolicyBundleDocument.model_validate(document)
    except ValidationError as error:
        raise PolicyGraphImportError(f"Not a {SCHEMA_VERSION} bundle: {error}") from None
    if content_sha256(document) != parsed.bundle.content_sha256:
        raise PolicyGraphImportError("content_sha256 does not match: edited after hashing.")
    for rule in parsed.rules:
        for quote in rule.source_quotations:
            if _sha256(quote.text) != quote.text_sha256:
                raise PolicyGraphImportError(f"{rule.rule_id}: a quotation's text_sha256 differs.")
    return parsed


def runtime_rules(parsed: PolicyBundleDocument) -> list[_Rule]:
    return [
        rule
        for rule in parsed.rules
        if rule.stage in RUNTIME_STAGES
        and rule.enabled_in_nrc
        and rule.approval.status != "REJECTED"
    ]


def quotation_ref(rule_id: str, index: int) -> str:
    # Same id as the reference contract's graph mapping (build_graph): <rule_id>-Q<n>.
    return f"{rule_id}-Q{index + 1}"


def _check_type(rule: _Rule) -> CheckType:
    parts = set(rule.check.method.split("+"))
    if "manual_attestation" in parts:
        return CheckType.MANUAL_ATTESTATION
    if any(part.startswith("semantic") for part in parts):
        return CheckType.SEMANTIC
    if parts <= {"deterministic", "lexical", "lexical_trigger"}:
        return CheckType.DETERMINISTIC
    raise PolicyGraphImportError(f"{rule.rule_id}: unknown check method {rule.check.method!r}.")


def _input_kinds(rule: _Rule) -> list[ValidationInputKind]:
    kinds = list(dict.fromkeys(_INPUT_KINDS[i] for i in rule.check.inputs if i in _INPUT_KINDS))
    if not kinds and _UNSUPPORTED_INPUTS & set(rule.check.inputs):
        raise PolicyGraphImportError(f"{rule.rule_id}: applies only to inputs not supported yet.")
    return kinds


def to_graph(parsed: PolicyBundleDocument) -> PolicyGraphImport:
    """The runtime part of the document in the graph-import format (see the module docstring)."""
    rules = runtime_rules(parsed)
    clauses = [
        ClauseImport(
            clause_ref=quotation_ref(rule.rule_id, i),
            # Display only; the full locator stays in the stored document.
            section=f"{quote.source_ref} {quote.locator}"[:_SECTION_MAX],
            text=quote.text,
        )
        for rule in rules
        for i, quote in enumerate(rule.source_quotations)
    ]
    if not clauses:
        raise PolicyGraphImportError("No runtime rule quotes a source.")
    meta = parsed.bundle
    nodes: list[NodeImport] = []
    edges: list[EdgeImport] = []
    for rule in rules:
        approved = rule.approval.status == "APPROVED"
        nodes.append(
            NodeImport(
                node_key=rule.rule_id,
                node_type=PolicyNodeType.RULE,
                text=rule.requirement,
                approved=approved,
                input_kinds=_input_kinds(rule),
            )
        )
        for i, quote in enumerate(rule.source_quotations):
            ref = quotation_ref(rule.rule_id, i)
            nodes.append(
                NodeImport(
                    node_key=ref,
                    node_type=PolicyNodeType.CLAUSE,
                    text=quote.text,
                    clause_ref=ref,
                    approved=approved,
                )
            )
            edges.append(
                EdgeImport(
                    edge_type=PolicyEdgeType.SUPPORTED_BY,
                    from_node_key=rule.rule_id,
                    to_node_key=ref,
                    approved=approved,
                    provenance_clause_ref=ref,
                )
            )
        for exception in rule.exceptions:
            if exception.status == "REJECTED":
                continue
            exception_approved = exception.status == "APPROVED"
            nodes.append(
                NodeImport(
                    node_key=exception.exception_id,
                    node_type=PolicyNodeType.EXCEPTION,
                    text=exception.text,
                    approved=exception_approved,
                )
            )
            edges.append(
                EdgeImport(
                    edge_type=PolicyEdgeType.EXCEPT_WHEN,
                    from_node_key=rule.rule_id,
                    to_node_key=exception.exception_id,
                    approved=exception_approved,
                    provenance_note=f"Exception {exception.exception_id} of {rule.rule_id} in "
                    f"{meta.bundle_id} {meta.bundle_version}.",
                )
            )
    return PolicyGraphImport(
        source=SourceImport(
            title=f"{meta.bundle_id} {meta.bundle_version}: source quotations",
            source_ref=f"{meta.bundle_id}@{meta.bundle_version}",
            is_synthetic=meta.is_synthetic,
            clauses=clauses,
        ),
        nodes=nodes,
        rules=[
            RuleImport(
                node_key=rule.rule_id,
                rule_id=rule.rule_id,
                rule_version=rule.rule_version,
                title=rule.title,
                requirement=rule.requirement,
                gate=rule.gate,
                severity=rule.severity,
                mandatory=rule.outcome_policy.mandatory,
                check_type=_check_type(rule),
            )
            for rule in rules
        ],
        edges=edges,
    )


async def import_policy_bundle(
    db: AsyncSession, tenant_id: UUID, bundle_id: UUID, document: dict[str, Any]
) -> ImportResult:
    """Import into a DRAFT bundle in the caller's transaction (tenant already bound).

    Writes nothing on a checksum mismatch, a bundle that is not a DRAFT, or one that already
    holds another graph or document. The same document again is a no-op (skipped).
    """
    parsed = verify_document(document)
    bundle = await db.scalar(
        select(PolicyBundle).where(
            PolicyBundle.tenant_id == tenant_id, PolicyBundle.id == bundle_id
        )
    )
    if bundle is None:
        raise PolicyGraphImportError(f"Policy bundle {bundle_id} not found for this tenant.")
    if bundle.contract_sha256 is not None:
        if bundle.contract_sha256 != parsed.bundle.content_sha256:
            raise PolicyGraphImportError(f"Policy bundle {bundle_id} holds another document.")
        return ImportResult(skipped=True)
    if bundle.lifecycle_state is not BundleLifecycleState.DRAFT:
        raise PolicyGraphImportError(
            f"Policy bundle {bundle_id} is {bundle.lifecycle_state.value}; import into a DRAFT."
        )
    rules = runtime_rules(parsed)
    result = await import_policy_graph(db, tenant_id, bundle_id, to_graph(parsed))
    if result.skipped:
        raise PolicyGraphImportError(f"Policy bundle {bundle_id} already has a graph.")
    rule_rows = {
        rule_id: row_id
        for rule_id, row_id in await db.execute(
            select(PolicyRuleRecord.rule_id, PolicyRuleRecord.id).where(
                PolicyRuleRecord.tenant_id == tenant_id,
                PolicyRuleRecord.policy_bundle_id == bundle_id,
            )
        )
    }
    records = {str(record["rule_id"]): record for record in document["rules"]}
    for rule in rules:
        await db.execute(
            update(PolicyRuleRecord)
            .where(PolicyRuleRecord.id == rule_rows[rule.rule_id])
            .values(contract=records[rule.rule_id])
        )
    db.add_all(
        PolicyRuleWording(
            tenant_id=tenant_id,
            policy_bundle_id=bundle_id,
            policy_rule_id=rule_rows[rule.rule_id],
            language=PolicyLanguage.EN,
            kind=WordingKind.STUDY_APPROVED_WORDING,
            text=rule.required_post_wording.source.value,
            origin=ProposalOrigin.IMPORT,
            review_state=ReviewState.PROPOSED,
        )
        for rule in rules
        if rule.required_post_wording is not None
    )
    bundle.contract_document = document
    bundle.contract_sha256 = parsed.bundle.content_sha256
    await db.flush()
    return result


async def export_policy_bundle(
    db: AsyncSession, tenant_id: UUID, bundle_id: UUID
) -> dict[str, Any]:
    """The imported document, checked against the stored content_sha256 before it is returned."""
    row = (
        await db.execute(
            select(PolicyBundle.contract_document, PolicyBundle.contract_sha256).where(
                PolicyBundle.tenant_id == tenant_id, PolicyBundle.id == bundle_id
            )
        )
    ).first()
    if row is None or row.contract_document is None:
        raise PolicyGraphImportError(f"Policy bundle {bundle_id} has no imported document.")
    document: dict[str, Any] = dict(row.contract_document)
    if content_sha256(document) != row.contract_sha256:
        raise PolicyGraphImportError(f"Policy bundle {bundle_id}: stored document changed.")
    return document


def load_document(path: Path) -> dict[str, Any]:
    document = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(document, dict):
        raise PolicyGraphImportError("A bundle document is a JSON object.")
    return document


async def _run(tenant_id: UUID, bundle_id: UUID, document: dict[str, Any]) -> ImportResult:
    engine = create_async_engine(
        OwnerDatabaseSettings().migration_database_url.get_secret_value(), poolclass=NullPool
    )
    try:
        async with AsyncSession(engine, sync_session_class=TenantSession) as db:
            await bind_tenant(db, tenant_id)
            result = await import_policy_bundle(db, tenant_id, bundle_id, document)
            await db.commit()
    finally:
        await engine.dispose()
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0] if __doc__ else None)
    parser.add_argument("--tenant-id", type=UUID, required=True)
    parser.add_argument("--bundle-id", type=UUID, required=True)
    parser.add_argument("--file", type=Path, default=BUNDLE_V03_PATH, help="bundle document JSON")
    args = parser.parse_args()
    result = asyncio.run(_run(args.tenant_id, args.bundle_id, load_document(args.file)))
    print(
        json.dumps(
            {"tenant_id": str(args.tenant_id), "policy_bundle_id": str(args.bundle_id)}
            | asdict(result),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
