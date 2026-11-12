"""Brief and caption validation (NRC-7): rule checks run in the request and are stored.

The checks are fast, deterministic string matches (app.services.rules), so a run finishes
before POST /validations returns and is stored as SUCCEEDED. The contract still returns a
job_id: until validation moves onto the job framework (NRC-6), it equals validation_run_id.

Image validation (D4) is not built here; the routes answer it from the demo stubs or with 501.
"""

import hashlib
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Final
from uuid import UUID, uuid4

from sqlalchemy import and_, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.auth.access import TenantContext
from app.core.errors import FeatureNotAvailableError, NoActivePolicyError, NotFoundError
from app.models import Creative, CreativeVersion, StudySnapshot, ValidationRun
from app.models.enums import ExecutionState, Outcome, ValidationInputKind
from app.schemas.common import ApiModel
from app.schemas.policies import PolicyBundleRead
from app.schemas.tenancy import StudyFacts
from app.schemas.validations import (
    CoverageGap,
    Finding,
    GateResult,
    RuleResult,
    ValidationAccepted,
    ValidationCreate,
    ValidationRunRead,
    ValidationRunSummary,
)
from app.services import idempotency, policies, rules
from app.services.creatives import visible_to

VALIDATE_ROUTE: Final = "POST /validations"


class _StoredResult(ApiModel):
    # The JSONB result column, validated on the way in and out.
    gates: list[GateResult]
    rule_results: list[RuleResult]
    findings: list[Finding]
    coverage_gaps: list[CoverageGap]


@dataclass(frozen=True, slots=True)
class BriefOutcome:
    # The latest finished brief run of a version, for the generation gate.
    validation_run_id: UUID
    outcome: Outcome
    policy_bundle_version: int


async def create_validation(
    db: AsyncSession, ctx: TenantContext, body: ValidationCreate, idempotency_key: str
) -> tuple[ValidationAccepted, bool]:
    """Check a brief or caption and store the run. Returns it and whether this was a replay.

    Raises:
        FeatureNotAvailableError: image input (D4 is not built).
        NotFoundError: no such creative or version visible to the caller.
        NoActivePolicyError: the tenant has no policy to validate against.
        IdempotencyKeyReusedError: the key was used before with a different body.
    """
    if body.input_kind is ValidationInputKind.IMAGE:
        raise FeatureNotAvailableError("Image validation is not implemented yet.")
    fingerprint = idempotency.request_hash(ctx.principal.sub, body)
    existing_id = await idempotency.claim(
        db, ctx.tenant_id, VALIDATE_ROUTE, idempotency_key, fingerprint
    )
    if existing_id is not None:
        return _accepted(existing_id), True

    target = await _target(db, ctx, body.creative_id, body.creative_version)
    policy = await _active_policy(db, ctx)
    text = target.prompt if body.input_kind is ValidationInputKind.BRIEF else target.caption or ""
    evaluation = rules.evaluate_text(body.input_kind, text, policy, target.facts)
    run = _new_run(ctx, body, text, target, policy, evaluation)
    db.add(run)
    await idempotency.complete(db, ctx.tenant_id, VALIDATE_ROUTE, idempotency_key, run.id)
    await db.commit()
    return _accepted(run.id), False


async def get_validation(
    db: AsyncSession, ctx: TenantContext, validation_run_id: UUID
) -> ValidationRunRead:
    """Return a stored run. is_current is false once the creative or the policy has changed.

    Raises:
        NotFoundError: no such run visible to the caller.
    """
    row = (
        await db.execute(
            select(ValidationRun, Creative.current_version)
            .join(
                Creative,
                and_(
                    Creative.tenant_id == ValidationRun.tenant_id,
                    Creative.id == ValidationRun.creative_id,
                ),
            )
            .where(*visible_to(ctx), ValidationRun.id == validation_run_id)
        )
    ).one_or_none()
    if row is None:
        raise NotFoundError("Validation run not found.")
    run, current_version = row
    active = await policies.get_active_policy_summary(db, ctx)
    result = _StoredResult.model_validate(run.result)
    return ValidationRunRead(
        validation_run_id=run.id,
        input_kind=run.input_kind,
        creative_version=run.creative_version,
        state=run.state,
        outcome=run.outcome,
        is_current=_is_current(
            run, current_version, active_version=None if active is None else active.version
        ),
        evaluated_at=run.evaluated_at,
        job_id=run.id,
        input_hash=run.input_hash,
        asset_id=None,
        study_snapshot_version=run.study_snapshot_version,
        policy_bundle_version=run.policy_bundle_version,
        gates=result.gates,
        rule_results=result.rule_results,
        findings=result.findings,
        coverage_gaps=result.coverage_gaps,
    )


async def latest_brief_outcome(
    db: AsyncSession, ctx: TenantContext, creative_id: UUID, version: int
) -> BriefOutcome | None:
    # The caller has already checked that the creative is visible.
    row = (
        await db.execute(
            select(ValidationRun.id, ValidationRun.outcome, ValidationRun.policy_bundle_version)
            .where(
                ValidationRun.tenant_id == ctx.tenant_id,
                ValidationRun.creative_id == creative_id,
                ValidationRun.creative_version == version,
                ValidationRun.input_kind == ValidationInputKind.BRIEF,
                ValidationRun.state == ExecutionState.SUCCEEDED,
            )
            .order_by(ValidationRun.created_at.desc(), ValidationRun.id.desc())
            .limit(1)
        )
    ).one_or_none()
    if row is None or row.outcome is None:
        return None
    return BriefOutcome(row.id, row.outcome, row.policy_bundle_version)


async def latest_summary(
    db: AsyncSession, ctx: TenantContext, creative_id: UUID, version: int
) -> ValidationRunSummary | None:
    # Latest stored run of any kind for the creative's current version (export shows it).
    # The caller has already checked that the creative is visible.
    run = await db.scalar(
        select(ValidationRun)
        .where(
            ValidationRun.tenant_id == ctx.tenant_id,
            ValidationRun.creative_id == creative_id,
            ValidationRun.creative_version == version,
        )
        .order_by(ValidationRun.created_at.desc(), ValidationRun.id.desc())
        .limit(1)
    )
    if run is None:
        return None
    active = await policies.get_active_policy_summary(db, ctx)
    return ValidationRunSummary(
        validation_run_id=run.id,
        input_kind=run.input_kind,
        creative_version=run.creative_version,
        state=run.state,
        outcome=run.outcome,
        is_current=_is_current(
            run, version, active_version=None if active is None else active.version
        ),
        evaluated_at=run.evaluated_at,
    )


@dataclass(frozen=True, slots=True)
class _Target:
    prompt: str
    caption: str | None
    facts: StudyFacts
    snapshot_version: int


async def _target(db: AsyncSession, ctx: TenantContext, creative_id: UUID, version: int) -> _Target:
    # The exact version, and the study snapshot it was written against, in one query.
    row = (
        await db.execute(
            select(
                CreativeVersion.prompt,
                CreativeVersion.caption,
                StudySnapshot.facts,
                StudySnapshot.version,
            )
            .join(
                Creative,
                and_(
                    Creative.tenant_id == CreativeVersion.tenant_id,
                    Creative.id == CreativeVersion.creative_id,
                ),
            )
            .join(
                StudySnapshot,
                and_(
                    StudySnapshot.tenant_id == CreativeVersion.tenant_id,
                    StudySnapshot.id == CreativeVersion.study_snapshot_id,
                ),
            )
            .where(
                *visible_to(ctx),
                CreativeVersion.creative_id == creative_id,
                CreativeVersion.version == version,
            )
        )
    ).one_or_none()
    if row is None:
        raise NotFoundError("Creative version not found.")
    return _Target(
        prompt=row.prompt,
        caption=row.caption,
        facts=StudyFacts.model_validate(row.facts),
        snapshot_version=row.version,
    )


def _new_run(
    ctx: TenantContext,
    body: ValidationCreate,
    text: str,
    target: _Target,
    policy: PolicyBundleRead,
    evaluation: rules.Evaluation,
) -> ValidationRun:
    return ValidationRun(
        id=uuid4(),
        tenant_id=ctx.tenant_id,
        creative_id=body.creative_id,
        creative_version=body.creative_version,
        input_kind=body.input_kind,
        state=ExecutionState.SUCCEEDED,
        outcome=evaluation.outcome,
        input_hash=_input_hash(body.input_kind, text),
        policy_bundle_id=policy.policy_bundle_id,
        policy_bundle_version=policy.version,
        study_snapshot_version=target.snapshot_version,
        result=_StoredResult(
            gates=evaluation.gates,
            rule_results=evaluation.rule_results,
            findings=evaluation.findings,
            coverage_gaps=evaluation.coverage_gaps,
        ).model_dump(mode="json"),
        created_by=ctx.principal.sub,
        evaluated_at=datetime.now(tz=UTC),
    )


async def _active_policy(db: AsyncSession, ctx: TenantContext) -> PolicyBundleRead:
    try:
        return await policies.get_active_policy(db, ctx)
    except NotFoundError:
        raise NoActivePolicyError() from None


def _is_current(run: ValidationRun, current_version: int, *, active_version: int | None) -> bool:
    return run.creative_version == current_version and run.policy_bundle_version == active_version


def _accepted(run_id: UUID) -> ValidationAccepted:
    return ValidationAccepted(
        job_id=run_id, validation_run_id=run_id, state=ExecutionState.SUCCEEDED
    )


def _input_hash(kind: ValidationInputKind, text: str) -> str:
    return hashlib.sha256(f"{kind.value}:{text}".encode()).hexdigest()
