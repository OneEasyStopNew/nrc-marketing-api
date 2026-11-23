"""Brief and caption validation (NRC-7).

Each run first gathers the policy context for its text (NRC-5: rules selected by their
conditions, their evidence and a search). The outcome comes from exactly the rules that context
selected: code checks first (phrases, exact wording, study facts), then the model check of the
semantic rules the code left open. A run with no context falls back to the legacy JSONB rules
and can't pass.

Where the run happens depends on VALIDATION_MODEL_MODE:
- off: no external call is needed, so the run finishes in the request and POST returns
  SUCCEEDED (semantic rules stay UNABLE_TO_ASSESS, so the run needs review).
- simulator or live: POST queues a job (202 QUEUED); the worker builds the context, runs the
  checks and the model (app.services.validation_jobs) and inserts the run under the job's id.
  Until then GET answers from the job. A finished PASS or FAIL with an equal reuse key (same
  content, configuration and dependency versions) is copied instead of judged again.

Runs are insert-only: a run is written once, finished. Image validation (D4) is not built here;
the routes answer it from the demo stubs or with 501.
"""

import hashlib
import json
import logging
from collections.abc import Iterable
from dataclasses import dataclass, replace
from datetime import UTC, date, datetime, timedelta
from types import MappingProxyType
from typing import Final
from uuid import UUID, uuid4

from sqlalchemy import ColumnElement, and_, select
from sqlalchemy import text as sql_text
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.ext.asyncio import AsyncSession

from app.auth.access import TenantContext
from app.core.config import Settings
from app.core.errors import (
    FeatureNotAvailableError,
    NoActivePolicyError,
    NotFoundError,
    PolicyContextUnavailableError,
)
from app.models import Creative, CreativeVersion, Job, OutboxEvent, StudySnapshot, ValidationRun
from app.models.enums import (
    Channel,
    ExecutionState,
    Language,
    Outcome,
    PolicyContextReason,
    PolicyLanguage,
    PostFormat,
    ValidationInputKind,
)
from app.schemas.common import ApiModel
from app.schemas.policies import PolicyBundleRead
from app.schemas.tenancy import StudyFacts
from app.schemas.validations import (
    CoverageGap,
    Finding,
    GateResult,
    RuleResult,
    SourceReference,
    ValidationAccepted,
    ValidationCreate,
    ValidationRunRead,
    ValidationRunSummary,
)
from app.services import idempotency, judging, policies, policy_evidence, rules
from app.services.creatives import visible_to
from app.services.jobs import RUN_EVENT
from app.services.policy_context import (
    ContextRequest,
    ContextScope,
    PolicyContextProvider,
    StoredManifest,
)
from app.services.policy_context.manifest import MAX_STATEMENT_CHARS, ContextManifest, Statement
from app.services.providers.text_judge import judge_model_id

VALIDATE_ROUTE: Final = "POST /validations"
# Postgres "query_canceled": a statement hit statement_timeout.
QUERY_CANCELED_SQLSTATE: Final = "57014"
# Recorded when the policy context couldn't be gathered; the rule checks still ran. The reason
# in brackets is a code (e.g. GRAPH_MISSING): CoverageGap has no code field.
CONTEXT_UNAVAILABLE_GAP: Final = (
    "Policy evidence was not gathered for this run ({reason}); the result comes from the rule "
    "checks only."
)
# Job kinds for validation: internal, never accepted by POST /creative-jobs (JobKind is that
# route's contract). Plain strings, like jobs.kind.
JOB_KIND_BY_INPUT: Final = MappingProxyType(
    {ValidationInputKind.BRIEF: "brief_validation", ValidationInputKind.TEXT: "text_validation"}
)
INPUT_BY_JOB_KIND: Final = MappingProxyType({v: k for k, v in JOB_KIND_BY_INPUT.items()})
# Only a definite result is reused: NEEDS_REVIEW may come from a failed model call.
_REUSABLE: Final = (Outcome.PASS, Outcome.FAIL)

logger = logging.getLogger(__name__)


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


@dataclass(frozen=True, slots=True)
class Target:
    # The exact creative version under check and the study snapshot it was written against.
    prompt: str
    caption: str | None
    language: Language
    channel: Channel
    post_format: PostFormat
    facts: StudyFacts
    snapshot_id: UUID
    snapshot_version: int

    def text(self, kind: ValidationInputKind) -> str:
        return self.prompt if kind is ValidationInputKind.BRIEF else self.caption or ""


@dataclass(frozen=True, slots=True)
class Judged:
    # How the model check went, for the run's audit columns.
    model_id: str | None = None
    prompt_version: str | None = None
    attempts: int | None = None


async def create_validation(
    db: AsyncSession,
    ctx: TenantContext,
    body: ValidationCreate,
    idempotency_key: str,
    context_provider: PolicyContextProvider,
    context_timeout: float,
    settings: Settings,
) -> tuple[ValidationAccepted, bool]:
    """Check a brief or caption now, or queue it for the worker. Returns it and whether this was
    a replay.

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
        return ValidationAccepted(
            job_id=existing_id,
            validation_run_id=existing_id,
            state=await _state_of(db, ctx.tenant_id, existing_id),
        ), True

    target = await load_target(db, visible_to(ctx), body.creative_id, body.creative_version)
    policy = await _active_policy(db, ctx)
    text = target.text(body.input_kind)
    now = datetime.now(tz=UTC)
    key = reuse_key(body.input_kind, text, target, policy, settings, now.date())
    if settings.validation_model_mode == "off":
        run = await _check_now(
            db, ctx, body, text, target, policy, context_provider, context_timeout, now
        )
        run.reuse_key = key
    else:
        reused = await _reusable(db, ctx.tenant_id, key)
        if reused is None:
            run_id = await _queue(db, ctx, body, target, policy, settings, now)
            await idempotency.complete(db, ctx.tenant_id, VALIDATE_ROUTE, idempotency_key, run_id)
            await db.commit()
            return _accepted(run_id, ExecutionState.QUEUED), False
        run = _copy(reused, ctx, body)
    db.add(run)
    await idempotency.complete(db, ctx.tenant_id, VALIDATE_ROUTE, idempotency_key, run.id)
    await db.commit()
    return _accepted(run.id, ExecutionState.SUCCEEDED), False


async def get_validation(
    db: AsyncSession, ctx: TenantContext, validation_run_id: UUID
) -> ValidationRunRead:
    """Return a stored run, or the queued job that will produce it. is_current is false once the
    creative or the policy has changed.

    Raises:
        NotFoundError: no such run or validation job visible to the caller.
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
    active = await policies.get_active_policy_summary(db, ctx.tenant_id)
    active_version = None if active is None else active.version
    if row is None:
        return await _pending_read(db, ctx, validation_run_id, active_version)
    run, current_version = row
    result = _StoredResult.model_validate(run.result)
    return ValidationRunRead(
        validation_run_id=run.id,
        input_kind=run.input_kind,
        creative_version=run.creative_version,
        state=run.state,
        outcome=run.outcome,
        is_current=_is_current(
            run.creative_version,
            run.policy_bundle_version,
            current_version,
            active_version=active_version,
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
    db: AsyncSession, tenant_id: UUID, creative_id: UUID, version: int
) -> BriefOutcome | None:
    # The caller has already checked that the creative is visible.
    row = (
        await db.execute(
            select(ValidationRun.id, ValidationRun.outcome, ValidationRun.policy_bundle_version)
            .where(
                ValidationRun.tenant_id == tenant_id,
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
    active = await policies.get_active_policy_summary(db, ctx.tenant_id)
    return ValidationRunSummary(
        validation_run_id=run.id,
        input_kind=run.input_kind,
        creative_version=run.creative_version,
        state=run.state,
        outcome=run.outcome,
        is_current=_is_current(
            run.creative_version,
            run.policy_bundle_version,
            version,
            active_version=None if active is None else active.version,
        ),
        evaluated_at=run.evaluated_at,
    )


async def load_target(
    db: AsyncSession, scope: Iterable[ColumnElement[bool]], creative_id: UUID, version: int
) -> Target:
    """The exact version and the study snapshot it was written against, in one query.

    scope filters Creative: visible_to(ctx) in a request, the job's tenant in the worker.

    Raises:
        NotFoundError: no such creative version within scope.
    """
    row = (
        await db.execute(
            select(
                CreativeVersion.prompt,
                CreativeVersion.caption,
                CreativeVersion.language,
                CreativeVersion.channel,
                CreativeVersion.post_format,
                StudySnapshot.facts,
                StudySnapshot.id.label("snapshot_id"),
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
                *scope,
                CreativeVersion.creative_id == creative_id,
                CreativeVersion.version == version,
            )
        )
    ).one_or_none()
    if row is None:
        raise NotFoundError("Creative version not found.")
    return Target(
        prompt=row.prompt,
        caption=row.caption,
        language=row.language,
        channel=row.channel,
        post_format=row.post_format,
        facts=StudyFacts.model_validate(row.facts),
        snapshot_id=row.snapshot_id,
        snapshot_version=row.version,
    )


def reuse_key(
    kind: ValidationInputKind,
    text: str,
    target: Target,
    policy: PolicyBundleRead,
    settings: Settings,
    day: date,
) -> str:
    """sha256 of the content, its scope and every version the result depends on.

    The day is included because rules carry effective dates: a result is never reused across
    one. Two runs with equal keys would be checked against the same rules by the same code,
    model and prompt.
    """
    parts = {
        "kind": kind.value,
        "text": hashlib.sha256(text.encode()).hexdigest(),
        "channel": target.channel,
        "post_format": target.post_format,
        "language": target.language,
        "policy_bundle": f"{policy.policy_bundle_id}:{policy.version}",
        "study_snapshot": f"{target.snapshot_id}:{target.snapshot_version}",
        "evaluation": rules.EVALUATION_VERSION,
        "judge": judge_model_id(settings),
        "prompt": judging.PROMPT_TEMPLATE_VERSION,
        "embedding": f"{settings.embedding_mode}:{settings.bedrock_embedding_model_id}",
        "day": day.isoformat(),
    }
    return hashlib.sha256(json.dumps(parts, sort_keys=True).encode()).hexdigest()


async def gather_context(
    db: AsyncSession,
    tenant_id: UUID,
    actor: str,
    provider: PolicyContextProvider,
    deadline_seconds: float,
    kind: ValidationInputKind,
    text: str,
    target: Target,
    policy: PolicyBundleRead,
    evaluated_at: datetime,
) -> tuple[StoredManifest | None, list[CoverageGap]]:
    """The stored context, or no context and a coverage gap saying why.

    Failing to gather it never fails the run: the run is stored with the gap, and the caller
    makes sure it can't pass. Each of its statements has a time limit (deadline_seconds). Uses
    the run's pinned bundle and snapshot, and the whole brief or caption as one statement; an
    identical earlier request reuses its stored manifest.
    """
    language = policy_language(target.language)
    # A PolicyContextReason; str, because the provider's reason arrives in the error's details.
    reason: str
    if language is None:
        reason = PolicyContextReason.LANGUAGE_NOT_SUPPORTED
    elif len(text) > MAX_STATEMENT_CHARS:
        # Possible after NFC normalisation of a text at the length limit.
        reason = PolicyContextReason.STATEMENT_TOO_LONG
    else:
        request = ContextRequest(
            policy_bundle_id=policy.policy_bundle_id,
            study_snapshot_id=target.snapshot_id,
            scope=ContextScope(
                input_kind=kind,
                channel=target.channel,
                post_format=target.post_format,
                language=language,
                evaluated_at=evaluated_at,
            ),
            statements=(Statement(text=text),) if text else (),
        )
        try:
            # A savepoint, so a database error here leaves the run's own transaction usable. The
            # deadline is Postgres's statement_timeout, not asyncio.timeout: cancelling asyncpg
            # mid-query invalidates the connection, and the run could then not be stored.
            # Rolling back to the savepoint undoes the limit; on success it is reset explicitly.
            async with db.begin_nested():
                await db.execute(
                    sql_text("SELECT set_config('statement_timeout', :ms, true)"),
                    {"ms": str(max(1, round(deadline_seconds * 1000)))},
                )
                stored = await provider.build(db, request, actor=actor)
                await db.execute(sql_text("SET LOCAL statement_timeout TO DEFAULT"))
        except PolicyContextUnavailableError as error:
            # E.g. an older bundle without a graph.
            reason = str((error.details or {}).get("reason", error.code))
        except SQLAlchemyError as error:
            sqlstate = getattr(getattr(error, "orig", None), "sqlstate", None)
            if sqlstate == QUERY_CANCELED_SQLSTATE:
                reason = PolicyContextReason.RETRIEVAL_TIMEOUT
            else:
                reason = PolicyContextReason.RETRIEVAL_ERROR
                # With the traceback, so the failing query can be found. The engine sets
                # hide_parameters, so SQLAlchemy's part of the error carries the SQL but no bound
                # values; asyncpg's own message can still quote a value it failed to encode.
                logger.exception(
                    "policy context failed tenant=%s bundle=%s sqlstate=%s",
                    tenant_id,
                    policy.policy_bundle_id,
                    sqlstate,
                )
        else:
            return stored, []
    logger.warning(
        "policy context unavailable tenant=%s bundle=%s reason=%s",
        tenant_id,
        policy.policy_bundle_id,
        reason,
    )
    return None, [CoverageGap(reason=CONTEXT_UNAVAILABLE_GAP.format(reason=reason))]


async def graph_rules(
    db: AsyncSession, tenant_id: UUID, manifest: ContextManifest, policy: PolicyBundleRead
) -> list[rules.GraphRule]:
    # Exactly the rules the manifest selected (mandatory and optional), read from the graph with
    # the sources the policy evidence endpoint shows. A selected rule with no row is not checked.
    stored = await policy_evidence.get_bundle_rules(db, tenant_id, policy.policy_bundle_id)
    rows = {(row.rule_id, row.rule_version): row for row in stored.rules}
    selected_rules = []
    for selected in [*manifest.required_rules, *manifest.optional_rules]:
        row = rows.get((selected.rule_id, selected.rule_version))
        source = SourceReference(source_id=policy.source_ref, clause=selected.rule_id)
        content = None
        if row is not None:
            if row.sources:
                first = row.sources[0]
                source = SourceReference(
                    source_id=first.source_title, clause=first.clause_ref, excerpt=first.excerpt
                )
            content = rules.RuleContent(
                title=row.title,
                requirement=row.requirement,
                exact_wording=row.exact_wording,
                prohibited_phrases=tuple(row.prohibited_phrases),
            )
        selected_rules.append(
            rules.GraphRule(
                rule_id=selected.rule_id,
                rule_version=selected.rule_version,
                node_key=selected.node_key,
                mandatory=selected.mandatory,
                gate=selected.gate,
                severity=selected.severity,
                source=source,
                check_type=selected.check_type,
                content=content,
            )
        )
    return selected_rules


def new_run(
    *,
    run_id: UUID,
    tenant_id: UUID,
    creative_id: UUID,
    creative_version: int,
    kind: ValidationInputKind,
    text: str,
    target: Target,
    policy: PolicyBundleRead,
    evaluation: rules.Evaluation,
    evaluated_at: datetime,
    context: StoredManifest | None,
    context_gaps: list[CoverageGap],
    created_by: str,
    judged: Judged,
    reuse_key: str | None,
) -> ValidationRun:
    return ValidationRun(
        id=run_id,
        tenant_id=tenant_id,
        creative_id=creative_id,
        creative_version=creative_version,
        input_kind=kind,
        state=ExecutionState.SUCCEEDED,
        outcome=evaluation.outcome,
        input_hash=input_hash(kind, text),
        policy_bundle_id=policy.policy_bundle_id,
        policy_bundle_version=policy.version,
        study_snapshot_version=target.snapshot_version,
        context_manifest_id=context.id if context else None,
        context_manifest_sha256=context.sha256 if context else None,
        judge_model_id=judged.model_id,
        judge_prompt_version=judged.prompt_version,
        judge_attempts=judged.attempts,
        evaluation_version=rules.EVALUATION_VERSION,
        reuse_key=reuse_key,
        result=_StoredResult(
            gates=evaluation.gates,
            rule_results=evaluation.rule_results,
            findings=evaluation.findings,
            coverage_gaps=[*evaluation.coverage_gaps, *context_gaps],
        ).model_dump(mode="json"),
        created_by=created_by,
        evaluated_at=evaluated_at,
    )


def without_pass(evaluation: rules.Evaluation) -> rules.Evaluation:
    # The run's outcome and gates, with every PASS turned into NEEDS_REVIEW.
    def demote(outcome: Outcome) -> Outcome:
        return Outcome.NEEDS_REVIEW if outcome is Outcome.PASS else outcome

    return replace(
        evaluation,
        outcome=demote(evaluation.outcome),
        gates=[
            gate.model_copy(update={"outcome": demote(gate.outcome)}) for gate in evaluation.gates
        ],
    )


def policy_language(language: Language | str) -> PolicyLanguage | None:
    # Policy text may support a language before the public enum does (F33), and the reverse must
    # not crash validation: a language without policy text gets no context and a gap.
    try:
        return PolicyLanguage(language)
    except ValueError:
        return None


def input_hash(kind: ValidationInputKind, text: str) -> str:
    return hashlib.sha256(f"{kind.value}:{text}".encode()).hexdigest()


async def _check_now(
    db: AsyncSession,
    ctx: TenantContext,
    body: ValidationCreate,
    text: str,
    target: Target,
    policy: PolicyBundleRead,
    provider: PolicyContextProvider,
    context_timeout: float,
    now: datetime,
) -> ValidationRun:
    # The run without a model: code checks only, so semantic rules stay UNABLE_TO_ASSESS.
    stored, context_gaps = await gather_context(
        db,
        ctx.tenant_id,
        ctx.principal.sub,
        provider,
        context_timeout,
        body.input_kind,
        text,
        target,
        policy,
        now,
    )
    if stored is None:
        # Without its policy context a run can't be shown to pass (17 Nov decision superseded on
        # 20 Nov, Dinesh's review): PASS becomes NEEDS_REVIEW; FAIL is already stricter.
        evaluation = without_pass(rules.evaluate_text(body.input_kind, text, policy, target.facts))
    else:
        selected = await graph_rules(db, ctx.tenant_id, stored.manifest, policy)
        evaluation = rules.evaluate_graph_rules(text, selected, target.facts)
    return new_run(
        run_id=uuid4(),
        tenant_id=ctx.tenant_id,
        creative_id=body.creative_id,
        creative_version=body.creative_version,
        kind=body.input_kind,
        text=text,
        target=target,
        policy=policy,
        evaluation=evaluation,
        evaluated_at=now,
        context=stored,
        context_gaps=context_gaps,
        created_by=ctx.principal.sub,
        judged=Judged(),
        reuse_key=None,
    )


async def _queue(
    db: AsyncSession,
    ctx: TenantContext,
    body: ValidationCreate,
    target: Target,
    policy: PolicyBundleRead,
    settings: Settings,
    now: datetime,
) -> UUID:
    # One job and its run event; the run will carry the job's id. The bundle version is pinned
    # now, so a policy activated meanwhile doesn't change what this run is checked against.
    run_id = uuid4()
    db.add(
        Job(
            id=run_id,
            tenant_id=ctx.tenant_id,
            kind=JOB_KIND_BY_INPUT[body.input_kind],
            creative_id=body.creative_id,
            creative_version=body.creative_version,
            study_snapshot_id=target.snapshot_id,
            study_snapshot_version=target.snapshot_version,
            policy_bundle_version=str(policy.version),
            state=ExecutionState.QUEUED,
            attempt_count=0,
            max_attempts=settings.validation_job_max_attempts,
            deadline_at=now + timedelta(seconds=settings.validation_job_deadline_seconds),
            model=judge_model_id(settings),
            created_by=ctx.principal.sub,
        )
    )
    # Flush the parent first: there is no ORM relationship to order the inserts.
    await db.flush()
    db.add(
        OutboxEvent(
            tenant_id=ctx.tenant_id,
            job_id=run_id,
            event_type=RUN_EVENT,
            payload={"job_id": str(run_id)},
            attempts=0,
        )
    )
    logger.info("validation queued job=%s tenant=%s", run_id, ctx.tenant_id)
    return run_id


async def _reusable(db: AsyncSession, tenant_id: UUID, key: str) -> ValidationRun | None:
    return await db.scalar(
        select(ValidationRun)
        .where(
            ValidationRun.tenant_id == tenant_id,
            ValidationRun.reuse_key == key,
            ValidationRun.state == ExecutionState.SUCCEEDED,
            ValidationRun.outcome.in_(_REUSABLE),
        )
        .order_by(ValidationRun.created_at.desc(), ValidationRun.id.desc())
        .limit(1)
    )


def _copy(source: ValidationRun, ctx: TenantContext, body: ValidationCreate) -> ValidationRun:
    # The same result for this creative version: same content, configuration and dependency
    # versions (equal reuse key), so no model call and no second charge. Provenance is kept.
    logger.info("validation reused run=%s tenant=%s", source.id, ctx.tenant_id)
    return ValidationRun(
        id=uuid4(),
        tenant_id=ctx.tenant_id,
        creative_id=body.creative_id,
        creative_version=body.creative_version,
        input_kind=source.input_kind,
        state=ExecutionState.SUCCEEDED,
        outcome=source.outcome,
        input_hash=source.input_hash,
        policy_bundle_id=source.policy_bundle_id,
        policy_bundle_version=source.policy_bundle_version,
        study_snapshot_version=source.study_snapshot_version,
        context_manifest_id=source.context_manifest_id,
        context_manifest_sha256=source.context_manifest_sha256,
        judge_model_id=source.judge_model_id,
        judge_prompt_version=source.judge_prompt_version,
        judge_attempts=source.judge_attempts,
        evaluation_version=source.evaluation_version,
        reuse_key=source.reuse_key,
        result=source.result,
        created_by=ctx.principal.sub,
        evaluated_at=source.evaluated_at,
    )


async def _pending_read(
    db: AsyncSession, ctx: TenantContext, job_id: UUID, active_version: int | None
) -> ValidationRunRead:
    # A queued, running or failed validation job: no results until the worker stores the run.
    row = (
        await db.execute(
            select(Job, Creative.current_version)
            .join(
                Creative, and_(Creative.tenant_id == Job.tenant_id, Creative.id == Job.creative_id)
            )
            .where(
                *visible_to(ctx),
                Job.id == job_id,
                Job.kind.in_(tuple(INPUT_BY_JOB_KIND)),
            )
        )
    ).one_or_none()
    if row is None:
        raise NotFoundError("Validation run not found.")
    job, current_version = row
    kind = INPUT_BY_JOB_KIND[job.kind]
    target = await load_target(db, visible_to(ctx), job.creative_id, job.creative_version)
    bundle_version = int(job.policy_bundle_version) if job.policy_bundle_version else None
    return ValidationRunRead(
        validation_run_id=job.id,
        input_kind=kind,
        creative_version=job.creative_version,
        state=job.state,
        outcome=None,
        is_current=_is_current(
            job.creative_version, bundle_version, current_version, active_version=active_version
        ),
        evaluated_at=None,
        job_id=job.id,
        input_hash=input_hash(kind, target.text(kind)),
        asset_id=None,
        study_snapshot_version=job.study_snapshot_version,
        policy_bundle_version=bundle_version,
        gates=[],
        rule_results=[],
        findings=[],
        coverage_gaps=[],
    )


async def _state_of(db: AsyncSession, tenant_id: UUID, run_id: UUID) -> ExecutionState:
    # A replay: the stored run's state, else its job's (still queued, running or failed).
    run_state = await db.scalar(
        select(ValidationRun.state).where(
            ValidationRun.tenant_id == tenant_id, ValidationRun.id == run_id
        )
    )
    if run_state is not None:
        return run_state
    job_state = await db.scalar(
        select(Job.state).where(Job.tenant_id == tenant_id, Job.id == run_id)
    )
    return job_state or ExecutionState.QUEUED


async def _active_policy(db: AsyncSession, ctx: TenantContext) -> PolicyBundleRead:
    try:
        return await policies.get_active_policy(db, ctx)
    except NotFoundError:
        raise NoActivePolicyError() from None


def _is_current(
    run_version: int,
    run_bundle_version: int | None,
    current_version: int,
    *,
    active_version: int | None,
) -> bool:
    return run_version == current_version and run_bundle_version == active_version


def _accepted(run_id: UUID, state: ExecutionState) -> ValidationAccepted:
    return ValidationAccepted(job_id=run_id, validation_run_id=run_id, state=state)


def judged_rules(selected: Iterable[rules.GraphRule]) -> list[judging.JudgedRule]:
    # The model's view of each rule: its id, node, title and requirement (rows loaded only).
    return [
        judging.JudgedRule(
            rule_id=rule.rule_id,
            node_key=rule.node_key,
            title=rule.content.title,
            requirement=rule.content.requirement,
        )
        for rule in selected
        if rule.content is not None
    ]
