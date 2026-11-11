"""Simulated validation, image generation, assets and export, enabled only by DEMO_STUBS=true.

The frontend integrates against these while D2-D5 are built. Responses use the real contract
schemas, but the results come from simple phrase checks against the tenant's active policy, not
from a real validator, and every validation run says so in coverage_gaps.

Nothing is persisted. Records live in this process's memory, are bounded, and disappear on
restart, so run a single worker. Each check-and-insert runs without an await in between, so
concurrent requests on the one event loop cannot interleave inside it.
"""

import hashlib
import logging
import re
import time
from collections import OrderedDict
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Final
from uuid import UUID, uuid4

from sqlalchemy.ext.asyncio import AsyncSession

from app.auth.access import TenantContext
from app.core.errors import (
    ExportNotReadyError,
    GenerationNotEligibleError,
    IdempotencyKeyReusedError,
    NotFoundError,
)
from app.models.enums import FORMAT_SPECS, ExecutionState
from app.schemas.assets import AssetOrigin, AssetRead, ExportManifest
from app.schemas.creatives import CreativeRead, CreativeVersionRead
from app.schemas.jobs import CreativeJobCreate, JobAccepted, JobKind, JobRead
from app.schemas.policies import PolicyBundleRead, PolicyRule
from app.schemas.validations import (
    CoverageGap,
    Finding,
    Gate,
    GateResult,
    ImageRegion,
    Outcome,
    RuleResult,
    RuleResultValue,
    TextSpan,
    ValidationAccepted,
    ValidationCreate,
    ValidationInputKind,
    ValidationRunRead,
    ValidationRunSummary,
)
from app.services import creatives, eligibility, idempotency, policies

logger = logging.getLogger(__name__)

MAX_RECORDS: Final = 1000
DOWNLOAD_URL_TTL: Final = timedelta(minutes=15)
DEMO_FILES_PATH: Final = "/demo-files"
STUB_PROVIDER: Final = "demo-stub"
STUB_MODEL: Final = "placeholder-svg"
STUB_MIME_TYPE: Final = "image/svg+xml"
STUB_GAP: Final = CoverageGap(
    reason="Demo stub: simulated result from phrase checks, not a real validation."
)
NO_POLICY_GAP: Final = CoverageGap(reason="The tenant has no active policy to validate against.")
NOT_CHECKED: Final = "Not checked by the demo stub."
VALIDATE_ROUTE: Final = "POST /validations"
GENERATE_ROUTE: Final = "POST /creative-jobs"

Clock = Callable[[], float]


@dataclass(frozen=True, slots=True)
class StubTimings:
    """How long simulated work takes, so the UI can exercise its progress and polling states."""

    validation_seconds: float = 2.0
    generation_seconds: float = 6.0


@dataclass(frozen=True, slots=True)
class _PhraseCheck:
    rule_id: str
    pattern: re.Pattern[str]
    explanation: str
    suggested_correction: str | None


# Each pattern is linear (no nested quantifiers) and inputs are capped at 4000 characters.
_PHRASE_CHECKS: Final[tuple[_PhraseCheck, ...]] = (
    _PhraseCheck(
        "REC-01",
        re.compile(r"\b(?:safe and effective|proven|cures?)\b", re.IGNORECASE),
        "States or implies that the investigational treatment is safe or works.",
        None,
    ),
    _PhraseCheck(
        "REC-02",
        re.compile(r"\b(?:better than|superior to|best treatment)\b", re.IGNORECASE),
        "Compares the investigational treatment favourably with other treatments.",
        None,
    ),
    _PhraseCheck(
        "REC-03",
        re.compile(r"\bnew (?:treatment|drug|medication|medicine)\b", re.IGNORECASE),
        "Calls the treatment new without saying it is investigational.",
        "investigational treatment",
    ),
    _PhraseCheck(
        "REC-04",
        re.compile(r"\bguarantee(?:d|s)?\b", re.IGNORECASE),
        "Promises a certain favourable outcome.",
        None,
    ),
    _PhraseCheck(
        "REC-05",
        re.compile(r"\bfree (?:treatment|medication|medicine|care)\b", re.IGNORECASE),
        "Promises free treatment.",
        "no cost to take part",
    ),
    _PhraseCheck(
        "REC-06",
        re.compile(r"\$\s?[0-9][0-9,]*|\b(?:get|earn) paid\b", re.IGNORECASE),
        "Emphasises payment for taking part.",
        None,
    ),
)
_PAYMENT_CHECK: Final = _PHRASE_CHECKS[5]
_DISCLOSURE_RULE: Final = "REC-07"
_DISCLOSURE_TEXT: Final = "This is a research study. Taking part is voluntary."
_DISCLOSURE_PATTERNS: Final = (
    re.compile(r"\bresearch study\b", re.IGNORECASE),
    re.compile(r"\bvoluntary\b", re.IGNORECASE),
)
_TEXT_RULES: Final = frozenset(check.rule_id for check in _PHRASE_CHECKS) | {_DISCLOSURE_RULE}
_IMAGE_RULES: Final = frozenset({_PAYMENT_CHECK.rule_id})
_PAYMENT_REGION: Final = ImageRegion(x=0.1, y=0.72, width=0.8, height=0.16)


class _BoundedMap[K, V]:
    # Insertion-ordered map that forgets its oldest entries beyond `limit`.

    def __init__(self, limit: int) -> None:
        self._items: OrderedDict[K, V] = OrderedDict()
        self._limit = limit

    def get(self, key: K) -> V | None:
        return self._items.get(key)

    def put(self, key: K, value: V) -> None:
        self._items[key] = value
        self._items.move_to_end(key)
        while len(self._items) > self._limit:
            self._items.popitem(last=False)


@dataclass(frozen=True, slots=True)
class _Replay:
    fingerprint: str
    resource_id: UUID


@dataclass(frozen=True, slots=True)
class _ValidationRecord:
    tenant_id: UUID
    creative_id: UUID
    started: float
    finished: ValidationRunRead


@dataclass(frozen=True, slots=True)
class _GenerationRecord:
    tenant_id: UUID
    creative_id: UUID
    creative_version: int
    job_id: UUID
    started: float
    created_at: datetime
    asset: AssetRead


@dataclass(frozen=True, slots=True)
class _Evaluation:
    outcome: Outcome
    gates: list[GateResult]
    rule_results: list[RuleResult]
    findings: list[Finding]
    coverage_gaps: list[CoverageGap]


# (tenant, creative, version, input kind or None for "any kind")
_RunKey = tuple[UUID, UUID, int, ValidationInputKind | None]


class DemoStubs:
    """In-memory stand-in for the validation, generation, asset and export services."""

    def __init__(
        self,
        public_base_url: str,
        *,
        timings: StubTimings | None = None,
        clock: Clock = time.monotonic,
    ) -> None:
        self._base_url = public_base_url.rstrip("/")
        self._timings = timings or StubTimings()
        self._clock = clock
        self._replays: _BoundedMap[tuple[UUID, str, str], _Replay] = _BoundedMap(MAX_RECORDS)
        self._runs: _BoundedMap[UUID, _ValidationRecord] = _BoundedMap(MAX_RECORDS)
        self._latest_runs: _BoundedMap[_RunKey, UUID] = _BoundedMap(MAX_RECORDS)
        self._jobs: _BoundedMap[UUID, _GenerationRecord] = _BoundedMap(MAX_RECORDS)
        self._job_by_asset: _BoundedMap[UUID, UUID] = _BoundedMap(MAX_RECORDS)
        self._latest_jobs: _BoundedMap[tuple[UUID, UUID, int], UUID] = _BoundedMap(MAX_RECORDS)

    # Validation

    async def start_validation(
        self, db: AsyncSession, ctx: TenantContext, body: ValidationCreate, idempotency_key: str
    ) -> tuple[ValidationAccepted, bool]:
        """Record a validation run. Returns the accepted run and whether this was a replay.

        Raises:
            NotFoundError: unknown creative, version or (for images) generated asset.
            IdempotencyKeyReusedError: the key was used before with a different body.
        """
        fingerprint = idempotency.request_hash(ctx.principal.sub, body)
        replayed = self._replayed_run(ctx, idempotency_key, fingerprint)
        if replayed is not None:
            return replayed, True

        creative = await creatives.get_creative(db, ctx, body.creative_id)
        version = await _version(db, ctx, body.creative_id, body.creative_version)
        asset = self._validation_asset(ctx, body)
        policy = await _active_policy(db, ctx)
        if asset is not None:
            evaluation = _evaluate_image(version.current_prompt, policy)
            input_hash = asset.checksum_sha256
        else:
            text = version.current_prompt
            if body.input_kind is ValidationInputKind.TEXT:
                text = version.caption or ""
            evaluation = _evaluate_text(body.input_kind, text, policy)
            input_hash = _sha256(f"{body.input_kind.value}:{text}".encode())

        # No await from here on (see the module docstring).
        replayed = self._replayed_run(ctx, idempotency_key, fingerprint)
        if replayed is not None:
            return replayed, True
        run_id = uuid4()
        finished = ValidationRunRead(
            validation_run_id=run_id,
            input_kind=body.input_kind,
            creative_version=body.creative_version,
            state=ExecutionState.SUCCEEDED,
            outcome=evaluation.outcome,
            is_current=body.creative_version == creative.version,
            evaluated_at=datetime.now(tz=UTC) + timedelta(seconds=self._timings.validation_seconds),
            job_id=uuid4(),
            input_hash=input_hash,
            asset_id=body.asset_id,
            study_snapshot_version=creative.study_snapshot_version,
            policy_bundle_version=None if policy is None else policy.version,
            gates=evaluation.gates,
            rule_results=evaluation.rule_results,
            findings=evaluation.findings,
            coverage_gaps=evaluation.coverage_gaps,
        )
        record = _ValidationRecord(ctx.tenant_id, body.creative_id, self._clock(), finished)
        self._runs.put(run_id, record)
        for kind in (body.input_kind, None):
            self._latest_runs.put(
                (ctx.tenant_id, body.creative_id, body.creative_version, kind), run_id
            )
        self._replays.put(
            (ctx.tenant_id, VALIDATE_ROUTE, idempotency_key), _Replay(fingerprint, run_id)
        )
        logger.info("demo validation %s recorded: %s", run_id, evaluation.outcome.value)
        return self._accepted_run(record), False

    async def get_validation(
        self, db: AsyncSession, ctx: TenantContext, validation_run_id: UUID
    ) -> ValidationRunRead:
        """Return the run as it looks now: RUNNING until its simulated duration has passed.

        Raises:
            NotFoundError: no such run visible to the caller.
        """
        record = self._runs.get(validation_run_id)
        if record is None or record.tenant_id != ctx.tenant_id:
            raise NotFoundError("Validation run not found.")
        try:
            # Also enforces study access: a creative the caller cannot see hides its runs.
            creative = await creatives.get_creative(db, ctx, record.creative_id)
        except NotFoundError:
            raise NotFoundError("Validation run not found.") from None
        return self._run_view(record, current_version=creative.version)

    def latest_brief_outcome(
        self, tenant_id: UUID, creative_id: UUID, version: int
    ) -> tuple[UUID, Outcome] | None:
        """Latest finished brief validation of one version, for the generation-eligibility check.

        A run that is still in progress counts as no run.
        """
        run_id = self._latest_runs.get((tenant_id, creative_id, version, ValidationInputKind.BRIEF))
        if run_id is None:
            return None
        record = self._runs.get(run_id)
        if record is None or not self._is_done(record.started, self._timings.validation_seconds):
            return None
        outcome = record.finished.outcome
        return None if outcome is None else (run_id, outcome)

    # Generation and assets

    async def start_generation(
        self, db: AsyncSession, ctx: TenantContext, body: CreativeJobCreate, idempotency_key: str
    ) -> tuple[JobAccepted, bool]:
        """Record a generation job for an eligible version. Returns the job and whether replayed.

        Raises:
            NotFoundError: unknown creative or version.
            GenerationNotEligibleError: no passing brief validation for this version, a stale
                version or a closed study; details.reason says which.
            IdempotencyKeyReusedError: the key was used before with a different body.
        """
        fingerprint = idempotency.request_hash(ctx.principal.sub, body)
        replayed = self._replayed_job(ctx, idempotency_key, fingerprint)
        if replayed is not None:
            return replayed, True

        check = await eligibility.check_generation_eligibility(
            db, ctx, body.creative_id, body.creative_version, brief_runs=self
        )
        if not check.eligible:
            raise GenerationNotEligibleError(check.message, details={"reason": check.reason.value})
        creative = await creatives.get_creative(db, ctx, body.creative_id)

        # No await from here on (see the module docstring).
        replayed = self._replayed_job(ctx, idempotency_key, fingerprint)
        if replayed is not None:
            return replayed, True
        record = self._new_generation(ctx, creative, body.creative_version)
        self._jobs.put(record.job_id, record)
        self._job_by_asset.put(record.asset.asset_id, record.job_id)
        self._latest_jobs.put(
            (ctx.tenant_id, body.creative_id, body.creative_version), record.job_id
        )
        self._replays.put(
            (ctx.tenant_id, GENERATE_ROUTE, idempotency_key), _Replay(fingerprint, record.job_id)
        )
        logger.info("demo generation job %s recorded", record.job_id)
        return JobAccepted(job_id=record.job_id, state=self._job_view(record).state), False

    async def get_job(self, db: AsyncSession, ctx: TenantContext, job_id: UUID) -> JobRead:
        """Return the job's simulated progress: QUEUED, RUNNING, WAITING_PROVIDER, SUCCEEDED.

        Raises:
            NotFoundError: no such job visible to the caller.
        """
        return self._job_view(await self._visible_job(db, ctx, job_id, "Job not found."))

    async def get_asset(self, db: AsyncSession, ctx: TenantContext, asset_id: UUID) -> AssetRead:
        """Return a generated asset once its job has succeeded, with a placeholder image URL.

        Raises:
            NotFoundError: no such asset visible to the caller, or its job is still running.
        """
        job_id = self._job_by_asset.get(asset_id)
        if job_id is None:
            raise NotFoundError("Asset not found.")
        record = await self._visible_job(db, ctx, job_id, "Asset not found.")
        if not self._is_done(record.started, self._timings.generation_seconds):
            raise NotFoundError("Asset not found.")
        return self._asset_view(record)

    def placeholder_file(self, asset_id: UUID) -> bytes | None:
        # Image bytes behind download_url. Generic content only: no tenant or creative data.
        record = self._finished_job_for_asset(asset_id)
        if record is None:
            return None
        return _placeholder_svg(asset_id, record.asset.width_px, record.asset.height_px)

    # Export

    async def export(
        self, db: AsyncSession, ctx: TenantContext, creative_id: UUID
    ) -> ExportManifest:
        """Bundle the current version, its generated image and its latest validation.

        Raises:
            NotFoundError: no such creative visible to the caller.
            ExportNotReadyError: the current version has no finished generated image yet.
        """
        creative = await creatives.get_creative(db, ctx, creative_id)
        job_id = self._latest_jobs.get((ctx.tenant_id, creative_id, creative.version))
        record = None if job_id is None else self._jobs.get(job_id)
        if record is None or not self._is_done(record.started, self._timings.generation_seconds):
            raise ExportNotReadyError()
        latest = None
        run_id = self._latest_runs.get((ctx.tenant_id, creative_id, creative.version, None))
        run = None if run_id is None else self._runs.get(run_id)
        if run is not None:
            view = self._run_view(run, current_version=creative.version)
            latest = ValidationRunSummary.model_validate(
                view.model_dump(include=set(ValidationRunSummary.model_fields))
            )
        policy = await policies.get_active_policy_summary(db, ctx)
        return ExportManifest(
            status="DRAFT",
            creative_id=creative_id,
            creative_version=creative.version,
            caption=creative.caption,
            image=self._asset_view(record),
            study_snapshot_version=creative.study_snapshot_version,
            policy_bundle_version=None if policy is None else policy.version,
            latest_validation=latest,
            exported_at=datetime.now(tz=UTC),
        )

    # Helpers

    def _is_done(self, started: float, duration: float) -> bool:
        return self._clock() - started >= duration

    def _replay_id(
        self, ctx: TenantContext, route: str, idempotency_key: str, fingerprint: str
    ) -> UUID | None:
        replay = self._replays.get((ctx.tenant_id, route, idempotency_key))
        if replay is None:
            return None
        if replay.fingerprint != fingerprint:
            raise IdempotencyKeyReusedError()
        return replay.resource_id

    def _replayed_run(
        self, ctx: TenantContext, idempotency_key: str, fingerprint: str
    ) -> ValidationAccepted | None:
        run_id = self._replay_id(ctx, VALIDATE_ROUTE, idempotency_key, fingerprint)
        record = None if run_id is None else self._runs.get(run_id)
        return None if record is None else self._accepted_run(record)

    def _replayed_job(
        self, ctx: TenantContext, idempotency_key: str, fingerprint: str
    ) -> JobAccepted | None:
        job_id = self._replay_id(ctx, GENERATE_ROUTE, idempotency_key, fingerprint)
        record = None if job_id is None else self._jobs.get(job_id)
        if record is None:
            return None
        return JobAccepted(job_id=record.job_id, state=self._job_view(record).state)

    def _accepted_run(self, record: _ValidationRecord) -> ValidationAccepted:
        return ValidationAccepted(
            job_id=record.finished.job_id,
            validation_run_id=record.finished.validation_run_id,
            state=self._run_view(record, current_version=None).state,
        )

    def _run_view(
        self, record: _ValidationRecord, *, current_version: int | None
    ) -> ValidationRunRead:
        is_current = record.finished.is_current
        if current_version is not None:
            is_current = record.finished.creative_version == current_version
        if self._is_done(record.started, self._timings.validation_seconds):
            return record.finished.model_copy(update={"is_current": is_current})
        return record.finished.model_copy(
            update={
                "state": ExecutionState.RUNNING,
                "outcome": None,
                "evaluated_at": None,
                "is_current": is_current,
                "gates": [],
                "rule_results": [],
                "findings": [],
                "coverage_gaps": [],
            }
        )

    def _finished_job_for_asset(self, asset_id: UUID) -> _GenerationRecord | None:
        job_id = self._job_by_asset.get(asset_id)
        record = None if job_id is None else self._jobs.get(job_id)
        if record is None or not self._is_done(record.started, self._timings.generation_seconds):
            return None
        return record

    def _validation_asset(self, ctx: TenantContext, body: ValidationCreate) -> AssetRead | None:
        if body.asset_id is None:
            return None
        record = self._finished_job_for_asset(body.asset_id)
        if (
            record is None
            or record.tenant_id != ctx.tenant_id
            or record.creative_id != body.creative_id
        ):
            raise NotFoundError("Asset not found.")
        return record.asset

    def _new_generation(
        self, ctx: TenantContext, creative: CreativeRead, version: int
    ) -> _GenerationRecord:
        spec = FORMAT_SPECS[creative.post_format]
        asset_id = uuid4()
        content = _placeholder_svg(asset_id, spec.width_px, spec.height_px)
        created_at = datetime.now(tz=UTC)
        asset = AssetRead(
            asset_id=asset_id,
            creative_id=creative.id,
            origin=AssetOrigin.GENERATED,
            mime_type=STUB_MIME_TYPE,
            width_px=spec.width_px,
            height_px=spec.height_px,
            size_bytes=len(content),
            checksum_sha256=_sha256(content),
            provider=STUB_PROVIDER,
            model=STUB_MODEL,
            created_at=created_at + timedelta(seconds=self._timings.generation_seconds),
            download_url=None,
            download_url_expires_at=None,
        )
        return _GenerationRecord(
            tenant_id=ctx.tenant_id,
            creative_id=creative.id,
            creative_version=version,
            job_id=uuid4(),
            started=self._clock(),
            created_at=created_at,
            asset=asset,
        )

    async def _visible_job(
        self, db: AsyncSession, ctx: TenantContext, job_id: UUID, not_found: str
    ) -> _GenerationRecord:
        record = self._jobs.get(job_id)
        if record is None or record.tenant_id != ctx.tenant_id:
            raise NotFoundError(not_found)
        try:
            # Also enforces study access: a creative the caller cannot see hides its jobs.
            await creatives.get_creative(db, ctx, record.creative_id)
        except NotFoundError:
            raise NotFoundError(not_found) from None
        return record

    def _job_view(self, record: _GenerationRecord) -> JobRead:
        total = self._timings.generation_seconds
        elapsed = max(0.0, min(self._clock() - record.started, total))
        if elapsed >= total:
            state, stage = ExecutionState.SUCCEEDED, "stored"
        elif elapsed >= total * 0.5:
            state, stage = ExecutionState.WAITING_PROVIDER, "generating"
        elif elapsed >= total * 0.15:
            state, stage = ExecutionState.RUNNING, "submitting"
        else:
            state, stage = ExecutionState.QUEUED, None
        done = state is ExecutionState.SUCCEEDED
        submitted = state is not ExecutionState.QUEUED
        return JobRead(
            job_id=record.job_id,
            kind=JobKind.IMAGE_GENERATION,
            creative_id=record.creative_id,
            creative_version=record.creative_version,
            state=state,
            stage=stage,
            provider_request_id=f"demo-{record.job_id.hex[:12]}" if submitted else None,
            failure_code=None,
            failure_message=None,
            result_asset_id=record.asset.asset_id if done else None,
            created_at=record.created_at,
            updated_at=record.created_at + timedelta(seconds=elapsed),
            completed_at=record.asset.created_at if done else None,
        )

    def _asset_view(self, record: _GenerationRecord) -> AssetRead:
        asset_id = record.asset.asset_id
        return record.asset.model_copy(
            update={
                "download_url": f"{self._base_url}{DEMO_FILES_PATH}/{asset_id}",
                "download_url_expires_at": datetime.now(tz=UTC) + DOWNLOAD_URL_TTL,
            }
        )


async def _version(
    db: AsyncSession, ctx: TenantContext, creative_id: UUID, version: int
) -> CreativeVersionRead:
    page = await creatives.list_versions(db, ctx, creative_id, before=version + 1, limit=1)
    if not page.items or page.items[0].version != version:
        raise NotFoundError("Creative version not found.")
    return page.items[0]


async def _active_policy(db: AsyncSession, ctx: TenantContext) -> PolicyBundleRead | None:
    try:
        return await policies.get_active_policy(db, ctx)
    except NotFoundError:
        return None


def _applicable(policy: PolicyBundleRead, kind: ValidationInputKind) -> dict[str, PolicyRule]:
    return {rule.rule_id: rule for rule in policy.rules if kind in rule.applies_to}


def _evaluate_text(
    kind: ValidationInputKind, text: str, policy: PolicyBundleRead | None
) -> _Evaluation:
    if policy is None:
        return _Evaluation(Outcome.NEEDS_REVIEW, [], [], [], [NO_POLICY_GAP, STUB_GAP])
    rules = _applicable(policy, kind)
    findings: list[Finding] = []
    for check in _PHRASE_CHECKS:
        rule = rules.get(check.rule_id)
        if rule is not None:
            findings.extend(
                _rule_finding(
                    rule,
                    check.explanation,
                    text_span=TextSpan(start=match.start(), end=match.end()),
                    offending_excerpt=match.group(),
                    suggested_correction=check.suggested_correction,
                )
                for match in check.pattern.finditer(text)
            )
    disclosure = rules.get(_DISCLOSURE_RULE)
    if disclosure is not None and not all(p.search(text) for p in _DISCLOSURE_PATTERNS):
        findings.append(
            _rule_finding(
                disclosure,
                "Does not say that this is a research study and that taking part is voluntary.",
                suggested_correction=_DISCLOSURE_TEXT,
            )
        )
    return _summarise(rules, findings, checked=_TEXT_RULES)


def _evaluate_image(prompt: str, policy: PolicyBundleRead | None) -> _Evaluation:
    # The placeholder image has no content to inspect, so the brief decides: a brief that
    # mentions payment gets a payment-emphasis finding on a fixed region of the image.
    if policy is None:
        return _Evaluation(Outcome.NEEDS_REVIEW, [], [], [], [NO_POLICY_GAP, STUB_GAP])
    rules = _applicable(policy, ValidationInputKind.IMAGE)
    findings: list[Finding] = []
    payment = rules.get(_PAYMENT_CHECK.rule_id)
    if payment is not None and _PAYMENT_CHECK.pattern.search(prompt):
        findings.append(
            _rule_finding(
                payment,
                "The payment amount is shown in large type.",
                image_region=_PAYMENT_REGION,
            )
        )
    return _summarise(rules, findings, checked=_IMAGE_RULES)


def _rule_finding(
    rule: PolicyRule,
    explanation: str,
    *,
    text_span: TextSpan | None = None,
    image_region: ImageRegion | None = None,
    offending_excerpt: str | None = None,
    suggested_correction: str | None = None,
) -> Finding:
    return Finding(
        rule_id=rule.rule_id,
        rule_version=rule.rule_version,
        severity=rule.severity,
        explanation=explanation,
        source=rule.source,
        text_span=text_span,
        image_region=image_region,
        offending_excerpt=offending_excerpt,
        suggested_correction=suggested_correction,
        needs_human_decision=not rule.mandatory,
    )


def _summarise(
    rules: Mapping[str, PolicyRule], findings: list[Finding], *, checked: frozenset[str]
) -> _Evaluation:
    failed = {finding.rule_id for finding in findings}
    rule_results = [
        RuleResult(
            rule_id=rule.rule_id,
            rule_version=rule.rule_version,
            mandatory=rule.mandatory,
            result=RuleResultValue.FAIL if rule.rule_id in failed else RuleResultValue.PASS,
            reason=None if rule.rule_id in checked else NOT_CHECKED,
        )
        for rule in rules.values()
    ]
    gates = [
        GateResult(
            gate=gate,
            outcome=_outcome(r for r in rules.values() if r.gate is gate and r.rule_id in failed),
        )
        for gate in Gate
        if any(rule.gate is gate for rule in rules.values())
    ]
    outcome = _outcome(rule for rule in rules.values() if rule.rule_id in failed)
    return _Evaluation(outcome, gates, rule_results, findings, [STUB_GAP])


def _outcome(failed_rules: Iterable[PolicyRule]) -> Outcome:
    # A failed mandatory rule fails the run; a failed optional rule needs a human decision.
    failed = list(failed_rules)
    if any(rule.mandatory for rule in failed):
        return Outcome.FAIL
    return Outcome.NEEDS_REVIEW if failed else Outcome.PASS


def _sha256(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def _placeholder_svg(asset_id: UUID, width: int, height: int) -> bytes:
    # Built only from integers and the asset ID's hex digits, so nothing needs escaping.
    label = asset_id.hex[:8]
    svg = (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}">'
        '<defs><linearGradient id="bg" x1="0" y1="0" x2="1" y2="1">'
        '<stop offset="0" stop-color="#1f4e79"/><stop offset="1" stop-color="#2e8b8b"/>'
        "</linearGradient></defs>"
        f'<rect width="{width}" height="{height}" fill="url(#bg)"/>'
        f'<text x="50%" y="47%" text-anchor="middle" font-family="Arial, sans-serif" '
        f'font-size="{width // 12}" fill="#ffffff">DEMO IMAGE</text>'
        f'<text x="50%" y="55%" text-anchor="middle" font-family="Arial, sans-serif" '
        f'font-size="{width // 32}" fill="#dfe9f3">Placeholder, not AI-generated · {label}</text>'
        "</svg>"
    )
    return svg.encode("utf-8")
