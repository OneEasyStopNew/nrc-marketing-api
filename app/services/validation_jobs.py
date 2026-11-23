"""The worker's half of brief and caption validation (NRC-7): one validation job, start to end.

POST /validations queued the job with everything pinned (creative version, study snapshot,
policy bundle version). Here the worker gathers the policy context, runs the code checks, asks
the model about the semantic rules the code left open, and inserts the finished run under the
job's id. Runs are insert-only, so a run is written exactly once, with the job's move to
SUCCEEDED, in one transaction.

Throttling and timeouts are retried with backoff until the job's attempts or deadline run out;
then the run is stored with those rules UNABLE_TO_ASSESS (NEEDS_REVIEW), never PASS. A model
answer that fails verification is retried inside judging.run_judge, then treated the same way.
Nothing here starts generation: the generation gate only reads stored brief runs.
"""

import logging
from collections.abc import Awaitable, Callable
from datetime import UTC, datetime
from typing import Final

from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import Settings
from app.core.errors import AppError, ValidationJobError
from app.models import Creative, Job
from app.models.enums import ExecutionState, JobStage, ModelCheckFailure
from app.services import judging, policies, rules, validations
from app.services.jobs import transition
from app.services.policy_context import PolicyContextProvider, StoredManifest
from app.services.providers.base import ProviderError
from app.services.providers.text_judge import TextJudge

_ACTIVE: Final = frozenset({ExecutionState.QUEUED, ExecutionState.RUNNING})

logger = logging.getLogger(__name__)


async def run(
    db: AsyncSession,
    job: Job,
    *,
    judge: TextJudge | None,
    context_provider: PolicyContextProvider,
    settings: Settings,
    now: datetime,
    heartbeat: Callable[[AsyncSession], Awaitable[None]],
    retry_delay: Callable[[int], float],
) -> float | None:
    """Run one validation job. Returns None when done, or the seconds until it should run again.

    Raises:
        ValidationJobError: the pinned creative version or policy version is unreadable or gone.
    """
    if job.state not in _ACTIVE:
        return None
    kind = validations.INPUT_BY_JOB_KIND[job.kind]
    version = validations.pinned_policy_version(job)
    if version is None:
        raise ValidationJobError("The job has no readable policy version.")
    await transition(db, job.id, allowed_from=_ACTIVE, to=ExecutionState.RUNNING, stage=None)
    try:
        target = await validations.load_target(
            db, (Creative.tenant_id == job.tenant_id,), job.creative_id, job.creative_version
        )
        policy = await policies.get_policy_version(db, job.tenant_id, version)
    except AppError:
        raise ValidationJobError(
            "The creative version or policy version to check is gone."
        ) from None
    text = target.text(kind)
    stored, context_gaps = await validations.gather_context(
        db,
        job.tenant_id,
        job.created_by,
        context_provider,
        settings.policy_context_timeout_seconds,
        kind,
        text,
        target,
        policy,
        now,
    )
    judged = validations.Judged()
    if stored is None:
        # Same rule as in the request: without its policy context a run can't pass.
        evaluation = validations.without_pass(rules.evaluate_text(kind, text, policy, target.facts))
    else:
        selected = await validations.graph_rules(db, job.tenant_id, stored.manifest, policy)
        open_rules = rules.rules_for_model(text, selected, target.facts)
        review = None
        if open_rules and judge is not None:
            outcome = await _review(
                db, job, judge, text, open_rules, stored, settings, now, heartbeat, retry_delay
            )
            if isinstance(outcome, float):
                return outcome
            review, judged = outcome
        evaluation = rules.evaluate_graph_rules(text, selected, target.facts, review)
    db.add(
        validations.new_run(
            run_id=job.id,
            tenant_id=job.tenant_id,
            creative_id=job.creative_id,
            creative_version=job.creative_version,
            kind=kind,
            text=text,
            target=target,
            policy=policy,
            evaluation=evaluation,
            evaluated_at=now,
            context=stored,
            context_gaps=context_gaps,
            created_by=job.created_by,
            judged=judged,
            reuse_key=validations.reuse_key(
                kind, text, target, policy, settings, job.created_at.astimezone(UTC).date()
            ),
        )
    )
    await transition(db, job.id, allowed_from={ExecutionState.RUNNING}, to=ExecutionState.SUCCEEDED)
    logger.info(
        "validation stored job=%s tenant=%s outcome=%s",
        job.id,
        job.tenant_id,
        evaluation.outcome.value,
    )
    return None


async def _review(
    db: AsyncSession,
    job: Job,
    judge: TextJudge,
    text: str,
    open_rules: list[rules.GraphRule],
    stored: StoredManifest,
    settings: Settings,
    now: datetime,
    heartbeat: Callable[[AsyncSession], Awaitable[None]],
    retry_delay: Callable[[int], float],
) -> tuple[rules.ModelReview, validations.Judged] | float:
    # The model's verified answers, a failed review to store, or a delay before trying again.
    judged = validations.Judged(
        model_id=judge.model_id, prompt_version=judging.PROMPT_TEMPLATE_VERSION
    )
    if now >= job.deadline_at:
        return _failed(judge, ModelCheckFailure.DEADLINE_PASSED), judged
    context = judging.judge_context(
        validations.judged_rules(open_rules), stored.manifest.evidence, stored.manifest.exceptions
    )
    try:
        judgement = await judging.run_judge(
            judge,
            text,
            context,
            max_output_tokens=settings.validation_model_max_output_tokens,
            max_attempts=settings.validation_model_max_attempts,
            before_call=lambda: heartbeat(db),
        )
    except ProviderError as error:
        attempts = job.attempt_count + 1
        if error.retryable and attempts < job.max_attempts:
            await transition(
                db,
                job.id,
                allowed_from={ExecutionState.RUNNING},
                to=ExecutionState.RUNNING,
                stage=JobStage.RETRY_WAIT,
                attempt_count=attempts,
            )
            logger.warning(
                "judge unavailable, retrying job=%s attempt=%d code=%s",
                job.id,
                attempts,
                error.code,
            )
            return max(error.retry_after_seconds or 0.0, retry_delay(attempts))
        logger.warning("judge failed job=%s code=%s", job.id, error.code)
        failure = (
            ModelCheckFailure.MODEL_UNAVAILABLE
            if error.retryable
            else ModelCheckFailure.MODEL_REJECTED
        )
        return _failed(judge, failure), judged
    answers = judgement.answers or {}
    review = rules.ModelReview(
        answers=answers,
        failure=judgement.failure,
        is_simulated=judge.is_simulated,
    )
    return review, validations.Judged(
        model_id=judgement.model_id,
        prompt_version=judgement.prompt_version,
        attempts=judgement.attempts,
    )


def _failed(judge: TextJudge, failure: ModelCheckFailure) -> rules.ModelReview:
    return rules.ModelReview(answers={}, failure=failure, is_simulated=judge.is_simulated)
