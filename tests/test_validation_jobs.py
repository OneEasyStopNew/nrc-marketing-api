# Brief and caption validation as a worker job (VALIDATION_MODEL_MODE on): POST queues, the
# worker checks with the SIMULATED model and stores the run, GET follows it.

import asyncio
from collections.abc import Iterator
from datetime import UTC, datetime, tzinfo
from typing import Any
from uuid import UUID, uuid4

import httpx
import pytest
from asgi_lifespan import LifespanManager
from cryptography.hazmat.primitives.asymmetric.rsa import RSAPrivateKey
from sqlalchemy import func, select, text
from sqlalchemy.ext.asyncio import AsyncConnection, create_async_engine
from sqlalchemy.pool import NullPool

from app.api.deps import get_signing_keys
from app.core.config import Settings, get_settings
from app.core.errors import ValidationJobError
from app.main import create_app
from app.models import Job, ValidationRun
from app.models.enums import ExecutionState, JudgeFailure, ModelCheckFailure, PolicyLanguage
from app.services import rules, validations
from app.services.jobs import RUN_EVENT
from app.services.providers import text_judge
from app.services.providers.base import ProviderRejectedError
from app.services.providers.text_judge import (
    NOVA_LITE_MODEL_ID,
    SIMULATED_JUDGE_MODEL_ID,
    JudgeReply,
    JudgeRequest,
    SimulatedJudgeScenario,
    SimulatedTextJudge,
)
from app.services.validations import INPUT_BY_JOB_KIND, JOB_KIND_BY_INPUT, VALIDATION_RUN_EVENT
from app.workers.common import WorkerConfigError
from app.workers.connector import (
    EVENT_HANDLERS,
    JOB_HANDLERS,
    JobRunHandler,
    ValidationJobHandler,
    check_worker_settings,
    run_once,
)
from scripts.import_policy_graph import PolicyGraphImport, load_synthetic_graph
from tests.conftest import (
    AuthHeaders,
    IntegrationEnv,
    SeededTenant,
    Sessions,
    StaticKeys,
    World,
)

API = "/v1/marketing"
CLEAN_PROMPT = "Invite adults with high blood pressure to learn about our research study."
RISKY = f"Guaranteed results. {CLEAN_PROMPT}"
SCENARIO = SimulatedJudgeScenario


@pytest.fixture
def graph() -> PolicyGraphImport:
    # The real synthetic graph: REC-01 to REC-05 are semantic, so the model decides them.
    return load_synthetic_graph()


@pytest.fixture(autouse=True)
def model_on() -> Iterator[None]:
    # The env is restored before the cache is cleared, so later tests never see these settings.
    with pytest.MonkeyPatch.context() as patch:
        patch.setenv("APP_ENV", "local")
        patch.setenv("VALIDATION_MODEL_MODE", "simulator")
        get_settings.cache_clear()
        yield
    get_settings.cache_clear()


async def _work(
    sessions: Sessions, scenario: SimulatedJudgeScenario = SCENARIO.PASS_ALL, **settings: Any
) -> int:
    # One worker pass with the SIMULATED judge in the given scenario.
    handler = ValidationJobHandler(judge=SimulatedTextJudge(scenario))
    handlers = dict.fromkeys(JOB_KIND_BY_INPUT.values(), handler)
    return await run_once(
        sessions,
        {},
        None,
        get_settings().model_copy(update=settings),
        event_handlers={VALIDATION_RUN_EVENT: JobRunHandler(handlers)},
    )


async def _queued(
    client: httpx.AsyncClient, auth: AuthHeaders, world: World, prompt: str = CLEAN_PROMPT
) -> tuple[dict[str, Any], str]:
    creative = await _creative(client, auth, world.a, prompt=prompt)
    response = await _validate(client, auth, world.a.tenant_id, creative)
    assert response.status_code == 202, response.text
    body = response.json()
    assert body["state"] == "QUEUED"
    assert body["job_id"] == body["validation_run_id"]
    return creative, body["validation_run_id"]


async def _read(client: httpx.AsyncClient, auth: AuthHeaders, world: World, run_id: str) -> Any:
    response = await _run(client, auth, world.a.tenant_id, run_id)
    assert response.status_code == 200, response.text
    return response.json()


async def test_a_generation_worker_pass_never_takes_a_validation(
    client: httpx.AsyncClient, auth: AuthHeaders, world: World, sessions: Sessions
) -> None:
    # Own event type: validation never queues behind, or runs as, a paid generation.
    _, run_id = await _queued(client, auth, world)

    ran = await run_once(
        sessions, {}, None, get_settings(), event_handlers={RUN_EVENT: JobRunHandler(JOB_HANDLERS)}
    )
    job = await _job(sessions, run_id)

    assert (ran, job.state) == (0, ExecutionState.QUEUED)
    assert set(EVENT_HANDLERS) >= {RUN_EVENT, VALIDATION_RUN_EVENT}


async def test_queued_run_is_readable_then_stored_by_the_worker(
    client: httpx.AsyncClient, auth: AuthHeaders, world: World, sessions: Sessions
) -> None:
    _, run_id = await _queued(client, auth, world)

    pending = await _read(client, auth, world, run_id)
    assert (pending["state"], pending["outcome"], pending["rule_results"]) == ("QUEUED", None, [])

    assert await _work(sessions) == 1
    done = await _read(client, auth, world, run_id)

    # Every semantic rule PASSed by the (SIMULATED) model and REC-08 by code.
    assert (done["state"], done["outcome"]) == ("SUCCEEDED", "PASS")
    assert {r["result"] for r in done["rule_results"]} == {"PASS"}
    assert rules.SIMULATED_REVIEW_GAP.reason in {g["reason"] for g in done["coverage_gaps"]}


async def test_the_run_records_the_model_prompt_and_versions(
    client: httpx.AsyncClient,
    auth: AuthHeaders,
    world: World,
    sessions: Sessions,
    connection: AsyncConnection,
) -> None:
    _, run_id = await _queued(client, auth, world)
    await _work(sessions)

    run = (
        await connection.execute(
            select(
                ValidationRun.judge_model_id,
                ValidationRun.judge_prompt_version,
                ValidationRun.judge_attempts,
                ValidationRun.evaluation_version,
                ValidationRun.reuse_key,
                ValidationRun.context_manifest_id,
            ).where(ValidationRun.id == UUID(run_id))
        )
    ).one()
    assert run.judge_model_id == SIMULATED_JUDGE_MODEL_ID
    assert run.judge_prompt_version == "judge-prompt-v1"
    assert run.judge_attempts == 1
    assert run.evaluation_version == rules.EVALUATION_VERSION
    assert run.reuse_key is not None
    assert run.context_manifest_id is not None


async def test_a_code_hit_fails_without_waiting_for_the_model(
    client: httpx.AsyncClient, auth: AuthHeaders, world: World, sessions: Sessions
) -> None:
    _, run_id = await _queued(client, auth, world, prompt=RISKY)

    await _work(sessions)
    done = await _read(client, auth, world, run_id)

    assert done["outcome"] == "FAIL"
    assert [f["rule_id"] for f in done["findings"]] == ["REC-04"]


async def test_a_model_fail_becomes_a_finding(
    client: httpx.AsyncClient, auth: AuthHeaders, world: World, sessions: Sessions
) -> None:
    _, run_id = await _queued(client, auth, world)

    await _work(sessions, SCENARIO.FAIL_FIRST)
    done = await _read(client, auth, world, run_id)

    assert done["outcome"] == "FAIL"
    (finding,) = done["findings"]
    assert finding["explanation"].startswith("SIMULATED: ")
    assert CLEAN_PROMPT[finding["text_span"]["start"] : finding["text_span"]["end"]] == "Invite"


@pytest.mark.parametrize("scenario", [SCENARIO.MALFORMED, SCENARIO.FAKE_CITATION])
async def test_a_bad_model_answer_never_passes(
    client: httpx.AsyncClient,
    auth: AuthHeaders,
    world: World,
    sessions: Sessions,
    scenario: SimulatedJudgeScenario,
) -> None:
    _, run_id = await _queued(client, auth, world)

    await _work(sessions, scenario)
    done = await _read(client, auth, world, run_id)

    assert (done["state"], done["outcome"]) == ("SUCCEEDED", "NEEDS_REVIEW")
    reasons = {r["reason"] for r in done["rule_results"] if r["result"] == "UNABLE_TO_ASSESS"}
    assert len(reasons) == 1
    assert reasons.pop().startswith("The model check failed (")


async def test_throttling_retries_then_needs_review_and_never_opens_generation(
    client: httpx.AsyncClient, auth: AuthHeaders, world: World, sessions: Sessions
) -> None:
    creative, run_id = await _queued(client, auth, world)
    # The cap is fixed on the job when it is queued (VALIDATION_JOB_MAX_ATTEMPTS).
    await _sql(sessions, "UPDATE jobs SET max_attempts = 2 WHERE id = :id", run_id)

    await _work(sessions, SCENARIO.THROTTLED)
    waiting = await _read(client, auth, world, run_id)
    # Recoverable: the job waits to try again, nothing is stored yet.
    assert (waiting["state"], waiting["outcome"]) == ("RUNNING", None)

    await _expire_backoff(sessions, run_id)
    await _work(sessions, SCENARIO.THROTTLED)
    done = await _read(client, auth, world, run_id)

    assert (done["state"], done["outcome"]) == ("SUCCEEDED", "NEEDS_REVIEW")
    reasons = {r["reason"] for r in done["rule_results"] if r["result"] == "UNABLE_TO_ASSESS"}
    assert reasons == {rules.MODEL_CHECK_FAILED.format(failure=ModelCheckFailure.MODEL_UNAVAILABLE)}
    eligibility = await client.get(
        f"/v1/marketing/creatives/{creative['id']}/generation-eligibility",
        headers=auth("creator-a", world.a.tenant_id),
    )
    assert eligibility.json()["eligible"] is False


async def test_past_the_deadline_the_model_is_skipped_and_the_run_needs_review(
    client: httpx.AsyncClient, auth: AuthHeaders, world: World, sessions: Sessions
) -> None:
    _, run_id = await _queued(client, auth, world)
    async with sessions() as db:
        await db.execute(
            text("UPDATE jobs SET deadline_at = now() - interval '1 minute' WHERE id = :id"),
            {"id": run_id},
        )
        await db.commit()

    await _work(sessions)
    done = await _read(client, auth, world, run_id)

    assert done["outcome"] == "NEEDS_REVIEW"
    assert done["rule_results"][0]["reason"] == rules.MODEL_CHECK_FAILED.format(
        failure=ModelCheckFailure.DEADLINE_PASSED
    )


class CountingJudge(SimulatedTextJudge):
    def __init__(self) -> None:
        super().__init__(SCENARIO.PASS_ALL)
        self.calls = 0

    async def judge(self, request: JudgeRequest) -> JudgeReply:
        self.calls += 1
        return await super().judge(request)


async def test_more_open_rules_than_one_call_covers_skips_the_model_and_needs_review(
    client: httpx.AsyncClient,
    auth: AuthHeaders,
    world: World,
    sessions: Sessions,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # The synthetic graph leaves 5 semantic rules open; a cap of 2 can't cover them.
    monkeypatch.setattr(text_judge, "MAX_RULES_PER_CALL", 2)
    _, run_id = await _queued(client, auth, world)
    judge = CountingJudge()
    handler = ValidationJobHandler(judge=judge)
    await run_once(
        sessions,
        {},
        None,
        get_settings(),
        event_handlers={
            VALIDATION_RUN_EVENT: JobRunHandler(dict.fromkeys(JOB_KIND_BY_INPUT.values(), handler))
        },
    )
    done = await _read(client, auth, world, run_id)

    assert judge.calls == 0
    assert done["outcome"] == "NEEDS_REVIEW"
    reasons = {r["reason"] for r in done["rule_results"] if r["result"] == "UNABLE_TO_ASSESS"}
    assert reasons == {rules.MODEL_CHECK_FAILED.format(failure=JudgeFailure.TOO_MANY_RULES)}


async def test_the_job_is_stamped_with_the_instant_its_reuse_key_used(
    client: httpx.AsyncClient,
    auth: AuthHeaders,
    world: World,
    sessions: Sessions,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # One millisecond before midnight UTC by the API's clock: the database clock may already be
    # on the next day, and the worker's key must still use the day the request looked up.
    before_midnight = datetime(2026, 11, 23, 23, 59, 59, 999_000, tzinfo=UTC)

    class _Clock(datetime):
        @classmethod
        def now(cls, tz: tzinfo | None = None) -> _Clock:
            return cls.fromtimestamp(before_midnight.timestamp(), tz=tz)

    monkeypatch.setattr(validations, "datetime", _Clock)
    _, run_id = await _queued(client, auth, world)

    job = await _job(sessions, run_id)
    assert job.created_at == before_midnight


@pytest.fixture
def two_active_validations() -> Iterator[None]:
    with pytest.MonkeyPatch.context() as patch:
        patch.setenv("VALIDATION_MAX_ACTIVE_JOBS_PER_TENANT", "2")
        get_settings.cache_clear()
        yield
    get_settings.cache_clear()


@pytest.mark.usefixtures("two_active_validations")
async def test_a_tenant_can_queue_up_to_its_limit_then_gets_429_until_one_finishes(
    client: httpx.AsyncClient, auth: AuthHeaders, world: World, sessions: Sessions
) -> None:
    # Different texts, so nothing is reused and each one would queue a job.
    for prompt in (CLEAN_PROMPT, f"{CLEAN_PROMPT} Learn more today."):
        await _queued(client, auth, world, prompt=prompt)

    third = await _creative(client, auth, world.a, prompt=f"{CLEAN_PROMPT} Call us.")
    refused = await _validate(client, auth, world.a.tenant_id, third)

    assert refused.status_code == 429
    body = refused.json()
    assert (body["code"], body["retryable"]) == ("VALIDATION_LIMIT_REACHED", True)
    assert await _work(sessions) == 2
    assert (await _validate(client, auth, world.a.tenant_id, third)).status_code == 202


@pytest.fixture
def one_active_validation() -> Iterator[None]:
    with pytest.MonkeyPatch.context() as patch:
        patch.setenv("VALIDATION_MAX_ACTIVE_JOBS_PER_TENANT", "1")
        get_settings.cache_clear()
        yield
    get_settings.cache_clear()


@pytest.mark.usefixtures("one_active_validation")
async def test_parallel_validations_never_take_more_than_the_last_slot(
    committed_tenant: SeededTenant,
    signing_key: RSAPrivateKey,
    test_env: IntegrationEnv,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Committed data and a connection per request: on one shared connection the admission lock
    # would never be contended. A pause between the count and the insert makes every request
    # count before any inserts, which is what the lock must prevent.
    queue = validations._queue

    async def slow_queue(*args: Any, **kwargs: Any) -> UUID:
        await asyncio.sleep(0.2)
        return await queue(*args, **kwargs)

    monkeypatch.setattr(validations, "_queue", slow_queue)
    app = create_app()
    app.dependency_overrides[get_signing_keys] = lambda: StaticKeys(signing_key)
    auth = AuthHeaders(signing_key)
    async with (
        LifespanManager(app),
        httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as http,
    ):
        creatives = [
            await _creative(http, auth, committed_tenant, prompt=f"{CLEAN_PROMPT} Variant {n}.")
            for n in range(5)
        ]
        responses = await asyncio.gather(
            *(_validate(http, auth, committed_tenant.tenant_id, c) for c in creatives)
        )

    assert sorted(r.status_code for r in responses) == [202, 429, 429, 429, 429]
    engine = create_async_engine(
        test_env.test_migration_database_url.get_secret_value(), poolclass=NullPool
    )
    try:
        async with engine.connect() as conn:
            queued = await conn.scalar(
                text("SELECT count(*) FROM jobs WHERE tenant_id = :t AND kind = ANY(:kinds)"),
                {"t": committed_tenant.tenant_id, "kinds": list(INPUT_BY_JOB_KIND)},
            )
    finally:
        await engine.dispose()
    assert queued == 1


def _spanish_only_graph() -> PolicyGraphImport:
    # Every rule limited to Spanish, so an English brief selects none of them.
    base = load_synthetic_graph()
    rule_nodes = {rule.node_key for rule in base.rules}
    return base.model_copy(
        update={
            "nodes": [
                node.model_copy(update={"languages": [PolicyLanguage.ES]})
                if node.node_key in rule_nodes
                else node
                for node in base.nodes
            ]
        }
    )


@pytest.mark.parametrize("graph", [_spanish_only_graph()])
async def test_a_text_no_rule_applies_to_needs_review_instead_of_passing(
    client: httpx.AsyncClient, auth: AuthHeaders, world: World, sessions: Sessions
) -> None:
    # Nothing checked is not the same as nothing broken: no unknown check becomes PASS.
    creative, run_id = await _queued(client, auth, world)

    await _work(sessions)
    done = await _read(client, auth, world, run_id)

    assert (done["state"], done["outcome"], done["rule_results"]) == (
        "SUCCEEDED",
        "NEEDS_REVIEW",
        [],
    )
    assert rules.NO_RULES_SELECTED_GAP.reason in {g["reason"] for g in done["coverage_gaps"]}
    eligibility = await client.get(
        f"/v1/marketing/creatives/{creative['id']}/generation-eligibility",
        headers=auth("creator-a", world.a.tenant_id),
    )
    assert eligibility.json()["eligible"] is False


class RejectingJudge:
    # A model refusal that retrying can't fix, like a disabled model or a rejected request.

    model_id = SIMULATED_JUDGE_MODEL_ID
    is_simulated = True

    async def judge(self, request: JudgeRequest) -> JudgeReply:
        raise ProviderRejectedError("The model rejected the request.")


async def test_a_refused_model_call_stores_a_fixed_code_not_the_provider_error_code(
    client: httpx.AsyncClient, auth: AuthHeaders, world: World, sessions: Sessions
) -> None:
    # The frontend branches on the code in brackets, so it is a ModelCheckFailure member.
    _, run_id = await _queued(client, auth, world)
    handler = ValidationJobHandler(judge=RejectingJudge())
    await run_once(
        sessions,
        {},
        None,
        get_settings(),
        event_handlers={
            VALIDATION_RUN_EVENT: JobRunHandler(dict.fromkeys(JOB_KIND_BY_INPUT.values(), handler))
        },
    )
    done = await _read(client, auth, world, run_id)

    assert (done["state"], done["outcome"]) == ("SUCCEEDED", "NEEDS_REVIEW")
    reasons = {r["reason"] for r in done["rule_results"] if r["result"] == "UNABLE_TO_ASSESS"}
    assert reasons == {rules.MODEL_CHECK_FAILED.format(failure=ModelCheckFailure.MODEL_REJECTED)}


async def test_identical_content_reuses_a_finished_result_without_a_job(
    client: httpx.AsyncClient,
    auth: AuthHeaders,
    world: World,
    sessions: Sessions,
    connection: AsyncConnection,
) -> None:
    _, first_id = await _queued(client, auth, world)
    await _work(sessions)

    second = await _creative(client, auth, world.a, prompt=CLEAN_PROMPT)
    response = await _validate(client, auth, world.a.tenant_id, second)

    body = response.json()
    assert body["state"] == "SUCCEEDED"
    assert body["validation_run_id"] != first_id
    reused = await _read(client, auth, world, body["validation_run_id"])
    assert reused["outcome"] == "PASS"
    jobs = await connection.scalar(
        select(func.count()).select_from(Job).where(Job.tenant_id == world.a.tenant_id)
    )
    assert jobs == 1


async def test_different_content_is_never_reused(
    client: httpx.AsyncClient, auth: AuthHeaders, world: World, sessions: Sessions
) -> None:
    await _queued(client, auth, world)
    await _work(sessions)

    # _queued asserts QUEUED: a new job, not a copy.
    await _queued(client, auth, world, prompt=f"{CLEAN_PROMPT} Ask your doctor.")


async def test_a_needs_review_result_is_not_reused(
    client: httpx.AsyncClient, auth: AuthHeaders, world: World, sessions: Sessions
) -> None:
    await _queued(client, auth, world)
    await _work(sessions, SCENARIO.MALFORMED)

    # A failed model answer may be transient: the same content is queued again (_queued asserts
    # QUEUED), not copied.
    await _queued(client, auth, world)


async def test_a_replay_returns_the_same_queued_run(
    client: httpx.AsyncClient, auth: AuthHeaders, world: World
) -> None:
    creative = await _creative(client, auth, world.a)
    first = await _validate(client, auth, world.a.tenant_id, creative, key="same-key-1")
    again = await _validate(client, auth, world.a.tenant_id, creative, key="same-key-1")

    assert again.json() == first.json()
    assert again.headers["Idempotent-Replayed"] == "true"


async def test_other_tenants_and_the_jobs_endpoint_cannot_read_a_queued_validation(
    client: httpx.AsyncClient, auth: AuthHeaders, world: World
) -> None:
    _, run_id = await _queued(client, auth, world)

    other = await client.get(
        f"/v1/marketing/validations/{run_id}", headers=auth("creator-b", world.b.tenant_id)
    )
    as_job = await client.get(
        f"/v1/marketing/jobs/{run_id}", headers=auth("creator-a", world.a.tenant_id)
    )

    assert other.status_code == 404
    # Validation jobs are internal; GET /jobs serves generation jobs only.
    assert as_job.status_code == 404


async def test_the_job_cancel_endpoint_cannot_cancel_a_validation(
    client: httpx.AsyncClient, auth: AuthHeaders, world: World, sessions: Sessions
) -> None:
    # The cancel path (NRC-6) adds a job.run event; a validation job must never get one.
    _, run_id = await _queued(client, auth, world)

    response = await client.post(
        f"/v1/marketing/jobs/{run_id}/cancel", headers=auth("creator-a", world.a.tenant_id)
    )
    job = await _job(sessions, run_id)

    assert response.status_code == 404
    assert job.state is ExecutionState.QUEUED


async def test_a_validation_job_whose_pinned_policy_is_gone_fails_cleanly(
    client: httpx.AsyncClient, auth: AuthHeaders, world: World, sessions: Sessions
) -> None:
    _, run_id = await _queued(client, auth, world)
    await _sql(sessions, "UPDATE jobs SET policy_bundle_version = '999' WHERE id = :id", run_id)

    await _work(sessions)
    job = await _job(sessions, run_id)

    assert (job.state, job.failure_code) == (
        ExecutionState.FAILED,
        ValidationJobError.code,
    )


async def test_a_validation_job_with_an_unreadable_policy_version_fails_at_once(
    client: httpx.AsyncClient, auth: AuthHeaders, world: World, sessions: Sessions
) -> None:
    # jobs.policy_bundle_version is text; a bad value must not crash the worker into retries.
    _, run_id = await _queued(client, auth, world)
    await _sql(sessions, "UPDATE jobs SET policy_bundle_version = 'v2' WHERE id = :id", run_id)

    await _work(sessions)
    job = await _job(sessions, run_id)

    assert (job.state, job.failure_code, job.attempt_count) == (
        ExecutionState.FAILED,
        ValidationJobError.code,
        0,
    )


async def _sql(sessions: Sessions, statement: str, job_id: str) -> None:
    # Test set-up on the worker's connection (it may update jobs).
    async with sessions() as db:
        await db.execute(text(statement), {"id": job_id})
        await db.commit()


async def _expire_backoff(sessions: Sessions, job_id: str) -> None:
    # Make the waiting run event due now instead of after its backoff.
    async with sessions() as db:
        await db.execute(
            text(
                "UPDATE outbox_events SET available_at = now() - interval '1 second' "
                "WHERE job_id = :id"
            ),
            {"id": job_id},
        )
        await db.commit()


async def _job(sessions: Sessions, job_id: str) -> Job:
    async with sessions() as db:
        job = await db.get(Job, UUID(job_id))
        assert job is not None
        return job


def test_the_worker_refuses_a_live_judge_without_a_model() -> None:
    # A call budget inside the lease, so only the missing model is wrong.
    settings = _live_judge(timeout=10, retries=3).model_copy(
        update={"bedrock_validation_model_id": None}
    )

    with pytest.raises(WorkerConfigError, match="missing BEDROCK_VALIDATION_MODEL_ID"):
        check_worker_settings(settings)


def _live_judge(timeout: float, retries: int) -> Settings:
    return get_settings().model_copy(
        update={
            "validation_model_mode": "live",
            "bedrock_validation_model_id": NOVA_LITE_MODEL_ID,
            "validation_model_timeout_seconds": timeout,
            "bedrock_max_retries": retries,
            "worker_lease_seconds": 60,
        }
    )


def test_the_worker_refuses_a_live_judge_call_budget_as_long_as_its_lease() -> None:
    # 15 s per attempt x 4 attempts = 60 s, the lease: a call could outlive it.
    with pytest.raises(WorkerConfigError, match=r"BEDROCK_MAX_RETRIES \+ 1\) must be below"):
        check_worker_settings(_live_judge(timeout=15, retries=3))


def test_a_live_judge_call_budget_just_under_the_lease_is_accepted() -> None:
    # 14.9 s x 4 = 59.6 s < 60 s.
    check_worker_settings(_live_judge(timeout=14.9, retries=3))


@pytest.mark.parametrize("mode", ["off", "simulator"])
def test_without_a_live_judge_the_call_budget_does_not_stop_the_worker(mode: str) -> None:
    # No Bedrock call happens, so its timeouts can't outlive a lease.
    settings = get_settings().model_copy(
        update={
            "validation_model_mode": mode,
            "validation_model_timeout_seconds": 60,
            "worker_lease_seconds": 60,
        }
    )

    check_worker_settings(settings)


async def _creative(
    client: httpx.AsyncClient, auth: AuthHeaders, tenant: SeededTenant, prompt: str = CLEAN_PROMPT
) -> dict[str, Any]:
    response = await client.post(
        f"{API}/creatives",
        headers=auth("creator-a", tenant.tenant_id, **{"Idempotency-Key": uuid4().hex}),
        json={
            "study_id": str(tenant.study_id),
            "prompt": prompt,
            "language": "en",
            "channel": "facebook",
            "post_format": "feed_image",
        },
    )
    assert response.status_code == 201, response.text
    body: dict[str, Any] = response.json()
    return body


async def _validate(
    client: httpx.AsyncClient,
    auth: AuthHeaders,
    tenant_id: UUID,
    creative: dict[str, Any],
    *,
    key: str | None = None,
) -> httpx.Response:
    return await client.post(
        f"{API}/validations",
        headers=auth("creator-a", tenant_id, **{"Idempotency-Key": key or uuid4().hex}),
        json={
            "creative_id": creative["id"],
            "creative_version": creative["version"],
            "input_kind": "brief",
        },
    )


async def _run(
    client: httpx.AsyncClient, auth: AuthHeaders, tenant_id: UUID, run_id: str
) -> httpx.Response:
    return await client.get(f"{API}/validations/{run_id}", headers=auth("creator-a", tenant_id))
