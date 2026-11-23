# Brief and caption validation as a worker job (VALIDATION_MODEL_MODE on): POST queues, the
# worker checks with the SIMULATED model and stores the run, GET follows it.

from collections.abc import Iterator
from typing import Any
from uuid import UUID, uuid4

import httpx
import pytest
from sqlalchemy import func, select, text
from sqlalchemy.ext.asyncio import AsyncConnection

from app.core.config import get_settings
from app.core.errors import ValidationJobError
from app.models import Job, ValidationRun
from app.models.enums import ExecutionState
from app.services import rules, validation_jobs
from app.services.jobs import RUN_EVENT
from app.services.providers.text_judge import (
    SIMULATED_JUDGE_MODEL_ID,
    SimulatedJudgeScenario,
    SimulatedTextJudge,
)
from app.services.validations import JOB_KIND_BY_INPUT, VALIDATION_RUN_EVENT
from app.workers.connector import (
    EVENT_HANDLERS,
    JOB_HANDLERS,
    JobRunHandler,
    ValidationJobHandler,
    WorkerConfigError,
    check_worker_settings,
    run_once,
)
from scripts.import_policy_graph import PolicyGraphImport, load_synthetic_graph
from tests.conftest import AuthHeaders, SeededTenant, Sessions, World

API = "/v1/marketing"
CLEAN_PROMPT = "Invite adults with high blood pressure to learn about our research study."
RISKY = f"Guaranteed results. {CLEAN_PROMPT}"
SCENARIO = SimulatedJudgeScenario


@pytest.fixture
def graph() -> PolicyGraphImport:
    # The real synthetic graph: REC-01 to REC-05 are semantic, so the model decides them.
    return load_synthetic_graph()


@pytest.fixture(autouse=True)
def model_on(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    monkeypatch.setenv("APP_ENV", "local")
    monkeypatch.setenv("VALIDATION_MODEL_MODE", "simulator")
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
    assert reasons == {rules.MODEL_CHECK_FAILED.format(failure=validation_jobs.ATTEMPTS_USED)}
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
    assert validation_jobs.DEADLINE_PASSED in done["rule_results"][0]["reason"]


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
    settings = get_settings().model_copy(
        update={"validation_model_mode": "live", "bedrock_validation_model_id": None}
    )

    with pytest.raises(WorkerConfigError, match="BEDROCK_VALIDATION_MODEL_ID"):
        check_worker_settings(settings)


def test_the_worker_refuses_a_judge_timeout_as_long_as_its_lease() -> None:
    settings = get_settings().model_copy(
        update={"validation_model_timeout_seconds": 60, "worker_lease_seconds": 60}
    )

    with pytest.raises(WorkerConfigError, match="VALIDATION_MODEL_TIMEOUT_SECONDS"):
        check_worker_settings(settings)


def test_with_the_model_off_the_judge_timeout_does_not_stop_the_worker() -> None:
    # No judge calls happen, so the judge timeout can't outlive a lease.
    settings = get_settings().model_copy(
        update={
            "validation_model_mode": "off",
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
