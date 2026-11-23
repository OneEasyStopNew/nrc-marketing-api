# One runner over the labelled SYNTHETIC NRC-7 cases in tests/data/nrc7_labelled_cases.json
# (developer-labelled, not the expert-labelled set the ticket requires). Each case runs through the
# real API: model off answers on POST; a SIMULATED judge scenario is queued and run by the worker.

import json
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any
from uuid import uuid4

import httpx
import pytest
from sqlalchemy import text

from app.core.config import get_settings
from app.services.jobs import RUN_EVENT
from app.services.providers.text_judge import SimulatedJudgeScenario, SimulatedTextJudge
from app.services.validations import JOB_KIND_BY_INPUT
from app.workers.connector import JOB_HANDLERS, JobRunHandler, ValidationJobHandler, run_once
from scripts.import_policy_graph import PolicyGraphImport, load_synthetic_graph
from tests.conftest import ROOT, AuthHeaders, Sessions, World

API = "/v1/marketing"
MODEL_OFF = "off"
CASES_FILE = ROOT / "tests" / "data" / "nrc7_labelled_cases.json"
ANY_RULE = "*"
# Results that may sit under an overall PASS (AC2: complete coverage).
PASSING_RESULTS = {"PASS", "NOT_APPLICABLE"}
# Passes a throttled run needs: one fails and waits, the next uses the last attempt.
THROTTLE_PASSES = 2
MAX_ATTEMPTS = 2


@dataclass(frozen=True, slots=True)
class Case:
    id: str
    kind: str
    prompt: str
    caption: str | None
    model: str
    expect_outcome: str
    expect_fail_rules: frozenset[str] | None
    expect_spans: list[dict[str, str]]

    @property
    def text(self) -> str:
        # The text the validation reads: the brief, or the caption (empty when there is none).
        return self.prompt if self.kind == "brief" else self.caption or ""


def _load_cases() -> list[Case]:
    raw = json.loads(CASES_FILE.read_text(encoding="utf-8"))
    return [
        Case(
            id=item["id"],
            kind=item["kind"],
            prompt=item["prompt"],
            caption=item.get("caption"),
            model=item["model"],
            expect_outcome=item["expect_outcome"],
            expect_fail_rules=(
                frozenset(item["expect_fail_rules"]) if "expect_fail_rules" in item else None
            ),
            expect_spans=item.get("expect_spans", []),
        )
        for item in raw["cases"]
    ]


CASES = _load_cases()


@pytest.fixture
def graph() -> PolicyGraphImport:
    # The real synthetic graph: the claim rules are semantic, so the model decides them.
    return load_synthetic_graph()


@pytest.fixture
def settings_cache() -> Iterator[None]:
    yield
    get_settings.cache_clear()


def test_the_case_file_is_labelled_and_complete() -> None:
    assert "SYNTHETIC" in json.loads(CASES_FILE.read_text(encoding="utf-8"))["_note"]
    assert len({case.id for case in CASES}) == len(CASES)
    scenarios = {s.value for s in SimulatedJudgeScenario} | {MODEL_OFF}
    assert {case.model for case in CASES} <= scenarios
    assert {case.kind for case in CASES} == {"brief", "text"}


@pytest.mark.usefixtures("settings_cache")
@pytest.mark.parametrize("case", CASES, ids=[case.id for case in CASES])
async def test_labelled_case(
    case: Case,
    client: httpx.AsyncClient,
    auth: AuthHeaders,
    world: World,
    sessions: Sessions,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("APP_ENV", "local")
    monkeypatch.setenv("VALIDATION_MODEL_MODE", "off" if case.model == MODEL_OFF else "simulator")
    get_settings.cache_clear()
    tenant_id = world.a.tenant_id
    headers = auth("creator-a", tenant_id)

    creative = await _creative(client, auth, world, case)
    accepted = await client.post(
        f"{API}/validations",
        headers=auth("creator-a", tenant_id, **{"Idempotency-Key": uuid4().hex}),
        json={
            "creative_id": creative["id"],
            "creative_version": creative["version"],
            "input_kind": case.kind,
        },
    )
    assert accepted.status_code == 202, accepted.text
    run_id = accepted.json()["validation_run_id"]

    if case.model == MODEL_OFF:
        assert accepted.json()["state"] == "SUCCEEDED"
    else:
        assert accepted.json()["state"] == "QUEUED"
        await _run_worker(sessions, case.model, run_id)

    response = await client.get(f"{API}/validations/{run_id}", headers=headers)
    assert response.status_code == 200, response.text
    body = response.json()

    assert body["state"] == "SUCCEEDED"
    assert body["outcome"] == case.expect_outcome
    findings = body["findings"]
    if case.expect_fail_rules is not None:
        assert {f["rule_id"] for f in findings} == case.expect_fail_rules
    for expected in case.expect_spans:
        assert _excerpts(case.text, findings, expected["rule_id"]) & {expected["excerpt"]}, (
            expected,
            findings,
        )

    # AC2: a PASS needs every rule PASS (or not applicable) and no rule left as a coverage gap.
    if body["outcome"] == "PASS":
        assert {r["result"] for r in body["rule_results"]} <= PASSING_RESULTS
        assert not [g for g in body["coverage_gaps"] if g.get("rule_id")]
    else:
        eligibility = await client.get(
            f"{API}/creatives/{creative['id']}/generation-eligibility", headers=headers
        )
        assert eligibility.json()["eligible"] is False

    # AC7: the stored result is the same after a refresh, and another tenant cannot see it.
    again = await client.get(f"{API}/validations/{run_id}", headers=headers)
    assert again.json() == body
    other = await client.get(
        f"{API}/validations/{run_id}", headers=auth("creator-b", world.b.tenant_id)
    )
    assert other.status_code == 404


def _excerpts(text_under_review: str, findings: list[dict[str, Any]], rule_id: str) -> set[str]:
    # Code-point slices of the reviewed text at each finding's span (ANY_RULE: all findings).
    return {
        text_under_review[f["text_span"]["start"] : f["text_span"]["end"]]
        for f in findings
        if f["text_span"] is not None and rule_id in (ANY_RULE, f["rule_id"])
    }


async def _creative(
    client: httpx.AsyncClient, auth: AuthHeaders, world: World, case: Case
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "study_id": str(world.a.study_id),
        "prompt": case.prompt,
        "language": "en",
        "channel": "facebook",
        "post_format": "feed_image",
    }
    if case.caption is not None:
        payload["caption"] = case.caption
    response = await client.post(
        f"{API}/creatives",
        headers=auth("creator-a", world.a.tenant_id, **{"Idempotency-Key": uuid4().hex}),
        json=payload,
    )
    assert response.status_code == 201, response.text
    body: dict[str, Any] = response.json()
    return body


async def _run_worker(sessions: Sessions, scenario: str, run_id: str) -> None:
    # One worker pass with the SIMULATED judge. A throttled run waits and tries again, so it
    # gets a second pass after its backoff is made due, with the cap lowered to two attempts.
    judge = SimulatedTextJudge(SimulatedJudgeScenario(scenario))
    handler = ValidationJobHandler(judge=judge)
    handlers = {**JOB_HANDLERS, **dict.fromkeys(JOB_KIND_BY_INPUT.values(), handler)}
    passes = THROTTLE_PASSES if scenario == SimulatedJudgeScenario.THROTTLED else 1
    if passes > 1:
        await _sql(sessions, "UPDATE jobs SET max_attempts = :n WHERE id = :id", run_id)
    for attempt in range(passes):
        if attempt:
            await _sql(
                sessions,
                "UPDATE outbox_events SET available_at = now() - interval '1 second' "
                "WHERE job_id = :id",
                run_id,
            )
        done = await run_once(
            sessions,
            {},
            None,
            get_settings(),
            event_handlers={RUN_EVENT: JobRunHandler(handlers)},
        )
        assert done == 1


async def _sql(sessions: Sessions, statement: str, job_id: str) -> None:
    # Test set-up on the worker's connection (it may update jobs and outbox events).
    async with sessions() as db:
        await db.execute(text(statement), {"id": job_id, "n": MAX_ATTEMPTS})
        await db.commit()
