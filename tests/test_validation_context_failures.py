# When policy context can't be gathered, a validation never passes (NRC-5): the worker stores
# the run, with a coverage gap, as NEEDS_REVIEW (or FAIL, if the rule checks already failed it).

import logging
from typing import Any
from uuid import uuid4

import httpx
import pytest
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import get_settings
from app.services.policy_context import ContextRequest, PolicyContextProvider, StoredManifest
from app.services.validations import INPUT_BY_JOB_KIND, VALIDATION_RUN_EVENT
from app.workers.connector import JobRunHandler, ValidationJobHandler, run_once
from tests.conftest import AuthHeaders, Sessions, World

API = "/v1/marketing"
CLEAN_PROMPT = "Invite adults with high blood pressure to learn about our research study."
RISKY_PROMPT = "Guaranteed results! This proven cure is free treatment."
# Short enough to keep the test fast, long enough that a normal build would never hit it.
TEST_TIMEOUT_SECONDS = 0.5


class BrokenProvider:
    # Fails inside the database, like a bad query or a lost connection would.

    async def build(
        self, db: AsyncSession, request: ContextRequest, *, actor: str
    ) -> StoredManifest:
        await db.execute(text("SELECT 1 / 0"))
        raise AssertionError("the query above always fails")


class StalledProvider:
    # Takes far longer than the deadline, like a slow query on a large bundle would.

    async def build(
        self, db: AsyncSession, request: ContextRequest, *, actor: str
    ) -> StoredManifest:
        await db.execute(text("SELECT pg_sleep(30)"))
        raise AssertionError("the deadline should have stopped the query above")


async def _validate(
    client: httpx.AsyncClient,
    auth: AuthHeaders,
    world: World,
    sessions: Sessions,
    prompt: str,
    provider: PolicyContextProvider,
) -> dict[str, Any]:
    creative = await client.post(
        f"{API}/creatives",
        headers=auth("creator-a", world.a.tenant_id, **{"Idempotency-Key": uuid4().hex}),
        json={
            "study_id": str(world.a.study_id),
            "prompt": prompt,
            "language": "en",
            "channel": "facebook",
            "post_format": "feed_image",
        },
    )
    assert creative.status_code == 201, creative.text
    response = await client.post(
        f"{API}/validations",
        headers=auth("creator-a", world.a.tenant_id, **{"Idempotency-Key": uuid4().hex}),
        json={"creative_id": creative.json()["id"], "creative_version": 1, "input_kind": "brief"},
    )
    assert response.status_code == 202, response.text
    run_id = response.json()["validation_run_id"]
    # The worker gathers the context with this provider; the savepoint keeps the run's own
    # transaction usable, so the run is stored.
    handler = ValidationJobHandler(context_provider=provider)
    settings = get_settings().model_copy(
        update={"policy_context_timeout_seconds": TEST_TIMEOUT_SECONDS}
    )
    ran = await run_once(
        sessions,
        {},
        None,
        settings,
        event_handlers={
            VALIDATION_RUN_EVENT: JobRunHandler(dict.fromkeys(INPUT_BY_JOB_KIND, handler))
        },
    )
    assert ran == 1
    run: dict[str, Any] = (
        await client.get(
            f"{API}/validations/{run_id}", headers=auth("creator-a", world.a.tenant_id)
        )
    ).json()
    return run


async def test_database_error_makes_a_clean_brief_need_review(
    client: httpx.AsyncClient, auth: AuthHeaders, world: World, sessions: Sessions
) -> None:
    run = await _validate(client, auth, world, sessions, CLEAN_PROMPT, BrokenProvider())

    assert run["outcome"] == "NEEDS_REVIEW"
    assert all(gate["outcome"] != "PASS" for gate in run["gates"])
    assert any("(RETRIEVAL_ERROR)" in gap["reason"] for gap in run["coverage_gaps"])
    context = await client.get(
        f"{API}/validations/{run['validation_run_id']}/context",
        headers=auth("creator-a", world.a.tenant_id),
    )
    assert context.json()["code"] == "CONTEXT_NOT_AVAILABLE"


async def test_database_error_is_logged_with_traceback_and_sqlstate(
    client: httpx.AsyncClient,
    auth: AuthHeaders,
    world: World,
    sessions: Sessions,
    caplog: pytest.LogCaptureFixture,
) -> None:
    with caplog.at_level(logging.ERROR, logger="app.services.validations"):
        await _validate(client, auth, world, sessions, CLEAN_PROMPT, BrokenProvider())

    [record] = [r for r in caplog.records if r.getMessage().startswith("policy context failed")]
    # 22012: division_by_zero, from BrokenProvider's query.
    assert "sqlstate=22012" in record.getMessage()
    assert record.exc_info is not None


async def test_timeout_makes_a_clean_brief_need_review(
    client: httpx.AsyncClient, auth: AuthHeaders, world: World, sessions: Sessions
) -> None:
    run = await _validate(client, auth, world, sessions, CLEAN_PROMPT, StalledProvider())

    assert run["outcome"] == "NEEDS_REVIEW"
    assert any("(RETRIEVAL_TIMEOUT)" in gap["reason"] for gap in run["coverage_gaps"])


async def test_failed_rule_checks_stay_failed_without_context(
    client: httpx.AsyncClient, auth: AuthHeaders, world: World, sessions: Sessions
) -> None:
    # FAIL is already stricter than NEEDS_REVIEW, so a missing context doesn't soften it.
    run = await _validate(client, auth, world, sessions, RISKY_PROMPT, BrokenProvider())

    assert run["outcome"] == "FAIL"
    assert any("(RETRIEVAL_ERROR)" in gap["reason"] for gap in run["coverage_gaps"])
