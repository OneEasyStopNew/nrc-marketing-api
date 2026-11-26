# Operator replay and settle of jobs that wait for an operator (NRC-6), and the audit trail.
#
# A job gets here UNKNOWN (the worker gave up: attempts exhausted) or CANCELLED with HELD credits
# (cancelled while the provider's outcome was unknown). Every scenario ends with
# assert_ledger_consistent. Images are SIMULATED.

import argparse
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from pathlib import Path
from typing import Any
from uuid import UUID, uuid4

import httpx
import pytest
from asgi_lifespan import LifespanManager
from cryptography.hazmat.primitives.asymmetric.rsa import RSAPrivateKey
from sqlalchemy import Row, text
from sqlalchemy.exc import DBAPIError
from sqlalchemy.ext.asyncio import AsyncConnection, create_async_engine
from sqlalchemy.pool import NullPool

from app.api.deps import get_signing_keys
from app.core.config import get_settings
from app.core.errors import JobOperationNotAllowedError, NotFoundError
from app.main import create_app
from app.models.enums import ExecutionState
from app.models.jobs import ReservationState, SpendState
from app.services import job_operations
from app.services.job_operations import replay, settle
from app.services.providers.base import ProviderOutcomeUnknownError
from app.workers.connector import run_once
from scripts import job_operations as script
from tests.conftest import (
    APP_ROLE,
    WORKER_ROLE,
    AuthHeaders,
    FakeSink,
    IntegrationEnv,
    ScriptedProvider,
    SeededTenant,
    Sessions,
    StaticKeys,
    World,
    _creative,
    _load,
    _post_job,
    _spend,
    as_role,
)
from tests.ledger import assert_ledger_consistent

ESTIMATE_MICROS = 94_000
ACTUAL_MICROS = 81_000
OPERATOR = "operator-1"
REASON = "Checked with the provider: no request under this key."
AUDIT_QUERY = (
    "SELECT action, operator, reason, job_state_before, job_state_after,"
    " reservation_state_before, reservation_state_after, outbox_event_id"
    " FROM job_operator_actions WHERE job_id = :j ORDER BY action"
)


@pytest.fixture
async def job_id(
    client: httpx.AsyncClient, auth: AuthHeaders, world: World, sessions: Sessions
) -> UUID:
    creative_id = await _creative(client, auth, world.a, sessions)
    response = await _post_job(client, auth, world, creative_id)
    assert response.status_code == 202
    return UUID(response.json()["job_id"])


def _priced() -> ScriptedProvider:
    # A price, so a wrong charge or release would show in provider_spend.
    provider = ScriptedProvider()
    provider.cost_usd = Decimal("0.094")
    return provider


async def _run(sessions: Sessions, provider: ScriptedProvider, sink: FakeSink) -> int:
    return await run_once(sessions, {"simulator": provider}, sink, get_settings())


async def _as_worker_in_tenant(connection: AsyncConnection, tenant_id: UUID) -> None:
    await as_role(connection, WORKER_ROLE)
    await connection.execute(
        text("SELECT set_config('app.tenant_id', :t, true)"), {"t": str(tenant_id)}
    )


async def _unknown_with_event(
    sessions: Sessions, provider: ScriptedProvider, world: World, job_id: UUID
) -> None:
    # The submit's answer was lost: UNKNOWN, and the worker will retry (an event is pending).
    provider.submit_errors = [ProviderOutcomeUnknownError("Simulated lost response.")]
    await _run(sessions, provider, FakeSink())
    job, reservation = await _load(sessions, world, job_id)
    assert (job.state, reservation) == (ExecutionState.UNKNOWN, ReservationState.HELD)


async def _worker_gave_up(
    sessions: Sessions,
    connection: AsyncConnection,
    provider: ScriptedProvider,
    world: World,
    job_id: UUID,
) -> None:
    # UNKNOWN with no unprocessed event: the attempt cap was reached and the worker stopped.
    await _unknown_with_event(sessions, provider, world, job_id)
    await _as_worker_in_tenant(connection, world.a.tenant_id)
    await connection.execute(
        text(
            "UPDATE outbox_events SET failed_attempts = :n, processed_at = NULL,"
            " available_at = clock_timestamp() - interval '1 second'"
        ),
        {"n": get_settings().generation_max_attempts},
    )
    await _run(sessions, provider, FakeSink())
    job, _ = await _load(sessions, world, job_id)
    assert job.state is ExecutionState.UNKNOWN
    assert await _pending_run_events(connection, world, job_id) == 0


async def _cancelled_with_held_credits(
    sessions: Sessions,
    connection: AsyncConnection,
    client: httpx.AsyncClient,
    auth: AuthHeaders,
    provider: ScriptedProvider,
    world: World,
    job_id: UUID,
) -> None:
    await _unknown_with_event(sessions, provider, world, job_id)
    response = await client.post(
        f"/v1/marketing/jobs/{job_id}/cancel", headers=auth("creator-a", world.a.tenant_id)
    )
    assert response.status_code == 202
    await _as_worker_in_tenant(connection, world.a.tenant_id)
    await connection.execute(
        text(
            "UPDATE outbox_events SET processed_at = NULL,"
            " available_at = clock_timestamp() - interval '1 second'"
        )
    )
    await _run(sessions, provider, FakeSink())
    job, reservation = await _load(sessions, world, job_id)
    assert (job.state, job.failure_code) == (ExecutionState.CANCELLED, "CANCELLED_OUTCOME_UNKNOWN")
    assert reservation is ReservationState.HELD


async def _pending_run_events(connection: AsyncConnection, world: World, job_id: UUID) -> int:
    await _as_worker_in_tenant(connection, world.a.tenant_id)
    count = await connection.scalar(
        text(
            "SELECT count(*) FROM outbox_events WHERE job_id = :j AND event_type = 'job.run'"
            " AND processed_at IS NULL"
        ),
        {"j": job_id},
    )
    return int(count or 0)


async def _audit(connection: AsyncConnection, tenant_id: UUID, job_id: UUID) -> list[Row[Any]]:
    await _as_worker_in_tenant(connection, tenant_id)
    result = await connection.execute(
        text(AUDIT_QUERY),
        {"j": job_id},
    )
    return list(result.all())


async def _replay(sessions: Sessions, tenant_id: UUID, job_id: UUID, reason: str = REASON) -> UUID:
    async with sessions() as db:
        return await replay(
            db, tenant_id, job_id, operator=OPERATOR, reason=reason, settings=get_settings()
        )


async def _settle(
    sessions: Sessions,
    tenant_id: UUID,
    job_id: UUID,
    *,
    billed: bool,
    actual_usd_micros: int | None = None,
) -> None:
    async with sessions() as db:
        await settle(
            db,
            tenant_id,
            job_id,
            billed=billed,
            actual_usd_micros=actual_usd_micros,
            operator=OPERATOR,
            reason=REASON,
        )


# ---- replay


async def test_replay_gives_the_stopped_job_a_new_run_event_and_reconciles_with_the_same_key(
    sessions: Sessions, job_id: UUID, world: World, connection: AsyncConnection
) -> None:
    spend_before = await _spend(connection)
    provider, sink = _priced(), FakeSink()
    await _worker_gave_up(sessions, connection, provider, world, job_id)
    await connection.execute(text("UPDATE jobs SET deadline_at = now() - interval '1 hour'"))
    stopped, _ = await _load(sessions, world, job_id)
    key = str(stopped.provider_idempotency_key)
    assert provider.submits == {key: 1}

    event_id = await _replay(sessions, world.a.tenant_id, job_id)

    job, reservation = await _load(sessions, world, job_id)
    assert (job.state, reservation) == (ExecutionState.UNKNOWN, ReservationState.HELD)
    longest_deadline = timedelta(seconds=get_settings().generation_deadline_seconds)
    assert job.deadline_at > datetime.now(UTC) + longest_deadline - timedelta(minutes=5)
    assert await _pending_run_events(connection, world, job_id) == 1
    [row] = await _audit(connection, world.a.tenant_id, job_id)
    assert tuple(row) == (
        "replay",
        OPERATOR,
        REASON,
        "UNKNOWN",
        "UNKNOWN",
        "HELD",
        "HELD",
        event_id,
    )

    await _run(sessions, provider, sink)

    job, reservation = await _load(sessions, world, job_id)
    # The same key again: the provider returns the original request, so there is one charge.
    assert provider.submits == {key: 2}
    assert job.state is ExecutionState.SUCCEEDED
    assert reservation is ReservationState.COMMITTED
    assert len(sink.stored) == 1
    await assert_ledger_consistent(connection, world.a.tenant_id, job_id, spend_before=spend_before)


async def test_replay_is_refused_while_the_worker_still_has_a_run_event(
    sessions: Sessions, job_id: UUID, world: World, connection: AsyncConnection
) -> None:
    spend_before = await _spend(connection)
    provider = _priced()
    await _unknown_with_event(sessions, provider, world, job_id)

    with pytest.raises(JobOperationNotAllowedError, match="not allowed") as refused:
        await _replay(sessions, world.a.tenant_id, job_id)

    assert refused.value.details == {"state": "UNKNOWN", "reason": "RUN_EVENT_PENDING"}
    assert await _audit(connection, world.a.tenant_id, job_id) == []
    await assert_ledger_consistent(connection, world.a.tenant_id, job_id, spend_before=spend_before)


async def test_replay_is_refused_for_a_job_that_is_not_unknown(
    sessions: Sessions, job_id: UUID, world: World, connection: AsyncConnection
) -> None:
    spend_before = await _spend(connection)

    with pytest.raises(JobOperationNotAllowedError, match="not allowed") as queued:
        await _replay(sessions, world.a.tenant_id, job_id)

    assert queued.value.details == {"state": "QUEUED", "reason": job_operations.NOT_UNKNOWN}
    await assert_ledger_consistent(connection, world.a.tenant_id, job_id, spend_before=spend_before)


async def test_replay_is_refused_for_a_finished_job(
    sessions: Sessions, job_id: UUID, world: World, connection: AsyncConnection
) -> None:
    spend_before = await _spend(connection)
    await _run(sessions, _priced(), FakeSink())

    with pytest.raises(JobOperationNotAllowedError, match="not allowed") as finished:
        await _replay(sessions, world.a.tenant_id, job_id)

    assert finished.value.details == {"state": "SUCCEEDED", "reason": job_operations.JOB_FINISHED}
    assert await _audit(connection, world.a.tenant_id, job_id) == []
    await assert_ledger_consistent(connection, world.a.tenant_id, job_id, spend_before=spend_before)


async def test_replay_and_settle_do_not_see_another_tenants_job(
    sessions: Sessions, job_id: UUID, world: World, connection: AsyncConnection
) -> None:
    spend_before = await _spend(connection)
    provider = _priced()
    await _worker_gave_up(sessions, connection, provider, world, job_id)

    with pytest.raises(NotFoundError, match="Job not found"):
        await _replay(sessions, world.b.tenant_id, job_id)
    with pytest.raises(NotFoundError, match="Job not found"):
        await _settle(sessions, world.b.tenant_id, job_id, billed=False)
    with pytest.raises(NotFoundError, match="Job not found"):
        await _replay(sessions, world.a.tenant_id, uuid4())

    job, reservation = await _load(sessions, world, job_id)
    assert (job.state, reservation) == (ExecutionState.UNKNOWN, ReservationState.HELD)
    assert await _audit(connection, world.a.tenant_id, job_id) == []
    await assert_ledger_consistent(connection, world.a.tenant_id, job_id, spend_before=spend_before)


# ---- settle


@pytest.mark.parametrize(
    ("billed", "actual", "code", "reservation_after", "spend_delta"),
    [
        (True, ACTUAL_MICROS, "OPERATOR_SETTLED_BILLED", ReservationState.COMMITTED, ACTUAL_MICROS),
        (True, None, "OPERATOR_SETTLED_BILLED", ReservationState.COMMITTED, ESTIMATE_MICROS),
        (False, None, "OPERATOR_SETTLED_NOT_BILLED", ReservationState.RELEASED, 0),
    ],
)
async def test_settle_ends_an_unknown_job_as_failed_and_settles_credits_and_spend_once(
    sessions: Sessions,
    job_id: UUID,
    world: World,
    connection: AsyncConnection,
    billed: bool,
    actual: int | None,
    code: str,
    reservation_after: ReservationState,
    spend_delta: int,
) -> None:
    spend_before = await _spend(connection)
    await _worker_gave_up(sessions, connection, _priced(), world, job_id)

    await _settle(sessions, world.a.tenant_id, job_id, billed=billed, actual_usd_micros=actual)

    job, reservation = await _load(sessions, world, job_id)
    assert (job.state, job.failure_code) == (ExecutionState.FAILED, code)
    assert job.failure_message
    assert job.completed_at is not None
    assert job.provider_cost_actual_usd_micros == actual
    assert reservation is reservation_after
    assert job.provider_spend_state is (SpendState.COMMITTED if billed else SpendState.RELEASED)
    reserved, committed = spend_before
    assert await _spend(connection) == (reserved, committed + spend_delta)
    [row] = await _audit(connection, world.a.tenant_id, job_id)
    assert tuple(row) == (
        "settle_billed" if billed else "settle_not_billed",
        OPERATOR,
        REASON,
        "UNKNOWN",
        "FAILED",
        "HELD",
        reservation_after.value,
        None,
    )
    await assert_ledger_consistent(connection, world.a.tenant_id, job_id, spend_before=spend_before)


@pytest.mark.parametrize(
    ("billed", "actual", "reservation_after", "spend_delta"),
    [
        (True, ACTUAL_MICROS, ReservationState.COMMITTED, ACTUAL_MICROS),
        (False, None, ReservationState.RELEASED, 0),
    ],
)
async def test_settle_keeps_a_cancelled_job_cancelled_and_settles_its_held_credits(
    client: httpx.AsyncClient,
    auth: AuthHeaders,
    sessions: Sessions,
    job_id: UUID,
    world: World,
    connection: AsyncConnection,
    billed: bool,
    actual: int | None,
    reservation_after: ReservationState,
    spend_delta: int,
) -> None:
    spend_before = await _spend(connection)
    await _cancelled_with_held_credits(sessions, connection, client, auth, _priced(), world, job_id)

    await _settle(sessions, world.a.tenant_id, job_id, billed=billed, actual_usd_micros=actual)

    job, reservation = await _load(sessions, world, job_id)
    # A terminal state never changes; the code still says it was cancelled with the outcome unknown.
    assert (job.state, job.failure_code) == (ExecutionState.CANCELLED, "CANCELLED_OUTCOME_UNKNOWN")
    assert reservation is reservation_after
    assert job.provider_cost_actual_usd_micros == actual
    reserved, committed = spend_before
    assert await _spend(connection) == (reserved, committed + spend_delta)
    [row] = await _audit(connection, world.a.tenant_id, job_id)
    assert (row.job_state_before, row.job_state_after) == ("CANCELLED", "CANCELLED")
    assert (row.reservation_state_before, row.reservation_state_after) == (
        "HELD",
        reservation_after.value,
    )
    await assert_ledger_consistent(connection, world.a.tenant_id, job_id, spend_before=spend_before)


async def test_a_second_settle_is_refused_and_changes_nothing(
    sessions: Sessions, job_id: UUID, world: World, connection: AsyncConnection
) -> None:
    spend_before = await _spend(connection)
    await _worker_gave_up(sessions, connection, _priced(), world, job_id)
    await _settle(sessions, world.a.tenant_id, job_id, billed=True, actual_usd_micros=ACTUAL_MICROS)
    after_first = await _spend(connection)

    with pytest.raises(JobOperationNotAllowedError, match="not allowed") as second:
        await _settle(sessions, world.a.tenant_id, job_id, billed=False)

    assert second.value.details == {"state": "FAILED", "reason": job_operations.JOB_FINISHED}
    assert await _spend(connection) == after_first
    assert len(await _audit(connection, world.a.tenant_id, job_id)) == 1
    await assert_ledger_consistent(connection, world.a.tenant_id, job_id, spend_before=spend_before)


async def test_a_second_settle_of_a_cancelled_job_finds_nothing_held(
    client: httpx.AsyncClient,
    auth: AuthHeaders,
    sessions: Sessions,
    job_id: UUID,
    world: World,
    connection: AsyncConnection,
) -> None:
    spend_before = await _spend(connection)
    await _cancelled_with_held_credits(sessions, connection, client, auth, _priced(), world, job_id)
    await _settle(sessions, world.a.tenant_id, job_id, billed=False)

    with pytest.raises(JobOperationNotAllowedError, match="not allowed") as second:
        await _settle(sessions, world.a.tenant_id, job_id, billed=True)

    assert second.value.details == {"state": "CANCELLED", "reason": job_operations.NOTHING_HELD}
    assert len(await _audit(connection, world.a.tenant_id, job_id)) == 1
    await assert_ledger_consistent(connection, world.a.tenant_id, job_id, spend_before=spend_before)


async def test_settle_is_refused_while_the_worker_still_has_a_run_event(
    sessions: Sessions, job_id: UUID, world: World, connection: AsyncConnection
) -> None:
    spend_before = await _spend(connection)
    await _unknown_with_event(sessions, _priced(), world, job_id)

    with pytest.raises(JobOperationNotAllowedError, match="not allowed") as refused:
        await _settle(sessions, world.a.tenant_id, job_id, billed=False)

    assert refused.value.details == {"state": "UNKNOWN", "reason": "RUN_EVENT_PENDING"}
    await assert_ledger_consistent(connection, world.a.tenant_id, job_id, spend_before=spend_before)


async def test_settle_is_refused_for_a_job_the_worker_is_still_running(
    sessions: Sessions, job_id: UUID, world: World, connection: AsyncConnection
) -> None:
    spend_before = await _spend(connection)

    with pytest.raises(JobOperationNotAllowedError, match="not allowed") as queued:
        await _settle(sessions, world.a.tenant_id, job_id, billed=False)

    assert queued.value.details == {
        "state": "QUEUED",
        "reason": job_operations.NOT_AWAITING_OPERATOR,
    }
    await assert_ledger_consistent(connection, world.a.tenant_id, job_id, spend_before=spend_before)


@pytest.mark.parametrize(("billed", "actual"), [(True, -1), (False, 0), (False, ACTUAL_MICROS)])
async def test_settle_refuses_an_actual_cost_that_is_negative_or_not_billed(
    sessions: Sessions,
    job_id: UUID,
    world: World,
    connection: AsyncConnection,
    billed: bool,
    actual: int,
) -> None:
    spend_before = await _spend(connection)
    await _worker_gave_up(sessions, connection, _priced(), world, job_id)

    with pytest.raises(JobOperationNotAllowedError, match="not allowed") as refused:
        await _settle(sessions, world.a.tenant_id, job_id, billed=billed, actual_usd_micros=actual)

    assert refused.value.details == {"state": None, "reason": job_operations.INVALID_ACTUAL_COST}
    job, reservation = await _load(sessions, world, job_id)
    assert (job.state, reservation) == (ExecutionState.UNKNOWN, ReservationState.HELD)
    await assert_ledger_consistent(connection, world.a.tenant_id, job_id, spend_before=spend_before)


# ---- audit table


async def test_one_tenant_cannot_read_or_write_another_tenants_audit_rows(
    sessions: Sessions, job_id: UUID, world: World, connection: AsyncConnection
) -> None:
    spend_before = await _spend(connection)
    await _worker_gave_up(sessions, connection, _priced(), world, job_id)
    await _replay(sessions, world.a.tenant_id, job_id)
    assert len(await _audit(connection, world.a.tenant_id, job_id)) == 1

    assert await _audit(connection, world.b.tenant_id, job_id) == []
    insert = text(
        "INSERT INTO job_operator_actions (id, tenant_id, job_id, action, operator,"
        " reason, job_state_before, job_state_after, reservation_state_before,"
        " reservation_state_after) VALUES (:id, :t, :j, 'replay', 'x', 'x',"
        " 'UNKNOWN', 'UNKNOWN', 'HELD', 'HELD')"
    )
    savepoint = await connection.begin_nested()
    await _as_worker_in_tenant(connection, world.b.tenant_id)
    with pytest.raises(DBAPIError, match="row-level security"):
        await connection.execute(insert, {"id": uuid4(), "t": world.a.tenant_id, "j": job_id})
    await savepoint.rollback()
    await assert_ledger_consistent(connection, world.a.tenant_id, job_id, spend_before=spend_before)


@pytest.mark.parametrize(
    ("role", "expected"),
    [(APP_ROLE, {"SELECT"}), (WORKER_ROLE, {"SELECT", "INSERT"})],
)
async def test_audit_table_grants_only_what_each_role_uses(
    connection: AsyncConnection, role: str, expected: set[str]
) -> None:
    rows = await connection.execute(
        text(
            "SELECT privilege_type FROM information_schema.role_table_grants"
            " WHERE table_name = 'job_operator_actions' AND grantee = :role"
        ),
        {"role": role},
    )

    assert {row.privilege_type for row in rows} == expected


async def test_api_role_cannot_write_an_audit_row(
    job_id: UUID, world: World, connection: AsyncConnection
) -> None:
    # No API code path writes audit rows, so the API role must not be able to forge one.
    insert = text(
        "INSERT INTO job_operator_actions (id, tenant_id, job_id, action, operator,"
        " reason, job_state_before, job_state_after, reservation_state_before,"
        " reservation_state_after) VALUES (:id, :t, :j, 'replay', 'x', 'x',"
        " 'UNKNOWN', 'UNKNOWN', 'HELD', 'HELD')"
    )
    savepoint = await connection.begin_nested()
    await as_role(connection, APP_ROLE)
    await connection.execute(
        text("SELECT set_config('app.tenant_id', :t, true)"), {"t": str(world.a.tenant_id)}
    )
    with pytest.raises(DBAPIError, match="permission denied for table job_operator_actions"):
        await connection.execute(insert, {"id": uuid4(), "t": world.a.tenant_id, "j": job_id})
    await savepoint.rollback()

    assert await _audit(connection, world.a.tenant_id, job_id) == []


@pytest.mark.parametrize("role", [APP_ROLE, WORKER_ROLE])
@pytest.mark.parametrize(
    "statement",
    ["UPDATE job_operator_actions SET reason = 'changed'", "DELETE FROM job_operator_actions"],
)
async def test_neither_role_can_change_or_delete_an_audit_row(
    sessions: Sessions,
    job_id: UUID,
    world: World,
    connection: AsyncConnection,
    role: str,
    statement: str,
) -> None:
    await _worker_gave_up(sessions, connection, _priced(), world, job_id)
    await _replay(sessions, world.a.tenant_id, job_id)

    savepoint = await connection.begin_nested()
    await as_role(connection, role)
    await connection.execute(
        text("SELECT set_config('app.tenant_id', :t, true)"), {"t": str(world.a.tenant_id)}
    )
    with pytest.raises(DBAPIError, match="permission denied"):
        await connection.execute(text(statement))
    await savepoint.rollback()

    assert len(await _audit(connection, world.a.tenant_id, job_id)) == 1


# ---- the script: arguments and exit codes


def _ids() -> list[str]:
    return ["--tenant", str(uuid4()), "--job", str(uuid4())]


def _who() -> list[str]:
    return ["--operator", OPERATOR, "--reason", REASON]


def test_script_parses_valid_arguments() -> None:
    tenant, job = uuid4(), uuid4()

    replayed = script.parse_args(["replay", "--tenant", str(tenant), "--job", str(job), *_who()])
    settled = script.parse_args(
        [
            *("settle", "--tenant", str(tenant), "--job", str(job)),
            *("--billed", "--actual-usd", "0.081"),
            *_who(),
        ]
    )

    assert (replayed.command, replayed.tenant, replayed.job) == ("replay", tenant, job)
    assert (settled.billed, settled.not_billed, settled.actual_usd) == (
        True,
        False,
        Decimal("0.081"),
    )


@pytest.mark.parametrize(
    ("argv", "message"),
    [
        (["replay", "--tenant", "nope", "--job", str(uuid4()), *_who()], "must be a UUID"),
        (["replay", *_ids(), "--operator", OPERATOR, "--reason", "   "], "must not be empty"),
        (["replay", *_ids(), "--operator", OPERATOR, "--reason", "x" * 501], "at most 500"),
        (["replay", *_ids(), "--operator", "o" * 129, "--reason", REASON], "at most 128"),
        (["replay", *_ids(), "--reason", REASON], "--operator"),
        (["settle", *_ids(), *_who()], "one of the arguments --billed --not-billed"),
        (["settle", *_ids(), "--billed", "--not-billed", *_who()], "not allowed with"),
        (["settle", *_ids(), "--not-billed", "--actual-usd", "0.01", *_who()], "only goes with"),
        (["settle", *_ids(), "--billed", "--actual-usd", "-1", *_who()], "must be between"),
        (["settle", *_ids(), "--billed", "--actual-usd", "NaN", *_who()], "must be between"),
        (["settle", *_ids(), "--billed", "--actual-usd", "1e9", *_who()], "must be between"),
        (["settle", *_ids(), "--billed", "--actual-usd", "abc", *_who()], "decimal number"),
        (["frobnicate", *_ids(), *_who()], "invalid choice"),
    ],
)
def test_script_rejects_invalid_arguments_with_exit_code_2(
    argv: list[str], message: str, capsys: pytest.CaptureFixture[str]
) -> None:
    with pytest.raises(SystemExit, match=str(script.EXIT_REFUSED)):
        script.parse_args(argv)

    assert message in capsys.readouterr().err


def test_script_main_stops_with_exit_code_2_without_the_worker_database_url(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    # No .env in this directory, and no variable.
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("WORKER_DATABASE_URL", raising=False)

    code = script.main(["replay", *_ids(), *_who()])

    assert code == script.EXIT_REFUSED
    assert "WORKER_DATABASE_URL" in capsys.readouterr().out


async def _accept_job(tenant: SeededTenant, signing_key: RSAPrivateKey, worker: Sessions) -> UUID:
    app = create_app()
    app.dependency_overrides[get_signing_keys] = lambda: StaticKeys(signing_key)
    auth = AuthHeaders(signing_key)
    async with (
        LifespanManager(app),
        httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as http,
    ):
        creative_id = await _creative(http, auth, tenant, worker)
        response = await _post_job(http, auth, World(tenant, tenant), creative_id)
    return UUID(response.json()["job_id"])


async def test_script_replays_then_settles_a_stopped_job_as_the_worker_role(
    committed_tenant: SeededTenant,
    signing_key: RSAPrivateKey,
    test_env: IntegrationEnv,
    capsys: pytest.CaptureFixture[str],
    committed_worker_sessions: Sessions,
) -> None:
    job = await _accept_job(committed_tenant, signing_key, committed_worker_sessions)
    owner = create_async_engine(
        test_env.test_migration_database_url.get_secret_value(), poolclass=NullPool
    )
    async with owner.begin() as conn:
        # The worker gave up: UNKNOWN, credits HELD, its run event processed.
        await conn.execute(text("UPDATE jobs SET state = 'UNKNOWN' WHERE id = :j"), {"j": job})
        await conn.execute(
            text("UPDATE credit_reservations SET state = 'HELD' WHERE job_id = :j"), {"j": job}
        )
        await conn.execute(
            text("UPDATE outbox_events SET processed_at = now() WHERE job_id = :j"), {"j": job}
        )
    url = test_env.test_worker_database_url.get_secret_value()
    ids = ["--tenant", str(committed_tenant.tenant_id), "--job", str(job)]
    settings = get_settings()

    replayed = await script.run(script.parse_args(["replay", *ids, *_who()]), url, settings)
    refused = await script.run(script.parse_args(["replay", *ids, *_who()]), url, settings)
    # The new run event is processed again, as if the worker had run it and stopped.
    async with owner.begin() as conn:
        await conn.execute(
            text("UPDATE outbox_events SET processed_at = now() WHERE job_id = :j"), {"j": job}
        )
    settled = await script.run(
        script.parse_args(["settle", *ids, "--not-billed", *_who()]), url, settings
    )

    async with owner.connect() as conn:
        state = await conn.scalar(text("SELECT state FROM jobs WHERE id = :j"), {"j": job})
        actions = (
            (
                await conn.execute(
                    text(
                        "SELECT action FROM job_operator_actions WHERE job_id = :j"
                        " ORDER BY created_at"
                    ),
                    {"j": job},
                )
            )
            .scalars()
            .all()
        )
    await owner.dispose()
    printed = capsys.readouterr().out
    assert (replayed, refused, settled) == (script.EXIT_OK, script.EXIT_REFUSED, script.EXIT_OK)
    assert f"replayed tenant={committed_tenant.tenant_id} job={job} event=" in printed
    assert (
        f"refused code=JOB_OPERATION_NOT_ALLOWED tenant={committed_tenant.tenant_id} job={job}"
        " state=UNKNOWN reason=RUN_EVENT_PENDING"
    ) in printed
    assert f"settled tenant={committed_tenant.tenant_id} job={job} billed=False" in printed
    assert REASON not in printed
    assert state == "FAILED"
    assert actions == ["replay", "settle_not_billed"]


async def test_script_reports_an_unknown_job_with_exit_code_2(
    committed_tenant: SeededTenant,
    test_env: IntegrationEnv,
    capsys: pytest.CaptureFixture[str],
) -> None:
    args: argparse.Namespace = script.parse_args(
        [
            *("settle", "--tenant", str(committed_tenant.tenant_id), "--job", str(uuid4())),
            "--billed",
            *_who(),
        ]
    )

    code = await script.run(
        args, test_env.test_worker_database_url.get_secret_value(), get_settings()
    )

    assert code == script.EXIT_REFUSED
    assert "refused code=NOT_FOUND" in capsys.readouterr().out
