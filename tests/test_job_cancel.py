# Job cancellation API (NRC-6): records a durable request; the worker side settles it.

from collections.abc import Iterator
from typing import Any
from uuid import UUID, uuid4

import httpx
import pytest
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncConnection

from app.core.config import get_settings
from app.db.session import bind_tenant
from app.models.enums import ExecutionState
from app.models.jobs import ReservationState
from app.services.jobs import transition
from tests.conftest import (
    AuthHeaders,
    Sessions,
    World,
    _creative,
    _load,
    _post_job,
    app_session,
)


def _error(response: httpx.Response) -> dict[str, Any]:
    body: dict[str, Any] = response.json()
    return body


async def _bind(connection: AsyncConnection, world: World) -> None:
    await connection.execute(
        text("SELECT set_config('app.tenant_id', :t, true)"), {"t": str(world.a.tenant_id)}
    )


async def _events(connection: AsyncConnection, world: World, *, pending: bool) -> list[Any]:
    await _bind(connection, world)
    query = (
        "SELECT available_at <= clock_timestamp() AS due, lease_owner, attempts"
        " FROM outbox_events WHERE event_type = 'job.run'"
    )
    if pending:
        query += " AND processed_at IS NULL"
    return list(await connection.execute(text(query)))


async def _force_state(
    connection: AsyncConnection, world: World, job_id: UUID, state: ExecutionState
) -> None:
    # What the worker would have done, via the same guarded update.
    async with app_session(connection) as db:
        await bind_tenant(db, world.a.tenant_id)
        moved = await transition(db, job_id, allowed_from=set(ExecutionState), to=state)
        await db.commit()
    assert moved, "test setup: the job must be able to move"


async def _drain_events(connection: AsyncConnection, world: World) -> None:
    await _bind(connection, world)
    await connection.execute(text("UPDATE outbox_events SET processed_at = now()"))


@pytest.fixture
async def job_id(client: httpx.AsyncClient, auth: AuthHeaders, world: World) -> UUID:
    creative_id = await _creative(client, auth, world.a)
    response = await _post_job(client, auth, world, creative_id)
    assert response.status_code == 202
    return UUID(response.json()["job_id"])


async def _cancel(
    client: httpx.AsyncClient,
    auth: AuthHeaders,
    tenant_id: UUID,
    job_id: UUID | str,
    *,
    sub: str = "creator-a",
) -> httpx.Response:
    return await client.post(f"/v1/marketing/jobs/{job_id}/cancel", headers=auth(sub, tenant_id))


async def test_cancel_queued_job_records_request_and_wakes_the_worker(
    client: httpx.AsyncClient,
    auth: AuthHeaders,
    world: World,
    sessions: Sessions,
    connection: AsyncConnection,
    job_id: UUID,
) -> None:
    # Pushed into the future, as after a backoff, to prove the cancel makes it due.
    await _bind(connection, world)
    await connection.execute(
        text("UPDATE outbox_events SET available_at = now() + interval '1 hour'")
    )
    assert not (await _events(connection, world, pending=True))[0].due

    response = await _cancel(client, auth, world.a.tenant_id, job_id)

    assert response.status_code == 202
    assert response.json()["state"] == "CANCEL_REQUESTED"
    assert response.json()["job_id"] == str(job_id)
    job, _ = await _load(sessions, world, job_id)
    assert job.state is ExecutionState.CANCEL_REQUESTED
    events = await _events(connection, world, pending=False)
    assert len(events) == 1
    assert events[0].due
    assert events[0].lease_owner is None


async def test_repeat_cancel_is_idempotent_and_adds_no_event(
    client: httpx.AsyncClient,
    auth: AuthHeaders,
    world: World,
    connection: AsyncConnection,
    job_id: UUID,
) -> None:
    first = await _cancel(client, auth, world.a.tenant_id, job_id)

    again = await _cancel(client, auth, world.a.tenant_id, job_id)

    assert again.status_code == 202
    assert again.json()["state"] == "CANCEL_REQUESTED"
    assert again.json()["updated_at"] == first.json()["updated_at"]
    assert len(await _events(connection, world, pending=False)) == 1


async def test_cancelled_job_is_returned_unchanged(
    client: httpx.AsyncClient,
    auth: AuthHeaders,
    world: World,
    connection: AsyncConnection,
    job_id: UUID,
) -> None:
    await _force_state(connection, world, job_id, ExecutionState.CANCELLED)

    response = await _cancel(client, auth, world.a.tenant_id, job_id)

    assert response.status_code == 202
    assert response.json()["state"] == "CANCELLED"


async def test_cancel_job_waiting_for_the_provider(
    client: httpx.AsyncClient,
    auth: AuthHeaders,
    world: World,
    sessions: Sessions,
    connection: AsyncConnection,
    job_id: UUID,
) -> None:
    await _force_state(connection, world, job_id, ExecutionState.WAITING_PROVIDER)

    response = await _cancel(client, auth, world.a.tenant_id, job_id)

    assert response.status_code == 202
    assert response.json()["state"] == "CANCEL_REQUESTED"
    job, _ = await _load(sessions, world, job_id)
    assert job.state is ExecutionState.CANCEL_REQUESTED


async def test_cancel_unknown_job_adds_a_run_event_for_the_worker(
    client: httpx.AsyncClient,
    auth: AuthHeaders,
    world: World,
    connection: AsyncConnection,
    job_id: UUID,
) -> None:
    # The worker gave up on it: UNKNOWN and no unprocessed event left.
    await _force_state(connection, world, job_id, ExecutionState.UNKNOWN)
    await _drain_events(connection, world)
    assert await _events(connection, world, pending=True) == []

    response = await _cancel(client, auth, world.a.tenant_id, job_id)

    assert response.status_code == 202
    assert response.json()["state"] == "CANCEL_REQUESTED"
    pending = await _events(connection, world, pending=True)
    assert len(pending) == 1
    assert pending[0].due
    assert pending[0].attempts == 0
    assert len(await _events(connection, world, pending=False)) == 2


@pytest.mark.parametrize("state", [ExecutionState.SUCCEEDED, ExecutionState.FAILED])
async def test_finished_job_cannot_be_cancelled(
    client: httpx.AsyncClient,
    auth: AuthHeaders,
    world: World,
    connection: AsyncConnection,
    job_id: UUID,
    state: ExecutionState,
) -> None:
    await _force_state(connection, world, job_id, state)

    response = await _cancel(client, auth, world.a.tenant_id, job_id)

    assert response.status_code == 409
    body = _error(response)
    assert (body["code"], body["retryable"]) == ("JOB_NOT_CANCELLABLE", False)
    assert body["details"] == {"state": state.value}
    assert "already finished" in body["message"]


async def test_other_tenant_gets_404(
    client: httpx.AsyncClient,
    auth: AuthHeaders,
    world: World,
    sessions: Sessions,
    job_id: UUID,
) -> None:
    response = await _cancel(client, auth, world.b.tenant_id, job_id, sub="creator-b")

    assert response.status_code == 404
    assert _error(response)["code"] == "NOT_FOUND"
    job, _ = await _load(sessions, world, job_id)
    assert job.state is ExecutionState.QUEUED


async def test_job_in_a_study_the_caller_cannot_see_gets_404(
    client: httpx.AsyncClient,
    auth: AuthHeaders,
    world: World,
    connection: AsyncConnection,
    job_id: UUID,
) -> None:
    # The membership loses its study grant after the job was started.
    await connection.execute(text("RESET ROLE"))
    await connection.execute(
        text("DELETE FROM membership_studies WHERE tenant_id = :t"), {"t": world.a.tenant_id}
    )
    await connection.execute(text("SET LOCAL ROLE marketing_app"))

    response = await _cancel(client, auth, world.a.tenant_id, job_id)

    assert response.status_code == 404
    assert _error(response)["message"] == "Job not found."


async def test_reviewer_cannot_cancel(
    client: httpx.AsyncClient, auth: AuthHeaders, world: World, job_id: UUID
) -> None:
    response = await _cancel(client, auth, world.a.tenant_id, job_id, sub="reviewer-a")

    assert response.status_code == 403


async def test_unknown_job_id_gets_404(
    client: httpx.AsyncClient, auth: AuthHeaders, world: World
) -> None:
    response = await _cancel(client, auth, world.a.tenant_id, uuid4())

    assert response.status_code == 404


async def test_malformed_job_id_gets_422(
    client: httpx.AsyncClient, auth: AuthHeaders, world: World
) -> None:
    response = await _cancel(client, auth, world.a.tenant_id, "not-a-uuid")

    assert response.status_code == 422


@pytest.fixture
def one_active_job_allowed() -> Iterator[None]:
    with pytest.MonkeyPatch.context() as patch:
        patch.setenv("GENERATION_MAX_ACTIVE_JOBS_PER_TENANT", "1")
        get_settings.cache_clear()
        yield
    get_settings.cache_clear()


@pytest.mark.usefixtures("one_active_job_allowed")
async def test_cancel_requested_job_still_takes_its_active_slot(
    client: httpx.AsyncClient, auth: AuthHeaders, world: World
) -> None:
    creative_id = await _creative(client, auth, world.a)
    first = await _post_job(client, auth, world, creative_id)
    assert first.status_code == 202
    cancel = await _cancel(client, auth, world.a.tenant_id, first.json()["job_id"])
    assert cancel.json()["state"] == "CANCEL_REQUESTED"

    response = await _post_job(client, auth, world, creative_id)

    assert response.status_code == 429
    assert _error(response)["code"] == "GENERATION_LIMIT_REACHED"
    assert _error(response)["details"] == {"reason": "ACTIVE_JOB_LIMIT"}


async def test_cancel_leaves_the_credit_reservation_to_the_worker(
    client: httpx.AsyncClient,
    auth: AuthHeaders,
    world: World,
    sessions: Sessions,
    job_id: UUID,
) -> None:
    await _cancel(client, auth, world.a.tenant_id, job_id)

    _, reservation = await _load(sessions, world, job_id)

    assert reservation is ReservationState.RESERVED
