"""Policy build queue (NRC-4): the worker turns an approved, queued bundle into a READY one.

Real Postgres (RLS, the lease function, the worker role's grants). Bundles are written as the
owner in the APPROVED state with one approved clause node and queued the way approval queues them
(the approval itself is tested in test_policy_lifecycle). Embedders are small fakes or the
SIMULATED one. All data is synthetic.
"""

from collections.abc import Sequence
from typing import Any
from uuid import UUID

import pytest
from sqlalchemy import Row, func, insert, select, update
from sqlalchemy.exc import DBAPIError
from sqlalchemy.ext.asyncio import AsyncConnection

from app.auth.access import Role, TenantContext, capabilities_for
from app.auth.tokens import Principal
from app.core.config import Settings, get_settings
from app.core.errors import PolicyIndexError
from app.db.session import bind_tenant
from app.models import PolicyBundle, PolicyNode
from app.models.enums import (
    BundleLifecycleState,
    Command,
    PolicyAuditAction,
    PolicyBuildFailure,
    PolicyLanguage,
    PolicyNodeType,
    ReviewState,
)
from app.models.policy_graph import SEARCHABLE_NODE_TYPES
from app.schemas.policy_admin import TransitionRequest
from app.services import policy_bundles as bundles
from app.services.policy_index import index_content_checksum
from app.services.providers.base import (
    ProviderError,
    ProviderRejectedError,
    ProviderUnavailableError,
)
from app.services.providers.embeddings import (
    SIMULATED_MODEL_ID,
    EmbeddingProvider,
    EmbeddingPurpose,
    EmbeddingResponseError,
    SimulatedEmbeddingProvider,
)
from app.workers import policy as policy_worker
from app.workers.common import WorkerConfigError
from app.workers.policy import BUILD_ACTOR, build_embedder, run_once
from tests.conftest import (
    WORKER_ROLE,
    Sessions,
    World,
    add_policy_bundle,
    app_session,
    as_role,
    as_tenant_owner,
    owner_sql,
)

LEXICAL_ONLY = {"index": "lexical_only"}
APPROVER = "policy-approver-a"


@pytest.fixture
def settings() -> Settings:
    return get_settings().model_copy(
        update={
            "worker_backoff_base_seconds": 30.0,
            "worker_backoff_max_seconds": 60.0,
            "policy_worker_max_attempts": 3,
        }
    )


class NoStorage:
    # The worker also serves the extraction queue; these tests leave it nothing to read.
    bucket = "nrc-test-policy"

    async def read(self, key: str, *, version_id: str | None = None) -> bytes:
        raise AssertionError("no source is queued in these tests")


class RaisingEmbedder:
    # An embedder whose model fails with the given provider error.
    model_id = "test:raising-embedder"

    def __init__(self, error_type: type[ProviderError]) -> None:
        self.error_type = error_type
        self.calls = 0

    async def embed(self, texts: Sequence[str], purpose: EmbeddingPurpose) -> list[list[float]]:
        self.calls += 1
        raise self.error_type("[SYNTHETIC] embedding failure")


async def queued_bundle(
    connection: AsyncConnection, tenant_id: UUID, version: int = 2, *, due: bool = True
) -> UUID:
    # APPROVED with one approved clause node (what the index covers); due as approval leaves it.
    async with as_tenant_owner(connection, tenant_id):
        bundle_id = await add_policy_bundle(
            connection, tenant_id, version, BundleLifecycleState.APPROVED
        )
        await connection.execute(
            insert(PolicyNode).values(
                tenant_id=tenant_id,
                policy_bundle_id=bundle_id,
                node_key="CL-1",
                node_type=PolicyNodeType.CLAUSE,
                text="[SYNTHETIC] Every post must say that this is a research study.",
                language=PolicyLanguage.EN,
                review_state=ReviewState.APPROVED,
                approved=True,
                reviewed_by="synthetic-fixture",
                reviewed_at=func.now(),
            )
        )
    if due:
        await make_due(connection, tenant_id, bundle_id)
    return bundle_id


async def work(
    sessions: Sessions,
    settings: Settings,
    embedder: EmbeddingProvider | None = None,
    owner: str = "w1",
) -> int:
    return await run_once(sessions, NoStorage(), settings, embedder=embedder, owner=owner)


async def bundle_row(connection: AsyncConnection, tenant_id: UUID, bundle_id: UUID) -> Row[Any]:
    rows = await owner_sql(
        connection,
        tenant_id,
        "SELECT lifecycle_state, available_at, last_error_code, leased_until, lease_owner,"
        " attempts, failed_attempts, updated_at, revision,"
        " coalesce(available_at > clock_timestamp(), false) AS deferred"
        " FROM policy_bundles WHERE id = :id",
        {"id": bundle_id},
    )
    assert len(rows) == 1
    return rows[0]


async def index_builds(
    connection: AsyncConnection, tenant_id: UUID, bundle_id: UUID
) -> list[Row[Any]]:
    return await owner_sql(
        connection,
        tenant_id,
        "SELECT id, model_id, state, node_count, content_checksum, failure_code"
        " FROM policy_index_builds WHERE policy_bundle_id = :id ORDER BY state",
        {"id": bundle_id},
    )


async def embedding_count(connection: AsyncConnection, tenant_id: UUID, bundle_id: UUID) -> int:
    rows = await owner_sql(
        connection,
        tenant_id,
        "SELECT count(*) AS n FROM policy_node_embeddings WHERE policy_bundle_id = :id",
        {"id": bundle_id},
    )
    return int(rows[0].n)


async def audit(
    connection: AsyncConnection, tenant_id: UUID, bundle_id: UUID, action: PolicyAuditAction
) -> list[Row[Any]]:
    return await owner_sql(
        connection,
        tenant_id,
        "SELECT action, from_state, to_state, actor, details, tenant_id FROM policy_audit_events"
        " WHERE policy_bundle_id = :id AND action = :action",
        {"id": bundle_id, "action": action.value},
    )


async def make_due(connection: AsyncConnection, tenant_id: UUID, bundle_id: UUID) -> None:
    await owner_sql(
        connection,
        tenant_id,
        "UPDATE policy_bundles SET available_at = clock_timestamp() - interval '1 second'"
        " WHERE id = :id",
        {"id": bundle_id},
    )


async def expire_lease(connection: AsyncConnection, tenant_id: UUID, bundle_id: UUID) -> None:
    await owner_sql(
        connection,
        tenant_id,
        "UPDATE policy_bundles SET leased_until = clock_timestamp() - interval '1 second'"
        " WHERE id = :id",
        {"id": bundle_id},
    )


async def test_the_simulator_builds_the_index_and_the_bundle_becomes_ready(
    connection: AsyncConnection,
    world: World,
    sessions: Sessions,
    settings: Settings,
) -> None:
    tenant = world.a.tenant_id
    bundle_id = await queued_bundle(connection, world.a.tenant_id)

    assert await work(sessions, settings, SimulatedEmbeddingProvider()) == 1

    ready = await bundle_row(connection, tenant, bundle_id)
    assert ready.lifecycle_state == "READY"
    assert (ready.available_at, ready.leased_until, ready.lease_owner) == (None, None, None)
    assert ready.last_error_code is None
    async with as_tenant_owner(connection, tenant):
        nodes = [
            (node_id, text)
            for node_id, text in await connection.execute(
                select(PolicyNode.id, PolicyNode.text).where(
                    PolicyNode.policy_bundle_id == bundle_id,
                    PolicyNode.approved.is_(True),
                    PolicyNode.node_type.in_(SEARCHABLE_NODE_TYPES),
                )
            )
        ]
    assert nodes
    (build,) = await index_builds(connection, tenant, bundle_id)
    assert (build.state, build.model_id, build.node_count) == ("READY", SIMULATED_MODEL_ID, 1)
    assert build.content_checksum == index_content_checksum(nodes)
    assert await embedding_count(connection, tenant, bundle_id) == len(nodes)
    (event,) = await audit(connection, tenant, bundle_id, PolicyAuditAction.BUILD_READY)
    assert (event.actor, event.from_state, event.to_state) == (BUILD_ACTOR, "BUILDING", "READY")
    assert event.tenant_id == tenant
    assert event.details == {
        "build_id": str(build.id),
        "model_id": SIMULATED_MODEL_ID,
        "node_count": len(nodes),
    }
    # Done: nothing left to lease.
    assert await work(sessions, settings, SimulatedEmbeddingProvider()) == 0


async def test_with_embeddings_off_the_bundle_is_ready_with_full_text_only(
    connection: AsyncConnection,
    world: World,
    sessions: Sessions,
    settings: Settings,
) -> None:
    tenant = world.a.tenant_id
    bundle_id = await queued_bundle(connection, world.a.tenant_id)

    await work(sessions, settings, None)

    assert (await bundle_row(connection, tenant, bundle_id)).lifecycle_state == "READY"
    assert await index_builds(connection, tenant, bundle_id) == []
    assert await embedding_count(connection, tenant, bundle_id) == 0
    (event,) = await audit(connection, tenant, bundle_id, PolicyAuditAction.BUILD_READY)
    assert event.details == LEXICAL_ONLY


# ---- Failures


async def test_a_permanent_failure_sends_the_bundle_back_and_is_not_leased_again(
    connection: AsyncConnection,
    world: World,
    sessions: Sessions,
    settings: Settings,
) -> None:
    tenant = world.a.tenant_id
    bundle_id = await queued_bundle(connection, world.a.tenant_id)
    embedder = RaisingEmbedder(EmbeddingResponseError)

    await work(sessions, settings, embedder)

    failed = await bundle_row(connection, tenant, bundle_id)
    assert (failed.lifecycle_state, failed.available_at) == ("APPROVED", None)
    assert failed.last_error_code == PolicyBuildFailure.EMBEDDING_FAILED
    assert (failed.leased_until, failed.lease_owner) == (None, None)
    (build,) = await index_builds(connection, tenant, bundle_id)
    assert (build.state, build.model_id, build.failure_code) == (
        "FAILED",
        embedder.model_id,
        EmbeddingResponseError.code,
    )
    assert await embedding_count(connection, tenant, bundle_id) == 0
    (event,) = await audit(connection, tenant, bundle_id, PolicyAuditAction.BUILD_FAILED)
    assert (event.actor, event.from_state, event.to_state) == (BUILD_ACTOR, "BUILDING", "APPROVED")
    assert event.details == {"error_code": PolicyBuildFailure.EMBEDDING_FAILED}
    assert await audit(connection, tenant, bundle_id, PolicyAuditAction.BUILD_READY) == []
    # Only BUILD queues it again.
    assert await work(sessions, settings, embedder) == 0
    assert embedder.calls == 1


async def test_build_queues_a_failed_bundle_again(
    connection: AsyncConnection, world: World, sessions: Sessions, settings: Settings
) -> None:
    tenant = world.a.tenant_id
    bundle_id = await queued_bundle(connection, tenant)
    await work(sessions, settings, RaisingEmbedder(EmbeddingResponseError))
    failed = await bundle_row(connection, tenant, bundle_id)
    assert (failed.lifecycle_state, failed.available_at) == ("APPROVED", None)
    approver = TenantContext(
        tenant_id=tenant,
        principal=Principal(sub=APPROVER, username=APPROVER, scopes=frozenset()),
        roles=frozenset({Role.POLICY_APPROVER}),
        capabilities=capabilities_for({Role.POLICY_APPROVER}),
        study_ids=frozenset(),
    )

    async with app_session(connection) as db:
        await bind_tenant(db, tenant)
        await bundles.transition(
            db,
            approver,
            bundle_id,
            failed.revision,
            Command.BUILD,
            TransitionRequest(),
            (PolicyLanguage.EN,),
        )

    queued = await bundle_row(connection, tenant, bundle_id)
    assert (queued.lifecycle_state, queued.last_error_code) == ("APPROVED", None)
    assert queued.available_at is not None
    assert (queued.attempts, queued.failed_attempts) == (0, 0)
    # Queued again, so the worker takes it and this time the embedder works.
    assert await work(sessions, settings, SimulatedEmbeddingProvider()) == 1
    assert (await bundle_row(connection, tenant, bundle_id)).lifecycle_state == "READY"


async def test_a_rejected_request_is_a_permanent_failure(
    connection: AsyncConnection,
    world: World,
    sessions: Sessions,
    settings: Settings,
) -> None:
    bundle_id = await queued_bundle(connection, world.a.tenant_id)
    embedder = RaisingEmbedder(ProviderRejectedError)

    await work(sessions, settings, embedder)

    failed = await bundle_row(connection, world.a.tenant_id, bundle_id)
    assert (failed.lifecycle_state, failed.available_at, failed.last_error_code) == (
        "APPROVED",
        None,
        PolicyBuildFailure.EMBEDDING_FAILED,
    )
    assert embedder.calls == 1


async def test_an_index_precondition_failure_records_no_build(
    connection: AsyncConnection,
    world: World,
    sessions: Sessions,
    settings: Settings,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def nothing_to_index(*args: object) -> None:
        raise PolicyIndexError("[SYNTHETIC] nothing", details={"reason": "NO_APPROVED_NODES"})

    monkeypatch.setattr(policy_worker, "build_policy_index", nothing_to_index)
    tenant = world.a.tenant_id
    bundle_id = await queued_bundle(connection, world.a.tenant_id)

    await work(sessions, settings, SimulatedEmbeddingProvider())

    failed = await bundle_row(connection, tenant, bundle_id)
    assert (failed.lifecycle_state, failed.available_at, failed.last_error_code) == (
        "APPROVED",
        None,
        "NO_APPROVED_NODES",
    )
    assert await index_builds(connection, tenant, bundle_id) == []
    assert len(await audit(connection, tenant, bundle_id, PolicyAuditAction.BUILD_FAILED)) == 1


async def test_an_outage_is_retried_with_backoff_then_fails_after_the_attempt_cap(
    connection: AsyncConnection,
    world: World,
    sessions: Sessions,
    settings: Settings,
) -> None:
    tenant = world.a.tenant_id
    bundle_id = await queued_bundle(connection, world.a.tenant_id)
    embedder = RaisingEmbedder(ProviderUnavailableError)

    await work(sessions, settings, embedder)

    waiting = await bundle_row(connection, tenant, bundle_id)
    assert (waiting.lifecycle_state, waiting.last_error_code) == (
        "APPROVED",
        PolicyBuildFailure.EMBEDDING_UNAVAILABLE,
    )
    assert waiting.deferred is True
    assert (waiting.leased_until, waiting.lease_owner, waiting.attempts) == (None, None, 1)
    assert await index_builds(connection, tenant, bundle_id) == []
    # Not due yet: nothing to lease until the backoff passes.
    assert await work(sessions, settings, embedder) == 0

    for attempt in (2, 3):
        await make_due(connection, tenant, bundle_id)
        assert await work(sessions, settings, embedder) == 1
        assert (await bundle_row(connection, tenant, bundle_id)).attempts == attempt

    failed = await bundle_row(connection, tenant, bundle_id)
    assert (failed.lifecycle_state, failed.available_at, failed.last_error_code) == (
        "APPROVED",
        None,
        PolicyBuildFailure.EMBEDDING_UNAVAILABLE,
    )
    (build,) = await index_builds(connection, tenant, bundle_id)
    assert (build.state, build.failure_code) == ("FAILED", ProviderUnavailableError.code)
    assert len(await audit(connection, tenant, bundle_id, PolicyAuditAction.BUILD_FAILED)) == 1
    assert embedder.calls == 3
    assert await work(sessions, settings, embedder) == 0


async def test_a_bundle_waits_and_then_becomes_ready_when_the_model_returns(
    connection: AsyncConnection,
    world: World,
    sessions: Sessions,
    settings: Settings,
) -> None:
    tenant = world.a.tenant_id
    bundle_id = await queued_bundle(connection, world.a.tenant_id)
    await work(sessions, settings, RaisingEmbedder(ProviderUnavailableError))

    await make_due(connection, tenant, bundle_id)
    await work(sessions, settings, SimulatedEmbeddingProvider())

    ready = await bundle_row(connection, tenant, bundle_id)
    assert (ready.lifecycle_state, ready.last_error_code, ready.available_at) == (
        "READY",
        None,
        None,
    )
    (build,) = await index_builds(connection, tenant, bundle_id)
    assert build.state == "READY"


# ---- Interrupted work, leases and the attempt cap


class Boom(Exception):
    # Stands in for a crash between the index build and the commit.
    pass


async def test_a_crash_after_the_index_is_built_leaves_no_ready_build_and_the_active_bundle_alone(
    connection: AsyncConnection,
    world: World,
    sessions: Sessions,
    settings: Settings,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    tenant = world.a.tenant_id
    bundle_id = await queued_bundle(connection, world.a.tenant_id)
    active_before = await owner_sql(
        connection,
        tenant,
        "SELECT id, lifecycle_state, updated_at, revision, rules FROM policy_bundles"
        " WHERE lifecycle_state = 'ACTIVE'",
    )
    assert len(active_before) == 1

    async def crash(*args: object) -> None:
        raise Boom

    # The index rows and READY are written by now; only the audit row and the commit are left.
    monkeypatch.setattr(policy_worker, "_audit", crash)
    await work(sessions, settings, SimulatedEmbeddingProvider())

    crashed = await bundle_row(connection, tenant, bundle_id)
    assert crashed.lifecycle_state == "BUILDING"
    assert crashed.lease_owner == "w1"
    assert await index_builds(connection, tenant, bundle_id) == []
    assert await embedding_count(connection, tenant, bundle_id) == 0
    assert await audit(connection, tenant, bundle_id, PolicyAuditAction.BUILD_READY) == []
    assert f"policy work failed item={bundle_id}" in caplog.text
    active_after = await owner_sql(
        connection,
        tenant,
        "SELECT id, lifecycle_state, updated_at, revision, rules FROM policy_bundles"
        " WHERE lifecycle_state = 'ACTIVE'",
    )
    assert [tuple(row) for row in active_after] == [tuple(row) for row in active_before]
    assert await index_builds(connection, tenant, active_before[0].id) == []

    # While the lease holds nobody else takes it; once it expires the next worker redoes it all.
    assert await work(sessions, settings, SimulatedEmbeddingProvider(), owner="w2") == 0
    monkeypatch.undo()
    await expire_lease(connection, tenant, bundle_id)
    assert await work(sessions, settings, SimulatedEmbeddingProvider(), owner="w2") == 1

    done = await bundle_row(connection, tenant, bundle_id)
    assert (done.lifecycle_state, done.failed_attempts, done.attempts) == ("READY", 1, 2)
    (build,) = await index_builds(connection, tenant, bundle_id)
    assert build.state == "READY"


async def test_a_bundle_that_keeps_crashing_is_failed_without_being_built(
    connection: AsyncConnection,
    world: World,
    sessions: Sessions,
    settings: Settings,
) -> None:
    tenant = world.a.tenant_id
    bundle_id = await queued_bundle(connection, world.a.tenant_id)
    embedder = RaisingEmbedder(ProviderUnavailableError)
    await owner_sql(
        connection,
        tenant,
        "UPDATE policy_bundles SET lifecycle_state = 'BUILDING', lease_owner = 'dead-worker',"
        " attempts = 3, failed_attempts = 2, leased_until = clock_timestamp() - interval '1 second'"
        " WHERE id = :id",
        {"id": bundle_id},
    )

    await work(sessions, settings, embedder)

    failed = await bundle_row(connection, tenant, bundle_id)
    # The lease counted the expired one: 2 + 1 reaches the cap of 3.
    assert (
        failed.lifecycle_state,
        failed.available_at,
        failed.last_error_code,
        failed.failed_attempts,
    ) == ("APPROVED", None, PolicyBuildFailure.BUILD_ABANDONED, 3)
    assert embedder.calls == 0
    assert await index_builds(connection, tenant, bundle_id) == []
    assert len(await audit(connection, tenant, bundle_id, PolicyAuditAction.BUILD_FAILED)) == 1


async def test_a_worker_that_lost_its_lease_writes_nothing(
    connection: AsyncConnection,
    world: World,
    sessions: Sessions,
    settings: Settings,
    caplog: pytest.LogCaptureFixture,
) -> None:
    tenant = world.a.tenant_id
    bundle_id = await queued_bundle(connection, world.a.tenant_id)
    async with as_tenant_owner(connection, world.b.tenant_id):
        b_bundle = await add_policy_bundle(
            connection, world.b.tenant_id, 1, BundleLifecycleState.APPROVED
        )
    b_before = await bundle_row(connection, world.b.tenant_id, b_bundle)
    simulator = SimulatedEmbeddingProvider()

    class LosingEmbedder:
        model_id = simulator.model_id

        async def embed(self, texts: Sequence[str], purpose: EmbeddingPurpose) -> list[list[float]]:
            # The lease runs out while the vectors are computed; no other worker is involved.
            await expire_lease(connection, tenant, bundle_id)
            await as_role(connection, WORKER_ROLE)
            return await simulator.embed(texts, purpose)

    await work(sessions, settings, LosingEmbedder())

    lost = await bundle_row(connection, tenant, bundle_id)
    assert lost.lifecycle_state == "BUILDING"
    assert await index_builds(connection, tenant, bundle_id) == []
    assert await embedding_count(connection, tenant, bundle_id) == 0
    assert await audit(connection, tenant, bundle_id, PolicyAuditAction.BUILD_READY) == []
    assert f"policy lease lost item={bundle_id}" in caplog.text
    assert tuple(await bundle_row(connection, world.b.tenant_id, b_bundle)) == tuple(b_before)
    assert await index_builds(connection, world.b.tenant_id, b_bundle) == []


# ---- The worker role and tenants


async def test_each_tenants_bundle_is_built_under_its_own_tenant(
    connection: AsyncConnection,
    world: World,
    sessions: Sessions,
    settings: Settings,
) -> None:
    bundle_id = await queued_bundle(connection, world.a.tenant_id)
    async with as_tenant_owner(connection, world.b.tenant_id):
        b_bundle = await add_policy_bundle(
            connection, world.b.tenant_id, 1, BundleLifecycleState.APPROVED
        )
    await owner_sql(
        connection,
        world.b.tenant_id,
        "UPDATE policy_bundles SET available_at = clock_timestamp() - interval '1 second'"
        " WHERE id = :id",
        {"id": b_bundle},
    )

    assert await work(sessions, settings, None) == 2

    for tenant, built in ((world.a.tenant_id, bundle_id), (world.b.tenant_id, b_bundle)):
        assert (await bundle_row(connection, tenant, built)).lifecycle_state == "READY"
        (event,) = await audit(connection, tenant, built, PolicyAuditAction.BUILD_READY)
        assert event.tenant_id == tenant


async def test_a_worker_session_cannot_touch_another_tenants_bundle(
    connection: AsyncConnection, world: World, sessions: Sessions
) -> None:
    async with as_tenant_owner(connection, world.b.tenant_id):
        b_bundle = await add_policy_bundle(
            connection, world.b.tenant_id, 1, BundleLifecycleState.APPROVED
        )

    async with sessions() as db:
        await bind_tenant(db, world.a.tenant_id)
        assert await db.get(PolicyBundle, b_bundle) is None
        changed = await db.execute(
            update(PolicyBundle)
            .where(PolicyBundle.id == b_bundle)
            .values(lifecycle_state=BundleLifecycleState.READY, updated_at=func.now())
        )
        assert changed.rowcount == 0  # type: ignore[attr-defined]
        with pytest.raises(DBAPIError, match="row-level security"):
            await policy_worker._audit(
                db,
                world.b.tenant_id,
                b_bundle,
                PolicyAuditAction.BUILD_READY,
                BundleLifecycleState.BUILDING,
                BundleLifecycleState.READY,
                {},
            )

    assert (await bundle_row(connection, world.b.tenant_id, b_bundle)).lifecycle_state == "APPROVED"
    assert await audit(connection, world.b.tenant_id, b_bundle, PolicyAuditAction.BUILD_READY) == []


# ---- Start-up


def test_the_worker_builds_the_embedder_from_settings(settings: Settings) -> None:
    assert build_embedder(settings.model_copy(update={"embedding_mode": "off"})) is None
    simulated = build_embedder(settings.model_copy(update={"embedding_mode": "simulator"}))
    assert isinstance(simulated, SimulatedEmbeddingProvider)


def test_live_embeddings_without_a_model_refuse_to_start(settings: Settings) -> None:
    live = settings.model_copy(
        update={"embedding_mode": "live", "bedrock_embedding_model_id": None}
    )

    with pytest.raises(WorkerConfigError, match="BEDROCK_EMBEDDING_MODEL_ID"):
        build_embedder(live)
