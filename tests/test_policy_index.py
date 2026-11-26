"""Policy index build (NRC-5): checksum, idempotency, immutability, all-or-nothing, tenancy."""

import hashlib
import subprocess
import sys
from collections.abc import Sequence
from uuid import UUID, uuid4

import pytest
from sqlalchemy import func, select, update
from sqlalchemy.ext.asyncio import AsyncConnection

from app.core.errors import PolicyIndexError
from app.models import PolicyIndexBuild, PolicyNode, PolicyNodeEmbedding
from app.models.enums import IndexBuildState, ReviewState
from app.models.policy_graph import EMBEDDING_DIMENSIONS, SEARCHABLE_NODE_TYPES
from app.services.policy_index import (
    EMBED_BATCH_SIZE,
    build_policy_index,
    index_content_checksum,
    record_failed_build,
)
from app.services.providers.embeddings import (
    SIMULATED_MODEL_ID,
    EmbeddingPurpose,
    EmbeddingResponseError,
    SimulatedEmbeddingProvider,
)
from scripts.import_policy_graph import PolicyGraphImport, import_policy_graph, load_synthetic_graph
from tests.conftest import (
    ROOT,
    World,
    add_policy_bundle,
    as_tenant_owner,
    bind_test_tenant,
    count_rows,
    owner_session,
)


@pytest.fixture
def graph() -> PolicyGraphImport:
    # These tests re-import and index the real synthetic graph into tenant A's bundle, so the
    # bundle must hold that graph, not conftest's code-checked default.
    return load_synthetic_graph()


# Searchable nodes in the synthetic rulebook (7 clauses, 2 disclosures, 1 interpretation), all
# approved. Its 9 rules and 2 exceptions are never embedded.
NODE_COUNT = 10


class _ShortVectors:
    # An embedder that breaks the contract: vectors of 3 numbers instead of 1024.
    model_id = "test:short-vectors"

    async def embed(self, texts: Sequence[str], purpose: EmbeddingPurpose) -> list[list[float]]:
        return [[0.1, 0.2, 0.3] for _ in texts]


class _FewerVectors:
    model_id = "test:fewer-vectors"

    async def embed(self, texts: Sequence[str], purpose: EmbeddingPurpose) -> list[list[float]]:
        return [[0.0] * EMBEDDING_DIMENSIONS for _ in texts[:-1]]


class _RecordingEmbedder(SimulatedEmbeddingProvider):
    def __init__(self) -> None:
        self.calls: list[tuple[int, EmbeddingPurpose]] = []

    async def embed(self, texts: Sequence[str], purpose: EmbeddingPurpose) -> list[list[float]]:
        self.calls.append((len(texts), purpose))
        return await super().embed(texts, purpose)


def _bundle_of(world: World) -> UUID:
    assert world.a.policy_bundle_id is not None
    return world.a.policy_bundle_id


async def _import_graph(connection: AsyncConnection, world: World) -> tuple[UUID, UUID]:
    tenant_id, bundle_id = world.a.tenant_id, _bundle_of(world)
    async with owner_session(connection, tenant_id) as db:
        await import_policy_graph(db, tenant_id, bundle_id, load_synthetic_graph())
    return tenant_id, bundle_id


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


# The checksum


def test_checksum_ignores_order_and_changes_with_id_or_text() -> None:
    a, b = uuid4(), uuid4()
    base = index_content_checksum([(a, "first"), (b, "second")])

    assert base == index_content_checksum([(b, "second"), (a, "first")])
    assert len(base) == 64
    assert base != index_content_checksum([(a, "first"), (b, "second!")])
    assert base != index_content_checksum([(a, "second"), (b, "first")])
    assert base != index_content_checksum([(a, "first")])


# Build


async def test_build_creates_one_ready_build_with_an_embedding_per_node(
    connection: AsyncConnection, world: World
) -> None:
    tenant_id, bundle_id = await _import_graph(connection, world)
    embedder = _RecordingEmbedder()
    async with owner_session(connection, tenant_id) as db:
        result = await build_policy_index(db, tenant_id, bundle_id, embedder)

        build = (await db.scalars(select(PolicyIndexBuild))).one()
        nodes = list(
            (
                await db.execute(
                    select(PolicyNode.id, PolicyNode.text).where(
                        PolicyNode.node_type.in_(SEARCHABLE_NODE_TYPES)
                    )
                )
            ).all()
        )
        embeddings = list(await db.scalars(select(PolicyNodeEmbedding)))

    assert result.skipped is False
    assert (result.build_id, result.model_id) == (build.id, SIMULATED_MODEL_ID)
    assert (result.node_count, result.content_checksum) == (
        NODE_COUNT,
        index_content_checksum(nodes),
    )
    assert build.state is IndexBuildState.READY
    assert (build.node_count, build.dimensions) == (NODE_COUNT, EMBEDDING_DIMENSIONS)
    assert build.content_checksum == result.content_checksum
    assert build.completed_at is not None
    assert build.failure_code is None
    assert len(embeddings) == NODE_COUNT
    assert {e.node_id for e in embeddings} == {node_id for node_id, _ in nodes}
    assert all(len(e.embedding) == EMBEDDING_DIMENSIONS for e in embeddings)
    assert all(e.model_id == SIMULATED_MODEL_ID for e in embeddings)
    hashes = {node_id: _sha(text) for node_id, text in nodes}
    assert all(e.content_sha256 == hashes[e.node_id] for e in embeddings)
    # Stored texts are embedded as documents, in batches no larger than the limit.
    assert sum(size for size, _ in embedder.calls) == NODE_COUNT
    assert all(size <= EMBED_BATCH_SIZE for size, _ in embedder.calls)
    assert {purpose for _, purpose in embedder.calls} == {EmbeddingPurpose.DOCUMENT}


async def test_unapproved_nodes_are_not_indexed(connection: AsyncConnection, world: World) -> None:
    tenant_id, bundle_id = await _import_graph(connection, world)
    async with owner_session(connection, tenant_id) as db:
        await db.execute(
            update(PolicyNode)
            .where(PolicyNode.policy_bundle_id == bundle_id, PolicyNode.node_key == "DISC-01-ES")
            .values(approved=False, review_state=ReviewState.PROPOSED)
        )
        result = await build_policy_index(db, tenant_id, bundle_id, SimulatedEmbeddingProvider())

    assert result.node_count == NODE_COUNT - 1
    assert await count_rows(connection, PolicyNodeEmbedding) == NODE_COUNT - 1


async def test_a_second_build_of_the_same_content_is_skipped(
    connection: AsyncConnection, world: World
) -> None:
    tenant_id, bundle_id = await _import_graph(connection, world)
    async with owner_session(connection, tenant_id) as db:
        first = await build_policy_index(db, tenant_id, bundle_id, SimulatedEmbeddingProvider())
    embedder = _RecordingEmbedder()
    async with owner_session(connection, tenant_id) as db:
        second = await build_policy_index(db, tenant_id, bundle_id, embedder)

    assert second.skipped is True
    assert (second.build_id, second.content_checksum) == (first.build_id, first.content_checksum)
    assert second.node_count == NODE_COUNT
    assert embedder.calls == []
    assert await count_rows(connection, PolicyIndexBuild) == 1
    assert await count_rows(connection, PolicyNodeEmbedding) == NODE_COUNT


async def test_changed_node_text_is_rejected_and_the_index_is_kept(
    connection: AsyncConnection, world: World
) -> None:
    tenant_id, bundle_id = await _import_graph(connection, world)
    async with owner_session(connection, tenant_id) as db:
        await build_policy_index(db, tenant_id, bundle_id, SimulatedEmbeddingProvider())
    async with owner_session(connection, tenant_id) as db:
        await db.execute(
            update(PolicyNode)
            .where(PolicyNode.policy_bundle_id == bundle_id, PolicyNode.node_key == "CL-FDA-1")
            .values(text="[SYNTHETIC] A different requirement.")
        )

    with pytest.raises(PolicyIndexError, match="nodes changed since its index was built") as caught:
        async with owner_session(connection, tenant_id) as db:
            await build_policy_index(db, tenant_id, bundle_id, SimulatedEmbeddingProvider())

    assert caught.value.details is not None
    assert caught.value.details["reason"] == "INDEX_CONTENT_CHANGED"
    assert await count_rows(connection, PolicyIndexBuild) == 1
    assert await count_rows(connection, PolicyNodeEmbedding) == NODE_COUNT


async def test_another_embedding_model_builds_its_own_index(
    connection: AsyncConnection, world: World
) -> None:
    class _OtherModel(SimulatedEmbeddingProvider):
        model_id = "test:other-model"

    tenant_id, bundle_id = await _import_graph(connection, world)
    async with owner_session(connection, tenant_id) as db:
        await build_policy_index(db, tenant_id, bundle_id, SimulatedEmbeddingProvider())
        other = await build_policy_index(db, tenant_id, bundle_id, _OtherModel())

    assert other.skipped is False
    assert await count_rows(connection, PolicyIndexBuild) == 2
    assert await count_rows(connection, PolicyNodeEmbedding) == 2 * NODE_COUNT


@pytest.mark.parametrize(
    ("embedder", "match"),
    [
        (_ShortVectors(), "one full vector per text"),
        (_FewerVectors(), "one full vector per text"),
    ],
)
async def test_a_bad_embedder_writes_nothing(
    connection: AsyncConnection,
    world: World,
    embedder: _ShortVectors | _FewerVectors,
    match: str,
) -> None:
    tenant_id, bundle_id = await _import_graph(connection, world)

    # The caller's transaction is rolled back when the build raises, as the owner script does.
    with pytest.raises(EmbeddingResponseError, match=match):
        async with owner_session(connection, tenant_id) as db:
            await build_policy_index(db, tenant_id, bundle_id, embedder)

    assert await count_rows(connection, PolicyIndexBuild) == 0
    assert await count_rows(connection, PolicyNodeEmbedding) == 0


async def test_a_bundle_without_approved_nodes_is_rejected(
    connection: AsyncConnection, world: World
) -> None:
    tenant_id = world.a.tenant_id
    async with as_tenant_owner(connection, tenant_id):
        empty_bundle = await add_policy_bundle(connection, tenant_id, version=2)

    with pytest.raises(PolicyIndexError, match="no approved nodes") as caught:
        async with owner_session(connection, tenant_id) as db:
            await build_policy_index(db, tenant_id, empty_bundle, SimulatedEmbeddingProvider())

    assert caught.value.details is not None
    assert caught.value.details["reason"] == "NO_APPROVED_NODES"
    assert await count_rows(connection, PolicyIndexBuild) == 0


async def test_another_tenants_bundle_cannot_be_indexed(
    connection: AsyncConnection, world: World
) -> None:
    # Bound to tenant A, tenant B's nodes are invisible (row-level security): nothing to index.
    async with as_tenant_owner(connection, world.b.tenant_id):
        bundle_b = await add_policy_bundle(connection, world.b.tenant_id, version=1)

    with pytest.raises(PolicyIndexError, match="no approved nodes"):
        async with owner_session(connection, world.a.tenant_id) as db:
            await build_policy_index(db, world.a.tenant_id, bundle_b, SimulatedEmbeddingProvider())


# Failed builds


async def test_record_failed_build_writes_a_failed_row_without_embeddings(
    connection: AsyncConnection, world: World
) -> None:
    tenant_id, bundle_id = await _import_graph(connection, world)
    async with owner_session(connection, tenant_id) as db:
        await record_failed_build(
            db, tenant_id, bundle_id, SIMULATED_MODEL_ID, "ProviderUnavailableError"
        )
        build = (await db.scalars(select(PolicyIndexBuild))).one()

    assert build.state is IndexBuildState.FAILED
    assert build.failure_code == "ProviderUnavailableError"
    assert (build.node_count, build.content_checksum) == (None, None)
    assert await count_rows(connection, PolicyNodeEmbedding) == 0


async def test_a_failed_build_does_not_block_a_later_successful_one(
    connection: AsyncConnection, world: World
) -> None:
    tenant_id, bundle_id = await _import_graph(connection, world)
    async with owner_session(connection, tenant_id) as db:
        await record_failed_build(db, tenant_id, bundle_id, SIMULATED_MODEL_ID, "X" * 100)
        result = await build_policy_index(db, tenant_id, bundle_id, SimulatedEmbeddingProvider())
        states = set(await db.scalars(select(PolicyIndexBuild.state)))

    assert result.skipped is False
    assert states == {IndexBuildState.READY, IndexBuildState.FAILED}
    # The long code was cut to the column's size instead of failing the insert.


# Tenancy


async def test_other_tenant_sees_none_of_the_index(
    connection: AsyncConnection, world: World
) -> None:
    tenant_id, bundle_id = await _import_graph(connection, world)
    async with owner_session(connection, tenant_id) as db:
        await build_policy_index(db, tenant_id, bundle_id, SimulatedEmbeddingProvider())
        await record_failed_build(db, tenant_id, bundle_id, "test:other", "SomeError")

    await bind_test_tenant(connection, world.b.tenant_id)
    assert await count_rows(connection, PolicyIndexBuild) == 0
    assert await count_rows(connection, PolicyNodeEmbedding) == 0
    await bind_test_tenant(connection, tenant_id)
    assert await count_rows(connection, PolicyIndexBuild) == 2
    assert (
        await connection.scalar(select(func.count()).select_from(PolicyNodeEmbedding)) == NODE_COUNT
    )


# The owner script


def test_build_script_refuses_embedding_mode_off() -> None:
    # The test environment runs with EMBEDDING_MODE=off. It must stop before opening a database.
    result = subprocess.run(  # noqa: S603  fixed interpreter and module, ids are random
        [
            sys.executable,
            "-m",
            "scripts.build_policy_index",
            "--tenant-id",
            str(uuid4()),
            "--bundle-id",
            str(uuid4()),
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode != 0
    assert "EMBEDDING_MODE=off builds no index" in result.stderr
