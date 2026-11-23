"""Build the embedding index of one policy bundle (NRC-5).

Runs in the caller's transaction (the owner script's), tenant already bound. A build is
all-or-nothing: the caller rolls back on any error, so no half-built index is ever visible, and
retrieval trusts only a READY build whose checksum matches the bundle's approved nodes.
"""

import hashlib
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Final
from uuid import UUID, uuid4

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.errors import PolicyIndexError
from app.models import PolicyIndexBuild, PolicyNode, PolicyNodeEmbedding
from app.models.enums import IndexBuildState
from app.models.policy_graph import EMBEDDING_DIMENSIONS, SEARCHABLE_NODE_TYPES
from app.services.providers.embeddings import (
    EmbeddingProvider,
    EmbeddingPurpose,
    EmbeddingResponseError,
)

# Node texts sent per embed call. The adapter splits further to the model's own limit; this only
# bounds how many vectors are held in flight.
EMBED_BATCH_SIZE: Final = 48
# policy_index_builds.failure_code is VARCHAR(64).
_FAILURE_CODE_MAX = 64


@dataclass(frozen=True, slots=True)
class IndexBuildResult:
    build_id: UUID
    model_id: str
    node_count: int
    content_checksum: str
    # True when a READY build with the same content already existed, so nothing was written.
    skipped: bool


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def index_content_checksum(nodes: Iterable[tuple[UUID, str]]) -> str:
    """sha256 over the sorted (node_id, sha256(text)) pairs of (node_id, text) inputs.

    The value stored in policy_index_builds.content_checksum. Retrieval compares it with the
    bundle's current approved nodes before it trusts the index, so both sides use this function.
    """
    pairs = sorted((str(node_id), _sha256(text)) for node_id, text in nodes)
    return _sha256("\n".join(f"{node_id}:{digest}" for node_id, digest in pairs))


async def build_policy_index(
    db: AsyncSession, tenant_id: UUID, bundle_id: UUID, embedder: EmbeddingProvider
) -> IndexBuildResult:
    """Embed the bundle's approved nodes with `embedder` and mark the build READY.

    A READY build for the same bundle and model with the same content is returned as skipped; one
    with different content is an error, because a bundle's graph is immutable and its index is
    never rebuilt silently.

    Raises:
        PolicyIndexError: no approved nodes (details.reason NO_APPROVED_NODES), or the content
            changed since the READY build (details.reason INDEX_CONTENT_CHANGED).
        EmbeddingResponseError: the embedder returned the wrong number or size of vectors.
        ProviderError: any other subclass the embedder raises (config, unavailable).
    """
    model_id = embedder.model_id
    # One build per bundle and model at a time: a concurrent run waits here, then sees READY.
    await db.execute(
        select(func.pg_advisory_xact_lock(func.hashtextextended(f"{bundle_id}:{model_id}", 0)))
    )
    nodes = [
        (node_id, text)
        for node_id, text in await db.execute(
            select(PolicyNode.id, PolicyNode.text)
            .where(
                PolicyNode.tenant_id == tenant_id,
                PolicyNode.policy_bundle_id == bundle_id,
                PolicyNode.approved.is_(True),
                # Only what search can return; embedding rules and exceptions would be paid for
                # and never read.
                PolicyNode.node_type.in_(SEARCHABLE_NODE_TYPES),
            )
            .order_by(PolicyNode.id)
        )
    ]
    if not nodes:
        raise PolicyIndexError(
            "The bundle has no approved nodes to index", details={"reason": "NO_APPROVED_NODES"}
        )
    checksum = index_content_checksum(nodes)

    ready = await db.scalar(
        select(PolicyIndexBuild).where(
            PolicyIndexBuild.tenant_id == tenant_id,
            PolicyIndexBuild.policy_bundle_id == bundle_id,
            PolicyIndexBuild.model_id == model_id,
            PolicyIndexBuild.state == IndexBuildState.READY,
        )
    )
    if ready is not None:
        if ready.content_checksum != checksum:
            raise PolicyIndexError(
                "The bundle's nodes changed since its index was built",
                details={"reason": "INDEX_CONTENT_CHANGED", "build_id": str(ready.id)},
            )
        return IndexBuildResult(
            build_id=ready.id,
            model_id=model_id,
            node_count=len(nodes),
            content_checksum=checksum,
            skipped=True,
        )

    build = PolicyIndexBuild(
        id=uuid4(),
        tenant_id=tenant_id,
        policy_bundle_id=bundle_id,
        model_id=model_id,
        dimensions=EMBEDDING_DIMENSIONS,
        state=IndexBuildState.BUILDING,
    )
    db.add(build)
    await db.flush()
    for start in range(0, len(nodes), EMBED_BATCH_SIZE):
        batch = nodes[start : start + EMBED_BATCH_SIZE]
        vectors = await embedder.embed([text for _, text in batch], EmbeddingPurpose.DOCUMENT)
        _check_vectors(vectors, len(batch))
        db.add_all(
            PolicyNodeEmbedding(
                tenant_id=tenant_id,
                policy_bundle_id=bundle_id,
                node_id=node_id,
                model_id=model_id,
                embedding=vector,
                content_sha256=_sha256(text),
            )
            for (node_id, text), vector in zip(batch, vectors, strict=True)
        )
        await db.flush()
    build.state = IndexBuildState.READY
    build.node_count = len(nodes)
    build.content_checksum = checksum
    build.completed_at = datetime.now(tz=UTC)
    await db.flush()
    return IndexBuildResult(
        build_id=build.id,
        model_id=model_id,
        node_count=len(nodes),
        content_checksum=checksum,
        skipped=False,
    )


def _check_vectors(vectors: list[list[float]], expected_count: int) -> None:
    # The contract says one vector of EMBEDDING_DIMENSIONS per text; a bad embedder must not be
    # able to store anything else, whatever the adapter checked.
    if len(vectors) != expected_count or any(len(v) != EMBEDDING_DIMENSIONS for v in vectors):
        raise EmbeddingResponseError("The embedder did not return one full vector per text")


async def record_failed_build(
    db: AsyncSession, tenant_id: UUID, bundle_id: UUID, model_id: str, failure_code: str
) -> None:
    # For the caller to run in a fresh transaction after rolling back a failed build. The code is
    # a short fixed value such as the error class name, never a message (those can hold text).
    now = datetime.now(tz=UTC)
    db.add(
        PolicyIndexBuild(
            id=uuid4(),
            tenant_id=tenant_id,
            policy_bundle_id=bundle_id,
            model_id=model_id,
            dimensions=EMBEDDING_DIMENSIONS,
            state=IndexBuildState.FAILED,
            failure_code=failure_code[:_FAILURE_CODE_MAX],
            completed_at=now,
        )
    )
    await db.flush()


async def stale_ready_builds(db: AsyncSession, tenant_id: UUID, bundle_id: UUID) -> list[UUID]:
    """READY builds of the bundle whose checksum no longer matches its approved searchable nodes.

    Activation (NRC-4) refuses a bundle with one: retrieval would distrust that index anyway.
    """
    nodes = [
        (node_id, text)
        for node_id, text in await db.execute(
            select(PolicyNode.id, PolicyNode.text)
            .where(
                PolicyNode.tenant_id == tenant_id,
                PolicyNode.policy_bundle_id == bundle_id,
                PolicyNode.approved.is_(True),
                PolicyNode.node_type.in_(SEARCHABLE_NODE_TYPES),
            )
            .order_by(PolicyNode.id)
        )
    ]
    checksum = index_content_checksum(nodes)
    return list(
        await db.scalars(
            select(PolicyIndexBuild.id).where(
                PolicyIndexBuild.tenant_id == tenant_id,
                PolicyIndexBuild.policy_bundle_id == bundle_id,
                PolicyIndexBuild.state == IndexBuildState.READY,
                PolicyIndexBuild.content_checksum != checksum,
            )
        )
    )
