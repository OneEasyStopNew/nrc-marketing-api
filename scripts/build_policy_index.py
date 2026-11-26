"""Build the embedding index of one policy bundle's approved nodes.

Usage (runs as the database owner, MIGRATION_DATABASE_URL):
    uv run python -m scripts.build_policy_index --tenant-id <uuid> --bundle-id <uuid>

Uses the embedding model of EMBEDDING_MODE (simulator, or live with BEDROCK_EMBEDDING_MODEL_ID);
refuses EMBEDDING_MODE=off. A bundle's index is built once: rerunning does nothing, and a bundle
whose nodes changed since its READY build is rejected. On failure nothing is kept except a FAILED
build row (a short error code), and the exit status is non-zero. Prints ids and counts only.
"""

import argparse
import asyncio
import json
from uuid import UUID

from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.pool import NullPool

from app.core.config import OwnerDatabaseSettings
from app.core.errors import PolicyIndexError
from app.db.session import TenantSession, bind_tenant
from app.services.policy_index import IndexBuildResult, build_policy_index, record_failed_build
from app.services.providers.embeddings import EmbeddingProvider
from scripts.embedding_setup import load_embedder


async def _run(tenant_id: UUID, bundle_id: UUID, embedder: EmbeddingProvider) -> IndexBuildResult:
    engine = create_async_engine(
        OwnerDatabaseSettings().migration_database_url.get_secret_value(), poolclass=NullPool
    )
    try:
        try:
            async with AsyncSession(engine, sync_session_class=TenantSession) as db:
                await bind_tenant(db, tenant_id)
                result = await build_policy_index(db, tenant_id, bundle_id, embedder)
                await db.commit()
            return result
        except PolicyIndexError:
            # A precondition (nothing to index, content changed), not a failed build attempt.
            raise
        except Exception as error:
            # The build's transaction is rolled back by now; the FAILED row goes in a new one.
            async with AsyncSession(engine, sync_session_class=TenantSession) as db:
                await bind_tenant(db, tenant_id)
                await record_failed_build(
                    db, tenant_id, bundle_id, embedder.model_id, type(error).__name__
                )
                await db.commit()
            raise
    finally:
        await engine.dispose()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0] if __doc__ else None)
    parser.add_argument("--tenant-id", type=UUID, required=True)
    parser.add_argument("--bundle-id", type=UUID, required=True)
    args = parser.parse_args()
    embedder = load_embedder()
    if embedder is None:
        raise SystemExit(
            "EMBEDDING_MODE=off builds no index. Set EMBEDDING_MODE to simulator (local or demo) "
            "or live, then run this again."
        )
    result = asyncio.run(_run(args.tenant_id, args.bundle_id, embedder))
    summary = {
        "tenant_id": str(args.tenant_id),
        "policy_bundle_id": str(args.bundle_id),
        "build_id": str(result.build_id),
        "model_id": result.model_id,
        "node_count": result.node_count,
        "skipped": result.skipped,
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
