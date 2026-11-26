"""Policy context retrieval (NRC-5) under parallel requests, on committed data."""

import asyncio
from datetime import UTC, datetime

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.pool import NullPool

from app.core.config import get_settings
from app.db.session import TenantSession, bind_tenant
from app.models import ContextManifestRecord
from app.models.enums import Channel, PolicyLanguage, ValidationInputKind
from app.services.policy_context import (
    AuroraPolicyContextProvider,
    ContextRequest,
    ContextScope,
    StoredManifest,
)
from scripts.import_policy_graph import import_policy_graph, load_synthetic_graph
from tests.conftest import IntegrationEnv, SeededTenant


async def test_parallel_identical_builds_store_one_manifest(
    committed_tenant: SeededTenant, test_env: IntegrationEnv
) -> None:
    tenant = committed_tenant.tenant_id
    assert committed_tenant.policy_bundle_id is not None
    owner = create_async_engine(
        test_env.test_migration_database_url.get_secret_value(), poolclass=NullPool
    )
    api = create_async_engine(test_env.test_database_url.get_secret_value(), poolclass=NullPool)
    request = ContextRequest(
        policy_bundle_id=committed_tenant.policy_bundle_id,
        study_snapshot_id=committed_tenant.snapshot_id,
        scope=ContextScope(
            input_kind=ValidationInputKind.TEXT,
            channel=Channel.FACEBOOK,
            language=PolicyLanguage.EN,
            evaluated_at=datetime(2026, 11, 18, tzinfo=UTC),
        ),
    )
    provider = AuroraPolicyContextProvider(get_settings())

    async def build() -> StoredManifest:
        # Each request on its own API-role connection, as two validation requests would be.
        async with AsyncSession(api, sync_session_class=TenantSession) as db:
            await bind_tenant(db, tenant)
            stored = await provider.build(db, request, actor="racer")
            await db.commit()
            return stored

    try:
        async with AsyncSession(owner, sync_session_class=TenantSession) as db:
            await bind_tenant(db, tenant)
            await import_policy_graph(
                db, tenant, committed_tenant.policy_bundle_id, load_synthetic_graph()
            )
            await db.commit()

        first, second = await asyncio.gather(build(), build())

        async with AsyncSession(owner, sync_session_class=TenantSession) as db:
            await bind_tenant(db, tenant)
            rows = await db.scalar(select(func.count()).select_from(ContextManifestRecord))
    finally:
        await api.dispose()
        await owner.dispose()

    assert first.id == second.id
    assert sorted([first.reused, second.reused]) == [False, True]
    assert first.manifest == second.manifest
    assert rows == 1
