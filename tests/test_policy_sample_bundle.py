# The reviewed sample bundle (NRC-4 evidence, README "Policy bundles") must stay clean: it is the
# file a reviewer imports to walk a bundle from review to ACTIVE.

import pytest
from sqlalchemy.ext.asyncio import AsyncConnection

from app.models.enums import PolicyLanguage, WordingKind
from app.services.policy_checks import load_snapshot, run_checks
from scripts.import_policy_graph import SYNTHETIC_GRAPH_PATH, import_policy_graph, load_graph
from tests.conftest import World, owner_session

SAMPLE_BUNDLE_PATH = SYNTHETIC_GRAPH_PATH.with_name("synthetic_sample_bundle.json")


@pytest.fixture
def seed_graph() -> bool:
    # These tests count or write their own sources, graph rows and wordings in tenant A, so its
    # bundle starts without NRC-5's synthetic graph.
    return False


async def test_the_sample_bundle_passes_every_check(
    connection: AsyncConnection, world: World
) -> None:
    graph = load_graph(SAMPLE_BUNDLE_PATH)
    tenant_id = world.a.tenant_id
    assert world.a.policy_bundle_id is not None
    async with owner_session(connection, tenant_id) as db:
        await import_policy_graph(db, tenant_id, world.a.policy_bundle_id, graph)
        snapshot = await load_snapshot(db, tenant_id, world.a.policy_bundle_id)

    assert graph.source.is_synthetic
    assert len(snapshot.rules) == len(graph.rules)
    # English only, so the default POLICY_SUPPORTED_LANGUAGES is enough.
    assert run_checks(snapshot, (PolicyLanguage.EN,)) == []
    # AC7: the exact wording is quoted from its clause, not reworded.
    exact = [w.text for w in snapshot.wordings if w.kind is WordingKind.EXACT_WORDING]
    assert exact == ["This is a research study. Taking part is voluntary."]
