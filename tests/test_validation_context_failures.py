# A failure while gathering policy context never fails the validation itself (NRC-5).

from collections.abc import Callable
from uuid import UUID, uuid4

import httpx
import pytest
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.deps import get_context_provider
from app.services.policy_context import ContextRequest, StoredManifest
from tests.conftest import AuthHeaders, World

API = "/v1/marketing"


class BrokenProvider:
    # Fails inside the database, like a bad query or a lost connection would.

    async def build(
        self, db: AsyncSession, tenant_id: UUID, request: ContextRequest, *, actor: str
    ) -> StoredManifest:
        await db.execute(text("SELECT 1 / 0"))
        raise AssertionError("the query above always fails")


@pytest.fixture
def app_overrides() -> dict[Callable[..., object], Callable[..., object]]:
    return {get_context_provider: BrokenProvider}


async def test_database_error_in_context_stores_run_with_gap(
    client: httpx.AsyncClient, auth: AuthHeaders, world: World
) -> None:
    headers = auth("creator-a", world.a.tenant_id, **{"Idempotency-Key": uuid4().hex})
    creative = await client.post(
        f"{API}/creatives",
        headers=headers,
        json={
            "study_id": str(world.a.study_id),
            "prompt": "Invite adults with high blood pressure to learn about our research study.",
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

    # The savepoint kept the run's own transaction usable: the run is stored, outcome intact.
    assert response.status_code == 202, response.text
    run_id = response.json()["validation_run_id"]
    run = (
        await client.get(
            f"{API}/validations/{run_id}", headers=auth("creator-a", world.a.tenant_id)
        )
    ).json()
    assert run["outcome"] == "PASS"
    assert any("(RETRIEVAL_ERROR)" in gap["reason"] for gap in run["coverage_gaps"])
    context = await client.get(
        f"{API}/validations/{run_id}/context", headers=auth("creator-a", world.a.tenant_id)
    )
    assert context.json()["code"] == "CONTEXT_NOT_AVAILABLE"
