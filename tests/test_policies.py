# Active policy, the policy summary in /context, and generation eligibility.

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from typing import Any
from uuid import UUID, uuid4

import httpx
import pytest
from pydantic import ValidationError
from sqlalchemy import insert, text
from sqlalchemy.ext.asyncio import AsyncConnection

from app.models import PolicyBundle, StudySnapshot
from app.models.enums import Language, RecruitmentStatus
from app.schemas.policies import PolicyRule
from scripts.seed_demo import DEMO_POLICY_RULES, SYNTHETIC_FACTS, demo_policy_rules
from tests.conftest import AuthHeaders, SeededTenant, World

API = "/v1/marketing"


@asynccontextmanager
async def as_owner(connection: AsyncConnection) -> AsyncIterator[None]:
    # Data the API role may not write (policies, snapshots) is added as the owner.
    await connection.execute(text("RESET ROLE"))
    try:
        yield
    finally:
        await connection.execute(text("SET LOCAL ROLE marketing_app"))


async def _create_creative(
    client: httpx.AsyncClient, auth: AuthHeaders, sub: str, tenant: SeededTenant
) -> dict[str, Any]:
    response = await client.post(
        f"{API}/creatives",
        headers=auth(sub, tenant.tenant_id, **{"Idempotency-Key": uuid4().hex}),
        json={
            "study_id": str(tenant.study_id),
            "prompt": "Invite adults with high blood pressure to learn about the study.",
            "language": "en",
            "channel": "facebook",
            "post_format": "feed_image",
        },
    )
    assert response.status_code == 201, response.text
    body: dict[str, Any] = response.json()
    return body


async def _eligibility(
    client: httpx.AsyncClient,
    auth: AuthHeaders,
    sub: str,
    tenant_id: UUID,
    creative_id: str,
    **params: int,
) -> httpx.Response:
    return await client.get(
        f"{API}/creatives/{creative_id}/generation-eligibility",
        headers=auth(sub, tenant_id),
        params=params,
    )


# Active policy


@pytest.mark.parametrize("sub", ["creator-a", "reviewer-a"])
async def test_active_policy_lists_every_rule(
    client: httpx.AsyncClient, world: World, auth: AuthHeaders, sub: str
) -> None:
    response = await client.get(f"{API}/policies/active", headers=auth(sub, world.a.tenant_id))

    assert response.status_code == 200
    body = response.json()
    assert body["policy_bundle_id"] == str(world.a.policy_bundle_id)
    assert body["version"] == 1
    assert body["is_synthetic"] is True
    assert [rule["rule_id"] for rule in body["rules"]] == [
        rule["rule_id"] for rule in DEMO_POLICY_RULES
    ]
    first = body["rules"][0]
    assert first["mandatory"] is True
    assert first["severity"] == "critical"
    assert first["source"]["clause"]


async def test_tenant_without_policy_gets_404(
    client: httpx.AsyncClient, world: World, auth: AuthHeaders
) -> None:
    response = await client.get(
        f"{API}/policies/active", headers=auth("creator-b", world.b.tenant_id)
    )

    assert response.status_code == 404
    assert response.json()["code"] == "NOT_FOUND"


async def test_highest_policy_version_is_active(
    client: httpx.AsyncClient, world: World, auth: AuthHeaders, connection: AsyncConnection
) -> None:
    async with as_owner(connection):
        await connection.execute(
            insert(PolicyBundle).values(
                id=uuid4(),
                tenant_id=world.a.tenant_id,
                version=2,
                name="[SYNTHETIC] Policy v2",
                source_ref="synthetic:test-fixture",
                rules=demo_policy_rules()[:2],
                is_synthetic=True,
                approved_by="synthetic-fixture",
                approved_at=datetime(2026, 11, 2, tzinfo=UTC),
            )
        )

    response = await client.get(
        f"{API}/policies/active", headers=auth("creator-a", world.a.tenant_id)
    )

    assert response.status_code == 200
    assert response.json()["version"] == 2
    assert len(response.json()["rules"]) == 2


async def test_policy_requires_a_token(client: httpx.AsyncClient, world: World) -> None:
    response = await client.get(
        f"{API}/policies/active", headers={"X-Tenant-Id": str(world.a.tenant_id)}
    )

    assert response.status_code == 401


# Policy summary in /context


async def test_context_includes_active_policy_summary(
    client: httpx.AsyncClient, world: World, auth: AuthHeaders
) -> None:
    response = await client.get(f"{API}/context", headers=auth("creator-a", world.a.tenant_id))

    assert response.status_code == 200
    assert response.json()["active_policy"] == {
        "policy_bundle_id": str(world.a.policy_bundle_id),
        "version": 1,
        "name": "[SYNTHETIC] Demo recruitment advertising rules",
        "is_synthetic": True,
        "rule_count": len(DEMO_POLICY_RULES),
    }


async def test_context_policy_is_null_without_policy(
    client: httpx.AsyncClient, world: World, auth: AuthHeaders
) -> None:
    response = await client.get(f"{API}/context", headers=auth("creator-b", world.b.tenant_id))

    assert response.status_code == 200
    assert response.json()["active_policy"] is None


# Generation eligibility


async def test_unvalidated_current_version_is_not_eligible(
    client: httpx.AsyncClient, world: World, auth: AuthHeaders
) -> None:
    creative = await _create_creative(client, auth, "creator-a", world.a)

    response = await _eligibility(client, auth, "creator-a", world.a.tenant_id, creative["id"])

    assert response.status_code == 200
    assert response.json() == {
        "creative_id": creative["id"],
        "creative_version": 1,
        "eligible": False,
        "reason": "NOT_VALIDATED",
        "message": "Validate the brief for this version before generating.",
        "policy_bundle_version": 1,
        "validation_run_id": None,
    }


async def test_older_version_is_stale(
    client: httpx.AsyncClient, world: World, auth: AuthHeaders
) -> None:
    creative = await _create_creative(client, auth, "creator-a", world.a)
    edit = await client.patch(
        f"{API}/creatives/{creative['id']}",
        headers=auth("creator-a", world.a.tenant_id, **{"If-Match": '"1"'}),
        json={"caption": "Learn about a research study near you."},
    )
    assert edit.status_code == 200, edit.text

    response = await _eligibility(
        client, auth, "creator-a", world.a.tenant_id, creative["id"], version=1
    )

    assert response.status_code == 200
    assert response.json()["reason"] == "STALE_VERSION"
    assert response.json()["eligible"] is False


async def test_study_closed_after_drafting_blocks_generation(
    client: httpx.AsyncClient, world: World, auth: AuthHeaders, connection: AsyncConnection
) -> None:
    creative = await _create_creative(client, auth, "creator-a", world.a)
    async with as_owner(connection):
        await connection.execute(
            insert(StudySnapshot).values(
                id=uuid4(),
                tenant_id=world.a.tenant_id,
                study_id=world.a.study_id,
                version=2,
                recruitment_status=RecruitmentStatus.CLOSED,
                language=Language.EN,
                facts=SYNTHETIC_FACTS,
                source_ref="synthetic:test-fixture",
                approved_by="synthetic-fixture",
                approved_at=datetime(2026, 11, 2, tzinfo=UTC),
                is_synthetic=True,
            )
        )

    response = await _eligibility(client, auth, "creator-a", world.a.tenant_id, creative["id"])

    assert response.json()["reason"] == "STUDY_CLOSED"


async def test_tenant_without_policy_is_not_eligible(
    client: httpx.AsyncClient, world: World, auth: AuthHeaders
) -> None:
    creative = await _create_creative(client, auth, "creator-b", world.b)

    response = await _eligibility(client, auth, "creator-b", world.b.tenant_id, creative["id"])

    assert response.json()["reason"] == "NO_ACTIVE_POLICY"
    assert response.json()["policy_bundle_version"] is None


async def test_other_tenants_creative_is_not_found(
    client: httpx.AsyncClient, world: World, auth: AuthHeaders
) -> None:
    creative = await _create_creative(client, auth, "creator-a", world.a)

    response = await _eligibility(client, auth, "creator-b", world.b.tenant_id, creative["id"])

    assert response.status_code == 404


async def test_future_version_is_not_found(
    client: httpx.AsyncClient, world: World, auth: AuthHeaders
) -> None:
    creative = await _create_creative(client, auth, "creator-a", world.a)

    response = await _eligibility(
        client, auth, "creator-a", world.a.tenant_id, creative["id"], version=2
    )

    assert response.status_code == 404


async def test_version_must_be_positive(
    client: httpx.AsyncClient, world: World, auth: AuthHeaders
) -> None:
    creative = await _create_creative(client, auth, "creator-a", world.a)

    response = await _eligibility(
        client, auth, "creator-a", world.a.tenant_id, creative["id"], version=0
    )

    assert response.status_code == 422


# Rule schema


def test_demo_rules_are_valid_and_unique() -> None:
    rules = demo_policy_rules()

    assert len({rule["rule_id"] for rule in rules}) == len(rules)


@pytest.mark.parametrize(
    ("field", "value"),
    [("rule_id", "rec 01"), ("applies_to", []), ("rule_version", 0), ("severity", "high")],
)
def test_rule_schema_rejects_invalid_values(field: str, value: object) -> None:
    rule = {**DEMO_POLICY_RULES[0], field: value}

    with pytest.raises(ValidationError, match=field):
        PolicyRule.model_validate(rule)
