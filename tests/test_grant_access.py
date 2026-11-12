"""Granting Cognito users roles in an existing tenant (scripts.grant_access)."""

import argparse
from uuid import uuid4

import pytest
from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncConnection, AsyncSession

from app.db.session import TenantSession
from app.models import Membership, MembershipStudy
from scripts.grant_access import UnknownTenantError, grant_access, parse_display_name
from scripts.seed_demo import parse_member
from tests.conftest import seed_tenant

COGNITO_SUB = "4c1d8f2e-9a7b-4e3c-b5d6-0f1e2a3b4c5d"


def _owner_session(conn: AsyncConnection) -> AsyncSession:
    return AsyncSession(
        bind=conn,
        join_transaction_mode="create_savepoint",
        expire_on_commit=False,
        sync_session_class=TenantSession,
    )


async def test_grants_roles_and_every_study_in_the_tenant(connection: AsyncConnection) -> None:
    async with _owner_session(connection) as db:
        tenant = await seed_tenant(db, name="Demo", members={}, with_policy=False)

        study_ids = await grant_access(db, tenant.tenant_id, [(COGNITO_SUB, ["policy_admin"])])

        roles = await db.scalar(select(Membership.roles).where(Membership.user_sub == COGNITO_SUB))
        granted = set(
            await db.scalars(
                select(MembershipStudy.study_id).where(MembershipStudy.user_sub == COGNITO_SUB)
            )
        )
    assert roles == ["policy_admin"]
    assert granted == {tenant.study_id, tenant.closed_study_id} == set(study_ids)


async def test_rerun_replaces_roles_and_reactivates(connection: AsyncConnection) -> None:
    async with _owner_session(connection) as db:
        tenant = await seed_tenant(
            db, name="Demo", members={COGNITO_SUB: ["creator"]}, with_policy=False
        )
        await db.execute(update(Membership).values(is_active=False))

        await grant_access(db, tenant.tenant_id, [(COGNITO_SUB, ["reviewer", "publisher"])])

        membership = (
            await db.execute(
                select(Membership.roles, Membership.is_active).where(
                    Membership.user_sub == COGNITO_SUB
                )
            )
        ).one()
        study_rows = await db.scalars(
            select(MembershipStudy.study_id).where(MembershipStudy.user_sub == COGNITO_SUB)
        )
        study_count = len(list(study_rows))
    assert membership == (["reviewer", "publisher"], True)
    assert study_count == 2


async def test_unknown_tenant_is_rejected(connection: AsyncConnection) -> None:
    async with _owner_session(connection) as db:
        with pytest.raises(UnknownTenantError, match="no tenant with id"):
            await grant_access(db, uuid4(), [(COGNITO_SUB, ["creator"])])


@pytest.mark.parametrize(
    ("value", "message"),
    [
        (f"{COGNITO_SUB}=", "expected <cognito-sub>="),
        ("=creator", "expected <cognito-sub>="),
        (f"{COGNITO_SUB}=policy-admin", "unknown roles"),
        ("bad sub=creator", "expected <cognito-sub>="),
    ],
)
def test_member_argument_is_validated(value: str, message: str) -> None:
    with pytest.raises(argparse.ArgumentTypeError, match=message):
        parse_member(value)


async def test_display_name_is_set_and_kept_on_rerun_without_one(
    connection: AsyncConnection,
) -> None:
    async with _owner_session(connection) as db:
        tenant = await seed_tenant(db, name="Demo", members={}, with_policy=False)
        await grant_access(
            db, tenant.tenant_id, [(COGNITO_SUB, ["creator"])], {COGNITO_SUB: "Priya Shah"}
        )

        await grant_access(db, tenant.tenant_id, [(COGNITO_SUB, ["reviewer"])])

        name = await db.scalar(
            select(Membership.display_name).where(Membership.user_sub == COGNITO_SUB)
        )
    assert name == "Priya Shah"


def test_display_name_argument_is_normalised() -> None:
    sub, name = parse_display_name(f"{COGNITO_SUB}=  José   Ruiz ")

    assert (sub, name) == (COGNITO_SUB, "José Ruiz")


@pytest.mark.parametrize(
    "value", [f"{COGNITO_SUB}=", f"{COGNITO_SUB}=   ", "=Priya", "x=" + "a" * 201]
)
def test_display_name_argument_is_validated(value: str) -> None:
    with pytest.raises(argparse.ArgumentTypeError, match="expected <cognito-sub>="):
        parse_display_name(value)
