# The deploy's migration job creates the services' login roles before migrating (NRC-26).

from collections.abc import AsyncIterator
from uuid import uuid4

import pytest
from pydantic import SecretStr
from sqlalchemy import text
from sqlalchemy.engine import make_url
from sqlalchemy.ext.asyncio import AsyncConnection, create_async_engine
from sqlalchemy.pool import NullPool

from scripts.ensure_db_roles import (
    APP_ROLE,
    WORKER_ROLE,
    UnsafeRoleError,
    WeakPasswordError,
    ensure_login_role,
    scram_verifier,
)
from tests.conftest import IntegrationEnv

PASSWORD = SecretStr("test-only-Pa55word-" + "x" * 8)


@pytest.fixture
async def owner(test_env: IntegrationEnv) -> AsyncIterator[AsyncConnection]:
    # The owner connection, in a transaction rolled back at the end: roles created here vanish.
    engine = create_async_engine(
        test_env.test_migration_database_url.get_secret_value(), poolclass=NullPool
    )
    async with engine.connect() as conn:
        transaction = await conn.begin()
        try:
            yield conn
        finally:
            await transaction.rollback()
    await engine.dispose()


async def _attributes(conn: AsyncConnection, role: str) -> dict[str, bool]:
    row = (
        await conn.execute(
            text(
                "SELECT rolcanlogin, rolsuper, rolcreatedb, rolcreaterole, rolbypassrls "
                "FROM pg_roles WHERE rolname = :role"
            ),
            {"role": role},
        )
    ).one()
    return dict(row._mapping)


async def _stored_password(conn: AsyncConnection, role: str) -> str:
    stored = await conn.scalar(
        text("SELECT rolpassword FROM pg_authid WHERE rolname = :role"), {"role": role}
    )
    return str(stored)


async def test_a_created_role_can_log_in_and_has_no_extra_privileges(
    test_env: IntegrationEnv,
) -> None:
    # Committed for real, because logging in needs the role visible to a new connection.
    role = f"nrc_test_{uuid4().hex[:12]}"
    owner_url = test_env.test_migration_database_url.get_secret_value()
    engine = create_async_engine(owner_url, poolclass=NullPool)
    try:
        async with engine.begin() as conn:
            result = await ensure_login_role(conn, role, PASSWORD, set_password=False)
            attributes = await _attributes(conn, role)
            stored = await _stored_password(conn, role)
        login_url = make_url(owner_url).set(username=role, password=PASSWORD.get_secret_value())
        login = create_async_engine(login_url, poolclass=NullPool)
        try:
            async with login.connect() as conn:
                who = await conn.scalar(text("SELECT current_user"))
        finally:
            await login.dispose()
    finally:
        async with engine.begin() as conn:
            await conn.exec_driver_sql(f'DROP ROLE IF EXISTS "{role}"')
        await engine.dispose()

    assert result.created
    assert who == role
    assert attributes == {
        "rolcanlogin": True,
        "rolsuper": False,
        "rolcreatedb": False,
        "rolcreaterole": False,
        "rolbypassrls": False,
    }
    assert stored.startswith("SCRAM-SHA-256$4096:")
    assert PASSWORD.get_secret_value() not in stored


async def test_an_existing_role_keeps_its_password_unless_asked(owner: AsyncConnection) -> None:
    role = f"nrc_test_{uuid4().hex[:12]}"
    await ensure_login_role(owner, role, PASSWORD, set_password=False)
    first = await _stored_password(owner, role)

    kept = await ensure_login_role(
        owner, role, SecretStr("another-Pa55word-xxxxxx"), set_password=False
    )
    unchanged = await _stored_password(owner, role)
    reset = await ensure_login_role(
        owner, role, SecretStr("another-Pa55word-xxxxxx"), set_password=True
    )
    changed = await _stored_password(owner, role)

    assert not kept.created
    assert not kept.password_set
    assert unchanged == first
    assert reset.password_set
    assert changed != first


@pytest.mark.parametrize("extra", ["CREATEDB", "CREATEROLE", "BYPASSRLS", "NOLOGIN"])
async def test_a_role_changed_by_hand_is_refused_not_repaired(
    owner: AsyncConnection, extra: str
) -> None:
    role = f"nrc_test_{uuid4().hex[:12]}"
    login = "" if extra == "NOLOGIN" else "LOGIN "
    await owner.exec_driver_sql(f'CREATE ROLE "{role}" {login}{extra}')

    with pytest.raises(UnsafeRoleError, match=role):
        await ensure_login_role(owner, role, PASSWORD, set_password=True)


async def test_the_real_service_roles_pass_the_safety_check(owner: AsyncConnection) -> None:
    # Local roles from docker/postgres-init.sql: the same attributes the job expects on staging.
    for role in (APP_ROLE, WORKER_ROLE):
        result = await ensure_login_role(owner, role, PASSWORD, set_password=False)
        assert not result.created


@pytest.mark.parametrize(
    "password", ["short", "x" * 129, "has space-in-it-long", "unicode-é-password-long"]
)
def test_passwords_outside_printable_ascii_16_to_128_are_refused(password: str) -> None:
    with pytest.raises(WeakPasswordError):
        scram_verifier(SecretStr(password))


def test_the_verifier_is_salted_and_never_contains_the_password() -> None:
    salt = bytes(range(16))

    fixed = scram_verifier(PASSWORD, salt=salt)
    again = scram_verifier(PASSWORD, salt=salt)
    fresh = scram_verifier(PASSWORD)

    assert fixed == again
    assert fresh != fixed  # a new random salt each time
    assert fixed.startswith("SCRAM-SHA-256$4096:AAECAwQFBgcICQoLDA0ODw==$")
    assert PASSWORD.get_secret_value() not in fresh
