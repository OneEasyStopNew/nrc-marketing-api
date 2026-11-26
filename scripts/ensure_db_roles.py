"""Create or update the two login roles the services use, before migrations run.

Usage (as the database owner, MIGRATION_DATABASE_URL; staging: the nrc_admin login):
    uv run python -m scripts.ensure_db_roles [--set-passwords]

marketing_app (the API) and marketing_worker (the connector worker) are created with LOGIN and
nothing else: no superuser, CREATEDB, CREATEROLE or BYPASSRLS, so row-level security applies to
both. Migrations grant their table rights, which is why this runs first (the deploy's migration
job runs it, then `alembic upgrade head`).

Passwords come from APP_DB_PASSWORD and WORKER_DB_PASSWORD. Only a SCRAM-SHA-256 verifier is
sent to the server, so the clear password never appears in statement logs. --set-passwords also
resets the password of roles that already exist (a rotated secret takes effect on the next
deploy); without it an existing role's password is left alone.

An existing role with any extra privilege is refused, not repaired: someone changed it by hand
and should look before the services run with it.
"""

import argparse
import asyncio
import base64
import hashlib
import hmac
import json
import re
import secrets
from dataclasses import asdict, dataclass
from typing import Final

from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncConnection, create_async_engine
from sqlalchemy.pool import NullPool

from app.core.config import OwnerDatabaseSettings

APP_ROLE: Final = "marketing_app"
WORKER_ROLE: Final = "marketing_worker"
SCRAM_ITERATIONS: Final = 4096
# Printable ASCII only: the server SASLpreps passwords at login, and for ASCII that is a no-op,
# so the verifier made here always matches what the server computes.
_PASSWORD: Final = re.compile(r"[\x21-\x7e]{16,128}")
_ROLE_NAME: Final = re.compile(r"[a-z_][a-z0-9_]{0,62}")
# In the order the query below selects them, after rolcanlogin.
_UNSAFE: Final = ("rolsuper", "rolcreatedb", "rolcreaterole", "rolbypassrls", "rolreplication")


class RolePasswords(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    app_db_password: SecretStr
    worker_db_password: SecretStr


class UnsafeRoleError(RuntimeError):
    pass


class WeakPasswordError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class RoleResult:
    role: str
    created: bool
    password_set: bool


def scram_verifier(password: SecretStr, *, salt: bytes | None = None) -> str:
    """The SCRAM-SHA-256 verifier Postgres stores (RFC 5802/7677), as `\\password` builds it."""
    secret = password.get_secret_value()
    if not _PASSWORD.fullmatch(secret):
        raise WeakPasswordError("role passwords must be 16-128 printable ASCII characters")
    salt = salt if salt is not None else secrets.token_bytes(16)
    salted = hashlib.pbkdf2_hmac("sha256", secret.encode(), salt, SCRAM_ITERATIONS)
    client_key = hmac.digest(salted, b"Client Key", "sha256")
    stored_key = hashlib.sha256(client_key).digest()
    server_key = hmac.digest(salted, b"Server Key", "sha256")

    def b64(raw: bytes) -> str:
        return base64.b64encode(raw).decode()

    return f"SCRAM-SHA-256${SCRAM_ITERATIONS}:{b64(salt)}${b64(stored_key)}:{b64(server_key)}"


async def _quoted(conn: AsyncConnection, statement: str, role: str, verifier: str) -> str:
    # Utility statements take no bind parameters, so the server quotes the values itself.
    quoted = await conn.scalar(
        text(
            "SELECT format(CAST(:statement AS text), CAST(:role AS text), CAST(:verifier AS text))"
        ),
        {"statement": statement, "role": role, "verifier": verifier},
    )
    return str(quoted)


async def ensure_login_role(
    conn: AsyncConnection, role: str, password: SecretStr, *, set_password: bool
) -> RoleResult:
    if not _ROLE_NAME.fullmatch(role):
        raise ValueError(f"invalid role name {role!r}")
    existing = (
        await conn.execute(
            text(
                "SELECT rolcanlogin, rolsuper, rolcreatedb, rolcreaterole, rolbypassrls, "
                "rolreplication FROM pg_roles WHERE rolname = :role"
            ),
            {"role": role},
        )
    ).one_or_none()
    if existing is None:
        statement = await _quoted(
            conn,
            "CREATE ROLE %I WITH LOGIN PASSWORD %L "
            "NOSUPERUSER NOCREATEDB NOCREATEROLE NOBYPASSRLS NOREPLICATION",
            role,
            scram_verifier(password),
        )
        await conn.exec_driver_sql(statement)
        return RoleResult(role, created=True, password_set=True)

    granted = [name for name, on in zip(_UNSAFE, existing[1:], strict=True) if on]
    if granted:
        raise UnsafeRoleError(f"role {role} has {', '.join(granted)}; remove it by hand first")
    if not existing.rolcanlogin:
        raise UnsafeRoleError(f"role {role} exists but cannot log in")
    if set_password:
        statement = await _quoted(
            conn, "ALTER ROLE %I WITH PASSWORD %L", role, scram_verifier(password)
        )
        await conn.exec_driver_sql(statement)
    return RoleResult(role, created=False, password_set=set_password)


async def main(*, set_passwords: bool) -> list[RoleResult]:
    passwords = RolePasswords()
    engine = create_async_engine(
        OwnerDatabaseSettings().migration_database_url.get_secret_value(), poolclass=NullPool
    )
    try:
        async with engine.begin() as conn:
            return [
                await ensure_login_role(
                    conn, APP_ROLE, passwords.app_db_password, set_password=set_passwords
                ),
                await ensure_login_role(
                    conn, WORKER_ROLE, passwords.worker_db_password, set_password=set_passwords
                ),
            ]
    finally:
        await engine.dispose()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--set-passwords",
        action="store_true",
        help="also reset the password of roles that already exist",
    )
    args = parser.parse_args()
    results = asyncio.run(main(set_passwords=args.set_passwords))
    print(json.dumps([asdict(result) for result in results]))
