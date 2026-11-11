"""Test harness: real Postgres (docker compose), real migrations, signed test tokens.

Each test runs in one outer transaction that is rolled back afterwards. Data is seeded as the
owner, then the connection switches to the marketing_app role so row-level security applies.
"""

import time
from collections.abc import AsyncIterator, Callable, Iterator
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from uuid import UUID, uuid4

import httpx
import jwt
import pytest
from alembic import command
from alembic.config import Config
from asgi_lifespan import LifespanManager
from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography.hazmat.primitives.asymmetric.rsa import RSAPrivateKey
from jwt.algorithms import RSAAlgorithm
from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncConnection, AsyncSession, create_async_engine
from sqlalchemy.pool import NullPool

from app.api.deps import get_db, get_signing_keys
from app.core.config import get_settings
from app.core.errors import AuthenticationError
from app.db.session import TenantSession
from app.main import create_app
from app.models import Membership, MembershipStudy, PolicyBundle, Study, StudySnapshot, Tenant
from app.models.enums import Language, RecruitmentStatus
from scripts.seed_demo import DEMO_POLICY_NAME, SYNTHETIC_FACTS, demo_policy_rules

ROOT = Path(__file__).resolve().parents[1]
REGION = "ap-south-1"
POOL_ID = "ap-south-1_TESTPOOL"
ISSUER = f"https://cognito-idp.{REGION}.amazonaws.com/{POOL_ID}"
CLIENT_ID = "test-client"
SCOPE = "marketing/api"
KID = "test-key"


class IntegrationEnv(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=ROOT / ".env", env_file_encoding="utf-8", extra="ignore"
    )

    test_database_url: SecretStr
    test_migration_database_url: SecretStr


@pytest.fixture(scope="session")
def test_env() -> IntegrationEnv:
    return IntegrationEnv()


@pytest.fixture(scope="session", autouse=True)
def app_environment(test_env: IntegrationEnv) -> Iterator[None]:
    with pytest.MonkeyPatch.context() as patch:
        patch.setenv("ENABLE_DOCS", "false")
        patch.setenv("DATABASE_URL", test_env.test_database_url.get_secret_value())
        patch.setenv("COGNITO_REGION", REGION)
        patch.setenv("COGNITO_USER_POOL_ID", POOL_ID)
        patch.setenv("COGNITO_CLIENT_IDS", CLIENT_ID)
        patch.setenv("REQUIRED_SCOPE", SCOPE)
        patch.setenv("CORS_ORIGINS", "")
        get_settings.cache_clear()
        yield
    get_settings.cache_clear()


@pytest.fixture(scope="session", autouse=True)
def migrated_database(test_env: IntegrationEnv) -> None:
    with pytest.MonkeyPatch.context() as patch:
        patch.setenv(
            "MIGRATION_DATABASE_URL", test_env.test_migration_database_url.get_secret_value()
        )
        config = Config(str(ROOT / "alembic.ini"))
        command.downgrade(config, "base")
        command.upgrade(config, "head")


@pytest.fixture(scope="session")
def signing_key() -> RSAPrivateKey:
    return rsa.generate_private_key(public_exponent=65537, key_size=2048)


class StaticKeys:
    # Stands in for Cognito's JWKS endpoint.

    def __init__(self, private_key: RSAPrivateKey) -> None:
        jwk = RSAAlgorithm.to_jwk(private_key.public_key(), as_dict=True)
        self._key = jwt.PyJWK({**jwk, "kid": KID, "alg": "RS256", "use": "sig"})

    async def get_signing_key(self, kid: str) -> jwt.PyJWK:
        if kid != KID:
            raise AuthenticationError("Access token was signed with an unknown key.")
        return self._key


def make_token(private_key: RSAPrivateKey, sub: str, **overrides: object) -> str:
    now = int(time.time())
    claims: dict[str, object] = {
        "sub": sub,
        "iss": ISSUER,
        "client_id": CLIENT_ID,
        "token_use": "access",
        "scope": f"openid {SCOPE}",
        "username": f"{sub}-name",
        "iat": now,
        "exp": now + 300,
    }
    claims.update(overrides)
    return jwt.encode(claims, private_key, algorithm="RS256", headers={"kid": KID})


@dataclass(frozen=True, slots=True)
class SeededTenant:
    tenant_id: UUID
    study_id: UUID
    closed_study_id: UUID
    snapshot_id: UUID
    policy_bundle_id: UUID | None


async def seed_tenant(
    db: AsyncSession, *, name: str, members: dict[str, list[str]], with_policy: bool
) -> SeededTenant:
    # Tenant with one recruiting and one closed synthetic study; every member gets both.
    tenant_id, study_id, closed_study_id, snapshot_id = uuid4(), uuid4(), uuid4(), uuid4()
    # Models have no ORM relationships, so each parent level is flushed before its children.
    db.add(Tenant(id=tenant_id, name=name))
    await db.flush()
    db.add_all(
        Membership(tenant_id=tenant_id, user_sub=sub, roles=roles) for sub, roles in members.items()
    )
    db.add_all(
        [
            Study(id=study_id, tenant_id=tenant_id, title="[SYNTHETIC] Open study"),
            Study(id=closed_study_id, tenant_id=tenant_id, title="[SYNTHETIC] Closed study"),
        ]
    )
    await db.flush()
    for study, status, snap_id in (
        (study_id, RecruitmentStatus.RECRUITING, snapshot_id),
        (closed_study_id, RecruitmentStatus.CLOSED, uuid4()),
    ):
        db.add(
            StudySnapshot(
                id=snap_id,
                tenant_id=tenant_id,
                study_id=study,
                version=1,
                recruitment_status=status,
                language=Language.EN,
                facts=SYNTHETIC_FACTS,
                source_ref="synthetic:test-fixture",
                approved_by="synthetic-fixture",
                approved_at=datetime(2026, 11, 1, tzinfo=UTC),
                is_synthetic=True,
            )
        )
        db.add_all(
            MembershipStudy(tenant_id=tenant_id, user_sub=sub, study_id=study) for sub in members
        )
    policy_bundle_id = uuid4() if with_policy else None
    if policy_bundle_id is not None:
        db.add(
            PolicyBundle(
                id=policy_bundle_id,
                tenant_id=tenant_id,
                version=1,
                name=DEMO_POLICY_NAME,
                source_ref="synthetic:test-fixture",
                rules=demo_policy_rules(),
                is_synthetic=True,
                approved_by="synthetic-fixture",
                approved_at=datetime(2026, 11, 1, tzinfo=UTC),
            )
        )
    await db.flush()
    return SeededTenant(tenant_id, study_id, closed_study_id, snapshot_id, policy_bundle_id)


@dataclass(frozen=True, slots=True)
class World:
    # Two tenants. creator-a and reviewer-a belong to A; creator-b to B. Only A has a policy.

    a: SeededTenant
    b: SeededTenant


@pytest.fixture
async def connection(test_env: IntegrationEnv) -> AsyncIterator[AsyncConnection]:
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


def _session(conn: AsyncConnection) -> AsyncSession:
    return AsyncSession(
        bind=conn,
        join_transaction_mode="create_savepoint",
        expire_on_commit=False,
        sync_session_class=TenantSession,
    )


@pytest.fixture
async def world(connection: AsyncConnection) -> World:
    async with _session(connection) as seed:
        a = await seed_tenant(
            seed,
            name="Tenant A",
            members={"creator-a": ["creator"], "reviewer-a": ["reviewer"]},
            with_policy=True,
        )
        b = await seed_tenant(
            seed, name="Tenant B", members={"creator-b": ["creator"]}, with_policy=False
        )
        await seed.commit()
    # From here on the connection behaves like the API's role: RLS applies, no owner rights.
    await connection.execute(text("SET LOCAL ROLE marketing_app"))
    return World(a, b)


@pytest.fixture
def app_overrides() -> dict[Callable[..., object], Callable[..., object]]:
    # Extra dependency overrides for the app under test; test modules redefine this fixture.
    return {}


@pytest.fixture
async def client(
    connection: AsyncConnection,
    world: World,
    signing_key: RSAPrivateKey,
    app_overrides: dict[Callable[..., object], Callable[..., object]],
) -> AsyncIterator[httpx.AsyncClient]:
    async def request_session() -> AsyncIterator[AsyncSession]:
        # One session per request, like production: uncommitted work is rolled back on close.
        async with _session(connection) as session:
            yield session

    app = create_app()
    app.dependency_overrides[get_db] = request_session
    app.dependency_overrides[get_signing_keys] = lambda: StaticKeys(signing_key)
    app.dependency_overrides.update(app_overrides)
    try:
        async with (
            LifespanManager(app),
            httpx.AsyncClient(
                transport=httpx.ASGITransport(app=app), base_url="http://test"
            ) as http,
        ):
            yield http
    finally:
        app.dependency_overrides.clear()


class AuthHeaders:
    def __init__(self, private_key: RSAPrivateKey) -> None:
        self._key = private_key

    def __call__(self, sub: str, tenant_id: UUID | None = None, **extra: str) -> dict[str, str]:
        headers = {"Authorization": f"Bearer {make_token(self._key, sub)}", **extra}
        if tenant_id is not None:
            headers["X-Tenant-Id"] = str(tenant_id)
        return headers


@pytest.fixture
def auth(signing_key: RSAPrivateKey) -> AuthHeaders:
    return AuthHeaders(signing_key)
