"""Test harness: real Postgres (docker compose), real migrations, signed test tokens.

Each test runs in one outer transaction that is rolled back afterwards. Data is seeded as the
owner, then the connection switches to the marketing_app role so row-level security applies.
API request sessions run as marketing_app and worker sessions (`sessions`) as marketing_worker,
each switching role when its transaction begins.
"""

import dataclasses
import os
import time
import warnings
from collections.abc import AsyncIterator, Awaitable, Callable, Iterator, Mapping
from contextlib import asynccontextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path
from typing import Any
from uuid import UUID, uuid4

import boto3
import httpx
import jwt
import pytest
from alembic import command
from alembic.config import Config
from asgi_lifespan import LifespanManager
from botocore.config import Config as BotoConfig
from botocore.exceptions import ClientError
from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography.hazmat.primitives.asymmetric.rsa import RSAPrivateKey
from jwt.algorithms import RSAAlgorithm
from PIL import Image
from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict
from sqlalchemy import Connection, Row, event, func, insert, select, text, update
from sqlalchemy.ext.asyncio import (
    AsyncConnection,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.orm import Session, SessionTransaction
from sqlalchemy.pool import NullPool

from app.api.deps import get_db, get_signing_keys
from app.core.config import get_settings
from app.core.errors import AuthenticationError
from app.db.base import Base
from app.db.session import TenantSession, bind_tenant, create_engine, create_sessionmaker
from app.main import create_app
from app.models import (
    Asset,
    CreditReservation,
    Job,
    MediaUpload,
    Membership,
    MembershipStudy,
    PolicyBundle,
    Study,
    StudySnapshot,
    Tenant,
)
from app.models.enums import (
    BundleLifecycleState,
    CheckType,
    IdentifiablePerson,
    Language,
    RecruitmentStatus,
    UploadPurpose,
    UploadState,
)
from app.models.jobs import ReservationState
from app.services.providers.base import (
    FetchedImage,
    ImageRequest,
    ProviderError,
    ProviderJobStatus,
    StatusResult,
    SubmitResult,
)
from app.services.providers.simulator import SimulatedImageProvider
from app.services.validations import VALIDATION_RUN_EVENT
from app.workers.connector import EVENT_HANDLERS, run_once
from scripts.import_policy_graph import PolicyGraphImport, import_policy_graph, load_synthetic_graph
from scripts.seed_demo import DEMO_POLICY_NAME, SYNTHETIC_FACTS, demo_policy_rules

ROOT = Path(__file__).resolve().parents[1]
REGION = "ap-south-1"
POOL_ID = "ap-south-1_TESTPOOL"
ISSUER = f"https://cognito-idp.{REGION}.amazonaws.com/{POOL_ID}"
CLIENT_ID = "test-client"
SCOPE = "marketing/api"
KID = "test-key"
# Local SeaweedFS from docker compose, and the versioned test bucket in it.
ENDPOINT = "http://127.0.0.1:8333"
BUCKET = "nrc-test-media"
# Policy source files (NRC-4) live in their own bucket, never the media one.
POLICY_BUCKET = "nrc-test-policy"
APP_ROLE = "marketing_app"
WORKER_ROLE = "marketing_worker"


class IntegrationEnv(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=ROOT / ".env", env_file_encoding="utf-8", extra="ignore"
    )

    test_database_url: SecretStr
    test_worker_database_url: SecretStr
    test_migration_database_url: SecretStr


@pytest.fixture(scope="session")
def test_env() -> IntegrationEnv:
    return IntegrationEnv()


@pytest.fixture(scope="session", autouse=True)
def app_environment(test_env: IntegrationEnv) -> Iterator[None]:
    with pytest.MonkeyPatch.context() as patch:
        patch.setenv("ENABLE_DOCS", "false")
        patch.setenv("DATABASE_URL", test_env.test_database_url.get_secret_value())
        patch.setenv("WORKER_DATABASE_URL", test_env.test_worker_database_url.get_secret_value())
        patch.setenv("COGNITO_REGION", REGION)
        patch.setenv("COGNITO_USER_POOL_ID", POOL_ID)
        patch.setenv("COGNITO_CLIENT_IDS", CLIENT_ID)
        patch.setenv("REQUIRED_SCOPE", SCOPE)
        patch.setenv("CORS_ORIGINS", "")
        # A developer's .env may switch these on; tests always start from production behaviour.
        patch.setenv("DEMO_STUBS", "false")
        patch.setenv("DEV_JWKS_FILE", "")
        patch.setenv("EMBEDDING_MODE", "off")
        patch.setenv("VALIDATION_MODEL_MODE", "off")
        # Local SeaweedFS; these credentials only open that container, not AWS.
        patch.setenv("AWS_S3_BUCKET", BUCKET)
        patch.setenv("AWS_ENDPOINT_URL_S3", ENDPOINT)
        patch.setenv("S3_FORCE_PATH_STYLE", "true")
        patch.setenv("AWS_ACCESS_KEY_ID", "localdev")
        patch.setenv("AWS_SECRET_ACCESS_KEY", "localdev-password")
        patch.delenv("AWS_PROFILE", raising=False)
        get_settings.cache_clear()
        yield
    get_settings.cache_clear()


@pytest.fixture(autouse=True)
def pillow_state() -> Iterator[None]:
    # The API lifespan and worker main() set Pillow's process-wide limit and warning filter;
    # restore both so no test depends on another having started the app.
    with pytest.MonkeyPatch.context() as patch, warnings.catch_warnings():
        patch.setattr(Image, "MAX_IMAGE_PIXELS", Image.MAX_IMAGE_PIXELS)
        yield


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


CREATOR_A_NAME = "Priya Shah"


async def seed_tenant(
    db: AsyncSession,
    *,
    name: str,
    members: dict[str, list[str]],
    with_policy: bool,
    display_names: Mapping[str, str] | None = None,
    with_graph: bool = True,
    graph: PolicyGraphImport | None = None,
) -> SeededTenant:
    # Tenant with one recruiting and one closed synthetic study; every member gets both.
    tenant_id, study_id, closed_study_id, snapshot_id = uuid4(), uuid4(), uuid4(), uuid4()
    # Models have no ORM relationships, so each parent level is flushed before its children.
    db.add(Tenant(id=tenant_id, name=name))
    await db.flush()
    names = display_names or {}
    db.add_all(
        Membership(tenant_id=tenant_id, user_sub=sub, roles=roles, display_name=names.get(sub))
        for sub, roles in members.items()
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
                lifecycle_state=BundleLifecycleState.ACTIVE,
            )
        )
        await db.flush()
        # With its rulebook graph, like seed_demo and the staging deploy step: without one every
        # validation needs review (no policy context), so a clean brief could never pass.
        if with_graph:
            await import_policy_graph(
                db, tenant_id, policy_bundle_id, graph or load_synthetic_graph()
            )
    await db.flush()
    return SeededTenant(tenant_id, study_id, closed_study_id, snapshot_id, policy_bundle_id)


@dataclass(frozen=True, slots=True)
class World:
    # Two tenants. creator-a and reviewer-a belong to A; creator-b to B. Only A has a policy.
    # Only creator-a has a display name.

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


Sessions = async_sessionmaker[AsyncSession]


class RoleSession(TenantSession):
    # A test session on the shared connection that acts as session.info["role"], if set.
    pass


@event.listens_for(RoleSession, "after_begin")
def _apply_role(session: Session, transaction: SessionTransaction, connection: Connection) -> None:
    role = session.info.get("role")
    if role in (APP_ROLE, WORKER_ROLE):
        # Lasts until the outer test transaction ends, or until a savepoint rollback.
        connection.execute(text(f"SET LOCAL ROLE {role}"))


def _session(conn: AsyncConnection, role: str | None = None) -> AsyncSession:
    return AsyncSession(
        bind=conn,
        join_transaction_mode="create_savepoint",
        expire_on_commit=False,
        sync_session_class=RoleSession,
        info={"role": role},
    )


def app_session(conn: AsyncConnection) -> AsyncSession:
    # As the API's role, on the test's connection.
    return _session(conn, APP_ROLE)


async def as_role(conn: AsyncConnection, role: str) -> None:
    # Switch the test's own statements to one of the two login roles.
    assert role in (APP_ROLE, WORKER_ROLE)
    await conn.execute(text(f"SET LOCAL ROLE {role}"))


async def bind_test_tenant(conn: AsyncConnection, tenant_id: UUID) -> None:
    # The tenant for the test's own statements, as bind_tenant does for sessions.
    await conn.execute(text("SELECT set_config('app.tenant_id', :t, true)"), {"t": str(tenant_id)})


@asynccontextmanager
async def as_tenant_owner(conn: AsyncConnection, tenant_id: UUID) -> AsyncIterator[None]:
    # Owner scripts (seed, policy graph import) write as the owner with the tenant bound; the
    # test's statements go back to the API role afterwards.
    await conn.execute(text("RESET ROLE"))
    await bind_test_tenant(conn, tenant_id)
    try:
        yield
    finally:
        await conn.execute(text(f"SET LOCAL ROLE {APP_ROLE}"))


async def owner_sql(
    conn: AsyncConnection,
    tenant_id: UUID,
    statement: str,
    params: dict[str, Any] | None = None,
) -> list[Row[Any]]:
    # One raw statement as the owner with the tenant bound (reads that RLS or grants would hide
    # from the role under test, and setup or time-travel UPDATEs).
    async with as_tenant_owner(conn, tenant_id):
        result = await conn.execute(text(statement), params or {})
        return list(result.all()) if result.returns_rows else []


@asynccontextmanager
async def owner_session(conn: AsyncConnection, tenant_id: UUID) -> AsyncIterator[AsyncSession]:
    # What owner scripts (seed, policy graph import) do: the owner role, the tenant bound, one
    # transaction (a savepoint of the test's).
    async with as_tenant_owner(conn, tenant_id), _session(conn) as db:
        await bind_tenant(db, tenant_id)
        yield db
        await db.commit()


async def count_rows(conn: AsyncConnection, model: type[Any]) -> int:
    return int(await conn.scalar(select(func.count()).select_from(model)) or 0)


async def add_policy_bundle(
    conn: AsyncConnection,
    tenant_id: UUID,
    version: int,
    state: BundleLifecycleState = BundleLifecycleState.ACTIVE,
    rules: list[dict[str, object]] | None = None,
) -> UUID:
    # A synthetic bundle without a graph (no JSONB rules unless given), written as the owner by
    # the caller. A tenant has at most one ACTIVE bundle, so a new ACTIVE one retires the
    # current one first (as activation does).
    if state is BundleLifecycleState.ACTIVE:
        await conn.execute(
            update(PolicyBundle)
            .where(
                PolicyBundle.tenant_id == tenant_id,
                PolicyBundle.lifecycle_state == BundleLifecycleState.ACTIVE,
            )
            .values(lifecycle_state=BundleLifecycleState.RETIRED)
        )
    bundle_id = uuid4()
    await conn.execute(
        insert(PolicyBundle).values(
            id=bundle_id,
            tenant_id=tenant_id,
            version=version,
            name=f"[SYNTHETIC] Policy v{version}",
            source_ref="synthetic:test-fixture",
            rules=rules or [],
            is_synthetic=True,
            approved_by="synthetic-fixture",
            approved_at=datetime(2026, 11, 17, tzinfo=UTC),
            lifecycle_state=state,
        )
    )
    return bundle_id


def make_pdf(pages: list[list[str | None]]) -> bytes:
    """A small PDF whose pages hold the given lines; None is an empty line (a paragraph gap)."""
    page_ids = [4 + 2 * index for index in range(len(pages))]
    objects = [
        "<< /Type /Catalog /Pages 2 0 R >>",
        "<< /Type /Pages /Kids [{}] /Count {} >>".format(
            " ".join(f"{page_id} 0 R" for page_id in page_ids), len(pages)
        ),
        "<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
    ]
    for page_id, lines in zip(page_ids, pages, strict=True):
        operators = ["BT", "/F1 12 Tf", "72 720 Td", "14 TL"]
        for line in lines:
            if line is None:
                operators.append("T*")
            else:
                escaped = line.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")
                operators.append(f"({escaped}) '")
        operators.append("ET")
        stream = "\n".join(operators)
        objects.append(
            f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents {page_id + 1} 0 R"
            " /Resources << /Font << /F1 3 0 R >> >> >>"
        )
        objects.append(f"<< /Length {len(stream)} >>\nstream\n{stream}\nendstream")
    out = b"%PDF-1.4\n"
    offsets = []
    for number, body in enumerate(objects, start=1):
        offsets.append(len(out))
        out += f"{number} 0 obj\n{body}\nendobj\n".encode("latin-1")
    xref = len(out)
    out += f"xref\n0 {len(objects) + 1}\n0000000000 65535 f \n".encode()
    out += "".join(f"{offset:010d} 00000 n \n" for offset in offsets).encode()
    out += (
        f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF\n".encode()
    )
    return out


async def add_policy_upload(
    conn: AsyncConnection,
    tenant_id: UUID,
    *,
    state: UploadState = UploadState.READY,
    purpose: UploadPurpose = UploadPurpose.POLICY_SOURCE,
    mime_type: str = "text/markdown",
    checksum: str | None = None,
    bucket: str = "nrc-test-policy",
    study_id: UUID | None = None,
) -> UUID:
    # A synthetic policy-source upload row as the media worker leaves it, written as the owner.
    # The caller switches the connection back to a login role afterwards.
    upload_id = uuid4()
    async with as_tenant_owner(conn, tenant_id):
        await conn.execute(
            insert(MediaUpload).values(
                id=upload_id,
                tenant_id=tenant_id,
                purpose=purpose,
                state=state,
                name="[SYNTHETIC] policy source",
                declared_mime_type=mime_type,
                declared_size_bytes=100,
                declared_checksum_sha256=checksum or upload_id.hex * 2,
                mime_type=mime_type if state is UploadState.READY else None,
                bucket=bucket,
                s3_key=f"policy/{upload_id}",
                s3_version_id="v1",
                upload_url_expires_at=datetime(2026, 11, 20, tzinfo=UTC),
                identifiable_person=IdentifiablePerson.UNKNOWN,
                study_id=study_id,
                created_by="policy-admin-a",
            )
        )
    return upload_id


@pytest.fixture
def sessions(connection: AsyncConnection) -> Sessions:
    # The worker's sessions: marketing_worker on the test's connection.
    return async_sessionmaker(
        bind=connection,
        join_transaction_mode="create_savepoint",
        expire_on_commit=False,
        sync_session_class=RoleSession,
        info={"role": WORKER_ROLE},
    )


@pytest.fixture
async def committed_worker_sessions(test_env: IntegrationEnv) -> AsyncIterator[Sessions]:
    # The worker's sessions on committed data (for tests that use committed_tenant).
    engine = create_engine(get_settings(), test_env.test_worker_database_url)
    try:
        yield create_sessionmaker(engine)
    finally:
        await engine.dispose()


async def store_validations(sessions: Sessions) -> int:
    # Validation is a worker job: one worker pass over the validation events stores the queued
    # runs. Generation events are left alone. Returns how many events ran.
    return await run_once(
        sessions,
        {},
        None,
        get_settings(),
        event_handlers={VALIDATION_RUN_EVENT: EVENT_HANDLERS[VALIDATION_RUN_EVENT]},
    )


@pytest.fixture
def seed_graph() -> bool:
    # Whether tenant A's bundle gets the synthetic graph. Modules that write their own graph rows
    # into that bundle override this with False.
    return True


@pytest.fixture
def graph() -> PolicyGraphImport:
    # The graph tenant A's bundle gets. By default every rule is checked by code, so a clean
    # brief passes and the generation gate opens (job, worker and media tests need that). Tests
    # of semantic rules, which can't pass until the model check exists, parametrize "graph" with
    # load_synthetic_graph().
    return deterministic_synthetic_graph()


def deterministic_synthetic_graph() -> PolicyGraphImport:
    # Every rule checked by code alone, so a clean brief can pass (the generation gate needs one).
    base = load_synthetic_graph()
    return base.model_copy(
        update={
            "rules": [
                rule.model_copy(update={"check_type": CheckType.DETERMINISTIC})
                for rule in base.rules
            ]
        }
    )


@pytest.fixture
async def world(connection: AsyncConnection, seed_graph: bool, graph: PolicyGraphImport) -> World:
    async with _session(connection) as seed:
        a = await seed_tenant(
            seed,
            name="Tenant A",
            members={"creator-a": ["creator"], "reviewer-a": ["reviewer"]},
            with_policy=True,
            display_names={"creator-a": CREATOR_A_NAME},
            with_graph=seed_graph,
            graph=graph,
        )
        b = await seed_tenant(
            seed, name="Tenant B", members={"creator-b": ["creator"]}, with_policy=False
        )
        await seed.commit()
    # From here on the connection behaves like the API's role: RLS applies, no owner rights.
    await as_role(connection, APP_ROLE)
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
        async with app_session(connection) as session:
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


@pytest.fixture
async def committed_tenant(
    test_env: IntegrationEnv, graph: PolicyGraphImport
) -> AsyncIterator[SeededTenant]:
    # Committed data (no shared test transaction), so parallel requests really race.
    engine = create_async_engine(
        test_env.test_migration_database_url.get_secret_value(), poolclass=NullPool
    )
    async with AsyncSession(engine, expire_on_commit=False) as db:
        seeded = await seed_tenant(
            db,
            name="Job race tenant",
            members={"creator-a": ["creator"]},
            with_policy=True,
            # The same graph fixture as world (by default every rule is checked by code).
            graph=graph,
        )
        await db.commit()
    try:
        yield seeded
    finally:
        async with AsyncSession(engine) as db:
            # jobs and assets reference each other: break the cycle before deleting.
            await db.execute(
                text("UPDATE jobs SET result_asset_id = NULL WHERE tenant_id = :t"),
                {"t": seeded.tenant_id},
            )
            for table in reversed(Base.metadata.sorted_tables):
                if table.name == "provider_spend":
                    # Not tenant data (one counter per provider).
                    continue
                column = table.c.id if table.name == "tenants" else table.c.tenant_id
                await db.execute(table.delete().where(column == seeded.tenant_id))
            await db.commit()
        await engine.dispose()


# ---- Helpers shared by test modules (no test module imports another).


JOBS = "/v1/marketing/creative-jobs"


async def _creative(
    client: httpx.AsyncClient,
    auth: AuthHeaders,
    tenant: SeededTenant,
    worker: Sessions,
    *,
    prompt: str = "Invite adults with high blood pressure to learn about the study.",
) -> str:
    # Creates a draft and validates its brief, running the validation job on the worker's
    # sessions: generation needs a stored run (NRC-7 gate).
    response = await client.post(
        "/v1/marketing/creatives",
        headers=auth("creator-a", tenant.tenant_id, **{"Idempotency-Key": uuid4().hex}),
        json={
            "study_id": str(tenant.study_id),
            "prompt": prompt,
            "language": "en",
            "channel": "facebook",
            "post_format": "feed_image",
        },
    )
    assert response.status_code == 201, response.text
    creative_id: str = response.json()["id"]
    validation = await client.post(
        "/v1/marketing/validations",
        headers=auth("creator-a", tenant.tenant_id, **{"Idempotency-Key": uuid4().hex}),
        json={"creative_id": creative_id, "creative_version": 1, "input_kind": "brief"},
    )
    assert validation.status_code == 202, validation.text
    await store_validations(worker)
    return creative_id


async def _post_job(
    client: httpx.AsyncClient,
    auth: AuthHeaders,
    world: World,
    creative_id: str,
    *,
    key: str | None = None,
    sub: str = "creator-a",
    version: int = 1,
) -> httpx.Response:
    headers = auth(sub, world.a.tenant_id, **{"Idempotency-Key": key or uuid4().hex})
    body = {"creative_id": creative_id, "creative_version": version, "kind": "image_generation"}
    return await client.post(JOBS, headers=headers, json=body)


class FakeSink:
    # Stands in for generated-image storage: a metadata row without any object.

    def __init__(self) -> None:
        self.stored: list[UUID] = []

    async def store(
        self, db: AsyncSession, job: Job, image: FetchedImage, provider_name: str, model: str
    ) -> UUID:
        assert image.content.startswith(b"\x89PNG")
        assert provider_name == "simulator"
        # A real row, because jobs.result_asset_id is a foreign key to assets.
        asset = Asset(
            tenant_id=job.tenant_id,
            creative_id=job.creative_id,
            creative_version=job.creative_version,
            job_id=job.id,
            origin="generated",
            bucket="fake",
            s3_key=f"fake/{uuid4()}.png",
            mime_type="image/png",
            width_px=1,
            height_px=1,
            size_bytes=len(image.content),
            checksum_sha256="0" * 64,
            created_by=job.created_by,
        )
        db.add(asset)
        await db.flush()
        self.stored.append(asset.id)
        return asset.id


async def _load(sessions: Sessions, world: World, job_id: UUID) -> tuple[Job, ReservationState]:
    async with sessions() as db:
        await bind_tenant(db, world.a.tenant_id)
        job = await db.get(Job, job_id)
        reservation = await db.scalar(
            select(CreditReservation.state).where(CreditReservation.job_id == job_id)
        )
    assert job is not None
    assert reservation is not None
    return job, reservation


async def _requeue(connection: AsyncConnection, world: World) -> None:
    # Make every outbox event due now, as if its backoff had passed.
    await connection.execute(
        text("SELECT set_config('app.tenant_id', :t, true)"), {"t": str(world.a.tenant_id)}
    )
    await connection.execute(
        text(
            "UPDATE outbox_events SET processed_at = NULL,"
            " available_at = clock_timestamp() - interval '1 second'"
        )
    )


async def _event(connection: AsyncConnection) -> Row[Any]:
    # The single outbox event of the test (RLS: tenant A is bound by now).
    return (
        await connection.execute(
            text(
                "SELECT processed_at, lease_owner, last_error_code, attempts, failed_attempts"
                " FROM outbox_events WHERE event_type = 'job.run'"
            )
        )
    ).one()


class ScriptedProvider(SimulatedImageProvider):
    # SIMULATED provider whose next calls can fail on cue.

    def __init__(self) -> None:
        super().__init__()
        # Raised by the next submits, after the request was recorded (accepted, response lost).
        self.submit_errors: list[Exception] = []
        self.status_error: ProviderError | None = None
        self.fetch_error: ProviderError | None = None
        # Reported in the COMPLETED status, as a provider stating its actual cost would.
        self.actual_cost_usd: Decimal | None = None
        self.estimates = 0
        self.before_estimate: Callable[[], Awaitable[None]] | None = None
        self.before_submit: Callable[[], Awaitable[None]] | None = None
        self.before_fetch: Callable[[], Awaitable[None]] | None = None
        # Status checks that still report the generation as in progress.
        self.pending_polls = 0
        # Reported in the COMPLETED status: when the provider deletes the output.
        self.outputs_expire_at: datetime | None = None

    async def estimate_cost_usd(self, request: ImageRequest) -> Decimal | None:
        self.estimates += 1
        if self.before_estimate is not None:
            await self.before_estimate()
        return await super().estimate_cost_usd(request)

    async def submit(self, request: ImageRequest) -> SubmitResult:
        if self.before_submit is not None:
            await self.before_submit()
        result = await super().submit(request)
        if self.submit_errors:
            raise self.submit_errors.pop(0)
        return result

    async def get_status(self, provider_request_id: str) -> StatusResult:
        if self.status_error is not None:
            raise self.status_error
        if self.pending_polls:
            self.pending_polls -= 1
            return StatusResult(provider_request_id, ProviderJobStatus.IN_PROGRESS)
        result = await super().get_status(provider_request_id)
        return dataclasses.replace(
            result, actual_cost_usd=self.actual_cost_usd, outputs_expire_at=self.outputs_expire_at
        )

    async def fetch_output(self, result: StatusResult) -> FetchedImage:
        if self.before_fetch is not None:
            await self.before_fetch()
        if self.fetch_error is not None:
            raise self.fetch_error
        return await super().fetch_output(result)


async def _spend(connection: AsyncConnection, provider: str = "simulator") -> tuple[int, int]:
    # (reserved, committed) micro-USD of the global spend counter; only the worker may read it.
    await as_role(connection, WORKER_ROLE)
    row = (
        await connection.execute(
            text(
                "SELECT reserved_usd_micros, committed_usd_micros FROM provider_spend"
                " WHERE provider = :p"
            ),
            {"p": provider},
        )
    ).one()
    return int(row.reserved_usd_micros), int(row.committed_usd_micros)


def _seaweedfs_reachable() -> bool:
    try:
        httpx.get(ENDPOINT, timeout=2)
    except httpx.HTTPError:
        return False
    return True


# Locally the tests skip without SeaweedFS; in CI (CI env var set) they run and so fail loudly.
needs_seaweedfs = pytest.mark.skipif(
    "CI" not in os.environ and not _seaweedfs_reachable(),
    reason=f"SeaweedFS is not reachable at {ENDPOINT}",
)


@pytest.fixture(scope="session")
def policy_bucket() -> str:
    # The versioned policy-source test bucket, created on first use. Tests that store objects
    # in it are skipped (needs_seaweedfs) when SeaweedFS is down, so then nothing is created.
    if _seaweedfs_reachable():
        client = boto3.client(
            "s3",
            endpoint_url=ENDPOINT,
            region_name="us-east-1",
            # Local-only dummy values that open the SeaweedFS container (docker/seaweedfs-s3.json).
            aws_access_key_id="localdev",
            aws_secret_access_key="localdev-password",  # noqa: S106
            config=BotoConfig(signature_version="s3v4", s3={"addressing_style": "path"}),
        )
        try:
            client.create_bucket(Bucket=POLICY_BUCKET)
        except ClientError as error:
            if error.response.get("Error", {}).get("Code") not in {
                "BucketAlreadyOwnedByYou",
                "BucketAlreadyExists",
            }:
                raise
        client.put_bucket_versioning(
            Bucket=POLICY_BUCKET, VersioningConfiguration={"Status": "Enabled"}
        )
    return POLICY_BUCKET
