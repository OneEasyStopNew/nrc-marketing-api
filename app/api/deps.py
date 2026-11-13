# Request dependencies: database session, caller identity, tenant context and command headers.

from collections.abc import AsyncIterator, Awaitable, Callable
from typing import Annotated
from uuid import UUID

from fastapi import Depends, Header, Request
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.auth.access import Capability, TenantContext
from app.auth.tokens import Principal, SigningKeyProvider, verify_access_token
from app.core.config import Settings, get_settings
from app.core.errors import (
    AuthenticationError,
    DependencyUnavailableError,
    PermissionDeniedError,
)
from app.services import tenancy
from app.services.storage import ObjectStorage


async def get_db(request: Request) -> AsyncIterator[AsyncSession]:
    sessionmaker: async_sessionmaker[AsyncSession] = request.app.state.sessionmaker
    async with sessionmaker() as session:
        yield session


async def get_signing_keys(request: Request) -> SigningKeyProvider:
    keys: SigningKeyProvider = request.app.state.signing_keys
    return keys


async def get_storage(request: Request) -> ObjectStorage:
    storage: ObjectStorage | None = request.app.state.storage
    if storage is None:
        raise DependencyUnavailableError("Image storage is not configured.")
    return storage


DbSession = Annotated[AsyncSession, Depends(get_db)]
StorageDep = Annotated[ObjectStorage, Depends(get_storage)]
SettingsDep = Annotated[Settings, Depends(get_settings)]

_bearer = HTTPBearer(auto_error=False, description="Cognito access token.")


async def get_principal(
    credentials: Annotated[HTTPAuthorizationCredentials | None, Depends(_bearer)],
    keys: Annotated[SigningKeyProvider, Depends(get_signing_keys)],
    settings: SettingsDep,
) -> Principal:
    if credentials is None:
        raise AuthenticationError("Send a Cognito access token as a Bearer token.")
    return await verify_access_token(
        credentials.credentials,
        keys,
        issuer=settings.cognito_issuer,
        client_ids=frozenset(settings.cognito_client_ids),
        required_scope=settings.required_scope,
    )


CurrentPrincipal = Annotated[Principal, Depends(get_principal)]


async def get_tenant_context(
    principal: CurrentPrincipal,
    db: DbSession,
    tenant_id: Annotated[
        UUID,
        Header(
            alias="X-Tenant-Id",
            description="Tenant to act in. Access is checked against your current membership.",
        ),
    ],
) -> TenantContext:
    return await tenancy.load_tenant_context(db, principal, tenant_id)


TenantCtx = Annotated[TenantContext, Depends(get_tenant_context)]


def require_capability(capability: Capability) -> Callable[..., Awaitable[TenantContext]]:
    async def check(ctx: TenantCtx) -> TenantContext:
        if capability not in ctx.capabilities:
            raise PermissionDeniedError(f"Your role does not allow {capability.value}.")
        return ctx

    return check


CreativeReader = Annotated[TenantContext, Depends(require_capability(Capability.CREATIVE_READ))]
CreativeWriter = Annotated[TenantContext, Depends(require_capability(Capability.CREATIVE_WRITE))]
ValidationRunner = Annotated[TenantContext, Depends(require_capability(Capability.VALIDATION_RUN))]
GenerationRunner = Annotated[TenantContext, Depends(require_capability(Capability.GENERATION_RUN))]

IdempotencyKey = Annotated[
    str,
    Header(
        alias="Idempotency-Key",
        min_length=8,
        max_length=128,
        pattern=r"^[A-Za-z0-9_-]+$",
        description="New value per intended command; resend the same value only when retrying it.",
    ),
]
IfMatch = Annotated[
    str | None,
    Header(alias="If-Match", description='Quoted version you edited, e.g. "3".'),
]
