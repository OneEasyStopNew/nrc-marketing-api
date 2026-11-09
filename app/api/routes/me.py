# Current user and tenant context.

from fastapi import APIRouter

from app.api.deps import CurrentPrincipal, DbSession, TenantCtx
from app.schemas.tenancy import ContextRead, MeRead
from app.services import tenancy

router = APIRouter(tags=["context"])


@router.get("/me", summary="Current user and the tenants they belong to")
async def get_me(principal: CurrentPrincipal, db: DbSession) -> MeRead:
    return await tenancy.get_me(db, principal)


@router.get("/context", summary="Roles, permitted studies and supported formats in one tenant")
async def get_context(ctx: TenantCtx, db: DbSession) -> ContextRead:
    return await tenancy.get_context(db, ctx)
