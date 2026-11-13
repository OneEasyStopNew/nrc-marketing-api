# The active policy: read-only rules that validation checks creatives against.

from fastapi import APIRouter

from app.api.deps import CreativeReader, DbSession
from app.schemas.common import ErrorBody
from app.schemas.policies import PolicyBundleRead
from app.services import policies

router = APIRouter(prefix="/policies", tags=["policies"])


@router.get(
    "/active",
    summary="Active policy version and its rules (read-only)",
    responses={404: {"model": ErrorBody, "description": "The tenant has no policy yet."}},
)
async def get_active_policy(ctx: CreativeReader, db: DbSession) -> PolicyBundleRead:
    return await policies.get_active_policy(db, ctx)
