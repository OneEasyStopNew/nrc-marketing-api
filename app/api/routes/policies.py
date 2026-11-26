# Policies: the active policy, and a bundle's rules with their sources (read-only).

from uuid import UUID

from fastapi import APIRouter

from app.api.deps import CreativeReader, DbSession
from app.schemas.common import ErrorBody
from app.schemas.policies import PolicyBundleRead
from app.schemas.policy_evidence import PolicyBundleRulesRead
from app.services import policies, policy_evidence

router = APIRouter(tags=["policies"])


@router.get(
    "/policies/active",
    summary="Active policy version and its rules (read-only)",
    responses={404: {"model": ErrorBody, "description": "The tenant has no policy yet."}},
)
async def get_active_policy(ctx: CreativeReader, db: DbSession) -> PolicyBundleRead:
    return await policies.get_active_policy(db, ctx)


@router.get(
    "/policy-bundles/{policy_bundle_id}/rules",
    summary="A policy bundle's rules with their sources, disclosures and exceptions",
    description="Approved rules of an active or retired bundle (the ones validation runs pin), "
    "for rule detail and evidence views. Shows rule ids, node keys and clause references.",
    responses={
        404: {
            "model": ErrorBody,
            "description": "No such bundle for the tenant, or it is not active or retired.",
        },
        409: {
            "model": ErrorBody,
            "description": "POLICY_CONTEXT_UNAVAILABLE: the bundle has no rule graph "
            "(details.reason GRAPH_MISSING).",
        },
    },
)
async def get_policy_bundle_rules(
    policy_bundle_id: UUID, ctx: CreativeReader, db: DbSession
) -> PolicyBundleRulesRead:
    return await policy_evidence.get_bundle_rules(db, ctx.tenant_id, policy_bundle_id)
