# Policy sources and bundles (NRC-4): register sources, read clauses, build a bundle's graph,
# review it and move it through its lifecycle. Services enforce the exact capability of each
# write; the routes only keep out roles that hold no policy capability at all.

from typing import Annotated, Any, Final
from uuid import UUID

from fastapi import APIRouter, Query, Request, Response, status
from pydantic import AwareDatetime

from app.api.deps import DbSession, IfMatch, PolicyManager, PolicyReader, SettingsDep
from app.api.etags import etag, expected_version
from app.auth.access import ROLE_CAPABILITIES
from app.models.enums import BundleLifecycleState
from app.schemas.common import ErrorBody
from app.schemas.policy_admin import (
    BundleCreate,
    EdgeCreate,
    NodeCreate,
    NodeUpdate,
    PolicyAuditPage,
    PolicyBundleChecks,
    PolicyBundleDetail,
    PolicyBundlePage,
    PolicyClauseList,
    PolicyClauseRead,
    PolicyEdgeList,
    PolicyEdgeRead,
    PolicyNodeList,
    PolicyNodeRead,
    PolicyRuleList,
    PolicyRuleRead,
    PolicySourceCreate,
    PolicySourcePage,
    PolicySourceRead,
    PolicyWordingRead,
    ReviewDecisionRequest,
    RuleCreate,
    RuleUpdate,
    TransitionCommandRequest,
    WordingCreate,
)
from app.services import policy_bundles, policy_reads, policy_sources

router = APIRouter(tags=["policy"])

_NOT_FOUND: dict[str, Any] = {"model": ErrorBody, "description": "Not found for this tenant."}
_EDIT_ERRORS: dict[int | str, dict[str, Any]] = {
    404: _NOT_FOUND,
    409: {
        "model": ErrorBody,
        "description": "POLICY_BUNDLE_STATE (details.state, details.allowed), "
        "POLICY_ITEM_INVALID, or POLICY_LIMIT_REACHED when adding an item, or approving or "
        "editing a rejected one, in a bundle that holds POLICY_MAX_ITEMS_PER_BUNDLE nodes, edges "
        "and wordings that are not rejected (details.limit).",
    },
    412: {"model": ErrorBody, "description": "Stale If-Match; details.current_version."},
    428: {"model": ErrorBody, "description": "If-Match missing or malformed."},
}
_Limit = Annotated[int, Query(ge=1, le=50)]
_Before = Annotated[
    AwareDatetime | None, Query(description="next_cursor from the last page (older rows).")
]


def _edited(response: Response, revision: int) -> None:
    # An edit succeeds only at exactly this revision and bumps it once. Re-reading the bundle
    # instead could hand out a later editor's revision for a change this client never saw.
    response.headers["ETag"] = etag(revision + 1)


def _location(request: Request, name: str, **params: str) -> str:
    return str(request.app.url_path_for(name, **params))


# ---- Sources


@router.post(
    "/policy-sources",
    status_code=status.HTTP_201_CREATED,
    summary="Register a READY policy upload as a policy source",
    description="201 when a source was created, 200 with the existing source when the same file "
    "(checksum) or upload was registered before; a source whose extraction FAILED is queued "
    "again. Clauses are extracted in the background: poll the source until extraction_state is "
    "EXTRACTED or FAILED.",
    responses={
        200: {"model": PolicySourceRead, "description": "The source already existed."},
        404: _NOT_FOUND,
        409: {"model": ErrorBody, "description": "POLICY_SOURCE_UPLOAD_NOT_READY."},
    },
)
async def register_policy_source(
    body: PolicySourceCreate,
    ctx: PolicyManager,
    db: DbSession,
    request: Request,
    response: Response,
) -> PolicySourceRead:
    source, created = await policy_reads.register_source(db, ctx, body)
    if not created:
        response.status_code = status.HTTP_200_OK
    response.headers["Location"] = _location(request, "get_policy_source", source_id=str(source.id))
    return PolicySourceRead.model_validate(source)


@router.get("/policy-sources", summary="Policy sources in this tenant, newest first")
async def list_policy_sources(
    ctx: PolicyReader, db: DbSession, before: _Before = None, limit: _Limit = 20
) -> PolicySourcePage:
    return await policy_reads.list_sources(db, ctx, before=before, limit=limit)


@router.get(
    "/policy-sources/{source_id}",
    summary="One policy source and its extraction status",
    responses={404: _NOT_FOUND},
)
async def get_policy_source(source_id: UUID, ctx: PolicyReader, db: DbSession) -> PolicySourceRead:
    return PolicySourceRead.model_validate(await policy_sources.get_source(db, ctx, source_id))


@router.get(
    "/policy-sources/{source_id}/clauses",
    summary="The clauses extracted from a source, in reading order",
    responses={404: _NOT_FOUND},
)
async def list_policy_source_clauses(
    source_id: UUID, ctx: PolicyReader, db: DbSession
) -> PolicyClauseList:
    clauses = await policy_sources.list_source_clauses(db, ctx, source_id)
    return PolicyClauseList(items=[PolicyClauseRead.model_validate(c) for c in clauses])


# ---- Bundles


@router.post(
    "/policy-bundles",
    status_code=status.HTTP_201_CREATED,
    summary="Create a draft policy bundle, empty or copied from another bundle",
    responses={
        404: _NOT_FOUND,
        409: {
            "model": ErrorBody,
            "description": "POLICY_LIMIT_REACHED: the tenant already has "
            "POLICY_MAX_OPEN_BUNDLES_PER_TENANT bundles in DRAFT or REVIEW, or the bundle to "
            "copy holds more than POLICY_MAX_ITEMS_PER_BUNDLE items (details.limit).",
        },
    },
)
async def create_policy_bundle(
    body: BundleCreate,
    ctx: PolicyReader,
    db: DbSession,
    settings: SettingsDep,
    request: Request,
    response: Response,
) -> PolicyBundleDetail:
    bundle = await policy_bundles.create_bundle(
        db,
        ctx,
        body,
        settings.policy_max_open_bundles_per_tenant,
        settings.policy_max_items_per_bundle,
    )
    response.headers["ETag"] = etag(bundle.revision)
    response.headers["Location"] = _location(
        request, "get_policy_bundle", policy_bundle_id=str(bundle.id)
    )
    return await policy_reads.bundle_detail(db, ctx, bundle.id)


@router.get("/policy-bundles", summary="Policy bundles in this tenant, newest version first")
async def list_policy_bundles(
    ctx: PolicyReader,
    db: DbSession,
    state: Annotated[
        BundleLifecycleState | None, Query(description="Only bundles in this state.")
    ] = None,
    before_version: Annotated[
        int | None, Query(ge=1, description="next_cursor from the last page.")
    ] = None,
    limit: _Limit = 20,
) -> PolicyBundlePage:
    return await policy_reads.list_bundles(
        db, ctx, state=state, before_version=before_version, limit=limit
    )


@router.get(
    "/policy-bundles/{policy_bundle_id}",
    summary="One policy bundle with its state and revision",
    description="The ETag is the revision to send as If-Match when editing or moving the bundle.",
    responses={404: _NOT_FOUND},
)
async def get_policy_bundle(
    policy_bundle_id: UUID, ctx: PolicyReader, db: DbSession, response: Response
) -> PolicyBundleDetail:
    detail = await policy_reads.bundle_detail(db, ctx, policy_bundle_id)
    response.headers["ETag"] = etag(detail.revision)
    return detail


# The older GET /policy-bundles/{id}/rules (active or retired bundles, for every creative reader)
# keeps its path; this editor view lives beside it.
@router.get(
    "/policy-bundles/{policy_bundle_id}/rule-records",
    summary="A bundle's rules with their node, review state and wordings (editor view)",
    description="Proposed rules and wordings are left out unless include_proposals=true, which "
    "needs policy:manage.",
    responses={
        403: {"model": ErrorBody, "description": "include_proposals without policy:manage."},
        404: _NOT_FOUND,
    },
)
async def list_policy_bundle_rules(
    policy_bundle_id: UUID,
    ctx: PolicyReader,
    db: DbSession,
    include_proposals: Annotated[bool, Query()] = False,
) -> PolicyRuleList:
    rules = await policy_reads.list_rules(
        db, ctx, policy_bundle_id, include_proposals=include_proposals
    )
    return PolicyRuleList(items=rules)


@router.get(
    "/policy-bundles/{policy_bundle_id}/nodes",
    summary="Every node of a bundle with its review state and origin",
    responses={404: _NOT_FOUND},
)
async def list_policy_bundle_nodes(
    policy_bundle_id: UUID, ctx: PolicyReader, db: DbSession
) -> PolicyNodeList:
    return PolicyNodeList(items=await policy_reads.list_nodes(db, ctx, policy_bundle_id))


@router.get(
    "/policy-bundles/{policy_bundle_id}/edges",
    summary="Every edge of a bundle with its review state and origin",
    responses={404: _NOT_FOUND},
)
async def list_policy_bundle_edges(
    policy_bundle_id: UUID, ctx: PolicyReader, db: DbSession
) -> PolicyEdgeList:
    return PolicyEdgeList(items=await policy_reads.list_edges(db, ctx, policy_bundle_id))


@router.get(
    "/policy-bundles/{policy_bundle_id}/checks",
    summary="Issues that would stop the bundle from being approved or activated",
    responses={404: _NOT_FOUND},
)
async def get_policy_bundle_checks(
    policy_bundle_id: UUID, ctx: PolicyReader, db: DbSession, settings: SettingsDep
) -> PolicyBundleChecks:
    return await policy_reads.bundle_checks(
        db, ctx, policy_bundle_id, settings.policy_supported_languages
    )


@router.get(
    "/policy-bundles/{policy_bundle_id}/audit",
    summary="Who did what to a bundle, newest first",
    responses={404: _NOT_FOUND},
)
async def list_policy_bundle_audit(
    policy_bundle_id: UUID,
    ctx: PolicyReader,
    db: DbSession,
    before: _Before = None,
    limit: _Limit = 20,
) -> PolicyAuditPage:
    return await policy_reads.list_audit(db, ctx, policy_bundle_id, before=before, limit=limit)


# ---- Edits (policy:manage, If-Match = bundle revision)


@router.post(
    "/policy-bundles/{policy_bundle_id}/nodes",
    status_code=status.HTTP_201_CREATED,
    summary="Add a clause, disclosure, exception, interpretation or example node",
    responses=_EDIT_ERRORS,
)
async def add_policy_node(
    policy_bundle_id: UUID,
    body: NodeCreate,
    ctx: PolicyReader,
    db: DbSession,
    settings: SettingsDep,
    response: Response,
    if_match: IfMatch = None,
) -> PolicyNodeRead:
    revision = expected_version(if_match)
    node = await policy_bundles.add_node(
        db, ctx, policy_bundle_id, revision, body, settings.policy_max_items_per_bundle
    )
    _edited(response, revision)
    return PolicyNodeRead.model_validate(node)


@router.patch(
    "/policy-bundles/{policy_bundle_id}/nodes/{node_id}",
    summary="Edit a node; it goes back to PROPOSED for review",
    responses=_EDIT_ERRORS,
)
async def update_policy_node(
    policy_bundle_id: UUID,
    node_id: UUID,
    body: NodeUpdate,
    ctx: PolicyReader,
    db: DbSession,
    settings: SettingsDep,
    response: Response,
    if_match: IfMatch = None,
) -> PolicyNodeRead:
    revision = expected_version(if_match)
    node = await policy_bundles.update_node(
        db, ctx, policy_bundle_id, revision, node_id, body, settings.policy_max_items_per_bundle
    )
    _edited(response, revision)
    return PolicyNodeRead.model_validate(node)


@router.post(
    "/policy-bundles/{policy_bundle_id}/rules",
    status_code=status.HTTP_201_CREATED,
    summary="Add a rule together with its RULE node",
    responses=_EDIT_ERRORS,
)
async def add_policy_rule(
    policy_bundle_id: UUID,
    body: RuleCreate,
    ctx: PolicyReader,
    db: DbSession,
    settings: SettingsDep,
    response: Response,
    if_match: IfMatch = None,
) -> PolicyRuleRead:
    revision = expected_version(if_match)
    rule = await policy_bundles.add_rule(
        db, ctx, policy_bundle_id, revision, body, settings.policy_max_items_per_bundle
    )
    _edited(response, revision)
    return await policy_reads.get_rule(db, ctx, policy_bundle_id, rule.id)


@router.patch(
    "/policy-bundles/{policy_bundle_id}/rules/{rule_id}",
    summary="Edit a rule; its node goes back to PROPOSED for review",
    description="rule_id in the path is the rule record id (PolicyRuleRead.id).",
    responses=_EDIT_ERRORS,
)
async def update_policy_rule(
    policy_bundle_id: UUID,
    rule_id: UUID,
    body: RuleUpdate,
    ctx: PolicyReader,
    db: DbSession,
    settings: SettingsDep,
    response: Response,
    if_match: IfMatch = None,
) -> PolicyRuleRead:
    revision = expected_version(if_match)
    rule = await policy_bundles.update_rule(
        db, ctx, policy_bundle_id, revision, rule_id, body, settings.policy_max_items_per_bundle
    )
    _edited(response, revision)
    return await policy_reads.get_rule(db, ctx, policy_bundle_id, rule.id)


@router.post(
    "/policy-bundles/{policy_bundle_id}/edges",
    status_code=status.HTTP_201_CREATED,
    summary="Link two nodes of the bundle",
    responses=_EDIT_ERRORS,
)
async def add_policy_edge(
    policy_bundle_id: UUID,
    body: EdgeCreate,
    ctx: PolicyReader,
    db: DbSession,
    settings: SettingsDep,
    response: Response,
    if_match: IfMatch = None,
) -> PolicyEdgeRead:
    revision = expected_version(if_match)
    edge = await policy_bundles.add_edge(
        db, ctx, policy_bundle_id, revision, body, settings.policy_max_items_per_bundle
    )
    _edited(response, revision)
    return PolicyEdgeRead.model_validate(edge)


@router.post(
    "/policy-bundles/{policy_bundle_id}/rules/{rule_id}/wordings",
    status_code=status.HTTP_201_CREATED,
    summary="Add exact required wording or a prohibited phrase to a rule",
    description="rule_id in the path is the rule record id. The text is kept exactly as sent.",
    responses=_EDIT_ERRORS,
)
async def add_policy_wording(
    policy_bundle_id: UUID,
    rule_id: UUID,
    body: WordingCreate,
    ctx: PolicyReader,
    db: DbSession,
    settings: SettingsDep,
    response: Response,
    if_match: IfMatch = None,
) -> PolicyWordingRead:
    revision = expected_version(if_match)
    wording = await policy_bundles.add_wording(
        db, ctx, policy_bundle_id, revision, rule_id, body, settings.policy_max_items_per_bundle
    )
    _edited(response, revision)
    return PolicyWordingRead.model_validate(wording)


@router.post(
    "/policy-bundles/{policy_bundle_id}/review-decisions",
    summary="Approve or reject one node, edge or wording (draft or in review)",
    responses=_EDIT_ERRORS,
)
async def decide_policy_item(
    policy_bundle_id: UUID,
    body: ReviewDecisionRequest,
    ctx: PolicyReader,
    db: DbSession,
    settings: SettingsDep,
    response: Response,
    if_match: IfMatch = None,
) -> PolicyBundleDetail:
    await policy_bundles.review_item(
        db,
        ctx,
        policy_bundle_id,
        expected_version(if_match),
        body.item_kind,
        body.item_id,
        body,
        settings.policy_max_items_per_bundle,
    )
    detail = await policy_reads.bundle_detail(db, ctx, policy_bundle_id)
    response.headers["ETag"] = etag(detail.revision)
    return detail


# Built from the service's transition table and the role map, so the contract can't drift from them.
_COMMAND_CAPABILITIES: Final = "; ".join(
    f"{command.value} needs "
    + " or ".join(
        f"{capability.value} (roles: "
        + ", ".join(
            sorted(role.value for role, held in ROLE_CAPABILITIES.items() if capability in held)
        )
        + ")"
        for capability in sorted(step.capabilities)
    )
    for command, step in policy_bundles.TRANSITIONS.items()
)


@router.post(
    "/policy-bundles/{policy_bundle_id}/transitions",
    summary="Move a bundle: submit, return_to_draft, approve, reject, build, activate, rollback "
    "or retire",
    description="Each command needs its own capability. Approval and activation are separate "
    f"(policy bundle D-16): {_COMMAND_CAPABILITIES}. A caller without it gets 403. "
    "409 POLICY_BUNDLE_STATE carries details.state and details.allowed; "
    "409 POLICY_BUNDLE_HAS_ISSUES carries details.issues. The reason is required for reject, "
    "activate, rollback and retire.",
    responses=_EDIT_ERRORS,
)
async def transition_policy_bundle(
    policy_bundle_id: UUID,
    body: TransitionCommandRequest,
    ctx: PolicyReader,
    db: DbSession,
    settings: SettingsDep,
    response: Response,
    if_match: IfMatch = None,
) -> PolicyBundleDetail:
    await policy_bundles.transition(
        db,
        ctx,
        policy_bundle_id,
        expected_version(if_match),
        body.command,
        body,
        settings.policy_supported_languages,
    )
    detail = await policy_reads.bundle_detail(db, ctx, policy_bundle_id)
    response.headers["ETag"] = etag(detail.revision)
    return detail
