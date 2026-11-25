# Connected social accounts and the Meta connect flow (NRC-13).

from typing import Annotated, Any
from uuid import UUID

from fastapi import APIRouter, Depends, Query, Request, status
from fastapi.responses import RedirectResponse

from app.api.deps import CreativeReader, DbSession, require_capability
from app.auth.access import Capability, TenantContext
from app.schemas.channel_accounts import (
    ChannelAccountList,
    ChannelAccountPage,
    ChannelAccountRead,
    MetaBindRequest,
    MetaConnectionRead,
    MetaConnectStart,
    MetaConnectStarted,
)
from app.schemas.common import ErrorBody
from app.services import channel_accounts
from app.services.channel_accounts import CALLBACK_PATH, MetaConfig, require_config
from app.services.meta_graph import MetaGraph

router = APIRouter(tags=["channel-accounts"])
callback_router = APIRouter(tags=["webhooks"])

# The tenant admin connects accounts. A separate capability would add an enum value the web
# client validates strictly, so it waits for a coordinated frontend change.
AccountAdmin = Annotated[TenantContext, Depends(require_capability(Capability.MEMBERSHIP_MANAGE))]


async def get_meta_config(request: Request) -> MetaConfig:
    config: MetaConfig | None = request.app.state.meta_config
    return require_config(config)


async def get_meta_graph(request: Request) -> MetaGraph:
    graph: MetaGraph = request.app.state.meta_graph
    return graph


MetaConfigDep = Annotated[MetaConfig, Depends(get_meta_config)]
MetaGraphDep = Annotated[MetaGraph, Depends(get_meta_graph)]
_NOT_CONFIGURED: dict[int | str, dict[str, Any]] = {
    501: {"model": ErrorBody, "description": "Meta is not configured here."}
}


@router.get(
    "/channel-accounts",
    summary="Connected Facebook Pages and Instagram accounts",
    description="Includes disconnected accounts (state DISCONNECTED) so history stays readable.",
    responses={404: {"model": ErrorBody, "description": "Unknown cursor."}},
)
async def list_channel_accounts(
    ctx: CreativeReader,
    db: DbSession,
    cursor: Annotated[UUID | None, Query(description="next_cursor from the last page.")] = None,
    limit: Annotated[int, Query(ge=1, le=100)] = 50,
) -> ChannelAccountPage:
    return await channel_accounts.list_accounts(db, ctx, after=cursor, limit=limit)


@router.post(
    "/integrations/meta/connect",
    status_code=status.HTTP_201_CREATED,
    summary="Start a Meta login to connect accounts",
    description="Send the browser to authorization_url. Meta returns it to this API, which "
    "redirects to return_path with ?meta_connection=<id>&meta_connect=select_accounts (or "
    "failed). Then read the connection and bind the accounts the user picks.",
    responses=_NOT_CONFIGURED,
)
async def start_meta_connect(
    body: MetaConnectStart, ctx: AccountAdmin, db: DbSession, config: MetaConfigDep
) -> MetaConnectStarted:
    return await channel_accounts.start_connect(db, ctx, config, body)


@router.get(
    "/integrations/meta/connections/{connection_id}",
    summary="Accounts a finished Meta login can bind",
    description="Only the user who started the login can read it.",
    responses={404: {"model": ErrorBody}},
)
async def get_meta_connection(
    connection_id: UUID, ctx: AccountAdmin, db: DbSession
) -> MetaConnectionRead:
    return await channel_accounts.get_connection(db, ctx, connection_id)


@router.post(
    "/integrations/meta/connections/{connection_id}/accounts",
    summary="Bind the accounts the user picked",
    description="Nothing is bound without this explicit choice. Each account must be one the "
    "login listed. A login binds once; start another to add more.",
    responses={
        404: {"model": ErrorBody},
        409: {"model": ErrorBody, "description": "Login closed, or account not offered."},
        503: {"model": ErrorBody, "description": "Meta did not answer."},
        **_NOT_CONFIGURED,
    },
)
async def bind_meta_accounts(
    connection_id: UUID,
    body: MetaBindRequest,
    ctx: AccountAdmin,
    db: DbSession,
    config: MetaConfigDep,
    graph: MetaGraphDep,
) -> ChannelAccountList:
    return await channel_accounts.bind_accounts(db, ctx, config, graph, connection_id, body)


@router.post(
    "/channel-accounts/{account_id}/reconnect",
    status_code=status.HTTP_201_CREATED,
    summary="Start a Meta login that refreshes one account",
    description="Same browser flow as connect; the callback refreshes this account directly and "
    "redirects with meta_connect=reconnected (or failed).",
    responses={404: {"model": ErrorBody}, **_NOT_CONFIGURED},
)
async def reconnect_channel_account(
    account_id: UUID,
    body: MetaConnectStart,
    ctx: AccountAdmin,
    db: DbSession,
    config: MetaConfigDep,
) -> MetaConnectStarted:
    return await channel_accounts.start_connect(
        db, ctx, config, body, reconnect_account_id=account_id
    )


@router.post(
    "/channel-accounts/{account_id}/check",
    summary="Ask Meta for the account's current access and permissions",
    responses={
        404: {"model": ErrorBody},
        503: {"model": ErrorBody, "description": "Meta did not answer."},
        **_NOT_CONFIGURED,
    },
)
async def check_channel_account(
    account_id: UUID,
    ctx: AccountAdmin,
    db: DbSession,
    config: MetaConfigDep,
    graph: MetaGraphDep,
) -> ChannelAccountRead:
    return await channel_accounts.check_account(db, ctx, config, graph, account_id)


@router.post(
    "/channel-accounts/{account_id}/disconnect",
    summary="Stop publishing to an account and forget its token",
    description="The account stays listed as DISCONNECTED. Posts already live are not removed.",
    responses={404: {"model": ErrorBody}},
)
async def disconnect_channel_account(
    account_id: UUID, ctx: AccountAdmin, db: DbSession
) -> ChannelAccountRead:
    return await channel_accounts.disconnect(db, ctx, account_id)


@callback_router.get(CALLBACK_PATH, include_in_schema=False)
async def meta_oauth_callback(
    db: DbSession,
    config: MetaConfigDep,
    graph: MetaGraphDep,
    state: Annotated[str | None, Query(max_length=200)] = None,
    code: Annotated[str | None, Query(max_length=2048)] = None,
    error: Annotated[str | None, Query(max_length=200)] = None,
) -> RedirectResponse:
    # Reached by the browser from Meta, without a Cognito token: the state is the credential.
    target = await channel_accounts.handle_callback(
        db, config, graph, state=state, code=code, error=error
    )
    response = RedirectResponse(target, status_code=status.HTTP_303_SEE_OTHER)
    # The code is single-use, but keep this URL out of caches and Referer headers anyway.
    response.headers["Cache-Control"] = "no-store"
    response.headers["Referrer-Policy"] = "no-referrer"
    return response
