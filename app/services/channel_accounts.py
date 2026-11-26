# Connecting Facebook Pages and Instagram professional accounts through Meta login (NRC-13).

import hashlib
import logging
import secrets
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from types import MappingProxyType
from typing import Final
from urllib.parse import urlencode
from uuid import UUID, uuid4

from pydantic import SecretStr
from sqlalchemy import select, tuple_
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.ext.asyncio import AsyncSession

from app.auth.access import TenantContext
from app.core.config import Settings
from app.core.crypto import TokenCipher, TokenDecryptError
from app.core.errors import (
    AccountNotOfferedError,
    ConnectSessionClosedError,
    DependencyUnavailableError,
    FeatureNotAvailableError,
    NotFoundError,
)
from app.db.session import bind_tenant
from app.models import ChannelAccount, MetaConnectSession
from app.models.channel_accounts import AccountChannel, AccountState, ConnectStatus
from app.models.enums import PostFormat
from app.schemas.channel_accounts import (
    AccountSelection,
    ChannelAccountList,
    ChannelAccountPage,
    ChannelAccountRead,
    ChannelCapabilities,
    ConnectCandidate,
    MetaBindRequest,
    MetaConnectionRead,
    MetaConnectStart,
    MetaConnectStarted,
)
from app.services.meta_graph import (
    FACEBOOK_SCOPES,
    INSTAGRAM_SCOPES,
    PUBLISH_TASK,
    REQUESTED_SCOPES,
    MetaGraph,
    MetaGraphError,
    MetaPage,
)

logger = logging.getLogger(__name__)

# Bump whenever a value below changes; publication manifests record the version they relied on.
CAPABILITY_VERSION: Final = "meta-v26.0-2026-11-25"
# Sources: Meta's Graph API Page feed/photos and IG User Media references (Nov 2026). Values
# Meta does not state are null rather than guessed. Publishing itself is NRC-14 (Facebook) and
# NRC-15 (Instagram); Instagram has no post format in this app yet.
CAPABILITIES: Final = MappingProxyType(
    {
        AccountChannel.FACEBOOK: ChannelCapabilities(
            version=CAPABILITY_VERSION,
            text_only=True,
            requires_media=False,
            image=True,
            video=False,
            formats=[PostFormat.FEED_IMAGE],
            image_mime_types=["image/jpeg", "image/png"],
            max_image_bytes=None,
            min_aspect_ratio=None,
            max_aspect_ratio=None,
            caption_max_chars=None,
            schedule=True,
            delete=False,
        ),
        AccountChannel.INSTAGRAM: ChannelCapabilities(
            version=CAPABILITY_VERSION,
            text_only=False,
            requires_media=True,
            image=True,
            video=False,
            formats=[],
            image_mime_types=["image/jpeg"],
            max_image_bytes=8 * 1024 * 1024,
            min_aspect_ratio="4:5",
            max_aspect_ratio="1.91:1",
            caption_max_chars=2200,
            schedule=False,
            delete=False,
        ),
    }
)
SCOPES_BY_CHANNEL: Final = MappingProxyType(
    {AccountChannel.FACEBOOK: FACEBOOK_SCOPES, AccountChannel.INSTAGRAM: INSTAGRAM_SCOPES}
)
PUBLISHABLE: Final = frozenset({AccountState.CONNECTED, AccountState.TEST_ONLY})
TASK_MISSING: Final = "page_task:CREATE_CONTENT"
CALLBACK_PATH: Final = "/webhooks/meta/oauth/callback"


@dataclass(frozen=True, slots=True)
class MetaConfig:
    # Everything connecting needs, present only when every setting is filled in.
    app_id: str
    redirect_uri: str
    login_base_url: str
    version: str
    web_app_url: str
    ttl: timedelta
    mode: str
    cipher: TokenCipher


def meta_config(settings: Settings) -> MetaConfig | None:
    if not (
        settings.meta_app_id
        and settings.meta_app_secret
        and settings.meta_oauth_redirect_uri
        and settings.web_app_url
        and settings.token_encryption_key
    ):
        return None
    return MetaConfig(
        app_id=settings.meta_app_id,
        redirect_uri=str(settings.meta_oauth_redirect_uri),
        login_base_url=str(settings.meta_login_base_url).rstrip("/"),
        version=settings.meta_graph_version,
        web_app_url=str(settings.web_app_url).rstrip("/"),
        ttl=timedelta(seconds=settings.meta_connect_ttl_seconds),
        mode=settings.meta_app_mode,
        cipher=TokenCipher(settings.token_encryption_key),
    )


def require_config(config: MetaConfig | None) -> MetaConfig:
    if config is None:
        raise FeatureNotAvailableError("Connecting Meta accounts is not configured here yet.")
    return config


def _hash(secret: str) -> str:
    return hashlib.sha256(secret.encode()).hexdigest()


def _now() -> datetime:
    return datetime.now(UTC)


def _session_context(tenant_id: UUID, session_id: UUID) -> str:
    return f"meta-connect:{tenant_id}:{session_id}"


def _account_context(tenant_id: UUID, account_id: UUID) -> str:
    return f"channel-account:{tenant_id}:{account_id}"


def derive_state(missing: Iterable[str], mode: str) -> tuple[AccountState, str | None]:
    missing = sorted(missing)
    if missing:
        return (
            AccountState.PERMISSION_MISSING,
            f"Meta did not grant: {', '.join(missing)}. Reconnect and allow them; a Page admin "
            "must give you the Create content task.",
        )
    if mode == "development":
        return (
            AccountState.TEST_ONLY,
            "The Meta app is in development mode: only accounts with a role on the app can "
            "publish.",
        )
    if mode == "in_review":
        return (
            AccountState.PENDING_APP_REVIEW,
            "Meta App Review for publishing permissions is pending; publishing is blocked "
            "until Meta approves it.",
        )
    return AccountState.CONNECTED, None


def _missing(channel: AccountChannel, page: MetaPage, granted: frozenset[str]) -> list[str]:
    missing = [scope for scope in SCOPES_BY_CHANNEL[channel] if scope not in granted]
    if PUBLISH_TASK not in page.tasks:
        missing.append(TASK_MISSING)
    return missing


def _candidates(pages: list[MetaPage], granted: frozenset[str]) -> list[dict[str, object]]:
    # Safe fields only: this list is stored and shown to the user.
    found: list[dict[str, object]] = []
    for page in pages:
        found.append(
            {
                "channel": AccountChannel.FACEBOOK.value,
                "provider_account_id": page.id,
                "name": page.name,
                "handle": None,
                "page_id": page.id,
                "missing_permissions": _missing(AccountChannel.FACEBOOK, page, granted),
            }
        )
        if page.instagram is not None:
            found.append(
                {
                    "channel": AccountChannel.INSTAGRAM.value,
                    "provider_account_id": page.instagram.id,
                    "name": page.instagram.name or page.instagram.username or page.name,
                    "handle": page.instagram.username,
                    "page_id": page.id,
                    "missing_permissions": _missing(AccountChannel.INSTAGRAM, page, granted),
                }
            )
    return found


def to_read(account: ChannelAccount) -> ChannelAccountRead:
    return ChannelAccountRead(
        id=account.id,
        channel=account.channel,
        provider_account_id=account.provider_account_id,
        name=account.name,
        handle=account.handle,
        state=account.state,
        state_reason=account.state_reason,
        can_publish=account.state in PUBLISHABLE,
        permissions=sorted(account.granted_scopes),
        missing_permissions=list(account.missing_permissions),
        connected_by=account.connected_by,
        connected_at=account.connected_at,
        expires_at=account.token_expires_at,
        last_checked_at=account.last_checked_at,
        disconnected_at=account.disconnected_at,
        capabilities=CAPABILITIES[account.channel],
    )


async def list_accounts(
    db: AsyncSession, ctx: TenantContext, *, after: UUID | None, limit: int
) -> ChannelAccountPage:
    order = (ChannelAccount.channel, ChannelAccount.name, ChannelAccount.id)
    query = select(ChannelAccount).where(ChannelAccount.tenant_id == ctx.tenant_id)
    if after is not None:
        # Keyset on (channel, name, id); an unknown cursor is a 404 rather than a silent restart.
        last = await _account(db, ctx, after)
        query = query.where(tuple_(*order) > tuple_(last.channel, last.name, last.id))
    rows = list(await db.scalars(query.order_by(*order).limit(limit + 1)))
    items = [to_read(row) for row in rows[:limit]]
    return ChannelAccountPage(items=items, next_cursor=items[-1].id if len(rows) > limit else None)


async def _account(
    db: AsyncSession, ctx: TenantContext, account_id: UUID, *, lock: bool = False
) -> ChannelAccount:
    query = select(ChannelAccount).where(
        ChannelAccount.tenant_id == ctx.tenant_id, ChannelAccount.id == account_id
    )
    account = await db.scalar(query.with_for_update() if lock else query)
    if account is None:
        raise NotFoundError("Account not found.")
    return account


async def start_connect(
    db: AsyncSession,
    ctx: TenantContext,
    config: MetaConfig,
    body: MetaConnectStart,
    *,
    reconnect_account_id: UUID | None = None,
) -> MetaConnectStarted:
    if reconnect_account_id is not None:
        await _account(db, ctx, reconnect_account_id)
    secret = secrets.token_urlsafe(32)
    session = MetaConnectSession(
        tenant_id=ctx.tenant_id,
        state_hash=_hash(secret),
        initiated_by=ctx.principal.sub,
        reconnect_account_id=reconnect_account_id,
        return_path=body.return_path,
        status=ConnectStatus.WAITING_FOR_META,
        expires_at=_now() + config.ttl,
    )
    db.add(session)
    await db.flush()
    query = urlencode(
        {
            "client_id": config.app_id,
            "redirect_uri": config.redirect_uri,
            # The tenant travels in the state so the callback can scope its lookup; the secret
            # part is what proves the request, and only its hash is stored.
            "state": f"{ctx.tenant_id.hex}.{secret}",
            "scope": ",".join(REQUESTED_SCOPES),
            "response_type": "code",
            # Ask again for anything declined last time (reconnect after a missing permission).
            "auth_type": "rerequest",
        }
    )
    await db.commit()
    return MetaConnectStarted(
        connection_id=session.id,
        authorization_url=f"{config.login_base_url}/{config.version}/dialog/oauth?{query}",
        expires_at=session.expires_at,
    )


def _landing(config: MetaConfig, path: str, **params: str) -> str:
    return f"{config.web_app_url}{path}?{urlencode(params)}"


async def handle_callback(
    db: AsyncSession,
    config: MetaConfig,
    graph: MetaGraph,
    *,
    state: str | None,
    code: str | None,
    error: str | None,
) -> str:
    """Finish the Meta side of a login and return where to send the browser.

    The answer never carries a token, only the connection id and an outcome. A state that is
    unknown, already used or malformed gets a generic failure and changes nothing.
    """
    tenant_hex, _, secret = (state or "").partition(".")
    try:
        tenant_id = UUID(hex=tenant_hex)
    except ValueError:
        return _landing(config, "/accounts", meta_connect="failed", reason="invalid_state")
    await bind_tenant(db, tenant_id)
    session = await db.scalar(
        select(MetaConnectSession)
        .where(MetaConnectSession.state_hash == _hash(secret))
        .with_for_update()
    )
    if session is None or not secret or session.status is not ConnectStatus.WAITING_FOR_META:
        await db.rollback()
        return _landing(config, "/accounts", meta_connect="failed", reason="invalid_state")

    session.callback_at = _now()
    outcome = await _finish_login(db, config, graph, session, code=code, error=error)
    await db.commit()
    return _landing(
        config, session.return_path, meta_connection=str(session.id), meta_connect=outcome
    )


async def _finish_login(
    db: AsyncSession,
    config: MetaConfig,
    graph: MetaGraph,
    session: MetaConnectSession,
    *,
    code: str | None,
    error: str | None,
) -> str:
    def fail(reason: str) -> str:
        session.status = ConnectStatus.FAILED
        session.error_code = reason
        return "failed"

    if session.expires_at <= _now():
        return fail("state_expired")
    if error is not None or not code:
        return fail("login_cancelled")
    try:
        short = await graph.exchange_code(code, config.redirect_uri)
        user_token = (await graph.long_lived_token(short.access_token)).access_token
        session.meta_user_id = await graph.me(user_token)
        granted = await graph.granted_scopes(user_token)
        pages = await graph.pages(user_token)
    except MetaGraphError as exc:
        logger.warning("meta login failed: code=%s subcode=%s", exc.code, exc.subcode)
        return fail("meta_error")
    if not pages:
        return fail("no_pages")

    session.granted_scopes = sorted(granted)
    session.candidates = _candidates(pages, granted)
    if session.reconnect_account_id is not None:
        return await _refresh_reconnected(db, config, session, pages, granted)
    session.user_token_encrypted = config.cipher.encrypt(
        user_token.get_secret_value(), context=_session_context(session.tenant_id, session.id)
    )
    session.status = ConnectStatus.AWAITING_SELECTION
    # The user picks accounts after landing back in the app: give them the full window again.
    session.expires_at = _now() + config.ttl
    return "select_accounts"


async def _refresh_reconnected(
    db: AsyncSession,
    config: MetaConfig,
    session: MetaConnectSession,
    pages: list[MetaPage],
    granted: frozenset[str],
) -> str:
    account = await db.scalar(
        select(ChannelAccount)
        .where(
            ChannelAccount.tenant_id == session.tenant_id,
            ChannelAccount.id == session.reconnect_account_id,
        )
        .with_for_update()
    )
    page = next((p for p in pages if account is not None and p.id == account.page_id), None)
    if (
        account is None
        or page is None
        or (
            account.channel is AccountChannel.INSTAGRAM
            and (page.instagram is None or page.instagram.id != account.provider_account_id)
        )
    ):
        session.status = ConnectStatus.FAILED
        session.error_code = "account_not_offered"
        return "failed"
    _apply_page(config, account, page, granted, session.initiated_by)
    session.status = ConnectStatus.COMPLETED
    session.completed_at = _now()
    return "reconnected"


def _apply_page(
    config: MetaConfig,
    account: ChannelAccount,
    page: MetaPage,
    granted: frozenset[str],
    user_sub: str,
) -> None:
    missing = _missing(account.channel, page, granted)
    account.state, account.state_reason = derive_state(missing, config.mode)
    account.granted_scopes = sorted(granted & set(SCOPES_BY_CHANNEL[account.channel]))
    account.missing_permissions = missing
    account.page_id = page.id
    if account.channel is AccountChannel.INSTAGRAM and page.instagram is not None:
        account.name = page.instagram.name or page.instagram.username or page.name
        account.handle = page.instagram.username
    else:
        account.name = page.name
    # Page tokens made from a long-lived user token have no fixed expiry; check_account() asks
    # Meta for the real status.
    account.token_encrypted = config.cipher.encrypt(
        page.access_token.get_secret_value(),
        context=_account_context(account.tenant_id, account.id),
    )
    account.token_expires_at = None
    account.capability_version = CAPABILITY_VERSION
    account.connected_by = user_sub
    account.connected_at = _now()
    account.last_checked_at = _now()
    account.disconnected_at = None
    account.disconnected_by = None
    account.updated_at = _now()


async def _own_session(
    db: AsyncSession, ctx: TenantContext, connection_id: UUID, *, lock: bool = False
) -> MetaConnectSession:
    # Only the person who started a login may see or use what it found.
    query = select(MetaConnectSession).where(
        MetaConnectSession.tenant_id == ctx.tenant_id,
        MetaConnectSession.id == connection_id,
        MetaConnectSession.initiated_by == ctx.principal.sub,
    )
    session = await db.scalar(query.with_for_update() if lock else query)
    if session is None:
        raise NotFoundError("Connection not found.")
    return session


async def get_connection(
    db: AsyncSession, ctx: TenantContext, connection_id: UUID
) -> MetaConnectionRead:
    session = await _own_session(db, ctx, connection_id)
    rows = await db.execute(
        select(ChannelAccount.channel, ChannelAccount.provider_account_id).where(
            ChannelAccount.tenant_id == ctx.tenant_id,
            ChannelAccount.state != AccountState.DISCONNECTED,
        )
    )
    connected = {(channel, account) for channel, account in rows}
    candidates = [
        ConnectCandidate.model_validate(
            {
                **item,
                "can_publish": not item["missing_permissions"],
                "already_connected": (
                    AccountChannel(str(item["channel"])),
                    item["provider_account_id"],
                )
                in connected,
            }
        )
        for item in (session.candidates or [])
    ]
    return MetaConnectionRead(
        connection_id=session.id,
        status=session.status,
        error_code=session.error_code,
        expires_at=session.expires_at,
        candidates=candidates if session.status is ConnectStatus.AWAITING_SELECTION else [],
    )


async def bind_accounts(
    db: AsyncSession,
    ctx: TenantContext,
    config: MetaConfig,
    graph: MetaGraph,
    connection_id: UUID,
    body: MetaBindRequest,
) -> ChannelAccountList:
    """Bind the accounts the user picked. Only accounts this login listed can be bound."""
    session = await _own_session(db, ctx, connection_id, lock=True)
    if session.status is not ConnectStatus.AWAITING_SELECTION:
        raise ConnectSessionClosedError(details={"status": session.status.value})
    if session.expires_at <= _now() or session.user_token_encrypted is None:
        raise ConnectSessionClosedError(details={"status": "expired"})
    offered = {
        (str(item["channel"]), str(item["provider_account_id"]))
        for item in (session.candidates or [])
    }
    selections = _unique(body.accounts)
    for pick in selections:
        if (pick.channel.value, pick.provider_account_id) not in offered:
            raise AccountNotOfferedError(
                details={"channel": pick.channel.value, "account": pick.provider_account_id}
            )

    try:
        user_token = SecretStr(
            config.cipher.decrypt(
                session.user_token_encrypted,
                context=_session_context(session.tenant_id, session.id),
            )
        )
    except TokenDecryptError:
        raise ConnectSessionClosedError(details={"status": "expired"}) from None
    try:
        # Fresh Page tokens and permissions: nothing from the callback is trusted for binding.
        granted = await graph.granted_scopes(user_token)
        pages = {page.id: page for page in await graph.pages(user_token)}
    except MetaGraphError as exc:
        logger.warning("meta bind failed: code=%s subcode=%s", exc.code, exc.subcode)
        raise DependencyUnavailableError("Meta did not answer; try again.") from None

    bound: list[ChannelAccount] = []
    for pick in selections:
        page = _page_for(pages, pick)
        if page is None:
            # Listed at login but gone now (access removed in between).
            raise AccountNotOfferedError(
                details={"channel": pick.channel.value, "account": pick.provider_account_id}
            )
        account = await _claim_account(db, ctx, pick, page)
        _apply_page(config, account, page, granted, ctx.principal.sub)
        bound.append(account)

    session.status = ConnectStatus.COMPLETED
    session.completed_at = _now()
    session.user_token_encrypted = None
    await db.commit()
    return ChannelAccountList(items=[to_read(account) for account in bound])


async def _claim_account(
    db: AsyncSession, ctx: TenantContext, pick: AccountSelection, page: MetaPage
) -> ChannelAccount:
    # Insert-if-absent, then lock: two admins binding the same account at once both end up on
    # one row instead of one of them hitting the unique constraint. The placeholder values are
    # overwritten by _apply_page() before commit.
    now = _now()
    await db.execute(
        pg_insert(ChannelAccount)
        .values(
            id=uuid4(),
            tenant_id=ctx.tenant_id,
            channel=pick.channel,
            provider_account_id=pick.provider_account_id,
            page_id=page.id,
            name=page.name,
            state=AccountState.DISCONNECTED,
            capability_version=CAPABILITY_VERSION,
            connected_by=ctx.principal.sub,
            connected_at=now,
        )
        .on_conflict_do_nothing(index_elements=["tenant_id", "channel", "provider_account_id"])
    )
    return (
        await db.scalars(
            select(ChannelAccount)
            .where(
                ChannelAccount.tenant_id == ctx.tenant_id,
                ChannelAccount.channel == pick.channel,
                ChannelAccount.provider_account_id == pick.provider_account_id,
            )
            .with_for_update()
            .execution_options(populate_existing=True)
        )
    ).one()


def _unique(selections: list[AccountSelection]) -> list[AccountSelection]:
    seen: dict[tuple[AccountChannel, str], AccountSelection] = {}
    for pick in selections:
        seen.setdefault((pick.channel, pick.provider_account_id), pick)
    return list(seen.values())


def _page_for(pages: dict[str, MetaPage], pick: AccountSelection) -> MetaPage | None:
    if pick.channel is AccountChannel.FACEBOOK:
        return pages.get(pick.provider_account_id)
    return next(
        (
            page
            for page in pages.values()
            if page.instagram is not None and page.instagram.id == pick.provider_account_id
        ),
        None,
    )


async def disconnect(db: AsyncSession, ctx: TenantContext, account_id: UUID) -> ChannelAccountRead:
    """Stop all new publishing to the account and forget its token. Idempotent.

    Posts already live stay live; this does not and cannot remove them. Publications that are
    queued but not sent will be paused by the dispatcher's state check (NRC-14).
    """
    account = await _account(db, ctx, account_id, lock=True)
    if account.state is not AccountState.DISCONNECTED:
        account.state = AccountState.DISCONNECTED
        account.state_reason = "Disconnected in Marketing. Connect it again to publish."
        account.token_encrypted = None
        account.token_expires_at = None
        account.disconnected_by = ctx.principal.sub
        account.disconnected_at = _now()
        account.updated_at = _now()
    await db.commit()
    return to_read(account)


async def check_account(
    db: AsyncSession, ctx: TenantContext, config: MetaConfig, graph: MetaGraph, account_id: UUID
) -> ChannelAccountRead:
    """Ask Meta whether the stored token still works and has the permissions publishing needs."""
    account = await _account(db, ctx, account_id, lock=True)
    if account.state is AccountState.DISCONNECTED or account.token_encrypted is None:
        await db.rollback()
        return to_read(account)
    try:
        token = SecretStr(
            config.cipher.decrypt(
                account.token_encrypted, context=_account_context(account.tenant_id, account.id)
            )
        )
        info = await graph.debug_token(token)
    except TokenDecryptError:
        info = None
    except MetaGraphError as exc:
        if not exc.token_invalid:
            await db.rollback()
            raise DependencyUnavailableError("Meta did not answer; try again.") from None
        info = None

    now = _now()
    if info is None or not info.is_valid:
        expired = info is not None and info.expires_at is not None and info.expires_at <= now
        account.state = AccountState.EXPIRED if expired else AccountState.REVOKED
        account.state_reason = (
            "Meta access has expired. Reconnect the account."
            if expired
            else "Meta no longer accepts this account's access (removed or password changed). "
            "Reconnect the account."
        )
    else:
        missing = [s for s in SCOPES_BY_CHANNEL[account.channel] if s not in info.scopes]
        account.state, account.state_reason = derive_state(missing, config.mode)
        account.missing_permissions = missing
        account.granted_scopes = sorted(info.scopes & set(SCOPES_BY_CHANNEL[account.channel]))
        account.token_expires_at = info.expires_at
    account.last_checked_at = now
    account.updated_at = now
    await db.commit()
    return to_read(account)
