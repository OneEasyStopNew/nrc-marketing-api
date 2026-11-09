# Access-token verification against the Cognito user pool's signing keys.

import asyncio
import logging
import time
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Final, Literal, Protocol

import httpx
import jwt
from pydantic import BaseModel, ValidationError

from app.core.errors import AuthenticationError, DependencyUnavailableError, PermissionDeniedError

logger = logging.getLogger(__name__)

ALGORITHMS: Final = ("RS256",)
REQUIRED_CLAIMS: Final = ("exp", "iat", "iss", "sub", "token_use", "client_id")
MIN_REFRESH_INTERVAL_SECONDS: Final = 60.0


@dataclass(frozen=True, slots=True)
class Principal:
    # The authenticated caller. Tenant access is decided separately, from memberships.

    sub: str
    username: str
    scopes: frozenset[str]


class _AccessTokenClaims(BaseModel):
    sub: str
    token_use: Literal["access"]
    client_id: str
    scope: str = ""
    username: str | None = None


class SigningKeyProvider(Protocol):
    async def get_signing_key(self, kid: str) -> jwt.PyJWK: ...


class JwksClient:
    # Caches the user pool's JWKS. Refetches on expiry, or for an unknown key ID once a minute.

    def __init__(self, http: httpx.AsyncClient, url: str, *, ttl_seconds: float) -> None:
        self._http = http
        self._url = url
        self._ttl = ttl_seconds
        self._keys: Mapping[str, jwt.PyJWK] = {}
        self._fetched_at: float | None = None
        self._lock = asyncio.Lock()

    async def get_signing_key(self, kid: str) -> jwt.PyJWK:
        for force_refresh in (False, True):
            keys = await self._load(force_refresh=force_refresh)
            if kid in keys:
                return keys[kid]
        raise AuthenticationError("Access token was signed with an unknown key.")

    async def _load(self, *, force_refresh: bool) -> Mapping[str, jwt.PyJWK]:
        async with self._lock:
            age = None if self._fetched_at is None else time.monotonic() - self._fetched_at
            if (
                age is None
                or age >= self._ttl
                or (force_refresh and age >= MIN_REFRESH_INTERVAL_SECONDS)
            ):
                self._keys = await self._fetch()
                self._fetched_at = time.monotonic()
            return self._keys

    async def _fetch(self) -> Mapping[str, jwt.PyJWK]:
        try:
            response = await self._http.get(self._url)
            response.raise_for_status()
            jwk_set = jwt.PyJWKSet.from_dict(response.json())
        except (httpx.HTTPError, ValueError, jwt.PyJWTError) as err:
            logger.warning("could not load signing keys: %s", type(err).__name__)
            raise DependencyUnavailableError("The identity provider is unavailable.") from err
        return {key.key_id: key for key in jwk_set.keys if key.key_id is not None}


async def verify_access_token(
    token: str,
    keys: SigningKeyProvider,
    *,
    issuer: str,
    client_ids: frozenset[str],
    required_scope: str,
) -> Principal:
    # Return the caller for a valid Cognito access token, or raise AuthenticationError.
    try:
        kid = jwt.get_unverified_header(token).get("kid")
    except jwt.PyJWTError as err:
        raise AuthenticationError("Malformed access token.") from err
    if not isinstance(kid, str):
        raise AuthenticationError("Access token has no key ID.")

    signing_key = await keys.get_signing_key(kid)
    try:
        payload = jwt.decode(
            token,
            signing_key,
            algorithms=list(ALGORITHMS),
            issuer=issuer,
            options={"require": list(REQUIRED_CLAIMS)},
        )
        claims = _AccessTokenClaims.model_validate(payload)
    except (jwt.PyJWTError, ValidationError) as err:
        raise AuthenticationError("Access token is invalid or expired.") from err

    # Cognito access tokens name their app client in client_id; they carry no aud claim.
    if claims.client_id not in client_ids:
        raise AuthenticationError("Access token was issued to a different application.")
    scopes = frozenset(claims.scope.split())
    if required_scope not in scopes:
        raise PermissionDeniedError("Access token lacks the required scope.")
    return Principal(sub=claims.sub, username=claims.username or claims.sub, scopes=scopes)
