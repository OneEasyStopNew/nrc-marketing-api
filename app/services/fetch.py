# SSRF-safe download of provider output URLs.

import asyncio
import ipaddress
import socket
from collections.abc import Awaitable, Callable
from urllib.parse import urlsplit

import httpx

from app.services.media import MediaRejectedError, inspect_image
from app.services.providers.base import FetchedImage, ProviderOutputError

_ALLOWED_TYPES = frozenset({"image/png", "image/jpeg"})
# Only the signature matters here; the caller enforces its own pixel limit later.
_SIGNATURE_MAX_PIXELS = 400_000_000


async def _default_resolver(host: str) -> list[str]:
    infos = await asyncio.get_running_loop().getaddrinfo(host, 443, type=socket.SOCK_STREAM)
    return [str(info[4][0]) for info in infos]


def _is_public(address: str) -> bool:
    try:
        ip = ipaddress.ip_address(address.split("%", 1)[0])
    except ValueError:
        return False
    if isinstance(ip, ipaddress.IPv6Address) and ip.ipv4_mapped is not None:
        ip = ip.ipv4_mapped
    return not (
        ip.is_private
        or ip.is_loopback
        or ip.is_link_local
        or ip.is_multicast
        or ip.is_reserved
        or ip.is_unspecified
        or not ip.is_global
    )


class SafeFetcher:
    # Residual risk: the host is resolved and vetted here, then httpx resolves it again when it
    # connects, so a DNS-rebinding attacker could swap the answer in between. Pinning the
    # connection to the vetted IP (custom transport, Host/SNI kept) is a production follow-up.

    def __init__(
        self,
        client: httpx.AsyncClient,
        *,
        resolver: Callable[[str], Awaitable[list[str]]] | None = None,
        timeout_seconds: float = 30,
    ) -> None:
        self._client = client
        self._resolver = resolver or _default_resolver
        self._timeout = timeout_seconds

    async def __call__(
        self, url: str, *, allowed_hosts: frozenset[str], max_bytes: int
    ) -> FetchedImage:
        host = self._validate_url(url, allowed_hosts)
        await self._check_addresses(host)
        try:
            return await asyncio.wait_for(self._download(url, host, max_bytes), self._timeout)
        except TimeoutError as exc:
            raise ProviderOutputError("Fetching the provider output timed out") from exc
        except httpx.HTTPError as exc:
            raise ProviderOutputError("Fetching the provider output failed") from exc

    @staticmethod
    def _validate_url(url: str, allowed_hosts: frozenset[str]) -> str:
        try:
            parts = urlsplit(url)
            host = parts.hostname
            _ = parts.port  # raises ValueError on a malformed port
        except ValueError as exc:
            raise ProviderOutputError("The provider output URL is invalid") from exc
        if parts.scheme != "https":
            raise ProviderOutputError("The provider output URL must use https")
        if parts.username or parts.password:
            raise ProviderOutputError("The provider output URL must not carry credentials")
        if not host or host not in allowed_hosts:
            raise ProviderOutputError("The provider output host is not allowed")
        return host

    async def _check_addresses(self, host: str) -> None:
        try:
            addresses = await self._resolver(host)
        except OSError as exc:
            raise ProviderOutputError("The provider output host could not be resolved") from exc
        if not addresses or not all(_is_public(address) for address in addresses):
            raise ProviderOutputError("The provider output host resolves to a blocked address")

    async def _download(self, url: str, host: str, max_bytes: int) -> FetchedImage:
        async with self._client.stream("GET", url, follow_redirects=False) as response:
            status = response.status_code
            if 300 <= status < 400:
                raise ProviderOutputError("The provider output redirected, which is not allowed")
            if status in (403, 404, 410):
                raise ProviderOutputError("The provider output has expired or is unavailable")
            if not 200 <= status < 300:
                raise ProviderOutputError(f"The provider output returned HTTP {status}")
            content_type = response.headers.get("content-type", "").split(";")[0].strip().lower()
            if content_type not in _ALLOWED_TYPES:
                raise ProviderOutputError("The provider output is not a PNG or JPEG image")
            declared = response.headers.get("content-length")
            if declared is not None and declared.isdigit() and int(declared) > max_bytes:
                raise ProviderOutputError("The provider output exceeds the size limit")
            chunks: list[bytes] = []
            received = 0
            async for chunk in response.aiter_bytes():
                received += len(chunk)
                if received > max_bytes:
                    raise ProviderOutputError("The provider output exceeds the size limit")
                chunks.append(chunk)
        content = b"".join(chunks)
        try:
            info = inspect_image(content, max_pixels=_SIGNATURE_MAX_PIXELS)
        except MediaRejectedError as exc:
            raise ProviderOutputError("The provider output is not a valid image") from exc
        if info.mime_type != content_type:
            raise ProviderOutputError("The provider output type does not match its content")
        return FetchedImage(content=content, content_type=content_type, source_host=host)
