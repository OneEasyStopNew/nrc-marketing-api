import asyncio
import struct
import time
import zlib
from collections.abc import AsyncIterator, Awaitable, Callable
from io import BytesIO

import httpx
import pytest
from PIL import Image

from app.services.fetch import SafeFetcher, build_fetch_client
from app.services.providers.base import FetchedImage, ProviderOutputError

HOST = "cdn.provider.example"
URL = f"https://{HOST}/out/1.png?X-Signature=secret-signature"
ALLOWED = frozenset({HOST})
CAP = 1_000_000


def _image(fmt: str) -> bytes:
    buffer = BytesIO()
    Image.new("RGB", (16, 16), (10, 120, 200)).save(buffer, format=fmt)
    return buffer.getvalue()


def _huge_png(width: int, height: int) -> bytes:
    def chunk(kind: bytes, data: bytes) -> bytes:
        crc = zlib.crc32(kind + data)
        return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", crc)

    header = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    return (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", header)
        + chunk(b"IDAT", zlib.compress(b"\x00"))
        + chunk(b"IEND", b"")
    )


PNG = _image("PNG")
JPEG = _image("JPEG")


def _resolver(*addresses: str) -> Callable[[str], Awaitable[list[str]]]:
    async def resolve(host: str) -> list[str]:
        return list(addresses)

    return resolve


def _fetcher(
    handler: Callable[[httpx.Request], httpx.Response],
    addresses: tuple[str, ...] = ("93.184.216.34",),
) -> tuple[SafeFetcher, list[httpx.Request]]:
    seen: list[httpx.Request] = []

    def record(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return handler(request)

    client = httpx.AsyncClient(transport=httpx.MockTransport(record))
    return SafeFetcher(client, resolver=_resolver(*addresses)), seen


def _ok(
    body: bytes = PNG, content_type: str = "image/png"
) -> Callable[[httpx.Request], httpx.Response]:
    return lambda request: httpx.Response(200, content=body, headers={"content-type": content_type})


async def _fetch(
    fetcher: SafeFetcher, url: str = URL, hosts: frozenset[str] = ALLOWED
) -> FetchedImage:
    return await fetcher(url, allowed_hosts=hosts, max_bytes=CAP)


async def test_success_returns_image_and_host() -> None:
    fetcher, seen = _fetcher(_ok())
    result = await _fetch(fetcher)
    assert result == FetchedImage(content=PNG, content_type="image/png", source_host=HOST)
    assert len(seen) == 1


async def test_jpeg_with_charset_parameter_is_accepted() -> None:
    fetcher, _ = _fetcher(_ok(JPEG, "image/jpeg; charset=binary"))
    assert (await _fetch(fetcher)).content_type == "image/jpeg"


async def test_http_scheme_is_rejected_without_a_request() -> None:
    fetcher, seen = _fetcher(_ok())
    with pytest.raises(ProviderOutputError):
        await _fetch(fetcher, f"http://{HOST}/out/1.png")
    assert seen == []


async def test_host_not_in_allowlist_is_rejected() -> None:
    fetcher, seen = _fetcher(_ok())
    with pytest.raises(ProviderOutputError):
        await _fetch(fetcher, "https://evil.example/out/1.png")
    assert seen == []


async def test_empty_allowlist_rejects_everything() -> None:
    fetcher, seen = _fetcher(_ok())
    with pytest.raises(ProviderOutputError):
        await _fetch(fetcher, hosts=frozenset())
    assert seen == []


async def test_userinfo_trick_is_rejected() -> None:
    fetcher, seen = _fetcher(_ok())
    with pytest.raises(ProviderOutputError):
        await _fetch(fetcher, f"https://evil@{HOST}/out/1.png")
    assert seen == []


@pytest.mark.parametrize("url", [f"https://{HOST}:notaport/x.png", "https://[::1/x.png"])
async def test_malformed_urls_are_rejected(url: str) -> None:
    fetcher, seen = _fetcher(_ok())
    with pytest.raises(ProviderOutputError):
        await _fetch(fetcher, url)
    assert seen == []


async def test_invalid_url_raised_by_httpx_is_wrapped() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.InvalidURL("bad url with secret-signature")

    fetcher, _ = _fetcher(handler)
    with pytest.raises(ProviderOutputError) as raised:
        await _fetch(fetcher)
    assert "secret-signature" not in str(raised.value)
    assert raised.value.__cause__ is None


async def test_slow_resolver_hits_the_overall_timeout() -> None:
    async def slow(host: str) -> list[str]:
        await asyncio.sleep(5)
        return ["93.184.216.34"]

    client = httpx.AsyncClient(transport=httpx.MockTransport(_ok()))
    fetcher = SafeFetcher(client, resolver=slow, timeout_seconds=0.05)
    with pytest.raises(ProviderOutputError, match="timed out"):
        await _fetch(fetcher)


async def test_tiny_file_with_huge_dimensions_is_rejected_without_decoding() -> None:
    # 50000 x 50000 PNG of a single colour is a few KB but would decode to 2.5 billion pixels.
    # Pillow's process-wide limit rejects it from the header alone.
    huge = _huge_png(50_000, 50_000)
    assert len(huge) < 100_000
    fetcher, _ = _fetcher(_ok(huge))
    started = time.perf_counter()
    with pytest.raises(ProviderOutputError):
        await _fetch(fetcher)
    assert time.perf_counter() - started < 2


def test_default_client_ignores_proxy_environment() -> None:
    assert build_fetch_client()._trust_env is False


@pytest.mark.parametrize(
    "address",
    [
        "10.0.0.5",
        "172.16.0.1",
        "192.168.1.1",
        "127.0.0.1",
        "169.254.169.254",
        "224.0.0.1",
        "240.0.0.1",
        "0.0.0.0",  # noqa: S104
        "::1",
        "fe80::1",
        "fc00::1",
        "fec0::1",
        "ff02::1",
        "::",
        "::ffff:127.0.0.1",
        "::ffff:10.0.0.1",
        "not-an-ip",
    ],
)
async def test_blocked_addresses_are_rejected(address: str) -> None:
    fetcher, seen = _fetcher(_ok(), addresses=(address,))
    with pytest.raises(ProviderOutputError):
        await _fetch(fetcher)
    assert seen == []


async def test_any_blocked_address_among_several_rejects() -> None:
    fetcher, seen = _fetcher(_ok(), addresses=("93.184.216.34", "10.0.0.5"))
    with pytest.raises(ProviderOutputError):
        await _fetch(fetcher)
    assert seen == []


async def test_public_ipv6_is_accepted() -> None:
    fetcher, _ = _fetcher(_ok(), addresses=("2606:2800:220:1:248:1893:25c8:1946",))
    assert (await _fetch(fetcher)).content == PNG


async def test_no_addresses_is_rejected() -> None:
    fetcher, _ = _fetcher(_ok(), addresses=())
    with pytest.raises(ProviderOutputError):
        await _fetch(fetcher)


async def test_redirect_is_rejected_and_not_followed() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(302, headers={"location": "https://evil.example/x.png"})

    fetcher, seen = _fetcher(handler)
    with pytest.raises(ProviderOutputError):
        await _fetch(fetcher)
    assert len(seen) == 1


async def test_content_length_over_cap_is_rejected_up_front() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200, content=PNG, headers={"content-type": "image/png", "content-length": str(CAP + 1)}
        )

    fetcher, _ = _fetcher(handler)
    with pytest.raises(ProviderOutputError, match="size limit"):
        await _fetch(fetcher)


def _superscript_length(body: bytes) -> Callable[[httpx.Request], httpx.Response]:
    # "²" passes str.isdigit() but int() rejects it; it must count as no Content-Length.
    # Raw bytes, as a server would send them; httpx decodes the value back to "²".
    headers = [(b"content-type", b"image/png"), (b"content-length", "²".encode())]

    def handler(request: httpx.Request) -> httpx.Response:
        response = httpx.Response(200, content=body, headers=headers)
        assert response.headers["content-length"] == "²"
        return response

    return handler


async def test_non_ascii_content_length_is_ignored_and_small_body_succeeds() -> None:
    fetcher, _ = _fetcher(_superscript_length(PNG))
    assert (await _fetch(fetcher)).content == PNG


async def test_non_ascii_content_length_still_applies_the_streaming_cap() -> None:
    fetcher, _ = _fetcher(_superscript_length(PNG))
    with pytest.raises(ProviderOutputError, match="size limit"):
        await fetcher(URL, allowed_hosts=ALLOWED, max_bytes=len(PNG) - 1)


async def test_streamed_body_over_cap_aborts_early() -> None:
    pulled: list[int] = []

    class Endless(httpx.AsyncByteStream):
        async def __aiter__(self) -> AsyncIterator[bytes]:
            for index in range(1000):
                pulled.append(index)
                yield b"x" * 1000

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, headers={"content-type": "image/png"}, stream=Endless())

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    fetcher = SafeFetcher(client, resolver=_resolver("93.184.216.34"))
    with pytest.raises(ProviderOutputError, match="size limit"):
        await fetcher(URL, allowed_hosts=ALLOWED, max_bytes=5000)
    assert len(pulled) < 10


@pytest.mark.parametrize("content_type", ["image/gif", "text/html", "image/webp", ""])
async def test_wrong_content_type_is_rejected(content_type: str) -> None:
    fetcher, _ = _fetcher(_ok(PNG, content_type))
    with pytest.raises(ProviderOutputError):
        await _fetch(fetcher)


async def test_signature_mismatch_is_rejected() -> None:
    fetcher, _ = _fetcher(_ok(JPEG, "image/png"))
    with pytest.raises(ProviderOutputError, match="does not match"):
        await _fetch(fetcher)


async def test_non_image_bytes_with_image_type_are_rejected() -> None:
    fetcher, _ = _fetcher(_ok(b"<html>not an image</html>", "image/png"))
    with pytest.raises(ProviderOutputError):
        await _fetch(fetcher)


@pytest.mark.parametrize("status", [403, 404, 410])
async def test_expired_output_is_rejected_without_leaking_the_url(status: int) -> None:
    fetcher, _ = _fetcher(lambda request: httpx.Response(status))
    with pytest.raises(ProviderOutputError) as raised:
        await _fetch(fetcher)
    assert "expired" in raised.value.message
    assert "secret-signature" not in str(raised.value)


async def test_server_error_is_rejected_without_leaking_the_url() -> None:
    fetcher, _ = _fetcher(lambda request: httpx.Response(500))
    with pytest.raises(ProviderOutputError) as raised:
        await _fetch(fetcher)
    assert "secret-signature" not in str(raised.value)


async def test_transport_error_is_wrapped_without_leaking_the_url() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError(f"cannot connect to {request.url}", request=request)

    fetcher, _ = _fetcher(handler)
    with pytest.raises(ProviderOutputError) as raised:
        await _fetch(fetcher)
    text = f"{raised.value} {raised.value.message} {raised.value.__cause__!r}"
    assert "secret-signature" not in text
    assert "X-Signature" not in text
    assert raised.value.__cause__ is None
