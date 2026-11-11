import struct
import subprocess
import sys
import warnings
import zlib
from concurrent.futures import ThreadPoolExecutor
from io import BytesIO
from pathlib import Path

import pytest
from asgi_lifespan import LifespanManager
from PIL import Image

from app.core.config import get_settings
from app.main import create_app
from app.services.media import MediaRejectedError, configure_pillow, inspect_header, inspect_image
from app.workers import connector

MAX_PIXELS = 1_000_000


def _image(fmt: str, size: tuple[int, int] = (20, 10)) -> bytes:
    buffer = BytesIO()
    Image.new("RGB", size, (200, 30, 30)).save(buffer, format=fmt)
    return buffer.getvalue()


def test_png_is_accepted() -> None:
    info = inspect_image(_image("PNG"), max_pixels=MAX_PIXELS)
    assert (info.mime_type, info.ext, info.width_px, info.height_px) == ("image/png", "png", 20, 10)


def test_jpeg_is_accepted() -> None:
    info = inspect_image(_image("JPEG"), max_pixels=MAX_PIXELS)
    assert (info.mime_type, info.ext, info.width_px, info.height_px) == (
        "image/jpeg",
        "jpg",
        20,
        10,
    )


@pytest.mark.parametrize("fmt", ["GIF", "WEBP"])
def test_other_image_formats_are_rejected(fmt: str) -> None:
    with pytest.raises(MediaRejectedError) as raised:
        inspect_image(_image(fmt), max_pixels=MAX_PIXELS)
    assert raised.value.code == "MEDIA_UNSUPPORTED_TYPE"


@pytest.mark.parametrize(
    "content",
    [b'<svg xmlns="http://www.w3.org/2000/svg"></svg>', b"hello, not an image", b"%PDF-1.7"],
)
def test_non_images_are_rejected(content: bytes) -> None:
    with pytest.raises(MediaRejectedError) as raised:
        inspect_image(content, max_pixels=MAX_PIXELS)
    assert raised.value.code == "MEDIA_CORRUPT"


def test_empty_is_rejected() -> None:
    with pytest.raises(MediaRejectedError) as raised:
        inspect_image(b"", max_pixels=MAX_PIXELS)
    assert raised.value.code == "MEDIA_EMPTY"


def test_decoded_type_wins_over_any_claimed_name() -> None:
    # A GIF is a GIF even if the caller calls it photo.png, and a PNG stays a PNG as .jpg.
    assert inspect_image(_image("PNG"), max_pixels=MAX_PIXELS).ext == "png"
    with pytest.raises(MediaRejectedError):
        inspect_image(_image("GIF"), max_pixels=MAX_PIXELS)


@pytest.mark.parametrize("fmt", ["PNG", "JPEG"])
def test_too_many_pixels_are_rejected(fmt: str) -> None:
    content = _image(fmt, (200, 100))
    with pytest.raises(MediaRejectedError) as raised:
        inspect_image(content, max_pixels=10_000)
    assert raised.value.code == "MEDIA_TOO_LARGE"


def test_pixel_limit_is_inclusive() -> None:
    assert inspect_image(_image("PNG", (100, 100)), max_pixels=10_000).width_px == 100


@pytest.mark.parametrize("fmt", ["PNG", "JPEG"])
def test_truncated_files_are_rejected(fmt: str) -> None:
    content = _image(fmt, (300, 300))
    with pytest.raises(MediaRejectedError) as raised:
        inspect_image(content[: len(content) // 2], max_pixels=MAX_PIXELS)
    assert raised.value.code == "MEDIA_CORRUPT"


def test_global_pillow_limit_is_never_mutated_by_inspection() -> None:
    before = Image.MAX_IMAGE_PIXELS
    inspect_image(_image("PNG"), max_pixels=MAX_PIXELS)
    with pytest.raises(MediaRejectedError):
        inspect_image(_image("PNG", (200, 100)), max_pixels=10_000)
    assert before == Image.MAX_IMAGE_PIXELS


def test_concurrent_calls_with_different_limits_are_independent() -> None:
    big = _image("PNG", (200, 100))

    def attempt(limit: int) -> bool:
        try:
            inspect_image(big, max_pixels=limit)
        except MediaRejectedError:
            return False
        return True

    before = Image.MAX_IMAGE_PIXELS
    limits = [10_000, 1_000_000] * 20
    with ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(attempt, limits))
    assert results == [limit >= 20_000 for limit in limits]
    assert before == Image.MAX_IMAGE_PIXELS


def _png_header_only(width: int, height: int) -> bytes:
    # A few bytes that declare width x height pixels; nothing large is ever allocated.
    def chunk(kind: bytes, data: bytes) -> bytes:
        return (
            struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data))
        )

    return (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
        + chunk(b"IDAT", zlib.compress(b"\x00"))
        + chunk(b"IEND", b"")
    )


def test_oversized_dimensions_are_rejected_before_decoding() -> None:
    with pytest.raises(MediaRejectedError) as raised:
        inspect_image(_png_header_only(50_000, 50_000), max_pixels=MAX_PIXELS)
    assert raised.value.code == "MEDIA_TOO_LARGE"


# 120 x 100 is in Pillow's warning range (over the limit); 200 x 100 is in its error range
# (over twice the limit). Both must be MEDIA_TOO_LARGE, not MEDIA_CORRUPT or an escape.
@pytest.mark.parametrize("size", [(120, 100), (200, 100)])
def test_header_over_pillow_limit_is_too_large(size: tuple[int, int]) -> None:
    configure_pillow(get_settings().model_copy(update={"max_image_pixels": 10_000}))
    with pytest.raises(MediaRejectedError) as raised:
        inspect_header(_png_header_only(*size))
    assert raised.value.code == "MEDIA_TOO_LARGE"


def test_importing_media_leaves_pillow_limit_alone() -> None:
    # A fresh interpreter, so the module-level code really runs.
    code = (
        "from PIL import Image\n"
        "before = Image.MAX_IMAGE_PIXELS\n"
        "import app.services.media\n"
        "assert before is not None and Image.MAX_IMAGE_PIXELS == before\n"
    )
    root = Path(__file__).resolve().parents[1]
    subprocess.run([sys.executable, "-c", code], check=True, cwd=root)  # noqa: S603


def test_configure_pillow_sets_limit_and_makes_bomb_warning_an_error() -> None:
    warnings.simplefilter("default")
    configure_pillow(get_settings().model_copy(update={"max_image_pixels": 1234}))
    assert Image.MAX_IMAGE_PIXELS == 1234
    with pytest.raises(Image.DecompressionBombWarning):
        warnings.warn("too many pixels", Image.DecompressionBombWarning, stacklevel=1)


async def test_api_lifespan_configures_pillow() -> None:
    Image.MAX_IMAGE_PIXELS = None
    async with LifespanManager(create_app()):
        assert get_settings().max_image_pixels == Image.MAX_IMAGE_PIXELS


def test_worker_main_configures_pillow(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: list[int | None] = []

    async def run_forever(settings: object) -> None:
        seen.append(Image.MAX_IMAGE_PIXELS)

    Image.MAX_IMAGE_PIXELS = None
    monkeypatch.setattr(connector, "run_forever", run_forever)
    connector.main()
    assert seen == [get_settings().max_image_pixels]


def test_mpo_is_accepted_as_jpeg() -> None:
    buffer = BytesIO()
    first = Image.new("RGB", (20, 10), (1, 2, 3))
    first.save(buffer, format="MPO", save_all=True, append_images=[first])
    info = inspect_image(buffer.getvalue(), max_pixels=MAX_PIXELS)
    assert (info.mime_type, info.ext) == ("image/jpeg", "jpg")
