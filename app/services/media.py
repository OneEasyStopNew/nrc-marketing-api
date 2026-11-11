# Decoded-signature image checks. The claimed type and file extension are never trusted.

import warnings
from dataclasses import dataclass
from io import BytesIO

from PIL import Image, ImageOps, UnidentifiedImageError

from app.core.config import Settings

_FORMATS = {
    "PNG": ("image/png", "png"),
    "JPEG": ("image/jpeg", "jpg"),
    # Some cameras and phones write JPEGs that Pillow reports as MPO.
    "MPO": ("image/jpeg", "jpg"),
}


@dataclass(frozen=True, slots=True)
class ImageInfo:
    mime_type: str
    ext: str
    width_px: int
    height_px: int


class MediaRejectedError(ValueError):
    def __init__(self, code: str, message: str) -> None:
        self.code = code
        super().__init__(message)


def configure_pillow(settings: Settings) -> None:
    # Process-wide, so every Image.open() in the API or worker has a limit, not only ours.
    # Call once at startup (API lifespan, worker main), before any image is opened.
    Image.MAX_IMAGE_PIXELS = settings.max_image_pixels
    # Pillow only warns between the limit and twice the limit; refuse those images too.
    warnings.simplefilter("error", Image.DecompressionBombWarning)


def inspect_header(content: bytes) -> ImageInfo:
    # Format and size from the header, plus Pillow's structural verify(). Never decodes pixels.
    if not content:
        raise MediaRejectedError("MEDIA_EMPTY", "The file is empty")
    try:
        with Image.open(BytesIO(content)) as probe:
            image_format = probe.format or ""
            if image_format not in _FORMATS:
                raise MediaRejectedError(
                    "MEDIA_UNSUPPORTED_TYPE", "Only PNG and JPEG images are accepted"
                )
            width, height = probe.size
            probe.verify()
    except MediaRejectedError:
        raise
    except (Image.DecompressionBombWarning, Image.DecompressionBombError) as exc:
        raise MediaRejectedError("MEDIA_TOO_LARGE", "The image has too many pixels") from exc
    except (UnidentifiedImageError, OSError, SyntaxError, ValueError) as exc:
        raise MediaRejectedError(
            "MEDIA_CORRUPT", "The file is not a valid, complete image"
        ) from exc
    mime_type, ext = _FORMATS[image_format]
    return ImageInfo(mime_type=mime_type, ext=ext, width_px=width, height_px=height)


def inspect_image(content: bytes, *, max_pixels: int) -> ImageInfo:
    # CPU-bound: call through asyncio.to_thread from async code.
    info = inspect_header(content)
    if info.width_px * info.height_px > max_pixels:
        raise MediaRejectedError("MEDIA_TOO_LARGE", "The image has too many pixels")
    try:
        with Image.open(BytesIO(content)) as image:
            # Decode fully so truncated data is caught, not just a valid header.
            image.load()
    except (UnidentifiedImageError, OSError, SyntaxError, ValueError) as exc:
        raise MediaRejectedError(
            "MEDIA_CORRUPT", "The file is not a valid, complete image"
        ) from exc
    return info


def fit_to_format(
    content: bytes, *, width_px: int, height_px: int, max_pixels: int
) -> tuple[bytes, ImageInfo]:
    """Center-crop to the format's aspect ratio, resize to its exact size, re-encode as PNG.

    The same input always gives the same bytes (so checksums are stable) and no metadata
    survives. CPU-bound: call through asyncio.to_thread from async code.
    """
    info = inspect_header(content)
    if info.width_px * info.height_px > max_pixels:
        raise MediaRejectedError("MEDIA_TOO_LARGE", "The image has too many pixels")
    try:
        with Image.open(BytesIO(content)) as opened:
            opened.load()
            image = ImageOps.exif_transpose(opened)
            # Pillow only writes ICC and EXIF data it finds in info, so a fresh copy drops them.
            has_alpha = "A" in image.getbands() or "transparency" in image.info
            image = image.convert("RGBA" if has_alpha else "RGB")
            image.info.clear()
            width, height = image.size
            # Integer math on purpose: no float rounding differences between machines.
            if width * height_px > height * width_px:
                crop_width = height * width_px // height_px
                left = (width - crop_width) // 2
                box = (left, 0, left + crop_width, height)
            else:
                crop_height = width * height_px // width_px
                top = (height - crop_height) // 2
                box = (0, top, width, top + crop_height)
            image = image.crop(box).resize((width_px, height_px), Image.Resampling.LANCZOS)
            out = BytesIO()
            image.save(out, format="PNG", compress_level=6)
    except (Image.DecompressionBombWarning, Image.DecompressionBombError) as exc:
        raise MediaRejectedError("MEDIA_TOO_LARGE", "The image has too many pixels") from exc
    except (UnidentifiedImageError, OSError, SyntaxError, ValueError) as exc:
        raise MediaRejectedError(
            "MEDIA_CORRUPT", "The file is not a valid, complete image"
        ) from exc
    return out.getvalue(), ImageInfo(
        mime_type="image/png", ext="png", width_px=width_px, height_px=height_px
    )
