# Decoded-signature image checks. The claimed type and file extension are never trusted.

from dataclasses import dataclass
from io import BytesIO

from PIL import Image, UnidentifiedImageError

# We enforce our own pixel cap before any decode. Pillow's limit is process-global and would
# warn or raise inconsistently across threads, so it is switched off once, at import.
Image.MAX_IMAGE_PIXELS = None

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
