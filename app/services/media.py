# Decoded-signature image checks. The claimed type and file extension are never trusted.

import warnings
from dataclasses import dataclass
from io import BytesIO

from PIL import Image, UnidentifiedImageError

_FORMATS = {"PNG": ("image/png", "png"), "JPEG": ("image/jpeg", "jpg")}


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


def inspect_image(content: bytes, *, max_pixels: int) -> ImageInfo:
    if not content:
        raise MediaRejectedError("MEDIA_EMPTY", "The file is empty")
    previous_limit = Image.MAX_IMAGE_PIXELS
    # Pillow warns above the limit and raises above twice the limit; both are rejections.
    Image.MAX_IMAGE_PIXELS = max_pixels
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            image_format, width, height = _decode(content)
    finally:
        Image.MAX_IMAGE_PIXELS = previous_limit
    if width * height > max_pixels:
        raise MediaRejectedError("MEDIA_TOO_LARGE", "The image has too many pixels")
    mime_type, ext = _FORMATS[image_format]
    return ImageInfo(mime_type=mime_type, ext=ext, width_px=width, height_px=height)


def _decode(content: bytes) -> tuple[str, int, int]:
    try:
        with Image.open(BytesIO(content)) as probe:
            image_format = probe.format or ""
            if image_format not in _FORMATS:
                raise MediaRejectedError(
                    "MEDIA_UNSUPPORTED_TYPE", "Only PNG and JPEG images are accepted"
                )
            probe.verify()
        with Image.open(BytesIO(content)) as image:
            width, height = image.size
            # Decode fully so truncated data is caught, not just a valid header.
            image.load()
    except MediaRejectedError:
        raise
    except (Image.DecompressionBombWarning, Image.DecompressionBombError) as exc:
        raise MediaRejectedError("MEDIA_TOO_LARGE", "The image has too many pixels") from exc
    except (UnidentifiedImageError, OSError, SyntaxError, ValueError) as exc:
        raise MediaRejectedError(
            "MEDIA_CORRUPT", "The file is not a valid, complete image"
        ) from exc
    return image_format, width, height
