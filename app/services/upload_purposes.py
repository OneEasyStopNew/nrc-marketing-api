# What differs between kinds of client upload: accepted types, size limit, how a file is checked
# and what a READY upload becomes. Creative media (NRC-3) and policy sources (NRC-4) share the
# rest: presigned URL, completion, quarantine, malware scan and the media worker's lease.

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Final

from app.core.config import Settings
from app.core.media_types import POLICY_SOURCE_TYPES
from app.models.enums import UploadPurpose
from app.services import media
from app.services.media import MediaRejectedError

PDF_TYPE: Final = "application/pdf"
_PDF_SIGNATURE: Final = b"%PDF-"
_TEXT_TYPES: Final = frozenset({"text/plain", "text/markdown"})


@dataclass(frozen=True, slots=True)
class InspectedFile:
    # What inspection found in the bytes; the declaration is compared against it afterwards.
    mime_type: str
    width_px: int | None = None
    height_px: int | None = None


@dataclass(frozen=True, slots=True)
class UploadPurposeSpec:
    allowed_types: Callable[[Settings], tuple[str, ...]]
    max_bytes: Callable[[Settings], int]
    # True: a READY upload is copied into an immutable assets row. False: it stays an upload.
    becomes_asset: bool
    # (content, declared mime type, settings). CPU-bound: call through asyncio.to_thread.
    # Raises MediaRejectedError carrying a code from uploads.REASONS.
    inspect: Callable[[bytes, str, Settings], InspectedFile]
    storage_unavailable: str


def _inspect_image(content: bytes, declared_mime_type: str, settings: Settings) -> InspectedFile:
    # The real type comes from decoding the file, never from its name or declared type.
    info = media.inspect_image(content, max_pixels=settings.max_image_pixels)
    return InspectedFile(info.mime_type, info.width_px, info.height_px)


def _inspect_policy_source(
    content: bytes, declared_mime_type: str, settings: Settings
) -> InspectedFile:
    # Signature checks only: the text of the file is extracted later, in the policy worker.
    if content.startswith(_PDF_SIGNATURE):
        return InspectedFile(PDF_TYPE)
    if b"\x00" not in content:
        try:
            content.decode("utf-8")
        except UnicodeDecodeError:
            pass
        else:
            # Plain text and Markdown cannot be told apart by content: a declared text type stands.
            return InspectedFile(
                declared_mime_type if declared_mime_type in _TEXT_TYPES else "text/plain"
            )
    raise MediaRejectedError("MEDIA_UNSUPPORTED_TYPE", "Neither a PDF nor UTF-8 text.")


UPLOAD_PURPOSES: Final[Mapping[UploadPurpose, UploadPurposeSpec]] = MappingProxyType(
    {
        UploadPurpose.CREATIVE_MEDIA: UploadPurposeSpec(
            allowed_types=lambda settings: settings.upload_image_types,
            max_bytes=lambda settings: settings.upload_image_max_bytes,
            becomes_asset=True,
            inspect=_inspect_image,
            storage_unavailable="Image storage is temporarily unavailable.",
        ),
        UploadPurpose.POLICY_SOURCE: UploadPurposeSpec(
            allowed_types=lambda settings: POLICY_SOURCE_TYPES,
            max_bytes=lambda settings: settings.policy_source_max_bytes,
            becomes_asset=False,
            inspect=_inspect_policy_source,
            storage_unavailable="Policy file storage is temporarily unavailable.",
        ),
    }
)
