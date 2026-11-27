# The image types this service can decode and store, and the extension each object key carries.
# The one place that lists them: settings validation, storage keys and the image checks read it.
# Adding a type needs a decoder in app.services.media first. Policy source files (NRC-4) are
# listed the same way; adding one needs a signature check in app.services.upload_purposes.

from collections.abc import Mapping
from types import MappingProxyType
from typing import Final

IMAGE_EXTENSIONS: Final[Mapping[str, str]] = MappingProxyType(
    {"image/png": "png", "image/jpeg": "jpg"}
)
POLICY_SOURCE_EXTENSIONS: Final[Mapping[str, str]] = MappingProxyType(
    {"application/pdf": "pdf", "text/plain": "txt", "text/markdown": "md"}
)
# The policy source types the inspector can check. A constant, not a setting: a new type needs
# code first.
POLICY_SOURCE_TYPES: Final[tuple[str, ...]] = tuple(POLICY_SOURCE_EXTENSIONS)
# Every type an upload object key can carry.
UPLOAD_EXTENSIONS: Final[Mapping[str, str]] = MappingProxyType(
    {**IMAGE_EXTENSIONS, **POLICY_SOURCE_EXTENSIONS}
)
