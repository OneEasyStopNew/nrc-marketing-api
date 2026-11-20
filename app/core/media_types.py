# The image types this service can decode and store, and the extension each object key carries.
# The one place that lists them: settings validation, storage keys and the image checks read it.
# Adding a type needs a decoder in app.services.media first.

from collections.abc import Mapping
from types import MappingProxyType
from typing import Final

IMAGE_EXTENSIONS: Final[Mapping[str, str]] = MappingProxyType(
    {"image/png": "png", "image/jpeg": "jpg"}
)
