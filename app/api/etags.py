# The If-Match / ETag convention shared by routes that edit a versioned resource.

import re

from app.core.errors import VersionRequiredError

_IF_MATCH = re.compile(r'"([1-9][0-9]{0,8})"')


def etag(version: int) -> str:
    return f'"{version}"'


def expected_version(if_match: str | None) -> int:
    if if_match is None:
        raise VersionRequiredError()
    match = _IF_MATCH.fullmatch(if_match.strip())
    if match is None:
        raise VersionRequiredError('If-Match must be one quoted version number, e.g. "3".')
    return int(match.group(1))
