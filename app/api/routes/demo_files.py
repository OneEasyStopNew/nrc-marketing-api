# Placeholder image bytes behind the demo stubs' download_url (not part of the API contract).

from uuid import UUID

from fastapi import APIRouter, Response

from app.api.deps import DemoStubsDep
from app.core.errors import NotFoundError
from app.services.demo_stubs import DEMO_FILES_PATH, STUB_MIME_TYPE

router = APIRouter(prefix=DEMO_FILES_PATH, include_in_schema=False)

# No auth, like a pre-signed S3 URL: an <img> tag cannot send a bearer token. The image is
# generic (no tenant or creative data) and the URL holds a random asset ID.
_HEADERS = {
    "Cache-Control": "private, max-age=900",
    "Content-Security-Policy": "default-src 'none'; style-src 'unsafe-inline'",
    "X-Content-Type-Options": "nosniff",
}


@router.get("/{asset_id}")
async def get_demo_file(asset_id: UUID, stubs: DemoStubsDep) -> Response:
    content = None if stubs is None else stubs.placeholder_file(asset_id)
    if content is None:
        raise NotFoundError("File not found.")
    return Response(content=content, media_type=STUB_MIME_TYPE, headers=_HEADERS)
