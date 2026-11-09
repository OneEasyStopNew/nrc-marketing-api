# Per-request correlation ID, available to logs and error bodies through a context variable.

import re
from contextvars import ContextVar
from uuid import uuid4

from starlette.datastructures import Headers, MutableHeaders
from starlette.types import ASGIApp, Message, Receive, Scope, Send

REQUEST_ID_HEADER = "X-Request-Id"
_VALID_REQUEST_ID = re.compile(r"[A-Za-z0-9._-]{8,128}")

request_id_var: ContextVar[str | None] = ContextVar("request_id", default=None)


class RequestContextMiddleware:
    # Pure ASGI middleware: assigns a request ID and marks every response as non-cacheable.

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        incoming = Headers(scope=scope).get(REQUEST_ID_HEADER)
        valid = incoming is not None and _VALID_REQUEST_ID.fullmatch(incoming) is not None
        request_id = incoming if incoming is not None and valid else str(uuid4())
        # Not reset afterwards: each request runs in its own task, and the outermost error
        # handler still needs the ID for the body of a 500 response.
        request_id_var.set(request_id)

        async def send_with_headers(message: Message) -> None:
            if message["type"] == "http.response.start":
                headers = MutableHeaders(scope=message)
                headers[REQUEST_ID_HEADER] = request_id
                headers["Cache-Control"] = "no-store"
            await send(message)

        await self.app(scope, receive, send_with_headers)
