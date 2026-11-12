# Application factory. Run with: uvicorn app.main:create_app --factory

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

import httpx
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.routing import APIRoute
from starlette.middleware.body_limit import RequestBodyLimitMiddleware

from app.api.errors import COMMON_ERROR_RESPONSES, register_exception_handlers
from app.api.routes import assets, creatives, health, jobs, me, workflows
from app.auth.tokens import JwksClient
from app.core.config import get_settings
from app.core.logging import configure_logging
from app.core.request_context import REQUEST_ID_HEADER, RequestContextMiddleware
from app.db.session import create_engine, create_sessionmaker

API_PREFIX = "/v1/marketing"
API_VERSION = "0.1.0"


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    settings = get_settings()
    engine = create_engine(settings)
    try:
        async with httpx.AsyncClient(timeout=settings.http_timeout_seconds) as http:
            app.state.sessionmaker = create_sessionmaker(engine)
            app.state.signing_keys = JwksClient(
                http, settings.cognito_jwks_url, ttl_seconds=settings.jwks_cache_ttl_seconds
            )
            yield
    finally:
        await engine.dispose()


def _operation_id(route: APIRoute) -> str:
    # Function names become the generated TypeScript client's method names.
    return route.name


def create_app() -> FastAPI:
    settings = get_settings()
    configure_logging(settings.log_level)
    app = FastAPI(
        title="NRC Marketing API",
        version=API_VERSION,
        lifespan=lifespan,
        docs_url="/docs" if settings.enable_docs else None,
        redoc_url=None,
        openapi_url="/openapi.json" if settings.enable_docs else None,
        generate_unique_id_function=_operation_id,
    )
    register_exception_handlers(app)
    for router in (me.router, creatives.router, workflows.router, jobs.router, assets.router):
        app.include_router(router, prefix=API_PREFIX, responses=COMMON_ERROR_RESPONSES)
    app.include_router(health.router)

    # The last middleware added runs first: CORS stays outermost so error responses carry it.
    app.add_middleware(RequestBodyLimitMiddleware, max_body_size=settings.max_request_body_bytes)
    app.add_middleware(RequestContextMiddleware)
    if settings.cors_origins:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=list(settings.cors_origins),
            allow_methods=["GET", "POST", "PATCH"],
            allow_headers=[
                "Authorization",
                "Content-Type",
                "Idempotency-Key",
                "If-Match",
                "X-Tenant-Id",
                REQUEST_ID_HEADER,
            ],
            expose_headers=["ETag", "Location", "Idempotent-Replayed", REQUEST_ID_HEADER],
        )
    return app
