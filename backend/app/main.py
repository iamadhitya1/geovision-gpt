"""FastAPI application factory and process lifecycle.

``create_app`` is a factory rather than a module-level singleton so tests can build an
isolated instance with overridden settings, and so a future worker process can reuse the
same wiring without inheriting the HTTP server.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.middleware import RequestContextMiddleware, SecurityHeadersMiddleware
from app.api.v1.router import api_router
from app.application.password_policy import PasswordPolicy
from app.core.config import Settings, get_settings
from app.core.errors import register_exception_handlers
from app.core.logging import configure_logging, get_logger
from app.infrastructure.bootstrap import ensure_first_admin
from app.infrastructure.cache.factory import build_cache
from app.infrastructure.db.session import create_all, create_engine, create_session_factory
from app.infrastructure.security.password import Argon2PasswordHasher
from app.infrastructure.security.tokens import JwtTokenService

logger = get_logger(__name__)

DESCRIPTION = """
**GeoVisionGPT** — a multi-modal Visual Question Answering system for satellite and Earth
observation imagery, built on vision-language models.

Module 1 (Authentication) is implemented. Modules 2-10 register additional routers under
the same `/api/v1` prefix.

**Roles** — `guest` (read-only), `researcher` (full analysis), `admin` (administration).
Role checks are hierarchical: an admin satisfies any researcher requirement.
"""


def build_state(app: FastAPI, settings: Settings) -> None:
    """Construct process-scoped singletons onto ``app.state``.

    These are built once because they are expensive (the Argon2 hasher precomputes its
    timing-equalisation hash) or must be shared (the connection pool, the cache client).
    """
    engine = create_engine(settings)

    app.state.settings = settings
    app.state.engine = engine
    app.state.session_factory = create_session_factory(engine)
    app.state.cache = build_cache(settings)
    app.state.password_hasher = Argon2PasswordHasher()
    app.state.token_service = JwtTokenService(
        secret_key=settings.secret_key.get_secret_value(),
        algorithm=settings.jwt_algorithm,
        access_token_expire_minutes=settings.access_token_expire_minutes,
        refresh_token_expire_days=settings.refresh_token_expire_days,
    )
    app.state.password_policy = PasswordPolicy(min_length=settings.password_min_length)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    settings: Settings = app.state.settings

    # Schema creation from metadata is a convenience for the local-lite profile only.
    # Every other environment applies Alembic migrations, so schema history stays
    # reviewable and reversible.
    if settings.is_sqlite and not settings.is_production:
        await create_all(app.state.engine)

    await ensure_first_admin(app.state.session_factory, settings, app.state.password_hasher)

    logger.info(
        "application_started",
        environment=settings.environment.value,
        database="sqlite" if settings.is_sqlite else "postgresql",
        cache=settings.cache_backend,
        inference_backend=settings.inference_backend,
    )

    yield

    await app.state.engine.dispose()
    close = getattr(app.state.cache, "close", None)
    if close is not None:
        await close()
    logger.info("application_stopped")


def create_app(settings: Settings | None = None) -> FastAPI:
    """Build a configured application instance."""
    settings = settings or get_settings()
    configure_logging(level=settings.log_level, json_output=settings.log_json)

    app = FastAPI(
        title=settings.project_name,
        description=DESCRIPTION,
        version="0.1.0",
        lifespan=lifespan,
        docs_url=None if settings.is_production else "/docs",
        redoc_url=None if settings.is_production else "/redoc",
        openapi_url=None if settings.is_production else "/openapi.json",
    )

    build_state(app, settings)

    # Middleware executes in reverse registration order, so security headers are added
    # last (outermost) and therefore apply to error responses too.
    app.add_middleware(RequestContextMiddleware)
    app.add_middleware(SecurityHeadersMiddleware, production=settings.is_production)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=True,
        allow_methods=["GET", "POST", "PATCH", "PUT", "DELETE", "OPTIONS"],
        allow_headers=["Authorization", "Content-Type", "X-Request-ID"],
        expose_headers=["X-Request-ID", "X-Response-Time-ms"],
    )

    register_exception_handlers(app)
    app.include_router(api_router, prefix=settings.api_v1_prefix)

    return app


app = create_app()
