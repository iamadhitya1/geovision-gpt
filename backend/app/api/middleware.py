"""HTTP middleware: request correlation, access logging, security headers."""

from __future__ import annotations

import time
from collections.abc import Awaitable, Callable
from uuid import uuid4

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

from app.core.logging import get_logger, request_id_ctx

logger = get_logger(__name__)

RequestHandler = Callable[[Request], Awaitable[Response]]


class RequestContextMiddleware(BaseHTTPMiddleware):
    """Assign a request id, bind it to the logging context, and echo it back.

    An inbound ``X-Request-ID`` is honoured so a trace can be followed across the
    frontend, NGINX and the API rather than restarting at our boundary.
    """

    async def dispatch(self, request: Request, call_next: RequestHandler) -> Response:
        request_id = request.headers.get("X-Request-ID") or uuid4().hex
        token = request_id_ctx.set(request_id)
        request.state.request_id = request_id

        started = time.perf_counter()
        try:
            response = await call_next(request)
        except Exception:
            duration_ms = (time.perf_counter() - started) * 1000
            logger.exception(
                "request_failed",
                method=request.method,
                path=request.url.path,
                duration_ms=round(duration_ms, 2),
            )
            raise
        finally:
            request_id_ctx.reset(token)

        duration_ms = (time.perf_counter() - started) * 1000
        response.headers["X-Request-ID"] = request_id
        response.headers["X-Response-Time-ms"] = f"{duration_ms:.2f}"

        logger.info(
            "request_completed",
            method=request.method,
            path=request.url.path,
            status=response.status_code,
            duration_ms=round(duration_ms, 2),
            client=request.client.host if request.client else None,
        )
        return response


class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    """Apply baseline security headers to every response.

    HSTS is emitted only in production: sending it from a local HTTP dev server pins the
    browser to HTTPS on ``localhost`` and breaks development for that port.
    """

    def __init__(self, app: object, *, production: bool = False) -> None:
        super().__init__(app)  # type: ignore[arg-type]
        self._production = production

    async def dispatch(self, request: Request, call_next: RequestHandler) -> Response:
        response = await call_next(request)
        response.headers.setdefault("X-Content-Type-Options", "nosniff")
        response.headers.setdefault("X-Frame-Options", "DENY")
        response.headers.setdefault("Referrer-Policy", "strict-origin-when-cross-origin")
        response.headers.setdefault(
            "Permissions-Policy", "geolocation=(self), camera=(), microphone=()"
        )
        if self._production:
            response.headers.setdefault(
                "Strict-Transport-Security", "max-age=31536000; includeSubDomains"
            )
        return response
