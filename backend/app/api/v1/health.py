"""Health and readiness probes."""

from __future__ import annotations

from typing import Annotated, Any

from fastapi import APIRouter, Depends, Response, status
from sqlalchemy import text

from app.api.deps import SessionDep, get_cache, get_settings_dep
from app.core.config import Settings
from app.core.logging import get_logger
from app.domain.interfaces import Cache

router = APIRouter(tags=["System"])
logger = get_logger(__name__)


@router.get("/health", summary="Liveness probe")
async def health(settings: Annotated[Settings, Depends(get_settings_dep)]) -> dict[str, Any]:
    """Answers "is the process up?" only.

    Kept free of dependency checks on purpose: if this returned unhealthy whenever the
    database blipped, the orchestrator would restart otherwise-healthy pods and turn a
    brief database outage into a restart storm.
    """
    return {
        "status": "ok",
        "service": settings.project_name,
        "environment": settings.environment.value,
        "version": "0.1.0",
    }


@router.get("/ready", summary="Readiness probe")
async def ready(
    session: SessionDep,
    cache: Annotated[Cache, Depends(get_cache)],
    response: Response,
) -> dict[str, Any]:
    """Answers "can this instance serve traffic?" by exercising each dependency.

    Returns 503 when any check fails so the load balancer removes this instance rather
    than routing requests it cannot fulfil.
    """
    checks: dict[str, str] = {}

    try:
        await session.execute(text("SELECT 1"))
        checks["database"] = "ok"
    except Exception as exc:
        logger.warning("readiness_database_failed", error=str(exc))
        checks["database"] = "error"

    try:
        await cache.set("healthcheck", "1", ttl_seconds=5)
        checks["cache"] = "ok" if await cache.exists("healthcheck") else "error"
    except Exception as exc:
        logger.warning("readiness_cache_failed", error=str(exc))
        checks["cache"] = "error"

    healthy = all(value == "ok" for value in checks.values())
    if not healthy:
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE

    return {"status": "ready" if healthy else "degraded", "checks": checks}
