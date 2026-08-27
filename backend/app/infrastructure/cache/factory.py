"""Cache backend selection."""

from __future__ import annotations

from app.core.config import Settings
from app.core.logging import get_logger
from app.domain.interfaces import Cache
from app.infrastructure.cache.memory import InMemoryCache

logger = get_logger(__name__)


def build_cache(settings: Settings) -> Cache:
    """Return the configured cache implementation.

    The in-memory backend is per-process, so in production it would silently give each
    replica its own rate-limit budget and its own refresh-token allow-list — a security
    regression that would be invisible until load-balanced. Selecting it in production is
    therefore a startup failure, not a warning.
    """
    if settings.cache_backend == "redis":
        from app.infrastructure.cache.redis_cache import RedisCache, create_redis_client

        logger.info("cache_backend_selected", backend="redis")
        return RedisCache(create_redis_client(settings.redis_url))

    if settings.is_production:
        raise RuntimeError(
            "GVG_CACHE_BACKEND=memory is not permitted in production: it is per-process, "
            "so rate limits and refresh-token revocation would not hold across replicas. "
            "Set GVG_CACHE_BACKEND=redis."
        )

    logger.info("cache_backend_selected", backend="memory")
    return InMemoryCache()
