"""Fixed-window rate limiting built on the :class:`Cache` port.

Fixed window is chosen over sliding-log for cost: one counter per key instead of a list of
timestamps. Its known weakness is burst tolerance at a window boundary (up to 2x the limit
across two adjacent windows), which is acceptable for login throttling where the objective
is defeating sustained credential-stuffing rather than perfectly smooth pacing.
"""

from __future__ import annotations

from app.core.logging import get_logger
from app.domain.errors import RateLimitExceededError
from app.domain.interfaces import Cache, RateLimiter

logger = get_logger(__name__)


class FixedWindowRateLimiter(RateLimiter):
    """Counts requests per key per window and rejects once the budget is exhausted."""

    def __init__(self, cache: Cache, *, namespace: str = "ratelimit") -> None:
        self._cache = cache
        self._namespace = namespace

    async def hit(self, key: str, *, limit: int, window_seconds: int) -> None:
        namespaced = f"{self._namespace}:{key}"
        count = await self._cache.increment(namespaced, ttl_seconds=window_seconds)

        if count > limit:
            retry_after = await self._cache.ttl(namespaced)
            retry_after = window_seconds if retry_after < 0 else retry_after
            logger.warning("rate_limit_exceeded", key=key, count=count, limit=limit)
            raise RateLimitExceededError(retry_after_seconds=retry_after)


class NullRateLimiter(RateLimiter):
    """No-op limiter used when rate limiting is disabled (tests, local debugging)."""

    async def hit(self, key: str, *, limit: int, window_seconds: int) -> None:
        return None
