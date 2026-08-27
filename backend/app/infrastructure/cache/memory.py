"""In-process cache.

Backs the ``local-lite`` profile and the test suite so that neither needs Redis running.
It is explicitly *not* production-safe: state is per-process, so with more than one worker
each replica would enforce its own rate-limit budget and hold its own refresh-token
allow-list. ``CacheFactory`` refuses to select it in production for that reason.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass

from app.domain.interfaces import Cache


@dataclass(slots=True)
class _Entry:
    value: str
    expires_at: float | None

    def is_expired(self, now: float) -> bool:
        return self.expires_at is not None and now >= self.expires_at


class InMemoryCache(Cache):
    """Dictionary-backed cache with TTL support."""

    def __init__(self) -> None:
        self._store: dict[str, _Entry] = {}
        # Serialises read-modify-write so increment() is atomic across concurrent
        # coroutines, matching the guarantee Redis INCR gives.
        self._lock = asyncio.Lock()

    async def get(self, key: str) -> str | None:
        async with self._lock:
            return self._get_live(key)

    async def set(self, key: str, value: str, *, ttl_seconds: int | None = None) -> None:
        async with self._lock:
            # `is not None` rather than a truthiness test: ttl_seconds=0 means "expire
            # immediately", and treating it as "no expiry" would store the value forever —
            # the exact inversion of what the caller asked for.
            expires_at = time.monotonic() + ttl_seconds if ttl_seconds is not None else None
            self._store[key] = _Entry(value=value, expires_at=expires_at)

    async def delete(self, key: str) -> None:
        async with self._lock:
            self._store.pop(key, None)

    async def exists(self, key: str) -> bool:
        async with self._lock:
            return self._get_live(key) is not None

    async def increment(self, key: str, *, ttl_seconds: int) -> int:
        async with self._lock:
            now = time.monotonic()
            entry = self._store.get(key)
            if entry is None or entry.is_expired(now):
                # First hit in this window sets the expiry; subsequent hits must not
                # extend it, or a steady stream of requests would never let it lapse.
                self._store[key] = _Entry(value="1", expires_at=now + ttl_seconds)
                return 1
            new_value = int(entry.value) + 1
            entry.value = str(new_value)
            return new_value

    async def ttl(self, key: str) -> int:
        async with self._lock:
            entry = self._store.get(key)
            now = time.monotonic()
            if entry is None or entry.is_expired(now) or entry.expires_at is None:
                return -1
            return max(0, int(entry.expires_at - now))

    async def clear(self) -> None:
        """Test helper — drop all state."""
        async with self._lock:
            self._store.clear()

    def _get_live(self, key: str) -> str | None:
        entry = self._store.get(key)
        if entry is None:
            return None
        if entry.is_expired(time.monotonic()):
            del self._store[key]
            return None
        return entry.value
