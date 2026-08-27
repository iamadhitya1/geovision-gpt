"""Redis-backed cache for the ``local-full`` and ``production`` profiles."""

from __future__ import annotations

from redis.asyncio import Redis

from app.domain.interfaces import Cache


class RedisCache(Cache):
    """Redis implementation of the :class:`Cache` port."""

    def __init__(self, client: Redis) -> None:
        self._client = client

    async def get(self, key: str) -> str | None:
        value = await self._client.get(key)
        return value.decode() if isinstance(value, bytes) else value

    async def set(self, key: str, value: str, *, ttl_seconds: int | None = None) -> None:
        await self._client.set(key, value, ex=ttl_seconds)

    async def delete(self, key: str) -> None:
        await self._client.delete(key)

    async def exists(self, key: str) -> bool:
        return bool(await self._client.exists(key))

    async def increment(self, key: str, *, ttl_seconds: int) -> int:
        """Increment and set the expiry only on the first hit of a window.

        Issued as a pipeline so the INCR and the EXPIRE cannot be interleaved by another
        client — without that, two concurrent first-hits can both see count 1 and one of
        them can leave the key with no TTL at all, making the limit permanent.
        """
        async with self._client.pipeline(transaction=True) as pipe:
            pipe.incr(key)
            pipe.ttl(key)
            count, current_ttl = await pipe.execute()

        if int(current_ttl) < 0:
            await self._client.expire(key, ttl_seconds)
        return int(count)

    async def ttl(self, key: str) -> int:
        return int(await self._client.ttl(key))

    async def close(self) -> None:
        await self._client.aclose()


def create_redis_client(url: str) -> Redis:
    return Redis.from_url(url, encoding="utf-8", decode_responses=True)
