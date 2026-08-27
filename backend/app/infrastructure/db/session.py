"""Async engine, session factory and unit of work."""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from app.core.config import Settings
from app.core.logging import get_logger
from app.domain.interfaces import UnitOfWork

logger = get_logger(__name__)


def create_engine(settings: Settings) -> AsyncEngine:
    """Build the async engine for the configured database.

    SQLite and PostgreSQL take different pooling arguments; passing PostgreSQL pool sizes
    to SQLite raises at connect time, so the branch is required rather than cosmetic.
    """
    if settings.is_sqlite:
        return create_async_engine(
            settings.database_url,
            echo=settings.db_echo,
            future=True,
            connect_args={"check_same_thread": False},
        )

    return create_async_engine(
        settings.database_url,
        echo=settings.db_echo,
        future=True,
        pool_size=settings.db_pool_size,
        max_overflow=settings.db_max_overflow,
        pool_pre_ping=True,  # drop connections killed by an idle proxy before use
        pool_recycle=1800,
    )


def create_session_factory(engine: AsyncEngine) -> async_sessionmaker[AsyncSession]:
    return async_sessionmaker(
        bind=engine,
        class_=AsyncSession,
        expire_on_commit=False,  # entities stay usable after commit inside a request
        autoflush=False,
    )


class SqlAlchemyUnitOfWork(UnitOfWork):
    """Transaction boundary over an :class:`AsyncSession`.

    Services call ``commit()`` at the end of a use case; the request-scoped dependency in
    ``api/deps.py`` rolls back on any unhandled exception, so a half-applied use case can
    never be persisted.
    """

    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def commit(self) -> None:
        await self._session.commit()

    async def rollback(self) -> None:
        await self._session.rollback()


@asynccontextmanager
async def session_scope(
    factory: async_sessionmaker[AsyncSession],
) -> AsyncIterator[AsyncSession]:
    """Provide a session that rolls back on failure and always closes."""
    session = factory()
    try:
        yield session
    except Exception:
        await session.rollback()
        raise
    finally:
        await session.close()


async def create_all(engine: AsyncEngine) -> None:
    """Create tables directly from metadata.

    Used only by the ``local-lite`` profile and the test suite. Every other environment
    goes through Alembic so that schema history is reviewable.
    """
    from app.infrastructure.db import models  # noqa: F401  (registers mappers)
    from app.infrastructure.db.base import Base

    async with engine.begin() as connection:
        await connection.run_sync(Base.metadata.create_all)
    logger.info("database_schema_created")
