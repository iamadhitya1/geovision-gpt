"""Alembic environment.

Reads the database URL from application settings rather than ``alembic.ini`` so a
migration can never be applied to a different database than the one the app is configured
for. Runs the async engine through ``run_sync`` because Alembic's API is synchronous.
"""

from __future__ import annotations

import asyncio
from logging.config import fileConfig

from sqlalchemy import Connection, pool
from sqlalchemy.ext.asyncio import async_engine_from_config

from alembic import context
from app.core.config import get_settings
from app.infrastructure.db import models  # noqa: F401  (registers mappers on the metadata)
from app.infrastructure.db.base import Base

config = context.config

if config.config_file_name is not None:
    fileConfig(config.config_file_name)

settings = get_settings()
config.set_main_option("sqlalchemy.url", settings.database_url)

target_metadata = Base.metadata


def _configure(connection: Connection) -> None:
    context.configure(
        connection=connection,
        target_metadata=target_metadata,
        compare_type=True,  # detect column type changes, not just added/removed columns
        compare_server_default=True,
        # Required for SQLite: it cannot ALTER most columns, so Alembic rebuilds the
        # table inside a batch operation instead.
        render_as_batch=settings.is_sqlite,
    )


def run_migrations_offline() -> None:
    """Emit SQL to stdout without connecting — used to hand a DBA a reviewable script."""
    context.configure(
        url=settings.database_url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
        render_as_batch=settings.is_sqlite,
    )
    with context.begin_transaction():
        context.run_migrations()


def _do_run(connection: Connection) -> None:
    _configure(connection)
    with context.begin_transaction():
        context.run_migrations()


async def run_migrations_online() -> None:
    connectable = async_engine_from_config(
        config.get_section(config.config_ini_section, {}),
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )
    async with connectable.connect() as connection:
        await connection.run_sync(_do_run)
    await connectable.dispose()


if context.is_offline_mode():
    run_migrations_offline()
else:
    asyncio.run(run_migrations_online())
