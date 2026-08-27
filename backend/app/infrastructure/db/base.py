"""SQLAlchemy declarative base and shared column conventions."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any, ClassVar

from sqlalchemy import JSON, DateTime, MetaData
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import DeclarativeBase

# Explicit naming convention so Alembic autogenerate produces stable, diffable
# constraint names instead of database-assigned ones.
NAMING_CONVENTION = {
    "ix": "ix_%(column_0_label)s",
    "uq": "uq_%(table_name)s_%(column_0_name)s",
    "ck": "ck_%(table_name)s_%(constraint_name)s",
    "fk": "fk_%(table_name)s_%(column_0_name)s_%(referred_table_name)s",
    "pk": "pk_%(table_name)s",
}

# JSONB on PostgreSQL (indexable, binary) degrading to plain JSON on SQLite.
JSONVariant = JSON().with_variant(JSONB(), "postgresql")

TZDateTime = DateTime(timezone=True)


class Base(DeclarativeBase):
    metadata = MetaData(naming_convention=NAMING_CONVENTION)

    # Maps Python annotations on `Mapped[...]` to column types, so models declare
    # `Mapped[dict[str, Any]]` and get JSONB/JSON without repeating the type each time.
    type_annotation_map: ClassVar[dict[Any, Any]] = {
        dict[str, Any]: JSONVariant,
        datetime: TZDateTime,
    }


def ensure_utc(value: datetime | None) -> datetime | None:
    """Return ``value`` as timezone-aware UTC.

    SQLite has no native timezone storage, so values written as aware come back naive.
    Every read path funnels through here to keep the domain layer's invariant — all
    datetimes are aware and in UTC — true regardless of which database is behind it.
    """
    if value is None:
        return None
    if value.tzinfo is None:
        return value.replace(tzinfo=UTC)
    return value.astimezone(UTC)
