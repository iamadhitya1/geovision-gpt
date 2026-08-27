"""ORM models — the persistence representation.

Deliberately distinct from ``app.domain.entities``. Mapping functions at the bottom of the
file are the only place the two shapes meet, which keeps schema decisions (index names,
column widths, enum storage) out of the business layer.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any
from uuid import UUID, uuid4

from sqlalchemy import Boolean, ForeignKey, Index, Integer, String, Uuid, func
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.domain.entities import AuditEvent, User
from app.domain.enums import AuditAction, AuditOutcome, Role
from app.infrastructure.db.base import Base, ensure_utc


class UserModel(Base):
    __tablename__ = "users"

    id: Mapped[UUID] = mapped_column(Uuid, primary_key=True, default=uuid4)
    email: Mapped[str] = mapped_column(String(320), unique=True, index=True, nullable=False)
    full_name: Mapped[str] = mapped_column(String(120), nullable=False)
    password_hash: Mapped[str] = mapped_column(String(255), nullable=False)

    # Stored as the enum's string value rather than a native DB enum: adding a role later
    # then needs no ALTER TYPE migration, which PostgreSQL cannot run inside a transaction.
    role: Mapped[str] = mapped_column(String(32), default=Role.RESEARCHER.value, nullable=False)

    is_active: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)
    is_verified: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    token_version: Mapped[int] = mapped_column(Integer, default=0, nullable=False)

    created_at: Mapped[datetime] = mapped_column(server_default=func.now(), nullable=False)
    # Timestamps are stamped in Python (see `apply_user_changes`) rather than with a
    # server-side `onupdate`. A server-side default expires the attribute after flush, so
    # the very next read of `updated_at` would emit an implicit SELECT — which under
    # asyncio raises MissingGreenlet, because that IO happens outside an awaited call.
    updated_at: Mapped[datetime] = mapped_column(server_default=func.now(), nullable=False)
    last_login_at: Mapped[datetime | None] = mapped_column(nullable=True)

    audit_events: Mapped[list[AuditEventModel]] = relationship(
        back_populates="user", cascade="all, delete-orphan", lazy="noload"
    )

    __table_args__ = (Index("ix_users_role_active", "role", "is_active"),)


class AuditEventModel(Base):
    __tablename__ = "audit_events"

    id: Mapped[UUID] = mapped_column(Uuid, primary_key=True, default=uuid4)
    user_id: Mapped[UUID | None] = mapped_column(
        Uuid, ForeignKey("users.id", ondelete="SET NULL"), nullable=True, index=True
    )
    action: Mapped[str] = mapped_column(String(64), nullable=False)
    outcome: Mapped[str] = mapped_column(String(16), nullable=False)

    # IPv6 in full textual form is 45 characters.
    ip_address: Mapped[str | None] = mapped_column(String(45), nullable=True)
    user_agent: Mapped[str | None] = mapped_column(String(512), nullable=True)
    detail: Mapped[dict[str, Any]] = mapped_column(default=dict, nullable=False)

    created_at: Mapped[datetime] = mapped_column(
        server_default=func.now(), nullable=False, index=True
    )

    user: Mapped[UserModel | None] = relationship(back_populates="audit_events", lazy="noload")

    # Supports the dominant analytics query: "recent events for this user".
    __table_args__ = (Index("ix_audit_user_created", "user_id", "created_at"),)


# --- Mappers -----------------------------------------------------------------


def user_to_domain(model: UserModel) -> User:
    created = ensure_utc(model.created_at)
    updated = ensure_utc(model.updated_at)
    assert created is not None and updated is not None  # NOT NULL columns
    return User(
        id=model.id,
        email=model.email,
        full_name=model.full_name,
        password_hash=model.password_hash,
        role=Role(model.role),
        is_active=model.is_active,
        is_verified=model.is_verified,
        token_version=model.token_version,
        created_at=created,
        updated_at=updated,
        last_login_at=ensure_utc(model.last_login_at),
    )


def user_to_model(entity: User) -> UserModel:
    return UserModel(
        id=entity.id,
        email=entity.email,
        full_name=entity.full_name,
        password_hash=entity.password_hash,
        role=entity.role.value,
        is_active=entity.is_active,
        is_verified=entity.is_verified,
        token_version=entity.token_version,
        created_at=entity.created_at,
        updated_at=entity.updated_at,
        last_login_at=entity.last_login_at,
    )


def apply_user_changes(model: UserModel, entity: User) -> UserModel:
    """Copy mutable domain state onto an already-attached ORM instance.

    ``updated_at`` is stamped here so every write advances it, including paths where a
    caller mutated a field directly without going through a domain method.
    """
    entity.updated_at = datetime.now(UTC)
    model.updated_at = entity.updated_at
    model.email = entity.email
    model.full_name = entity.full_name
    model.password_hash = entity.password_hash
    model.role = entity.role.value
    model.is_active = entity.is_active
    model.is_verified = entity.is_verified
    model.token_version = entity.token_version
    model.last_login_at = entity.last_login_at
    return model


def audit_to_domain(model: AuditEventModel) -> AuditEvent:
    created = ensure_utc(model.created_at)
    assert created is not None
    return AuditEvent(
        id=model.id,
        user_id=model.user_id,
        action=AuditAction(model.action),
        outcome=AuditOutcome(model.outcome),
        ip_address=model.ip_address,
        user_agent=model.user_agent,
        detail=dict(model.detail or {}),
        created_at=created,
    )


def audit_to_model(entity: AuditEvent) -> AuditEventModel:
    return AuditEventModel(
        id=entity.id,
        user_id=entity.user_id,
        action=entity.action.value,
        outcome=entity.outcome.value,
        ip_address=entity.ip_address,
        user_agent=entity.user_agent,
        detail=entity.detail,
        created_at=entity.created_at,
    )
