"""ORM models — the persistence representation.

Deliberately distinct from ``app.domain.entities``. Mapping functions at the bottom of the
file are the only place the two shapes meet, which keeps schema decisions (index names,
column widths, enum storage) out of the business layer.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any
from uuid import UUID, uuid4

from sqlalchemy import JSON, BigInteger, Boolean, ForeignKey, Index, Integer, String, Uuid, func
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.domain.entities import (
    AnswerAlternative,
    AuditEvent,
    ImageMetadata,
    SatelliteImage,
    User,
    VqaAnswer,
    VqaQuery,
)
from app.domain.enums import (
    AuditAction,
    AuditOutcome,
    FusionType,
    ImageFormat,
    ImageStatus,
    Role,
)
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


class ImageModel(Base):
    __tablename__ = "images"

    id: Mapped[UUID] = mapped_column(Uuid, primary_key=True, default=uuid4)
    # CASCADE rather than audit's SET NULL: an image with no owner isn't meaningful data
    # to retain, unlike an audit trail entry, where "someone did X" stays worth keeping
    # even after the account is gone.
    owner_id: Mapped[UUID] = mapped_column(
        Uuid, ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True
    )
    original_filename: Mapped[str] = mapped_column(String(255), nullable=False)
    image_format: Mapped[str] = mapped_column(String(16), nullable=False)
    status: Mapped[str] = mapped_column(
        String(16), default=ImageStatus.UPLOADED.value, nullable=False
    )
    storage_key: Mapped[str] = mapped_column(String(512), unique=True, nullable=False)
    thumbnail_key: Mapped[str | None] = mapped_column(String(512), nullable=True)
    file_size_bytes: Mapped[int] = mapped_column(BigInteger, nullable=False)
    checksum_sha256: Mapped[str] = mapped_column(String(64), index=True, nullable=False)
    # Named `metadata_json`, not `metadata` — SQLAlchemy's declarative base already owns
    # the `metadata` attribute name (the table registry), so reusing it here would shadow
    # that rather than define a column.
    metadata_json: Mapped[dict[str, Any] | None] = mapped_column(nullable=True)
    error_message: Mapped[str | None] = mapped_column(String(1024), nullable=True)

    created_at: Mapped[datetime] = mapped_column(server_default=func.now(), nullable=False)
    updated_at: Mapped[datetime] = mapped_column(server_default=func.now(), nullable=False)

    # Supports the dominant listing query: "this owner's images, newest first".
    __table_args__ = (Index("ix_images_owner_created", "owner_id", "created_at"),)


class VqaQueryModel(Base):
    __tablename__ = "vqa_queries"

    id: Mapped[UUID] = mapped_column(Uuid, primary_key=True, default=uuid4)
    image_id: Mapped[UUID] = mapped_column(
        Uuid, ForeignKey("images.id", ondelete="CASCADE"), nullable=False, index=True
    )
    asked_by: Mapped[UUID] = mapped_column(
        Uuid, ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True
    )
    question: Mapped[str] = mapped_column(String(500), nullable=False)
    # Every fusion variant's answer to this one question, stored together rather than as
    # child rows — see VqaQuery's own docstring: the comparison is the point, so the two
    # (or however many) answers are one fact, not rows that could drift out of sync.
    answers_json: Mapped[list[dict[str, Any]]] = mapped_column(
        JSON, default=list, nullable=False
    )

    created_at: Mapped[datetime] = mapped_column(
        server_default=func.now(), nullable=False, index=True
    )

    # Supports the dominant listing query: "this image's questions, newest first".
    __table_args__ = (Index("ix_vqa_queries_image_created", "image_id", "created_at"),)


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


def _metadata_to_json(metadata: ImageMetadata | None) -> dict[str, Any] | None:
    if metadata is None:
        return None
    return {
        "width_px": metadata.width_px,
        "height_px": metadata.height_px,
        "band_count": metadata.band_count,
        "dtype": metadata.dtype,
        "crs": metadata.crs,
        "pixel_size_x": metadata.pixel_size_x,
        "pixel_size_y": metadata.pixel_size_y,
        "gsd_meters_approx": metadata.gsd_meters_approx,
        "bounds_native": list(metadata.bounds_native) if metadata.bounds_native else None,
        "bounds_wgs84": list(metadata.bounds_wgs84) if metadata.bounds_wgs84 else None,
        "cloud_cover_percent": metadata.cloud_cover_percent,
    }


def _metadata_from_json(data: dict[str, Any] | None) -> ImageMetadata | None:
    if data is None:
        return None
    bounds_native = data.get("bounds_native")
    bounds_wgs84 = data.get("bounds_wgs84")
    return ImageMetadata(
        width_px=data["width_px"],
        height_px=data["height_px"],
        band_count=data["band_count"],
        dtype=data["dtype"],
        crs=data.get("crs"),
        pixel_size_x=data.get("pixel_size_x"),
        pixel_size_y=data.get("pixel_size_y"),
        gsd_meters_approx=data.get("gsd_meters_approx"),
        bounds_native=tuple(bounds_native) if bounds_native else None,
        bounds_wgs84=tuple(bounds_wgs84) if bounds_wgs84 else None,
        cloud_cover_percent=data.get("cloud_cover_percent"),
    )


def image_to_domain(model: ImageModel) -> SatelliteImage:
    created = ensure_utc(model.created_at)
    updated = ensure_utc(model.updated_at)
    assert created is not None and updated is not None  # NOT NULL columns
    return SatelliteImage(
        id=model.id,
        owner_id=model.owner_id,
        original_filename=model.original_filename,
        image_format=ImageFormat(model.image_format),
        storage_key=model.storage_key,
        file_size_bytes=model.file_size_bytes,
        checksum_sha256=model.checksum_sha256,
        status=ImageStatus(model.status),
        thumbnail_key=model.thumbnail_key,
        metadata=_metadata_from_json(model.metadata_json),
        error_message=model.error_message,
        created_at=created,
        updated_at=updated,
    )


def image_to_model(entity: SatelliteImage) -> ImageModel:
    return ImageModel(
        id=entity.id,
        owner_id=entity.owner_id,
        original_filename=entity.original_filename,
        image_format=entity.image_format.value,
        status=entity.status.value,
        storage_key=entity.storage_key,
        thumbnail_key=entity.thumbnail_key,
        file_size_bytes=entity.file_size_bytes,
        checksum_sha256=entity.checksum_sha256,
        metadata_json=_metadata_to_json(entity.metadata),
        error_message=entity.error_message,
        created_at=entity.created_at,
        updated_at=entity.updated_at,
    )


def apply_image_changes(model: ImageModel, entity: SatelliteImage) -> ImageModel:
    entity.updated_at = datetime.now(UTC)
    model.updated_at = entity.updated_at
    model.status = entity.status.value
    model.thumbnail_key = entity.thumbnail_key
    model.metadata_json = _metadata_to_json(entity.metadata)
    model.error_message = entity.error_message
    return model


def _answer_to_json(answer: VqaAnswer) -> dict[str, Any]:
    return {
        "fusion": answer.fusion.value,
        "answer": answer.answer,
        "confidence": answer.confidence,
        "alternatives": [
            {"answer": alt.answer, "confidence": alt.confidence} for alt in answer.alternatives
        ],
    }


def _answer_from_json(data: dict[str, Any]) -> VqaAnswer:
    return VqaAnswer(
        fusion=FusionType(data["fusion"]),
        answer=data["answer"],
        confidence=data["confidence"],
        alternatives=tuple(
            AnswerAlternative(answer=alt["answer"], confidence=alt["confidence"])
            for alt in data["alternatives"]
        ),
    )


def vqa_query_to_domain(model: VqaQueryModel) -> VqaQuery:
    created = ensure_utc(model.created_at)
    assert created is not None  # NOT NULL column
    return VqaQuery(
        id=model.id,
        image_id=model.image_id,
        asked_by=model.asked_by,
        question=model.question,
        answers=tuple(_answer_from_json(a) for a in model.answers_json),
        created_at=created,
    )


def vqa_query_to_model(entity: VqaQuery) -> VqaQueryModel:
    return VqaQueryModel(
        id=entity.id,
        image_id=entity.image_id,
        asked_by=entity.asked_by,
        question=entity.question,
        answers_json=[_answer_to_json(a) for a in entity.answers],
        created_at=entity.created_at,
    )
