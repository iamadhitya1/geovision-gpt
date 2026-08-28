"""Module 2 — satellite images.

Revision ID: 0002_images
Revises: 0001_initial_auth
Created: 2026-08-28
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import JSONB

from alembic import op

revision: str = "0002_images"
down_revision: str | None = "0001_initial_auth"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

JSON_VARIANT = sa.JSON().with_variant(JSONB(), "postgresql")


def upgrade() -> None:
    op.create_table(
        "images",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("owner_id", sa.Uuid(), nullable=False),
        sa.Column("original_filename", sa.String(length=255), nullable=False),
        sa.Column("image_format", sa.String(length=16), nullable=False),
        sa.Column("status", sa.String(length=16), server_default="uploaded", nullable=False),
        sa.Column("storage_key", sa.String(length=512), nullable=False),
        sa.Column("thumbnail_key", sa.String(length=512), nullable=True),
        sa.Column("file_size_bytes", sa.BigInteger(), nullable=False),
        sa.Column("checksum_sha256", sa.String(length=64), nullable=False),
        sa.Column("metadata_json", JSON_VARIANT, nullable=True),
        sa.Column("error_message", sa.String(length=1024), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.PrimaryKeyConstraint("id", name="pk_images"),
        sa.UniqueConstraint("storage_key", name="uq_images_storage_key"),
        # CASCADE, not audit_events' SET NULL: an image with no owner isn't meaningful
        # data to retain, unlike an audit trail entry.
        sa.ForeignKeyConstraint(
            ["owner_id"], ["users.id"], name="fk_images_owner_id_users", ondelete="CASCADE"
        ),
    )
    op.create_index("ix_images_owner_id", "images", ["owner_id"])
    op.create_index("ix_images_checksum_sha256", "images", ["checksum_sha256"])
    # Supports the dominant listing query: "this owner's images, newest first".
    op.create_index("ix_images_owner_created", "images", ["owner_id", "created_at"])


def downgrade() -> None:
    op.drop_index("ix_images_owner_created", table_name="images")
    op.drop_index("ix_images_checksum_sha256", table_name="images")
    op.drop_index("ix_images_owner_id", table_name="images")
    op.drop_table("images")
