"""Module 3/4 — VQA queries.

Revision ID: 0003_vqa_queries
Revises: 0002_images
Created: 2026-09-17
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import JSONB

from alembic import op

revision: str = "0003_vqa_queries"
down_revision: str | None = "0002_images"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

JSON_VARIANT = sa.JSON().with_variant(JSONB(), "postgresql")


def upgrade() -> None:
    op.create_table(
        "vqa_queries",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("image_id", sa.Uuid(), nullable=False),
        sa.Column("asked_by", sa.Uuid(), nullable=False),
        sa.Column("question", sa.String(length=500), nullable=False),
        # Every fusion variant's answer to this one question, stored together — see
        # VqaQuery's own docstring: the baseline-vs-cross-attention comparison is one
        # fact, not rows that could drift out of sync with each other.
        sa.Column("answers_json", JSON_VARIANT, server_default="[]", nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.PrimaryKeyConstraint("id", name="pk_vqa_queries"),
        sa.ForeignKeyConstraint(
            ["image_id"], ["images.id"], name="fk_vqa_queries_image_id_images", ondelete="CASCADE"
        ),
        sa.ForeignKeyConstraint(
            ["asked_by"], ["users.id"], name="fk_vqa_queries_asked_by_users", ondelete="CASCADE"
        ),
    )
    op.create_index("ix_vqa_queries_image_id", "vqa_queries", ["image_id"])
    op.create_index("ix_vqa_queries_asked_by", "vqa_queries", ["asked_by"])
    # Supports the dominant listing query: "this image's questions, newest first".
    op.create_index("ix_vqa_queries_image_created", "vqa_queries", ["image_id", "created_at"])


def downgrade() -> None:
    op.drop_index("ix_vqa_queries_image_created", table_name="vqa_queries")
    op.drop_index("ix_vqa_queries_asked_by", table_name="vqa_queries")
    op.drop_index("ix_vqa_queries_image_id", table_name="vqa_queries")
    op.drop_table("vqa_queries")
