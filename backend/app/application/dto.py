"""Data Transfer Objects — the wire contract.

Separate from domain entities on purpose: ``UserRead`` must never grow a
``password_hash`` field just because ``User`` has one. Keeping the types distinct makes
that class of leak impossible rather than merely discouraged.
"""

from __future__ import annotations

import base64
from datetime import datetime
from typing import Annotated, Any, Generic, TypeVar
from uuid import UUID

from pydantic import BaseModel, ConfigDict, EmailStr, Field, field_validator

from app.domain.concepts import question_kind, resolve_targets
from app.domain.entities import (
    AnswerAlternative,
    AttentionHeatmap,
    AuditEvent,
    ConceptHighlight,
    HighlightLayer,
    ImageMetadata,
    SatelliteImage,
    TokenPair,
    User,
    VqaAnswer,
    VqaQuery,
)
from app.domain.enums import AuditAction, AuditOutcome, FusionType, ImageFormat, ImageStatus, Role

T = TypeVar("T")

PasswordStr = Annotated[str, Field(min_length=8, max_length=128)]
NameStr = Annotated[str, Field(min_length=1, max_length=120)]


# --- Requests ----------------------------------------------------------------


class RegisterRequest(BaseModel):
    model_config = ConfigDict(
        str_strip_whitespace=True,
        json_schema_extra={
            "example": {
                "email": "researcher@geovisiongpt.io",
                "full_name": "Ada Lovelace",
                "password": "Ge0Vision_Strong_2026",
            }
        },
    )

    email: EmailStr
    full_name: NameStr
    password: PasswordStr

    @field_validator("email")
    @classmethod
    def _normalise_email(cls, value: str) -> str:
        # Stored lowercase so that lookups and the unique constraint agree.
        return value.lower()


class LoginRequest(BaseModel):
    # The example matches the bootstrap admin created on first startup, so a first-time
    # reader can click "Try it out" -> "Execute" with zero edits and see a real result.
    model_config = ConfigDict(
        str_strip_whitespace=True,
        json_schema_extra={
            "example": {
                "email": "admin@geovisiongpt.io",
                "password": "ChangeMe_Please_2026",
            }
        },
    )

    email: EmailStr
    password: str

    @field_validator("email")
    @classmethod
    def _normalise_email(cls, value: str) -> str:
        return value.lower()


class RefreshRequest(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "example": {"refresh_token": "<paste the refresh_token from your login response>"}
        }
    )

    refresh_token: str = Field(min_length=1)


class ChangePasswordRequest(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "current_password": "ChangeMe_Please_2026",
                "new_password": "An0ther_Strong_2026",
            }
        }
    )

    current_password: str = Field(min_length=1)
    new_password: PasswordStr


class UpdateProfileRequest(BaseModel):
    model_config = ConfigDict(
        str_strip_whitespace=True,
        json_schema_extra={"example": {"full_name": "Ada Lovelace"}},
    )

    full_name: NameStr | None = None


class AdminUpdateUserRequest(BaseModel):
    """Admin-only mutations, kept apart from self-service updates so that role
    escalation can never ride in on a profile edit."""

    model_config = ConfigDict(
        json_schema_extra={"example": {"role": "admin", "is_active": True, "is_verified": True}}
    )

    role: Role | None = None
    is_active: bool | None = None
    is_verified: bool | None = None


class CaptureSceneRequest(BaseModel):
    """A point on the map to fetch a satellite scene for. Latitude is limited to +-80
    degrees: beyond that the Web Mercator map and the Sentinel-2 mosaic both thin out."""

    model_config = ConfigDict(
        json_schema_extra={"example": {"latitude": 52.9554, "longitude": 5.7743}}
    )

    latitude: Annotated[float, Field(ge=-80.0, le=80.0)]
    longitude: Annotated[float, Field(ge=-180.0, le=180.0)]


class AskQuestionRequest(BaseModel):
    """``fusions``, not ``fusion``: this project's whole original contribution is the
    baseline-vs-cross-attention comparison, so the default is to answer with every model
    variant that's actually loaded, not just one. A client that only wants one still can,
    by passing a single-item list."""

    model_config = ConfigDict(
        str_strip_whitespace=True,
        json_schema_extra={
            "example": {
                "question": "Is this a rural or an urban area?",
                "fusions": ["elementwise", "cross_attention"],
            }
        },
    )

    question: Annotated[str, Field(min_length=1, max_length=500)]
    fusions: list[FusionType] | None = None  # None means "every fusion currently loaded"


class ExplainRequest(BaseModel):
    """Body for the attention-heatmap endpoint — just the question, since this always
    targets cross-attention specifically (see `AttentionHeatmapRead`)."""

    model_config = ConfigDict(
        str_strip_whitespace=True,
        json_schema_extra={"example": {"question": "Is this a rural or an urban area?"}},
    )

    question: Annotated[str, Field(min_length=1, max_length=500)]


# --- Responses ---------------------------------------------------------------


class TokenResponse(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "access_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
                "refresh_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
                "token_type": "bearer",
                "expires_in": 1800,
            }
        }
    )

    access_token: str
    refresh_token: str
    token_type: str = "bearer"
    expires_in: int

    @classmethod
    def from_domain(cls, pair: TokenPair) -> TokenResponse:
        return cls(
            access_token=pair.access_token,
            refresh_token=pair.refresh_token,
            token_type=pair.token_type,
            expires_in=pair.expires_in,
        )


class UserRead(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "id": "6bdc72d1-dc8a-4dc0-a972-082f42c25c76",
                "email": "researcher@geovisiongpt.io",
                "full_name": "Ada Lovelace",
                "role": "researcher",
                "is_active": True,
                "is_verified": False,
                "created_at": "2026-08-27T09:15:13.280499Z",
                "last_login_at": "2026-08-27T09:19:12.817502Z",
            }
        }
    )

    id: UUID
    email: EmailStr
    full_name: str
    role: Role
    is_active: bool
    is_verified: bool
    created_at: datetime
    last_login_at: datetime | None = None

    @classmethod
    def from_domain(cls, user: User) -> UserRead:
        return cls(
            id=user.id,
            email=user.email,
            full_name=user.full_name,
            role=user.role,
            is_active=user.is_active,
            is_verified=user.is_verified,
            created_at=user.created_at,
            last_login_at=user.last_login_at,
        )


class AuthenticatedUserResponse(BaseModel):
    """Returned by registration and login: identity plus credentials in one round trip."""

    user: UserRead
    tokens: TokenResponse


class AuditEventRead(BaseModel):
    id: UUID
    user_id: UUID | None
    action: AuditAction
    outcome: AuditOutcome
    ip_address: str | None
    user_agent: str | None
    detail: dict[str, Any]
    created_at: datetime

    @classmethod
    def from_domain(cls, event: AuditEvent) -> AuditEventRead:
        return cls(
            id=event.id,
            user_id=event.user_id,
            action=event.action,
            outcome=event.outcome,
            ip_address=event.ip_address,
            user_agent=event.user_agent,
            detail=event.detail,
            created_at=event.created_at,
        )


class ImageMetadataRead(BaseModel):
    """Everything extracted from the raster itself. Geospatial fields are ``null`` for a
    PNG/JPEG, or for a TIFF that carries no embedded CRS — that's a legitimate state, not
    an error, and the API says so explicitly rather than omitting the fields."""

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "width_px": 512,
                "height_px": 512,
                "band_count": 3,
                "dtype": "uint8",
                "crs": "EPSG:4326",
                "pixel_size_x": 0.0001,
                "pixel_size_y": 0.0001,
                "gsd_meters_approx": 11.13,
                "bounds_native": [72.5, 23.0968, 72.5512, 23.148],
                "bounds_wgs84": [72.5, 23.0968, 72.5512, 23.148],
                "cloud_cover_percent": 4.2,
            }
        }
    )

    width_px: int
    height_px: int
    band_count: int
    dtype: str
    crs: str | None = None
    pixel_size_x: float | None = None
    pixel_size_y: float | None = None
    gsd_meters_approx: float | None = None
    bounds_native: tuple[float, float, float, float] | None = None
    bounds_wgs84: tuple[float, float, float, float] | None = None
    cloud_cover_percent: float | None = None

    @classmethod
    def from_domain(cls, metadata: ImageMetadata) -> ImageMetadataRead:
        return cls(
            width_px=metadata.width_px,
            height_px=metadata.height_px,
            band_count=metadata.band_count,
            dtype=metadata.dtype,
            crs=metadata.crs,
            pixel_size_x=metadata.pixel_size_x,
            pixel_size_y=metadata.pixel_size_y,
            gsd_meters_approx=metadata.gsd_meters_approx,
            bounds_native=metadata.bounds_native,
            bounds_wgs84=metadata.bounds_wgs84,
            cloud_cover_percent=metadata.cloud_cover_percent,
        )


class ImageRead(BaseModel):
    id: UUID
    owner_id: UUID
    original_filename: str
    image_format: ImageFormat
    status: ImageStatus
    file_size_bytes: int
    checksum_sha256: str
    metadata: ImageMetadataRead | None = None
    error_message: str | None = None
    thumbnail_available: bool
    created_at: datetime
    updated_at: datetime

    @classmethod
    def from_domain(cls, image: SatelliteImage) -> ImageRead:
        return cls(
            id=image.id,
            owner_id=image.owner_id,
            original_filename=image.original_filename,
            image_format=image.image_format,
            status=image.status,
            file_size_bytes=image.file_size_bytes,
            checksum_sha256=image.checksum_sha256,
            metadata=ImageMetadataRead.from_domain(image.metadata) if image.metadata else None,
            error_message=image.error_message,
            thumbnail_available=image.thumbnail_key is not None,
            created_at=image.created_at,
            updated_at=image.updated_at,
        )


class AnswerAlternativeRead(BaseModel):
    answer: str
    confidence: float

    @classmethod
    def from_domain(cls, alternative: AnswerAlternative) -> AnswerAlternativeRead:
        return cls(answer=alternative.answer, confidence=alternative.confidence)


class VqaAnswerRead(BaseModel):
    fusion: FusionType
    answer: str
    confidence: float
    alternatives: list[AnswerAlternativeRead]
    note: str | None = None

    @classmethod
    def from_domain(cls, answer: VqaAnswer) -> VqaAnswerRead:
        return cls(
            fusion=answer.fusion,
            answer=answer.answer,
            confidence=answer.confidence,
            alternatives=[AnswerAlternativeRead.from_domain(a) for a in answer.alternatives],
            note=answer.note,
        )


class AttentionHeatmapRead(BaseModel):
    """A `grid_size` x `grid_size` grid of attention weights, flattened row-major, each in
    [0, 1] and summing to ~1.0 across the grid — render as a semi-transparent overlay on
    the source image. Interpretability only: shows where the model attended, not a
    verified ground-truth location."""

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "fusion": "cross_attention",
                "answer": "urban",
                "confidence": 0.88,
                "grid_size": 14,
                "weights": [0.004, 0.004, 0.009, "... 196 values total"],
            }
        }
    )

    fusion: FusionType
    answer: str
    confidence: float
    grid_size: int
    weights: list[float]

    @classmethod
    def from_domain(cls, heatmap: AttentionHeatmap) -> AttentionHeatmapRead:
        return cls(
            fusion=heatmap.fusion,
            answer=heatmap.answer,
            confidence=heatmap.confidence,
            grid_size=heatmap.grid_size,
            weights=list(heatmap.weights),
        )


class HighlightLayerRead(BaseModel):
    """One thing the question named, and how it was found."""

    concept: str
    label: str  # what to call it on screen
    color: str  # "#rrggbb" — the colour it is outlined in on the overlay
    coverage: float  # fraction of the image it covers
    method: str
    caveat: str | None = None

    @classmethod
    def from_domain(cls, layer: HighlightLayer) -> HighlightLayerRead:
        return cls(
            concept=layer.concept,
            label=layer.label,
            color=layer.color,
            coverage=layer.coverage,
            method=layer.method,
            caveat=layer.caveat,
        )


class HighlightRead(BaseModel):
    """Where the thing(s) a question names appear in the image, as one transparent overlay
    PNG the same shape as the image: everything outside them dimmed, each outlined in its own
    colour (see `layers` for the legend).

    Produced by a separate step, not by the trained VQA model (which cannot locate anything)
    — each layer's `method` says which technique (a colour index, a zero-shot segmenter, or
    OpenStreetMap map data) and `caveat` says how far to trust it.
    """

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "layers": [
                    {
                        "concept": "roads",
                        "label": "roads",
                        "color": "#ff5a5a",
                        "coverage": 0.04,
                        "method": "OpenStreetMap map data — not detected from the image",
                        "caveat": "Drawn from map data at this scene's coordinates…",
                    }
                ],
                "overlay_png_base64": "iVBORw0KGgo... (PNG, base64)",
                "caveat": None,
            }
        }
    )

    layers: list[HighlightLayerRead]
    overlay_png_base64: str
    caveat: str | None = None

    @classmethod
    def from_domain(cls, highlight: ConceptHighlight) -> HighlightRead:
        return cls(
            layers=[HighlightLayerRead.from_domain(layer) for layer in highlight.layers],
            overlay_png_base64=base64.b64encode(highlight.overlay_png).decode("ascii"),
            caveat=highlight.caveat,
        )


class VqaQueryRead(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "id": "b2f6a1e4-9d3c-4f8a-9e21-6a4b7c8d9e0f",
                "image_id": "6bdc72d1-dc8a-4dc0-a972-082f42c25c76",
                "question": "Is this a rural or an urban area?",
                "answers": [
                    {
                        "fusion": "elementwise",
                        "answer": "urban",
                        "confidence": 0.91,
                        "alternatives": [
                            {"answer": "urban", "confidence": 0.91},
                            {"answer": "rural", "confidence": 0.07},
                        ],
                    },
                    {
                        "fusion": "cross_attention",
                        "answer": "urban",
                        "confidence": 0.88,
                        "alternatives": [
                            {"answer": "urban", "confidence": 0.88},
                            {"answer": "rural", "confidence": 0.1},
                        ],
                    },
                ],
                "created_at": "2026-09-17T10:03:44.120000Z",
            }
        }
    )

    id: UUID
    image_id: UUID
    question: str
    answers: list[VqaAnswerRead]
    created_at: datetime
    # What in the question can be highlighted on the image (one thing, or two for a
    # comparison), and what kind of question it is — a "no" to a presence question means there is
    # nothing to point at, a "no" to a comparison does not. Worked out from the text, not stored.
    highlight_targets: list[str] = Field(default_factory=list)
    highlight_kind: str = "presence"

    @classmethod
    def from_domain(cls, query: VqaQuery) -> VqaQueryRead:
        return cls(
            id=query.id,
            image_id=query.image_id,
            question=query.question,
            answers=[VqaAnswerRead.from_domain(a) for a in query.answers],
            created_at=query.created_at,
            highlight_targets=[c.value for c in resolve_targets(query.question)],
            highlight_kind=question_kind(query.question),
        )


class FusionUsageRead(BaseModel):
    fusion: FusionType
    answered: int
    mean_confidence: float


class AgreementRead(BaseModel):
    """How often the two model variants gave the same top answer, over the questions both
    of them answered — the live, in-app view of the project's baseline-vs-cross-attention
    comparison (the held-out test-set accuracy lives in ``ml/README.md``)."""

    compared: int
    agreed: int


class VqaStatsRead(BaseModel):
    total_questions: int
    # Everything below is computed over the most recent ``sample_size`` questions, so
    # ``sample_size < total_questions`` means the figures describe the latest slice.
    sample_size: int
    images_queried: int
    by_fusion: list[FusionUsageRead]
    agreement: AgreementRead
    recent: list[VqaQueryRead]


class Page(BaseModel, Generic[T]):
    """Uniform pagination envelope reused by every list endpoint in the project."""

    items: list[T]
    total: int
    offset: int
    limit: int

    @property
    def has_more(self) -> bool:
        return self.offset + len(self.items) < self.total


class MessageResponse(BaseModel):
    message: str


class ErrorResponse(BaseModel):
    """The single error shape emitted by the API.

    Every error the API returns — validation, auth, not-found, rate limit, anything —
    has exactly this shape, so a client only ever needs one error-handling code path.
    """

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "code": "invalid_credentials",
                "message": "Incorrect email or password.",
                "detail": {},
                "request_id": "4833200cf6764c68b9198cb2f674427e",
            }
        }
    )

    code: str
    message: str
    detail: dict[str, Any] = Field(default_factory=dict)
    request_id: str | None = None


# --- Internal command objects ------------------------------------------------


class RequestContext(BaseModel):
    """Ambient caller information threaded into services for auditing.

    Passing this explicitly keeps services free of any dependency on the web request
    object, which is what allows them to be unit-tested without FastAPI.
    """

    ip_address: str | None = None
    user_agent: str | None = None
