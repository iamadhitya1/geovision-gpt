"""Domain entities — framework-free business objects.

Deliberately plain dataclasses rather than SQLAlchemy models or Pydantic schemas. The
persistence model (``infrastructure/db/models.py``) and the wire model
(``application/dto.py``) are separate types that map onto these, so a column rename or an
API field change cannot silently alter business rules.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any, Literal
from uuid import UUID, uuid4

from app.domain.enums import AuditAction, AuditOutcome, FusionType, ImageFormat, ImageStatus, Role


def _now() -> datetime:
    return datetime.now(UTC)


@dataclass(slots=True)
class User:
    """A platform user.

    ``token_version`` is the global revocation lever: every refresh token embeds the
    version current at issue time, so incrementing it invalidates all outstanding
    sessions for this user in one write, without a token table to sweep.
    """

    email: str
    full_name: str
    password_hash: str
    role: Role = Role.RESEARCHER
    id: UUID = field(default_factory=uuid4)
    is_active: bool = True
    is_verified: bool = False
    token_version: int = 0
    created_at: datetime = field(default_factory=_now)
    updated_at: datetime = field(default_factory=_now)
    last_login_at: datetime | None = None

    # --- Business rules ------------------------------------------------------

    @property
    def is_admin(self) -> bool:
        return self.role is Role.ADMIN

    def can_login(self) -> bool:
        return self.is_active

    def has_role(self, required: Role) -> bool:
        """Hierarchical check: an admin satisfies a researcher requirement."""
        return self.role.outranks_or_equals(required)

    def record_login(self) -> None:
        self.last_login_at = _now()
        self.updated_at = self.last_login_at

    def revoke_all_sessions(self) -> None:
        """Invalidate every outstanding refresh token for this user."""
        self.token_version += 1
        self.updated_at = _now()

    def change_password(self, new_hash: str) -> None:
        """A password change must also end existing sessions, or a stolen refresh token
        survives the very rotation intended to defeat it."""
        self.password_hash = new_hash
        self.revoke_all_sessions()

    def deactivate(self) -> None:
        self.is_active = False
        self.revoke_all_sessions()

    def activate(self) -> None:
        self.is_active = True
        self.updated_at = _now()

    def change_role(self, new_role: Role) -> None:
        self.role = new_role
        self.updated_at = _now()


@dataclass(frozen=True, slots=True)
class TokenPair:
    """Issued credential pair returned to the client."""

    access_token: str
    refresh_token: str
    token_type: str = "bearer"
    expires_in: int = 0  # seconds until the access token expires


@dataclass(frozen=True, slots=True)
class TokenClaims:
    """Decoded, validated JWT payload."""

    subject: UUID
    role: Role
    token_type: str
    jti: str
    issued_at: datetime
    expires_at: datetime
    token_version: int = 0


@dataclass(frozen=True, slots=True)
class ImageMetadata:
    """Everything extracted by inspecting the raster itself, as opposed to what the
    uploader claimed. ``width_px``/``height_px``/``band_count``/``dtype`` apply to every
    format; the geospatial fields are populated only when the source carries embedded
    georeferencing, which in practice means GeoTIFF with a CRS actually set.
    """

    width_px: int
    height_px: int
    band_count: int
    dtype: str  # e.g. "uint8", "uint16", "float32" — the raster's native pixel type

    crs: str | None = None  # e.g. "EPSG:4326"; None when the source has no georeferencing
    pixel_size_x: float | None = None  # native-CRS units per pixel (not always metres)
    pixel_size_y: float | None = None
    gsd_meters_approx: float | None = None  # approximate ground sample distance, in metres
    bounds_native: tuple[float, float, float, float] | None = None  # minx,miny,maxx,maxy
    bounds_wgs84: tuple[float, float, float, float] | None = None  # west,south,east,north

    # A brightness-ratio heuristic, not a real cloud mask — see the SGV research doc for
    # why this project doesn't trust a VLM's own cloud-cover guess either. None when the
    # image has fewer than 3 bands (the heuristic needs RGB-like channels to be meaningful).
    cloud_cover_percent: float | None = None


@dataclass(slots=True)
class SatelliteImage:
    """An uploaded raster and everything known about it.

    ``metadata`` is ``None`` until processing finishes (or forever, if it fails) — the
    entity is created and persisted immediately on upload so the record exists even if
    extraction subsequently fails, and ``error_message`` explains why.
    """

    owner_id: UUID
    original_filename: str
    image_format: ImageFormat
    storage_key: str
    file_size_bytes: int
    checksum_sha256: str
    id: UUID = field(default_factory=uuid4)
    status: ImageStatus = ImageStatus.UPLOADED
    thumbnail_key: str | None = None
    metadata: ImageMetadata | None = None
    error_message: str | None = None
    created_at: datetime = field(default_factory=_now)
    updated_at: datetime = field(default_factory=_now)

    def mark_processing(self) -> None:
        self.status = ImageStatus.PROCESSING
        self.updated_at = _now()

    def mark_ready(self, metadata: ImageMetadata, *, thumbnail_key: str | None) -> None:
        """``thumbnail_key`` is Optional: thumbnailing is best-effort and a failure there
        must not turn a successful upload with valid metadata into a failed one."""
        self.status = ImageStatus.READY
        self.metadata = metadata
        self.thumbnail_key = thumbnail_key
        self.error_message = None
        self.updated_at = _now()

    def mark_failed(self, reason: str) -> None:
        self.status = ImageStatus.FAILED
        self.error_message = reason
        self.updated_at = _now()

    def is_owned_by(self, user_id: UUID) -> bool:
        return self.owner_id == user_id


@dataclass(frozen=True, slots=True)
class AnswerAlternative:
    """One entry in a prediction's ranked shortlist — see :class:`VqaAnswer.alternatives`."""

    answer: str
    confidence: float  # softmax probability, 0.0-1.0


@dataclass(frozen=True, slots=True)
class VqaAnswer:
    """One trained model variant's prediction for one question.

    ``confidence`` is the top prediction's own softmax probability — genuinely reported,
    not smoothed or hidden, so a low-confidence answer (most often on ``count`` questions;
    see the project report's evaluation results) is visibly a low-confidence answer rather
    than presented with false certainty.
    """

    fusion: FusionType
    answer: str
    confidence: float
    alternatives: tuple[AnswerAlternative, ...]  # ranked, most confident first
    # A plain-English caveat shown beside the answer when it needs one — currently only
    # counting questions, which are answered as a range rather than an exact number.
    note: str | None = None


@dataclass(frozen=True, slots=True)
class AttentionHeatmap:
    """Where the cross-attention model looked when answering a question — only this fusion
    variant has per-patch attention to show; elementwise fusion pools the whole image into
    one vector before combining, so there's nothing comparable to visualise for it.

    ``weights`` is a flattened ``grid_size`` x ``grid_size`` grid, row-major, averaged over
    the question's real (non-padding) word tokens and normalised to sum to ~1.0 across the
    grid. This shows where the model attended, not a verified ground-truth location for any
    object in the scene — an interpretability aid, not a detector.
    """

    fusion: FusionType
    answer: str
    confidence: float
    grid_size: int
    weights: tuple[float, ...]  # length grid_size * grid_size


@dataclass(frozen=True, slots=True)
class MapFeature:
    """One piece of map geometry in lon/lat: a line (a road) or a polygon (a building)."""

    kind: Literal["line", "polygon"]
    points: tuple[tuple[float, float], ...]  # (longitude, latitude)


@dataclass(frozen=True, slots=True)
class HighlightLayer:
    """One thing a question named, and where it was found.

    ``method`` names the technique honestly — a colour index, a zero-shot segmenter, or
    OpenStreetMap map data — because none of them is the trained VQA model, which cannot locate
    anything. ``caveat`` says how far to trust it, and is set wherever a method is weak.
    """

    concept: str
    label: str  # what to call it on screen, e.g. "roads"
    color: str  # "#rrggbb", the colour it is outlined in
    coverage: float  # fraction of the image it covers (0.0-1.0)
    method: str
    caveat: str | None = None


@dataclass(frozen=True, slots=True)
class ConceptHighlight:
    """What to lay over an image to point at the thing(s) a question is about.

    ``overlay_png`` is a transparent PNG the same shape as the analysed image: everything
    *outside* every layer is dimmed, and each layer is tinted and outlined in its own colour, so
    the matching areas stand out in the foreground. ``caveat`` is a note about the whole
    highlight (e.g. that size and shape words in the question were not used).
    """

    layers: tuple[HighlightLayer, ...]
    overlay_png: bytes
    caveat: str | None = None


@dataclass(slots=True)
class VqaQuery:
    """A question asked about a satellite image, and every trained model variant's answer
    to it — kept together so the baseline-vs-cross-attention comparison (the project's own
    contribution) is a fact recorded once, not something recomputed from two rows.
    """

    image_id: UUID
    asked_by: UUID
    question: str
    answers: tuple[VqaAnswer, ...]
    id: UUID = field(default_factory=uuid4)
    created_at: datetime = field(default_factory=_now)

    def answer_for(self, fusion: FusionType) -> VqaAnswer | None:
        return next((a for a in self.answers if a.fusion is fusion), None)


@dataclass(slots=True)
class AuditEvent:
    """An immutable record of a security-relevant action.

    Written on both success and failure — a log that only records successes cannot
    evidence an attack.
    """

    action: AuditAction
    outcome: AuditOutcome
    id: UUID = field(default_factory=uuid4)
    user_id: UUID | None = None
    ip_address: str | None = None
    user_agent: str | None = None
    detail: dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=_now)
