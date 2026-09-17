"""Domain enumerations.

Pure Python. No framework imports — see the dependency rule in docs/00-architecture.md.
"""

from __future__ import annotations

from enum import StrEnum


class Role(StrEnum):
    """Access roles, ordered from least to most privileged.

    ``GUEST``      — read-only demo access; may query public sample imagery.
    ``RESEARCHER`` — full analysis workflow; owns projects, images and conversations.
    ``ADMIN``      — platform administration, user management, audit inspection.
    """

    GUEST = "guest"
    RESEARCHER = "researcher"
    ADMIN = "admin"

    @property
    def level(self) -> int:
        return _ROLE_LEVELS[self]

    def outranks_or_equals(self, other: Role) -> bool:
        """True when this role satisfies a requirement for ``other``."""
        return self.level >= other.level


_ROLE_LEVELS: dict[Role, int] = {
    Role.GUEST: 0,
    Role.RESEARCHER: 10,
    Role.ADMIN: 100,
}


class ImageFormat(StrEnum):
    """Accepted raster formats.

    ``GEOTIFF`` is the only format that can carry embedded georeferencing (CRS, affine
    transform). PNG and JPEG are accepted for quick-look imagery that has no geospatial
    metadata of its own — their :class:`~app.domain.entities.ImageMetadata` will always
    have ``crs``, ``bounds_native`` and ``bounds_wgs84`` set to ``None``.
    """

    GEOTIFF = "geotiff"
    PNG = "png"
    JPEG = "jpeg"


# Extensions accepted per format, lower-cased and without the leading dot. Kept as the
# single source of truth so the upload endpoint's whitelist and the content-sniffing step
# can never silently drift apart from each other.
IMAGE_FORMAT_EXTENSIONS: dict[ImageFormat, frozenset[str]] = {
    ImageFormat.GEOTIFF: frozenset({"tif", "tiff"}),
    ImageFormat.PNG: frozenset({"png"}),
    ImageFormat.JPEG: frozenset({"jpg", "jpeg"}),
}


class FusionType(StrEnum):
    """Which of the two trained model variants answered a question — see
    ``ml/models/vqa_model.py``: ``ELEMENTWISE`` reproduces LiT-4-RSVQA's own published
    baseline; ``CROSS_ATTENTION`` is this project's own experimental fusion mechanism.
    Comparing the two is the project's original contribution, not just an implementation
    detail — see docs/02-research-contribution.md.
    """

    ELEMENTWISE = "elementwise"
    CROSS_ATTENTION = "cross_attention"


class ImageStatus(StrEnum):
    """Processing lifecycle of an uploaded image.

    Currently every transition happens synchronously within the upload request, but the
    states are recorded regardless — that is what lets a later module move metadata
    extraction to a background worker without changing anything a client observes.
    """

    UPLOADED = "uploaded"
    PROCESSING = "processing"
    READY = "ready"
    FAILED = "failed"


class TokenType(StrEnum):
    """Discriminator embedded in the JWT ``typ`` claim.

    Without it, a refresh token would be accepted wherever an access token is expected —
    a long-lived credential used as a short-lived one.
    """

    ACCESS = "access"
    REFRESH = "refresh"


class AuditAction(StrEnum):
    """Auditable events. Module 1 defines the authentication subset."""

    USER_REGISTERED = "user.registered"
    LOGIN_SUCCEEDED = "auth.login.succeeded"
    LOGIN_FAILED = "auth.login.failed"
    LOGOUT = "auth.logout"
    TOKEN_REFRESHED = "auth.token.refreshed"
    TOKEN_REUSE_DETECTED = "auth.token.reuse_detected"
    PASSWORD_CHANGED = "user.password.changed"
    ROLE_CHANGED = "user.role.changed"
    USER_DEACTIVATED = "user.deactivated"
    USER_ACTIVATED = "user.activated"
    VERIFICATION_CHANGED = "user.verification.changed"
    ACCESS_DENIED = "authz.access_denied"
    IMAGE_UPLOADED = "image.uploaded"
    IMAGE_PROCESSING_FAILED = "image.processing.failed"
    IMAGE_DELETED = "image.deleted"
    VQA_QUERY_ANSWERED = "vqa.query.answered"
    VQA_QUERY_FAILED = "vqa.query.failed"


class AuditOutcome(StrEnum):
    SUCCESS = "success"
    FAILURE = "failure"
