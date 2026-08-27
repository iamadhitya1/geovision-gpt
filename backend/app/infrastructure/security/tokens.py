"""JWT issuing and verification."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from uuid import UUID, uuid4

import jwt
from jwt import ExpiredSignatureError
from jwt import InvalidTokenError as PyJWTInvalidTokenError

from app.core.logging import get_logger
from app.domain.entities import TokenClaims, User
from app.domain.enums import Role, TokenType
from app.domain.errors import InvalidTokenError
from app.domain.interfaces import TokenService

logger = get_logger(__name__)


class JwtTokenService(TokenService):
    """HMAC-signed JWTs for access and refresh credentials.

    Design notes:

    * ``typ`` distinguishes access from refresh. Verification requires the expected type,
      so a long-lived refresh token cannot be presented as a bearer credential.
    * ``tv`` carries the user's ``token_version`` at issue time, giving a one-write global
      revocation lever (see :meth:`User.revoke_all_sessions`).
    * ``jti`` is unique per refresh token and is tracked as an allow-list entry, which is
      what makes single-use rotation and replay detection possible.
    """

    def __init__(
        self,
        *,
        secret_key: str,
        algorithm: str = "HS256",
        access_token_expire_minutes: int = 30,
        refresh_token_expire_days: int = 14,
        issuer: str = "geovisiongpt",
    ) -> None:
        self._secret = secret_key
        self._algorithm = algorithm
        self._access_ttl = timedelta(minutes=access_token_expire_minutes)
        self._refresh_ttl = timedelta(days=refresh_token_expire_days)
        self._issuer = issuer

    @property
    def refresh_ttl_seconds(self) -> int:
        return int(self._refresh_ttl.total_seconds())

    def create_access_token(self, user: User) -> tuple[str, int]:
        token, _ = self._encode(user, TokenType.ACCESS, self._access_ttl)
        return token, int(self._access_ttl.total_seconds())

    def create_refresh_token(self, user: User) -> tuple[str, str]:
        return self._encode(user, TokenType.REFRESH, self._refresh_ttl)

    def decode(self, token: str, *, expected_type: TokenType) -> TokenClaims:
        try:
            payload = jwt.decode(
                token,
                self._secret,
                algorithms=[self._algorithm],
                issuer=self._issuer,
                options={"require": ["exp", "iat", "sub", "jti"]},
            )
        except ExpiredSignatureError as exc:
            raise InvalidTokenError("The token has expired.") from exc
        except PyJWTInvalidTokenError as exc:
            raise InvalidTokenError() from exc

        if payload.get("typ") != expected_type.value:
            logger.warning(
                "token_type_mismatch", expected=expected_type.value, got=payload.get("typ")
            )
            raise InvalidTokenError("Token type mismatch.")

        try:
            subject = UUID(str(payload["sub"]))
            role = Role(payload["role"])
        except (KeyError, ValueError) as exc:
            raise InvalidTokenError("Malformed token claims.") from exc

        return TokenClaims(
            subject=subject,
            role=role,
            token_type=expected_type.value,
            jti=str(payload["jti"]),
            issued_at=datetime.fromtimestamp(payload["iat"], tz=UTC),
            expires_at=datetime.fromtimestamp(payload["exp"], tz=UTC),
            token_version=int(payload.get("tv", 0)),
        )

    def _encode(self, user: User, token_type: TokenType, ttl: timedelta) -> tuple[str, str]:
        now = datetime.now(UTC)
        jti = uuid4().hex
        payload = {
            "sub": str(user.id),
            "role": user.role.value,
            "typ": token_type.value,
            "tv": user.token_version,
            "jti": jti,
            "iat": int(now.timestamp()),
            "exp": int((now + ttl).timestamp()),
            "iss": self._issuer,
        }
        return jwt.encode(payload, self._secret, algorithm=self._algorithm), jti
