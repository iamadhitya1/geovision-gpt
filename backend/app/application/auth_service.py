"""Authentication use cases — Module 1's core.

Every collaborator arrives through the constructor as a Protocol from
``app.domain.interfaces``. This module imports no database driver, no web framework and no
cache client, which is why the whole service is unit-testable with plain in-memory fakes.
"""

from __future__ import annotations

from typing import Any
from uuid import UUID

from app.application.audit_service import AuditService
from app.application.dto import (
    ChangePasswordRequest,
    LoginRequest,
    RegisterRequest,
    RequestContext,
)
from app.application.password_policy import PasswordPolicy
from app.core.logging import get_logger
from app.domain.entities import TokenPair, User
from app.domain.enums import AuditAction, AuditOutcome, Role, TokenType
from app.domain.errors import (
    EmailAlreadyRegisteredError,
    InactiveUserError,
    InvalidCredentialsError,
    InvalidTokenError,
    TokenReuseError,
)
from app.domain.interfaces import (
    Cache,
    PasswordHasher,
    TokenService,
    UnitOfWork,
    UserRepository,
)

logger = get_logger(__name__)

_REFRESH_KEY_PREFIX = "refresh"


class AuthService:
    """Registration, login, token rotation, logout and password change."""

    def __init__(
        self,
        *,
        users: UserRepository,
        hasher: PasswordHasher,
        tokens: TokenService,
        cache: Cache,
        policy: PasswordPolicy,
        audit: AuditService,
        uow: UnitOfWork,
        refresh_ttl_seconds: int,
        default_role: Role = Role.RESEARCHER,
    ) -> None:
        self._users = users
        self._hasher = hasher
        self._tokens = tokens
        self._cache = cache
        self._policy = policy
        self._audit = audit
        self._uow = uow
        self._refresh_ttl_seconds = refresh_ttl_seconds
        self._default_role = default_role

    # --- Registration --------------------------------------------------------

    async def register(
        self, request: RegisterRequest, context: RequestContext | None = None
    ) -> tuple[User, TokenPair]:
        """Create an account and immediately issue credentials.

        The password policy runs *before* the uniqueness check so that a weak password is
        rejected without revealing whether the email is already registered.
        """
        self._policy.validate(request.password, email=request.email)

        if await self._users.exists_by_email(request.email):
            await self._audit.record(
                action=AuditAction.USER_REGISTERED,
                outcome=AuditOutcome.FAILURE,
                context=context,
                reason="email_taken",
                email=request.email,
            )
            await self._uow.commit()
            raise EmailAlreadyRegisteredError()

        user = User(
            email=request.email,
            full_name=request.full_name,
            password_hash=self._hasher.hash(request.password),
            role=self._default_role,
        )
        user = await self._users.add(user)

        await self._audit.record(
            action=AuditAction.USER_REGISTERED,
            user_id=user.id,
            context=context,
            role=user.role.value,
        )
        tokens = await self._issue_tokens(user)
        await self._uow.commit()

        logger.info("user_registered", user_id=str(user.id), role=user.role.value)
        return user, tokens

    # --- Login ---------------------------------------------------------------

    async def authenticate(
        self, request: LoginRequest, context: RequestContext | None = None
    ) -> tuple[User, TokenPair]:
        """Verify credentials and issue a token pair."""
        user = await self._users.get_by_email(request.email)

        if user is None:
            # Hash a throwaway value so an unknown email costs the same wall-clock time
            # as a known one; otherwise response latency enumerates the user table.
            self._hasher.dummy_verify()
            await self._record_failed_login(context, email=request.email, reason="unknown_email")
            raise InvalidCredentialsError()

        if not self._hasher.verify(request.password, user.password_hash):
            await self._record_failed_login(
                context, email=request.email, reason="bad_password", user_id=user.id
            )
            raise InvalidCredentialsError()

        if not user.can_login():
            await self._record_failed_login(
                context, email=request.email, reason="inactive", user_id=user.id
            )
            raise InactiveUserError()

        # Transparent upgrade when the hashing cost parameters have been raised.
        if self._hasher.needs_rehash(user.password_hash):
            user.password_hash = self._hasher.hash(request.password)
            logger.info("password_hash_upgraded", user_id=str(user.id))

        user.record_login()
        await self._users.update(user)

        tokens = await self._issue_tokens(user)
        await self._audit.record(
            action=AuditAction.LOGIN_SUCCEEDED, user_id=user.id, context=context
        )
        await self._uow.commit()

        logger.info("login_succeeded", user_id=str(user.id))
        return user, tokens

    # --- Token rotation ------------------------------------------------------

    async def refresh(
        self, refresh_token: str, context: RequestContext | None = None
    ) -> tuple[User, TokenPair]:
        """Rotate a refresh token, detecting replay.

        Refresh tokens are tracked as an allow-list of ``jti`` values. A cryptographically
        valid token whose ``jti`` is absent has either been used already or been revoked —
        both mean the token is in the wrong hands, so every session for the user is
        terminated rather than merely rejecting this request.
        """
        claims = self._tokens.decode(refresh_token, expected_type=TokenType.REFRESH)

        user = await self._users.get_by_id(claims.subject)
        if user is None:
            raise InvalidTokenError()

        if claims.token_version != user.token_version:
            await self._audit.record(
                action=AuditAction.TOKEN_REUSE_DETECTED,
                outcome=AuditOutcome.FAILURE,
                user_id=user.id,
                context=context,
                reason="stale_token_version",
            )
            await self._uow.commit()
            raise InvalidTokenError()

        key = self._refresh_key(claims.jti)
        if not await self._cache.exists(key):
            user.revoke_all_sessions()
            await self._users.update(user)
            await self._audit.record(
                action=AuditAction.TOKEN_REUSE_DETECTED,
                outcome=AuditOutcome.FAILURE,
                user_id=user.id,
                context=context,
                jti=claims.jti,
            )
            await self._uow.commit()
            logger.warning("refresh_token_reuse", user_id=str(user.id), jti=claims.jti)
            raise TokenReuseError()

        if not user.can_login():
            raise InactiveUserError()

        await self._cache.delete(key)  # single-use: consume before re-issuing
        tokens = await self._issue_tokens(user)
        await self._audit.record(
            action=AuditAction.TOKEN_REFRESHED, user_id=user.id, context=context
        )
        await self._uow.commit()
        return user, tokens

    # --- Logout --------------------------------------------------------------

    async def logout(self, refresh_token: str, context: RequestContext | None = None) -> None:
        """Invalidate a single session.

        A malformed or expired token is not an error here — the caller's intent (end the
        session) is already satisfied, and returning 401 on logout only strands clients.
        """
        try:
            claims = self._tokens.decode(refresh_token, expected_type=TokenType.REFRESH)
        except InvalidTokenError:
            return

        await self._cache.delete(self._refresh_key(claims.jti))
        await self._audit.record(action=AuditAction.LOGOUT, user_id=claims.subject, context=context)
        await self._uow.commit()

    async def logout_everywhere(self, user: User, context: RequestContext | None = None) -> None:
        """Terminate every session for ``user`` in a single write."""
        user.revoke_all_sessions()
        await self._users.update(user)
        await self._audit.record(
            action=AuditAction.LOGOUT, user_id=user.id, context=context, scope="all_sessions"
        )
        await self._uow.commit()

    # --- Password change -----------------------------------------------------

    async def change_password(
        self,
        user: User,
        request: ChangePasswordRequest,
        context: RequestContext | None = None,
    ) -> None:
        """Change a password, then end every existing session."""
        if not self._hasher.verify(request.current_password, user.password_hash):
            await self._audit.record(
                action=AuditAction.PASSWORD_CHANGED,
                outcome=AuditOutcome.FAILURE,
                user_id=user.id,
                context=context,
                reason="bad_current_password",
            )
            await self._uow.commit()
            raise InvalidCredentialsError("Current password is incorrect.")

        self._policy.validate(request.new_password, email=user.email)

        user.change_password(self._hasher.hash(request.new_password))
        await self._users.update(user)
        await self._audit.record(
            action=AuditAction.PASSWORD_CHANGED, user_id=user.id, context=context
        )
        await self._uow.commit()
        logger.info("password_changed", user_id=str(user.id))

    # --- Internals -----------------------------------------------------------

    async def _issue_tokens(self, user: User) -> TokenPair:
        access_token, expires_in = self._tokens.create_access_token(user)
        refresh_token, jti = self._tokens.create_refresh_token(user)
        await self._cache.set(
            self._refresh_key(jti),
            str(user.id),
            ttl_seconds=self._refresh_ttl_seconds,
        )
        return TokenPair(
            access_token=access_token,
            refresh_token=refresh_token,
            expires_in=expires_in,
        )

    async def _record_failed_login(
        self,
        context: RequestContext | None,
        *,
        user_id: UUID | None = None,
        **detail: Any,
    ) -> None:
        await self._audit.record(
            action=AuditAction.LOGIN_FAILED,
            outcome=AuditOutcome.FAILURE,
            user_id=user_id,
            context=context,
            **detail,
        )
        await self._uow.commit()

    @staticmethod
    def _refresh_key(jti: str) -> str:
        return f"{_REFRESH_KEY_PREFIX}:{jti}"
