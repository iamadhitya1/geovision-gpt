"""Authentication endpoints."""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, Request, status

from app.api.deps import (
    CurrentUser,
    client_ip,
    get_auth_service,
    get_rate_limiter,
    get_request_context,
    get_settings_dep,
)
from app.application.auth_service import AuthService
from app.application.dto import (
    AuthenticatedUserResponse,
    ChangePasswordRequest,
    LoginRequest,
    MessageResponse,
    RefreshRequest,
    RegisterRequest,
    RequestContext,
    TokenResponse,
    UserRead,
)
from app.core.config import Settings
from app.domain.interfaces import RateLimiter

router = APIRouter(prefix="/auth", tags=["Authentication"])

AuthServiceDep = Annotated[AuthService, Depends(get_auth_service)]
ContextDep = Annotated[RequestContext, Depends(get_request_context)]
LimiterDep = Annotated[RateLimiter, Depends(get_rate_limiter)]
SettingsDep = Annotated[Settings, Depends(get_settings_dep)]


async def _throttle_credentials(
    request: Request, limiter: RateLimiter, settings: Settings, bucket: str
) -> None:
    """Throttle credential-submitting endpoints per client address.

    Keyed by IP rather than by email: an attacker spraying one password across many
    accounts would otherwise never trip a per-account counter, and keying by email also
    lets a third party lock a known user out of their own account.
    """
    await limiter.hit(
        f"{bucket}:{client_ip(request)}",
        limit=settings.rate_limit_login_attempts,
        window_seconds=settings.rate_limit_login_window_seconds,
    )


@router.post(
    "/register",
    response_model=AuthenticatedUserResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Register a new account",
    responses={
        409: {"description": "Email already registered"},
        422: {"description": "Password fails the strength policy"},
        429: {"description": "Too many registration attempts"},
    },
)
async def register(
    payload: RegisterRequest,
    request: Request,
    service: AuthServiceDep,
    context: ContextDep,
    limiter: LimiterDep,
    settings: SettingsDep,
) -> AuthenticatedUserResponse:
    """Create an account with the default `researcher` role and return credentials.

    Elevation to `admin` is deliberately not self-service; it is an admin-only mutation
    on `PATCH /users/{id}`.
    """
    await _throttle_credentials(request, limiter, settings, "register")
    user, tokens = await service.register(payload, context)
    return AuthenticatedUserResponse(
        user=UserRead.from_domain(user), tokens=TokenResponse.from_domain(tokens)
    )


@router.post(
    "/login",
    response_model=AuthenticatedUserResponse,
    summary="Exchange credentials for an access/refresh token pair",
    responses={
        401: {"description": "Incorrect email or password, or the account is inactive"},
        429: {"description": "Too many login attempts"},
    },
)
async def login(
    payload: LoginRequest,
    request: Request,
    service: AuthServiceDep,
    context: ContextDep,
    limiter: LimiterDep,
    settings: SettingsDep,
) -> AuthenticatedUserResponse:
    await _throttle_credentials(request, limiter, settings, "login")
    user, tokens = await service.authenticate(payload, context)
    return AuthenticatedUserResponse(
        user=UserRead.from_domain(user), tokens=TokenResponse.from_domain(tokens)
    )


@router.post(
    "/refresh",
    response_model=TokenResponse,
    summary="Rotate a refresh token",
    responses={401: {"description": "Token invalid, expired, revoked, or replayed"}},
)
async def refresh(
    payload: RefreshRequest,
    service: AuthServiceDep,
    context: ContextDep,
) -> TokenResponse:
    """Refresh tokens are single-use. The previous token is invalidated as the new pair is
    issued; presenting it again is treated as theft and revokes every session."""
    _user, tokens = await service.refresh(payload.refresh_token, context)
    return TokenResponse.from_domain(tokens)


@router.post("/logout", response_model=MessageResponse, summary="End the current session")
async def logout(
    payload: RefreshRequest,
    service: AuthServiceDep,
    context: ContextDep,
) -> MessageResponse:
    """Idempotent: an already-invalid token still returns 200, because the caller's intent
    is satisfied either way."""
    await service.logout(payload.refresh_token, context)
    return MessageResponse(message="Logged out.")


@router.post(
    "/logout-all",
    response_model=MessageResponse,
    summary="End every session for the current user",
)
async def logout_all(
    current_user: CurrentUser,
    service: AuthServiceDep,
    context: ContextDep,
) -> MessageResponse:
    """Use this after a device is lost or a credential may have leaked — it invalidates
    every refresh token issued to this account, on every device, in one call."""
    await service.logout_everywhere(current_user, context)
    return MessageResponse(message="All sessions have been ended.")


@router.post(
    "/change-password",
    response_model=MessageResponse,
    summary="Change the current user's password",
    responses={
        401: {"description": "Current password is incorrect"},
        422: {"description": "New password fails the strength policy"},
    },
)
async def change_password(
    payload: ChangePasswordRequest,
    current_user: CurrentUser,
    service: AuthServiceDep,
    context: ContextDep,
) -> MessageResponse:
    """Succeeding here ends every existing session, including the caller's — otherwise a
    stolen refresh token would outlive the password change made to defeat it."""
    await service.change_password(current_user, payload, context)
    return MessageResponse(message="Password changed. Please sign in again.")


@router.get("/me", response_model=UserRead, summary="Current authenticated user")
async def me(current_user: CurrentUser) -> UserRead:
    """A quick way to confirm a bearer token is valid and see who it belongs to."""
    return UserRead.from_domain(current_user)
