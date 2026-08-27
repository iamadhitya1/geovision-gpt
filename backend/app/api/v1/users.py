"""User endpoints: self-service profile and admin administration."""

from __future__ import annotations

from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, Depends, Query

from app.api.deps import (
    AdminUser,
    CurrentUser,
    get_audit_service,
    get_request_context,
    get_user_service,
)
from app.application.audit_service import AuditService
from app.application.dto import (
    AdminUpdateUserRequest,
    AuditEventRead,
    Page,
    RequestContext,
    UpdateProfileRequest,
    UserRead,
)
from app.application.user_service import UserService
from app.domain.enums import Role

router = APIRouter(prefix="/users", tags=["Users"])

UserServiceDep = Annotated[UserService, Depends(get_user_service)]
AuditServiceDep = Annotated[AuditService, Depends(get_audit_service)]
ContextDep = Annotated[RequestContext, Depends(get_request_context)]


@router.get("/me", response_model=UserRead, summary="Own profile")
async def read_own_profile(current_user: CurrentUser) -> UserRead:
    return UserRead.from_domain(current_user)


@router.patch("/me", response_model=UserRead, summary="Update own profile")
async def update_own_profile(
    payload: UpdateProfileRequest,
    current_user: CurrentUser,
    service: UserServiceDep,
    context: ContextDep,
) -> UserRead:
    """Only fields a user may set about themselves. Role and activation are absent from
    the request model, so privilege escalation is not merely rejected — it is unrepresentable."""
    updated = await service.update_profile(current_user, payload, context)
    return UserRead.from_domain(updated)


@router.get(
    "",
    response_model=Page[UserRead],
    summary="List users (admin)",
    responses={403: {"description": "Requires the admin role"}},
)
async def list_users(
    _admin: AdminUser,
    service: UserServiceDep,
    offset: Annotated[int, Query(ge=0)] = 0,
    limit: Annotated[int, Query(ge=1, le=200)] = 50,
    role: Role | None = None,
    is_active: bool | None = None,
    search: Annotated[str | None, Query(max_length=120)] = None,
) -> Page[UserRead]:
    users, total = await service.list_users(
        offset=offset, limit=limit, role=role, is_active=is_active, search=search
    )
    return Page[UserRead](
        items=[UserRead.from_domain(u) for u in users],
        total=total,
        offset=offset,
        limit=limit,
    )


@router.get(
    "/{user_id}",
    response_model=UserRead,
    summary="Read a user (admin)",
    responses={404: {"description": "User not found"}},
)
async def read_user(
    user_id: UUID,
    _admin: AdminUser,
    service: UserServiceDep,
) -> UserRead:
    return UserRead.from_domain(await service.get(user_id))


@router.patch(
    "/{user_id}",
    response_model=UserRead,
    summary="Update a user's role or activation (admin)",
    responses={
        403: {"description": "Requires admin, or the actor is targeting their own privileges"},
        404: {"description": "User not found"},
    },
)
async def admin_update_user(
    user_id: UUID,
    payload: AdminUpdateUserRequest,
    admin: AdminUser,
    service: UserServiceDep,
    context: ContextDep,
) -> UserRead:
    """Deactivating a user also revokes their outstanding sessions immediately."""
    updated = await service.admin_update(
        actor=admin, target_id=user_id, request=payload, context=context
    )
    return UserRead.from_domain(updated)


@router.get(
    "/{user_id}/audit",
    response_model=Page[AuditEventRead],
    summary="Read a user's audit trail (admin)",
    responses={404: {"description": "User not found"}},
)
async def read_user_audit(
    user_id: UUID,
    _admin: AdminUser,
    service: UserServiceDep,
    audit: AuditServiceDep,
    offset: Annotated[int, Query(ge=0)] = 0,
    limit: Annotated[int, Query(ge=1, le=200)] = 50,
) -> Page[AuditEventRead]:
    """Resolves the user first so an unknown id returns 404 rather than an empty page,
    which would otherwise be indistinguishable from a user with no recorded activity."""
    await service.get(user_id)
    events, total = await audit.list_events(user_id=user_id, offset=offset, limit=limit)
    return Page[AuditEventRead](
        items=[AuditEventRead.from_domain(e) for e in events],
        total=total,
        offset=offset,
        limit=limit,
    )
