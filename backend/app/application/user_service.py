"""User management use cases (self-service profile and admin administration)."""

from __future__ import annotations

from uuid import UUID

from app.application.audit_service import AuditService
from app.application.dto import (
    AdminUpdateUserRequest,
    RequestContext,
    UpdateProfileRequest,
)
from app.core.logging import get_logger
from app.domain.entities import User
from app.domain.enums import AuditAction, Role
from app.domain.errors import AuthorizationError, NotFoundError
from app.domain.interfaces import UnitOfWork, UserRepository

logger = get_logger(__name__)


class UserService:
    """Reads and mutations over user records."""

    def __init__(
        self,
        *,
        users: UserRepository,
        audit: AuditService,
        uow: UnitOfWork,
    ) -> None:
        self._users = users
        self._audit = audit
        self._uow = uow

    async def get(self, user_id: UUID) -> User:
        user = await self._users.get_by_id(user_id)
        if user is None:
            raise NotFoundError("User not found.", user_id=str(user_id))
        return user

    async def list_users(
        self,
        *,
        offset: int = 0,
        limit: int = 50,
        role: Role | None = None,
        is_active: bool | None = None,
        search: str | None = None,
    ) -> tuple[list[User], int]:
        return await self._users.list_users(
            offset=offset, limit=limit, role=role, is_active=is_active, search=search
        )

    async def update_profile(
        self,
        user: User,
        request: UpdateProfileRequest,
        context: RequestContext | None = None,
    ) -> User:
        """Self-service update. Intentionally cannot touch role or activation."""
        if request.full_name is not None:
            user.full_name = request.full_name
        updated = await self._users.update(user)
        await self._uow.commit()
        return updated

    async def admin_update(
        self,
        *,
        actor: User,
        target_id: UUID,
        request: AdminUpdateUserRequest,
        context: RequestContext | None = None,
    ) -> User:
        """Administrative mutation of another account.

        Two guards exist because losing the last administrator is unrecoverable without
        direct database access: an admin may neither demote nor deactivate themselves.
        """
        target = await self.get(target_id)

        if request.role is not None and request.role is not target.role:
            if actor.id == target.id and request.role is not Role.ADMIN:
                raise AuthorizationError("An administrator cannot demote their own account.")
            previous = target.role
            target.change_role(request.role)
            await self._audit.record(
                action=AuditAction.ROLE_CHANGED,
                user_id=target.id,
                context=context,
                actor_id=str(actor.id),
                previous_role=previous.value,
                new_role=request.role.value,
            )

        if request.is_active is not None and request.is_active is not target.is_active:
            if actor.id == target.id and not request.is_active:
                raise AuthorizationError("An administrator cannot deactivate their own account.")
            if request.is_active:
                target.activate()
                action = AuditAction.USER_ACTIVATED
            else:
                target.deactivate()  # also revokes outstanding sessions
                action = AuditAction.USER_DEACTIVATED
            await self._audit.record(
                action=action,
                user_id=target.id,
                context=context,
                actor_id=str(actor.id),
            )

        if request.is_verified is not None:
            target.is_verified = request.is_verified

        updated = await self._users.update(target)
        await self._uow.commit()
        logger.info("user_admin_updated", target_id=str(target.id), actor_id=str(actor.id))
        return updated
