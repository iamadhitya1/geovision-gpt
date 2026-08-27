"""Audit trail use cases."""

from __future__ import annotations

from typing import Any
from uuid import UUID

from app.application.dto import RequestContext
from app.domain.entities import AuditEvent
from app.domain.enums import AuditAction, AuditOutcome
from app.domain.interfaces import AuditRepository


class AuditService:
    """Records security-relevant events.

    Recording never raises into the caller's path: a failed audit write must not turn a
    successful login into a 500. Failures are logged and swallowed, and the repository is
    responsible for its own durability guarantees.
    """

    def __init__(self, repository: AuditRepository) -> None:
        self._repository = repository

    async def record(
        self,
        *,
        action: AuditAction,
        outcome: AuditOutcome = AuditOutcome.SUCCESS,
        user_id: UUID | None = None,
        context: RequestContext | None = None,
        **detail: Any,
    ) -> None:
        event = AuditEvent(
            action=action,
            outcome=outcome,
            user_id=user_id,
            ip_address=context.ip_address if context else None,
            user_agent=context.user_agent if context else None,
            detail=detail,
        )
        await self._repository.add(event)

    async def list_events(
        self,
        *,
        user_id: UUID | None = None,
        offset: int = 0,
        limit: int = 50,
    ) -> tuple[list[AuditEvent], int]:
        return await self._repository.list_events(user_id=user_id, offset=offset, limit=limit)
