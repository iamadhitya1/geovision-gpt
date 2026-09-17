"""SQLAlchemy implementations of the domain persistence ports.

Repositories translate between ORM models and domain entities and never commit — the
transaction boundary belongs to the use case via :class:`SqlAlchemyUnitOfWork`. That split
is what allows a single service method to write a user row and an audit row atomically.
"""

from __future__ import annotations

from uuid import UUID

from sqlalchemy import Select, func, or_, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.domain.entities import AuditEvent, SatelliteImage, User, VqaQuery
from app.domain.enums import ImageFormat, ImageStatus, Role
from app.domain.interfaces import (
    AuditRepository,
    ImageRepository,
    UserRepository,
    VqaQueryRepository,
)
from app.infrastructure.db.models import (
    AuditEventModel,
    ImageModel,
    UserModel,
    VqaQueryModel,
    apply_image_changes,
    apply_user_changes,
    audit_to_domain,
    audit_to_model,
    image_to_domain,
    image_to_model,
    user_to_domain,
    user_to_model,
    vqa_query_to_domain,
    vqa_query_to_model,
)


class SqlAlchemyUserRepository(UserRepository):
    """User persistence backed by SQLAlchemy."""

    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def add(self, user: User) -> User:
        model = user_to_model(user)
        self._session.add(model)
        # Flush (not commit) so the row is visible to later statements in this
        # transaction and any constraint violation surfaces here, not at commit time.
        await self._session.flush()
        return user_to_domain(model)

    async def get_by_id(self, user_id: UUID) -> User | None:
        model = await self._session.get(UserModel, user_id)
        return user_to_domain(model) if model else None

    async def get_by_email(self, email: str) -> User | None:
        stmt = select(UserModel).where(UserModel.email == email.lower())
        model = (await self._session.execute(stmt)).scalar_one_or_none()
        return user_to_domain(model) if model else None

    async def update(self, user: User) -> User:
        model = await self._session.get(UserModel, user.id)
        if model is None:
            # Re-attach a detached entity rather than silently dropping the write.
            model = user_to_model(user)
            self._session.add(model)
        else:
            apply_user_changes(model, user)
        await self._session.flush()
        return user_to_domain(model)

    async def exists_by_email(self, email: str) -> bool:
        stmt = select(func.count()).select_from(UserModel).where(UserModel.email == email.lower())
        return bool((await self._session.execute(stmt)).scalar_one())

    async def list_users(
        self,
        *,
        offset: int = 0,
        limit: int = 50,
        role: Role | None = None,
        is_active: bool | None = None,
        search: str | None = None,
    ) -> tuple[list[User], int]:
        stmt: Select[tuple[UserModel]] = select(UserModel)
        stmt = self._apply_filters(stmt, role=role, is_active=is_active, search=search)

        count_stmt = self._apply_filters(
            select(func.count()).select_from(UserModel),
            role=role,
            is_active=is_active,
            search=search,
        )
        total = int((await self._session.execute(count_stmt)).scalar_one())

        stmt = stmt.order_by(UserModel.created_at.desc()).offset(offset).limit(limit)
        models = (await self._session.execute(stmt)).scalars().all()
        return [user_to_domain(m) for m in models], total

    @staticmethod
    def _apply_filters(
        stmt: Select[tuple[UserModel]] | Select[tuple[int]],
        *,
        role: Role | None,
        is_active: bool | None,
        search: str | None,
    ) -> Select:  # type: ignore[type-arg]
        if role is not None:
            stmt = stmt.where(UserModel.role == role.value)
        if is_active is not None:
            stmt = stmt.where(UserModel.is_active.is_(is_active))
        if search:
            pattern = f"%{search.lower()}%"
            stmt = stmt.where(
                or_(
                    func.lower(UserModel.email).like(pattern),
                    func.lower(UserModel.full_name).like(pattern),
                )
            )
        return stmt


class SqlAlchemyAuditRepository(AuditRepository):
    """Append-only audit persistence."""

    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def add(self, event: AuditEvent) -> AuditEvent:
        model = audit_to_model(event)
        self._session.add(model)
        await self._session.flush()
        return audit_to_domain(model)

    async def list_events(
        self,
        *,
        user_id: UUID | None = None,
        offset: int = 0,
        limit: int = 50,
    ) -> tuple[list[AuditEvent], int]:
        stmt = select(AuditEventModel)
        count_stmt = select(func.count()).select_from(AuditEventModel)
        if user_id is not None:
            stmt = stmt.where(AuditEventModel.user_id == user_id)
            count_stmt = count_stmt.where(AuditEventModel.user_id == user_id)

        total = int((await self._session.execute(count_stmt)).scalar_one())
        stmt = stmt.order_by(AuditEventModel.created_at.desc()).offset(offset).limit(limit)
        models = (await self._session.execute(stmt)).scalars().all()
        return [audit_to_domain(m) for m in models], total


class SqlAlchemyImageRepository(ImageRepository):
    """Satellite image persistence backed by SQLAlchemy."""

    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def add(self, image: SatelliteImage) -> SatelliteImage:
        model = image_to_model(image)
        self._session.add(model)
        await self._session.flush()
        return image_to_domain(model)

    async def get_by_id(self, image_id: UUID) -> SatelliteImage | None:
        model = await self._session.get(ImageModel, image_id)
        return image_to_domain(model) if model else None

    async def update(self, image: SatelliteImage) -> SatelliteImage:
        model = await self._session.get(ImageModel, image.id)
        if model is None:
            model = image_to_model(image)
            self._session.add(model)
        else:
            apply_image_changes(model, image)
        await self._session.flush()
        return image_to_domain(model)

    async def delete(self, image_id: UUID) -> None:
        model = await self._session.get(ImageModel, image_id)
        if model is not None:
            await self._session.delete(model)
            await self._session.flush()

    async def list_images(
        self,
        *,
        offset: int = 0,
        limit: int = 50,
        owner_id: UUID | None = None,
        image_format: ImageFormat | None = None,
        status: ImageStatus | None = None,
    ) -> tuple[list[SatelliteImage], int]:
        stmt: Select[tuple[ImageModel]] = select(ImageModel)
        stmt = self._apply_filters(
            stmt, owner_id=owner_id, image_format=image_format, status=status
        )

        count_stmt = self._apply_filters(
            select(func.count()).select_from(ImageModel),
            owner_id=owner_id,
            image_format=image_format,
            status=status,
        )
        total = int((await self._session.execute(count_stmt)).scalar_one())

        stmt = stmt.order_by(ImageModel.created_at.desc()).offset(offset).limit(limit)
        models = (await self._session.execute(stmt)).scalars().all()
        return [image_to_domain(m) for m in models], total

    @staticmethod
    def _apply_filters(
        stmt: Select[tuple[ImageModel]] | Select[tuple[int]],
        *,
        owner_id: UUID | None,
        image_format: ImageFormat | None,
        status: ImageStatus | None,
    ) -> Select:  # type: ignore[type-arg]
        if owner_id is not None:
            stmt = stmt.where(ImageModel.owner_id == owner_id)
        if image_format is not None:
            stmt = stmt.where(ImageModel.image_format == image_format.value)
        if status is not None:
            stmt = stmt.where(ImageModel.status == status.value)
        return stmt


class SqlAlchemyVqaQueryRepository(VqaQueryRepository):
    """VQA question/answer persistence backed by SQLAlchemy."""

    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def add(self, query: VqaQuery) -> VqaQuery:
        model = vqa_query_to_model(query)
        self._session.add(model)
        await self._session.flush()
        return vqa_query_to_domain(model)

    async def get_by_id(self, query_id: UUID) -> VqaQuery | None:
        model = await self._session.get(VqaQueryModel, query_id)
        return vqa_query_to_domain(model) if model else None

    async def list_for_image(
        self, image_id: UUID, *, offset: int = 0, limit: int = 50
    ) -> tuple[list[VqaQuery], int]:
        stmt = select(VqaQueryModel).where(VqaQueryModel.image_id == image_id)
        count_stmt = (
            select(func.count())
            .select_from(VqaQueryModel)
            .where(VqaQueryModel.image_id == image_id)
        )
        total = int((await self._session.execute(count_stmt)).scalar_one())

        stmt = stmt.order_by(VqaQueryModel.created_at.desc()).offset(offset).limit(limit)
        models = (await self._session.execute(stmt)).scalars().all()
        return [vqa_query_to_domain(m) for m in models], total
