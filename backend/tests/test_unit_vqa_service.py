"""Unit tests for VqaService, against hand-written fakes rather than the real database or
a real trained model — this exercises the service's own branching logic (ownership,
readiness, partial-fusion-failure handling) in isolation, the same way
test_unit_security.py isolates password/token logic from the rest of the stack.
"""

from __future__ import annotations

from uuid import UUID, uuid4

import pytest

from app.application.audit_service import AuditService
from app.application.vqa_service import VqaService
from app.domain.entities import (
    AnswerAlternative,
    AuditEvent,
    SatelliteImage,
    User,
    VqaAnswer,
    VqaQuery,
)
from app.domain.enums import FusionType, ImageFormat, ImageStatus, Role
from app.domain.errors import ImageNotReadyError, ModelUnavailableError, NotFoundError

# --- Fakes ---------------------------------------------------------------------


class FakeImageRepository:
    def __init__(self, images: list[SatelliteImage]) -> None:
        self._images = {i.id: i for i in images}

    async def add(self, image: SatelliteImage) -> SatelliteImage:
        self._images[image.id] = image
        return image

    async def get_by_id(self, image_id: UUID) -> SatelliteImage | None:
        return self._images.get(image_id)

    async def update(self, image: SatelliteImage) -> SatelliteImage:
        self._images[image.id] = image
        return image

    async def delete(self, image_id: UUID) -> None:
        self._images.pop(image_id, None)

    async def list_images(self, **_kwargs: object) -> tuple[list[SatelliteImage], int]:
        values = list(self._images.values())
        return values, len(values)


class FakeObjectStorage:
    def __init__(self) -> None:
        self._objects: dict[str, bytes] = {}

    async def put(self, key: str, data: bytes, *, content_type: str) -> None:
        self._objects[key] = data

    async def get(self, key: str) -> bytes:
        return self._objects.get(key, b"fake-image-bytes")

    async def delete(self, key: str) -> None:
        self._objects.pop(key, None)

    async def exists(self, key: str) -> bool:
        return key in self._objects


class FakeVqaModel:
    """Answers deterministically per fusion type; ``failing_fusions`` simulates a model
    variant erroring on ``predict`` (e.g. a corrupt checkpoint) without touching torch."""

    def __init__(
        self,
        *,
        loaded: frozenset[FusionType],
        failing_fusions: frozenset[FusionType] = frozenset(),
    ) -> None:
        self._loaded = loaded
        self._failing = failing_fusions

    def available_fusions(self) -> frozenset[FusionType]:
        return self._loaded

    async def predict(
        self, image_data: bytes, question: str, *, fusion: FusionType, top_k: int = 3
    ) -> VqaAnswer:
        if fusion not in self._loaded:
            raise ModelUnavailableError(fusion=fusion.value)
        if fusion in self._failing:
            raise RuntimeError("simulated inference failure")
        return VqaAnswer(
            fusion=fusion,
            answer="urban",
            confidence=0.9,
            alternatives=(AnswerAlternative(answer="urban", confidence=0.9),),
        )


class FakeVqaQueryRepository:
    def __init__(self) -> None:
        self._queries: dict[UUID, VqaQuery] = {}

    async def add(self, query: VqaQuery) -> VqaQuery:
        self._queries[query.id] = query
        return query

    async def get_by_id(self, query_id: UUID) -> VqaQuery | None:
        return self._queries.get(query_id)

    async def list_for_image(
        self, image_id: UUID, *, offset: int = 0, limit: int = 50
    ) -> tuple[list[VqaQuery], int]:
        matches = [q for q in self._queries.values() if q.image_id == image_id]
        return matches, len(matches)

    async def list_recent_by_user(
        self, user_id: UUID | None, *, limit: int
    ) -> tuple[list[VqaQuery], int]:
        matches = [q for q in self._queries.values() if user_id is None or q.asked_by == user_id]
        matches.sort(key=lambda q: q.created_at, reverse=True)
        return matches[:limit], len(matches)


class FakeAuditRepository:
    def __init__(self) -> None:
        self.events: list[AuditEvent] = []

    async def add(self, event: AuditEvent) -> AuditEvent:
        self.events.append(event)
        return event

    async def list_events(self, **_kwargs: object) -> tuple[list[AuditEvent], int]:
        return self.events, len(self.events)


class FakeUnitOfWork:
    def __init__(self) -> None:
        self.committed = False

    async def commit(self) -> None:
        self.committed = True

    async def rollback(self) -> None:
        pass


# --- Fixtures --------------------------------------------------------------------


def _owner() -> User:
    return User(email="owner@geovisiongpt.io", full_name="Owner", password_hash="h", id=uuid4())


def _ready_image(owner_id: UUID) -> SatelliteImage:
    return SatelliteImage(
        owner_id=owner_id,
        original_filename="scene.tif",
        image_format=ImageFormat.GEOTIFF,
        storage_key="k",
        file_size_bytes=100,
        checksum_sha256="abc",
        status=ImageStatus.READY,
    )


def _build_service(
    *,
    image: SatelliteImage,
    model: FakeVqaModel,
    audit_repo: FakeAuditRepository | None = None,
) -> VqaService:
    audit_repo = audit_repo or FakeAuditRepository()
    return VqaService(
        images=FakeImageRepository([image]),  # type: ignore[arg-type]
        storage=FakeObjectStorage(),  # type: ignore[arg-type]
        model=model,  # type: ignore[arg-type]
        queries=FakeVqaQueryRepository(),  # type: ignore[arg-type]
        audit=AuditService(audit_repo),  # type: ignore[arg-type]
        uow=FakeUnitOfWork(),  # type: ignore[arg-type]
        default_top_k=3,
    )


# --- Tests -------------------------------------------------------------------------


async def test_ask_returns_an_answer_from_every_loaded_fusion() -> None:
    owner = _owner()
    image = _ready_image(owner.id)
    model = FakeVqaModel(loaded=frozenset({FusionType.ELEMENTWISE, FusionType.CROSS_ATTENTION}))
    service = _build_service(image=image, model=model)

    query = await service.ask(image_id=image.id, question="Is this urban?", requesting_user=owner)

    assert len(query.answers) == 2
    assert query.answer_for(FusionType.ELEMENTWISE) is not None
    assert query.answer_for(FusionType.CROSS_ATTENTION) is not None


async def test_ask_scopes_to_requested_fusions_only() -> None:
    owner = _owner()
    image = _ready_image(owner.id)
    model = FakeVqaModel(loaded=frozenset({FusionType.ELEMENTWISE, FusionType.CROSS_ATTENTION}))
    service = _build_service(image=image, model=model)

    query = await service.ask(
        image_id=image.id,
        question="Is this urban?",
        requesting_user=owner,
        fusions=[FusionType.ELEMENTWISE],
    )

    assert len(query.answers) == 1
    assert query.answers[0].fusion is FusionType.ELEMENTWISE


async def test_ask_raises_not_found_for_someone_elses_image() -> None:
    owner = _owner()
    stranger = User(email="s@geovisiongpt.io", full_name="S", password_hash="h", id=uuid4())
    image = _ready_image(owner.id)
    model = FakeVqaModel(loaded=frozenset({FusionType.ELEMENTWISE}))
    service = _build_service(image=image, model=model)

    with pytest.raises(NotFoundError):
        await service.ask(image_id=image.id, question="Q", requesting_user=stranger)


async def test_ask_raises_not_ready_for_an_unprocessed_image() -> None:
    owner = _owner()
    image = _ready_image(owner.id)
    image.status = ImageStatus.PROCESSING
    model = FakeVqaModel(loaded=frozenset({FusionType.ELEMENTWISE}))
    service = _build_service(image=image, model=model)

    with pytest.raises(ImageNotReadyError):
        await service.ask(image_id=image.id, question="Q", requesting_user=owner)


async def test_ask_keeps_successful_answers_when_one_fusion_fails() -> None:
    """The whole point of comparing two variants is that one failing shouldn't hide the
    other's working result."""
    owner = _owner()
    image = _ready_image(owner.id)
    model = FakeVqaModel(
        loaded=frozenset({FusionType.ELEMENTWISE, FusionType.CROSS_ATTENTION}),
        failing_fusions=frozenset({FusionType.CROSS_ATTENTION}),
    )
    audit_repo = FakeAuditRepository()
    service = _build_service(image=image, model=model, audit_repo=audit_repo)

    query = await service.ask(image_id=image.id, question="Q", requesting_user=owner)

    assert len(query.answers) == 1
    assert query.answers[0].fusion is FusionType.ELEMENTWISE
    # Still recorded as an overall success — one working answer is a useful result.
    assert audit_repo.events[-1].outcome.value == "success"


async def test_ask_admin_can_query_any_users_image() -> None:
    owner = _owner()
    admin = User(
        email="admin@geovisiongpt.io",
        full_name="Admin",
        password_hash="h",
        role=Role.ADMIN,
        id=uuid4(),
    )
    image = _ready_image(owner.id)
    model = FakeVqaModel(loaded=frozenset({FusionType.ELEMENTWISE}))
    service = _build_service(image=image, model=model)

    query = await service.ask(image_id=image.id, question="Q", requesting_user=admin)
    assert len(query.answers) == 1


def _answer(fusion: FusionType, text: str, confidence: float) -> VqaAnswer:
    return VqaAnswer(
        fusion=fusion,
        answer=text,
        confidence=confidence,
        alternatives=(AnswerAlternative(answer=text, confidence=confidence),),
    )


async def test_stats_counts_agreement_and_mean_confidence() -> None:
    owner = _owner()
    image = _ready_image(owner.id)
    service = _build_service(
        image=image,
        model=FakeVqaModel(loaded=frozenset({FusionType.ELEMENTWISE, FusionType.CROSS_ATTENTION})),
    )
    repo = service._queries
    for elementwise, cross in [("yes", "yes"), ("no", "yes"), ("urban", "urban")]:
        await repo.add(
            VqaQuery(
                image_id=image.id,
                asked_by=owner.id,
                question="q",
                answers=(
                    _answer(FusionType.ELEMENTWISE, elementwise, 0.6),
                    _answer(FusionType.CROSS_ATTENTION, cross, 0.8),
                ),
            )
        )

    stats = await service.stats(owner)

    assert stats.total_questions == 3
    assert stats.images_queried == 1
    assert (stats.agreement.compared, stats.agreement.agreed) == (3, 2)
    by_fusion = {u.fusion: u for u in stats.by_fusion}
    assert by_fusion[FusionType.ELEMENTWISE].mean_confidence == pytest.approx(0.6)
    assert by_fusion[FusionType.CROSS_ATTENTION].mean_confidence == pytest.approx(0.8)


async def test_stats_are_scoped_to_the_requesting_researcher_but_admins_see_all() -> None:
    owner, other = _owner(), _owner()
    image = _ready_image(owner.id)
    service = _build_service(image=image, model=FakeVqaModel(loaded=frozenset()))
    repo = service._queries
    for asker in (owner, other, other):
        await repo.add(
            VqaQuery(
                image_id=image.id,
                asked_by=asker.id,
                question="q",
                answers=(_answer(FusionType.ELEMENTWISE, "yes", 0.5),),
            )
        )
    admin = User(email="root@geovisiongpt.io", full_name="Root", password_hash="h", role=Role.ADMIN)

    assert (await service.stats(owner)).total_questions == 1
    assert (await service.stats(other)).total_questions == 2
    assert (await service.stats(admin)).total_questions == 3
