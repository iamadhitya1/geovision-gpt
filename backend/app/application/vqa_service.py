"""Visual question answering use cases.

Like every other service in this project, this imports no ML framework directly — only
the domain :class:`~app.domain.interfaces.VqaModelPort` port. ``torch``/``transformers``/
``timm`` live behind ``infrastructure/vqa/``, the same separation
:class:`~app.application.image_service.ImageService` keeps around ``rasterio``.
"""

from __future__ import annotations

import asyncio
from dataclasses import replace
from uuid import UUID

from app.application.audit_service import AuditService
from app.application.dto import (
    AgreementRead,
    FusionUsageRead,
    RequestContext,
    VqaQueryRead,
    VqaStatsRead,
)
from app.core.logging import get_logger
from app.domain.concepts import ignored_modifiers, resolve_targets
from app.domain.entities import (
    AttentionHeatmap,
    ConceptHighlight,
    SatelliteImage,
    User,
    VqaAnswer,
    VqaQuery,
)
from app.domain.enums import AuditAction, AuditOutcome, FusionType, ImageStatus
from app.domain.errors import (
    ImageNotReadyError,
    ModelUnavailableError,
    NotFoundError,
    NothingToHighlightError,
    UnsupportedQuestionError,
)
from app.domain.interfaces import (
    ImageRepository,
    ObjectStorage,
    SegmentationPort,
    UnitOfWork,
    VqaModelPort,
    VqaQueryRepository,
)
from app.domain.question_guard import check_question

logger = get_logger(__name__)

_STATS_SAMPLE = 1000
_RECENT_COUNT = 8


def _with_note(answer: VqaAnswer, note: str | None) -> VqaAnswer:
    """Put the "read as ..." note ahead of any note the model runner already attached."""
    if note is None:
        return answer
    return replace(answer, note=f"{note} {answer.note}" if answer.note else note)


class VqaService:
    """Answer a question about an already-uploaded, already-processed satellite image."""

    def __init__(
        self,
        *,
        images: ImageRepository,
        storage: ObjectStorage,
        model: VqaModelPort,
        queries: VqaQueryRepository,
        audit: AuditService,
        uow: UnitOfWork,
        default_top_k: int,
        segmenter: SegmentationPort | None = None,
    ) -> None:
        self._images = images
        self._storage = storage
        self._model = model
        self._queries = queries
        self._audit = audit
        self._uow = uow
        self._default_top_k = default_top_k
        self._segmenter = segmenter

    def available_fusions(self) -> frozenset[FusionType]:
        return self._model.available_fusions()

    async def ask(
        self,
        *,
        image_id: UUID,
        question: str,
        requesting_user: User,
        fusions: list[FusionType] | None = None,
        context: RequestContext | None = None,
    ) -> VqaQuery:
        """Run ``question`` through every requested (or every loaded) model variant and
        persist the result as one :class:`~app.domain.entities.VqaQuery`.

        A partial failure — one fusion variant erroring while another succeeds — still
        returns the answers that did succeed rather than discarding the whole request;
        this matters specifically because the two variants are meant to be compared, and a
        transient failure on one side shouldn't hide a working result on the other.
        """
        image = await self._images.get_by_id(image_id)
        if image is None or not self._can_access(image, requesting_user):
            raise NotFoundError("Image not found.", image_id=str(image_id))
        if image.status is not ImageStatus.READY:
            raise ImageNotReadyError(image_id=str(image_id), status=image.status.value)

        # The models only know RSVQA's question templates: reword a loosely-worded question to
        # the nearest one, and turn away one of a different kind rather than guess confidently.
        check = check_question(question)
        if not check.supported:
            raise UnsupportedQuestionError(check.reason, question=question)

        requested = frozenset(fusions) if fusions else self._model.available_fusions()
        image_bytes = await self._storage.get(image.storage_key)

        results = await asyncio.gather(
            *(
                self._model.predict(
                    image_bytes, check.text, fusion=fusion, top_k=self._default_top_k
                )
                for fusion in requested
            ),
            return_exceptions=True,
        )

        answers: list[VqaAnswer] = []
        failures: list[str] = []
        for fusion, result in zip(requested, results, strict=True):
            if isinstance(result, BaseException):
                logger.warning(
                    "vqa_prediction_failed",
                    image_id=str(image_id),
                    fusion=fusion.value,
                    error=str(result),
                )
                failures.append(fusion.value)
            else:
                answers.append(_with_note(result, check.note))

        query = VqaQuery(
            image_id=image_id,
            asked_by=requesting_user.id,
            question=question,
            answers=tuple(answers),
        )
        query = await self._queries.add(query)

        await self._audit.record(
            action=AuditAction.VQA_QUERY_ANSWERED if answers else AuditAction.VQA_QUERY_FAILED,
            outcome=AuditOutcome.SUCCESS if answers else AuditOutcome.FAILURE,
            user_id=requesting_user.id,
            context=context,
            image_id=str(image_id),
            query_id=str(query.id),
            fusions_answered=[a.fusion.value for a in answers],
            fusions_failed=failures,
        )
        await self._uow.commit()

        logger.info(
            "vqa_query_answered",
            image_id=str(image_id),
            query_id=str(query.id),
            fusions_answered=len(answers),
            fusions_failed=len(failures),
        )
        return query

    async def explain(
        self, *, image_id: UUID, question: str, requesting_user: User
    ) -> AttentionHeatmap:
        """Cross-attention's attention weights as a heatmap — interpretability only, not
        persisted as a query (unlike :meth:`ask`): it's a view onto an existing answer, not
        a new recorded event. Same ownership/readiness checks as asking a question."""
        image = await self._images.get_by_id(image_id)
        if image is None or not self._can_access(image, requesting_user):
            raise NotFoundError("Image not found.", image_id=str(image_id))
        if image.status is not ImageStatus.READY:
            raise ImageNotReadyError(image_id=str(image_id), status=image.status.value)

        image_bytes = await self._storage.get(image.storage_key)
        check = check_question(question)
        return await self._model.explain(image_bytes, check.text if check.supported else question)

    async def highlight(
        self, *, image_id: UUID, question: str, requesting_user: User
    ) -> ConceptHighlight:
        """Where the thing(s) the question is about appear in the image.

        Not persisted, like :meth:`explain`: it's a view onto an existing answer. Whether a
        thing is actually *present* is the VQA model's call (the client only offers the
        highlight when the answer isn't "no"/"none") — this step only locates it. Comparison
        questions name two things, and both are highlighted, each in its own colour.
        """
        concepts = resolve_targets(question)
        if not concepts:
            raise NothingToHighlightError(question=question)
        if self._segmenter is None:
            raise ModelUnavailableError(
                "Highlighting isn't enabled on this server.", feature="highlight"
            )

        image = await self._images.get_by_id(image_id)
        if image is None or not self._can_access(image, requesting_user):
            raise NotFoundError("Image not found.", image_id=str(image_id))
        if image.status is not ImageStatus.READY:
            raise ImageNotReadyError(image_id=str(image_id), status=image.status.value)

        image_bytes = await self._storage.get(image.storage_key)
        bounds = image.metadata.bounds_wgs84 if image.metadata is not None else None
        result = await self._segmenter.highlight(image_bytes, concepts, bounds_wgs84=bounds)

        ignored = ignored_modifiers(question)
        if ignored:
            words = ", ".join(f'"{w}"' for w in ignored)
            note = f"The question's size/shape words ({words}) aren't taken into account."
            result = replace(result, caveat=note)
        return result

    async def list_for_image(
        self, image_id: UUID, requesting_user: User, *, offset: int = 0, limit: int = 50
    ) -> tuple[list[VqaQuery], int]:
        image = await self._images.get_by_id(image_id)
        if image is None or not self._can_access(image, requesting_user):
            raise NotFoundError("Image not found.", image_id=str(image_id))
        return await self._queries.list_for_image(image_id, offset=offset, limit=limit)

    async def stats(self, requesting_user: User) -> VqaStatsRead:
        """Usage figures for the dashboard: a researcher sees their own questions, an admin
        sees everyone's. Aggregated in Python over the latest ``_STATS_SAMPLE`` questions —
        plenty for a research tool, and it keeps the port a simple "give me recent rows"
        rather than pushing JSON-column aggregation into every database backend."""
        owner_scope = None if requesting_user.is_admin else requesting_user.id
        queries, total = await self._queries.list_recent_by_user(owner_scope, limit=_STATS_SAMPLE)

        confidences: dict[FusionType, list[float]] = {}
        compared = agreed = 0
        for query in queries:
            for answer in query.answers:
                confidences.setdefault(answer.fusion, []).append(answer.confidence)
            first = query.answer_for(FusionType.ELEMENTWISE)
            second = query.answer_for(FusionType.CROSS_ATTENTION)
            if first is not None and second is not None:
                compared += 1
                agreed += int(first.answer == second.answer)

        return VqaStatsRead(
            total_questions=total,
            sample_size=len(queries),
            images_queried=len({q.image_id for q in queries}),
            by_fusion=[
                FusionUsageRead(
                    fusion=fusion,
                    answered=len(values),
                    mean_confidence=sum(values) / len(values),
                )
                for fusion, values in sorted(confidences.items(), key=lambda kv: kv[0].value)
            ],
            agreement=AgreementRead(compared=compared, agreed=agreed),
            recent=[VqaQueryRead.from_domain(q) for q in queries[:_RECENT_COUNT]],
        )

    @staticmethod
    def _can_access(image: SatelliteImage, user: User) -> bool:
        return user.is_admin or image.is_owned_by(user.id)
