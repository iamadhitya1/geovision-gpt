"""Visual question answering endpoints — Module 3/4."""

from __future__ import annotations

from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, Depends, Query

from app.api.deps import ResearcherUser, get_request_context, get_vqa_service
from app.application.dto import AskQuestionRequest, Page, RequestContext, VqaQueryRead
from app.application.vqa_service import VqaService
from app.domain.enums import FusionType

router = APIRouter(tags=["VQA"])

VqaServiceDep = Annotated[VqaService, Depends(get_vqa_service)]
ContextDep = Annotated[RequestContext, Depends(get_request_context)]


@router.get(
    "/vqa/fusions",
    summary="List which model variants are currently loaded",
    response_model=list[FusionType],
)
async def list_available_fusions(
    _researcher: ResearcherUser, service: VqaServiceDep
) -> list[FusionType]:
    """Reflects which trained checkpoints actually exist on this server right now — see
    `ml/README.md`. It's normal for this to return an empty list, or just one fusion, on
    a machine the checkpoints haven't been copied to yet."""
    return sorted(service.available_fusions(), key=lambda f: f.value)


@router.post(
    "/images/{image_id}/ask",
    response_model=VqaQueryRead,
    status_code=201,
    summary="Ask a question about an image",
    responses={
        404: {"description": "Image not found, or not owned by the caller"},
        422: {"description": "The image hasn't finished processing, or processing failed"},
        503: {"description": "None of the requested model variant(s) are loaded"},
    },
)
async def ask_question(
    image_id: UUID,
    request: AskQuestionRequest,
    researcher: ResearcherUser,
    service: VqaServiceDep,
    context: ContextDep,
) -> VqaQueryRead:
    """Runs the question through every fusion variant in `request.fusions` (or every
    variant currently loaded, if omitted) and returns all of their answers together —
    comparing the published baseline against this project's own cross-attention
    experiment is the point, so both come back from one call by default."""
    query = await service.ask(
        image_id=image_id,
        question=request.question,
        requesting_user=researcher,
        fusions=request.fusions,
        context=context,
    )
    return VqaQueryRead.from_domain(query)


@router.get(
    "/images/{image_id}/vqa",
    response_model=Page[VqaQueryRead],
    summary="List past questions asked about this image",
    responses={404: {"description": "Image not found, or not owned by the caller"}},
)
async def list_queries(
    image_id: UUID,
    researcher: ResearcherUser,
    service: VqaServiceDep,
    offset: Annotated[int, Query(ge=0)] = 0,
    limit: Annotated[int, Query(ge=1, le=200)] = 50,
) -> Page[VqaQueryRead]:
    queries, total = await service.list_for_image(
        image_id, researcher, offset=offset, limit=limit
    )
    return Page[VqaQueryRead](
        items=[VqaQueryRead.from_domain(q) for q in queries],
        total=total,
        offset=offset,
        limit=limit,
    )
