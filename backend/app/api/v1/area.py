"""Area analysis: ask the models about a place bigger than one image (sliding window)."""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends

from app.api.deps import ResearcherUser, get_area_service
from app.application.area_service import MIN_SIZE_KM, AreaAnalysisService
from app.application.dto import AreaAnalysisRead, AreaAnalysisRequest, AreaCapabilitiesRead
from app.domain.area import WINDOW_PX

router = APIRouter(prefix="/area", tags=["Area analysis"])

AreaServiceDep = Annotated[AreaAnalysisService, Depends(get_area_service)]


@router.get(
    "/capabilities",
    response_model=AreaCapabilitiesRead,
    summary="Which imagery sources and area sizes are available",
)
async def area_capabilities(
    _researcher: ResearcherUser, service: AreaServiceDep
) -> AreaCapabilitiesRead:
    return AreaCapabilitiesRead(
        sources=service.available_sources(),
        max_size_km=service.max_size_km,
        min_size_km=MIN_SIZE_KM,
        window_km=WINDOW_PX * 10.0 / 1000.0,
        fusions=sorted(service.available_fusions(), key=lambda f: f.value),
    )


@router.post(
    "/analyse",
    response_model=AreaAnalysisRead,
    summary="Ask a question about an area, with a sliding window",
    responses={
        422: {"description": "Unsupported question, or an area outside the allowed sizes"},
        502: {"description": "The imagery service couldn't be reached, or has nothing there"},
        503: {"description": "No model variant is loaded"},
    },
)
async def analyse_area(
    request: AreaAnalysisRequest, _researcher: ResearcherUser, service: AreaServiceDep
) -> AreaAnalysisRead:
    """Fetches a picture of the square area at the models' ~10 m scale, slides their 256 px
    window across it (half-window steps) and asks the question of every window. The answers are
    combined into a coarse grid and a share-of-area summary per model.

    A rough map of what the windows answered, not an outline of anything. How well the models
    answer on a large area has not been measured against labels."""
    analysis = await service.analyse(
        question=request.question,
        latitude=request.latitude,
        longitude=request.longitude,
        size_km=request.size_km,
        source=request.source,
        date_from=request.date_from,
        date_to=request.date_to,
        fusions=request.fusions,
    )
    return AreaAnalysisRead.from_domain(analysis)
