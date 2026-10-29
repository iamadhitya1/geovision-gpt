"""Ask the models a question about an area bigger than one image: the sliding-window use case.

Like :class:`~app.application.vqa_service.VqaService`, this imports no ML or imaging library. It
fetches a picture of the area through an :class:`~app.domain.interfaces.AreaImageryProvider`,
asks the :class:`~app.domain.interfaces.VqaModelPort` about each 256 px window of it, and
combines the answers with the bookkeeping in :mod:`app.domain.area`.

Nothing is stored: like the attention heatmap, an area analysis is a view onto the models'
answers, not a recorded event.
"""

from __future__ import annotations

import asyncio
from collections.abc import Mapping
from datetime import date

from app.core.logging import get_logger
from app.domain.area import (
    WINDOW_PX,
    AreaAnalysis,
    AreaCell,
    FusionAreaResult,
    WindowAnswer,
    WindowPlan,
    aggregate_cells,
    cell_pixel_box,
    plan_windows,
    summarise,
)
from app.domain.enums import FusionType, ImagerySource
from app.domain.errors import (
    ImageryUnavailableError,
    ModelUnavailableError,
    UnsupportedQuestionError,
    ValidationError,
)
from app.domain.interfaces import AreaImageryProvider, AreaScenePort, VqaModelPort
from app.domain.question_guard import check_question

logger = get_logger(__name__)

# The models read about 10 m of ground per pixel, so a pixel count is a distance.
_GROUND_M = 10.0
# Smallest area worth asking about: exactly one window (2.56 km).
MIN_SIZE_KM = WINDOW_PX * _GROUND_M / 1000.0


class AreaAnalysisService:
    def __init__(
        self,
        *,
        model: VqaModelPort,
        providers: Mapping[ImagerySource, AreaImageryProvider],
        max_size_km: float,
        max_windows: int,
        concurrency: int = 2,
    ) -> None:
        self._model = model
        self._providers = dict(providers)
        self._max_size_km = max_size_km
        self._max_windows = max_windows
        self._gate = asyncio.Semaphore(concurrency)

    @property
    def max_size_km(self) -> float:
        return self._max_size_km

    def available_sources(self) -> list[ImagerySource]:
        """Earth Engine first (recent imagery, any dates) when it is set up, then the fixed
        2020 mosaic."""
        order = (ImagerySource.EARTH_ENGINE, ImagerySource.SENTINEL_CLOUDLESS)
        return [source for source in order if source in self._providers]

    def available_fusions(self) -> frozenset[FusionType]:
        return self._model.available_fusions()

    async def analyse(
        self,
        *,
        question: str,
        latitude: float,
        longitude: float,
        size_km: float,
        source: ImagerySource | None = None,
        date_from: date | None = None,
        date_to: date | None = None,
        fusions: list[FusionType] | None = None,
    ) -> AreaAnalysis:
        check = check_question(question)
        if not check.supported:
            raise UnsupportedQuestionError(check.reason, question=question)

        if not MIN_SIZE_KM <= size_km <= self._max_size_km:
            raise ValidationError(
                f"The area must be between {MIN_SIZE_KM:.2f} and {self._max_size_km:g} km wide.",
                size_km=size_km,
            )
        if date_from is not None and date_to is not None and date_from > date_to:
            raise ValidationError("The start date must not be after the end date.")

        size_px = round(size_km * 1000.0 / _GROUND_M)
        plan = plan_windows(size_px, size_px)
        if plan.window_count > self._max_windows:
            raise ValidationError(
                f"That area needs {plan.window_count} windows; at most {self._max_windows} are "
                "allowed. Choose a smaller area.",
                windows=plan.window_count,
            )

        requested = sorted(
            frozenset(fusions) if fusions else self._model.available_fusions(),
            key=lambda f: f.value,
        )
        if not requested:
            raise ModelUnavailableError()

        available = self.available_sources()
        chosen = source if source is not None else (available[0] if available else None)
        provider = self._providers.get(chosen) if chosen is not None else None
        if chosen is None or provider is None:
            raise ImageryUnavailableError(
                "No satellite imagery source is available on this server for that request."
            )

        scene = await provider.fetch_area(
            latitude, longitude, size_px=size_px, date_from=date_from, date_to=date_to
        )

        answers: dict[FusionType, list[WindowAnswer]] = {f: [] for f in requested}
        failed: dict[FusionType, int] = {f: 0 for f in requested}

        async def ask_window(x: int, y: int) -> None:
            async with self._gate:
                crop = await asyncio.to_thread(scene.crop_png, x, y, plan.window_px)
                results = await asyncio.gather(
                    *(
                        self._model.predict(crop, check.text, fusion=fusion, top_k=1)
                        for fusion in requested
                    ),
                    return_exceptions=True,
                )
            for fusion, result in zip(requested, results, strict=True):
                if isinstance(result, BaseException):
                    failed[fusion] += 1
                    logger.warning("area_window_failed", fusion=fusion.value, error=str(result))
                else:
                    answers[fusion].append(WindowAnswer(x, y, result.answer, result.confidence))

        await asyncio.gather(*(ask_window(x, y) for x, y in plan.origins))

        if all(not answers[f] for f in requested):
            raise ModelUnavailableError("No model could answer for this area.")

        results = tuple(
            self._fusion_result(scene, plan, fusion, answers[fusion], failed[fusion])
            for fusion in requested
        )
        logger.info(
            "area_analysed",
            size_km=size_km,
            windows=plan.window_count,
            source=chosen.value,
            fusions=[f.value for f in requested],
            failed=sum(failed.values()),
        )
        return AreaAnalysis(
            question=question,
            question_used=check.text,
            note=check.note,
            source=chosen,
            captured=scene.captured,
            bounds_wgs84=scene.bounds_wgs84,
            width_px=scene.width_px,
            height_px=scene.height_px,
            window_px=plan.window_px,
            stride_px=plan.stride_px,
            window_count=plan.window_count,
            results=results,
        )

    @staticmethod
    def _fusion_result(
        scene: AreaScenePort,
        plan: WindowPlan,
        fusion: FusionType,
        window_answers: list[WindowAnswer],
        failed: int,
    ) -> FusionAreaResult:
        cells = aggregate_cells(plan, window_answers)
        placed = tuple(
            AreaCell(
                row=c.row,
                col=c.col,
                bounds_wgs84=scene.box_wgs84(*cell_pixel_box(plan, c.row, c.col)),
                answer=c.answer,
                agreement=c.agreement,
                windows=c.windows,
                mean_confidence=c.mean_confidence,
            )
            for c in cells
        )
        return FusionAreaResult(
            fusion=fusion,
            share_by_answer=tuple(summarise(cells)),
            cells=placed,
            failed_windows=failed,
        )
