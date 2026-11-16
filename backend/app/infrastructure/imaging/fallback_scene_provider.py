"""A single-place scene from the best source available, with a fallback.

Implements :class:`~app.domain.interfaces.ImageryProvider`. "Analyse this place" asks for one scene.
The preferred source is Google Earth Engine (the newest clear Sentinel-2 view, rendered exactly like
the pictures the Gujarat models were fine-tuned on); if it cannot be reached the older fixed EOX
composite is used instead, and the scene says which source it came from, so what is shown on
screen about the imagery is always true.
"""

from __future__ import annotations

from typing import Any

from app.core.logging import get_logger
from app.domain.entities import CapturedScene
from app.domain.errors import ImageryUnavailableError
from app.domain.interfaces import ImageryProvider

logger = get_logger(__name__)


class FallbackSceneProvider:
    def __init__(self, *, primary: ImageryProvider, fallback: ImageryProvider) -> None:
        self._primary = primary
        self._fallback = fallback

    async def fetch_scene(self, latitude: float, longitude: float) -> CapturedScene:
        try:
            return await self._primary.fetch_scene(latitude, longitude)
        except ImageryUnavailableError as error:
            logger.warning("scene_primary_failed_using_fallback", error=error.message)
            return await self._fallback.fetch_scene(latitude, longitude)

    async def aclose(self) -> None:
        for source in (self._primary, self._fallback):
            close: Any = getattr(source, "aclose", None)
            if close is not None:
                await close()
