"""Forest, grass and farmland masks from ESA WorldCover, served through Earth Engine.

Implements :class:`~app.domain.interfaces.LandCoverPort`. Why it exists: those three highlights drew
OpenStreetMap through the free public Overpass servers, which are often at capacity, and when they
failed the highlight fell back to marking *all* green vegetation, so "show the forest" could
light up every field. WorldCover is a 10 m land-cover map (the same scale as the pictures the models
read) with separate classes for tree cover, grassland and cropland, and it is reached through the
Earth Engine sign-in the app already uses.

It is a model-made map of 2021, not a survey: trees outside forests count as "tree cover", and land
can have changed since. The description attached to the highlight says so.
"""

from __future__ import annotations

import asyncio
import hashlib
from pathlib import Path
from typing import Any

from app.core.logging import get_logger
from app.domain.concepts import HighlightConcept
from app.domain.errors import ImageryUnavailableError

logger = get_logger(__name__)

COLLECTION = "ESA/WorldCover/v200"
# WorldCover's class codes for the things a question can name.
CLASS_CODES = {
    HighlightConcept.FOREST: 10,  # tree cover
    HighlightConcept.GRASS: 30,  # grassland
    HighlightConcept.FARMLAND: 40,  # cropland
}
_NAMES = {
    HighlightConcept.FOREST: "tree cover",
    HighlightConcept.GRASS: "grassland",
    HighlightConcept.FARMLAND: "cropland",
}
METHOD = "ESA WorldCover land-cover map (10 m classes, 2021) — not found by the answering model"

Bounds = tuple[float, float, float, float]


class WorldCoverLandCover:
    """``earth_engine`` returns the signed-in module (a callable, so tests can inject one)."""

    def __init__(self, *, earth_engine: Any, cache_dir: Path | None = None) -> None:
        self._earth_engine = earth_engine
        self._cache_dir = cache_dir

    def describe(self, concept: HighlightConcept) -> tuple[str, str]:
        name = _NAMES[concept]
        caveat = (
            f'Shows WorldCover\'s "{name}" class from a 2021 land-cover map made by a model, so '
            + ("it includes trees outside forests, " if concept is HighlightConcept.FOREST else "")
            + "the land can have changed since and some patches will be mislabelled. It is not "
            "something the answering model found."
        )
        return METHOD, caveat

    async def mask(
        self, concept: HighlightConcept, bounds_wgs84: Bounds, size: tuple[int, int]
    ) -> bytes:
        if concept not in CLASS_CODES:
            raise ImageryUnavailableError(
                f"WorldCover has no '{concept.value}' class.", concept=concept.value
            )
        path = self._disk_path(concept, bounds_wgs84, size)
        if path is not None and path.exists():
            return path.read_bytes()
        try:
            png = await asyncio.to_thread(self._fetch_sync, concept, bounds_wgs84, size)
        except ImageryUnavailableError:
            raise
        except Exception as exc:
            logger.warning("worldcover_failed", error=str(exc))
            raise ImageryUnavailableError(
                "The land-cover map couldn't be reached through Earth Engine right now.",
                concept=concept.value,
            ) from exc
        if path is not None:
            try:
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(png)
            except OSError:
                logger.warning("worldcover_cache_unwritable", path=str(path))
        return png

    # --- Earth Engine (blocking) ----------------------------------------------------------

    def _fetch_sync(
        self, concept: HighlightConcept, bounds: Bounds, size: tuple[int, int]
    ) -> bytes:
        ee = self._earth_engine() if callable(self._earth_engine) else self._earth_engine
        west, south, east, north = bounds
        width, height = size
        present = (
            ee.ImageCollection(COLLECTION)
            .first()
            .select("Map")
            .eq(CLASS_CODES[concept])
            .multiply(255)
            .toUint8()
        )
        grid = {
            "dimensions": {"width": width, "height": height},
            "affineTransform": {
                "scaleX": (east - west) / width,
                "shearX": 0,
                "translateX": west,
                "shearY": 0,
                "scaleY": -(north - south) / height,  # row 0 is the north edge
                "translateY": north,
            },
            "crsCode": "EPSG:4326",
        }
        png: bytes = ee.data.computePixels(
            {"expression": present, "fileFormat": "PNG", "grid": grid}
        )
        return png

    # --- disk cache -----------------------------------------------------------------------

    def _disk_path(
        self, concept: HighlightConcept, bounds: Bounds, size: tuple[int, int]
    ) -> Path | None:
        if self._cache_dir is None:
            return None
        key = (concept.value, tuple(round(v, 4) for v in bounds), size)
        digest = hashlib.sha1(repr(key).encode("utf-8")).hexdigest()[:20]
        return self._cache_dir / f"worldcover-{concept.value}-{digest}.png"
