"""Building footprints for India from Google's Open Buildings, served through Earth Engine.

Implements :class:`~app.domain.interfaces.MapDataPort` for **buildings only**. Why it exists: the
"show where the buildings are" highlight used OpenStreetMap, which maps almost no buildings in rural
Gujarat (on 9 Gujarat tiles it had 42 buildings where Open Buildings had 10,523), so the highlight
drew almost nothing on a picture full of buildings. Open Buildings (version 3) is a set of building
outlines detected by a Google model from higher-resolution satellite imagery, covering South Asia
among other regions.

These footprints are detected, not surveyed: they come from imagery taken at a different date, they
can miss small or covered buildings, and they are not something the answering model found. The
description attached to the highlight says so.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from app.core.logging import get_logger
from app.domain.concepts import HighlightConcept
from app.domain.entities import MapFeature
from app.domain.errors import ImageryUnavailableError

logger = get_logger(__name__)

COLLECTION = "GOOGLE/Research/open-buildings/v3/polygons"
# Detections below this confidence are mostly not buildings (Google suggests 0.65-0.70).
MIN_CONFIDENCE = 0.7
# A footprint count above this fails an Earth Engine request, so a dense area is split further.
_ELEMENT_LIMIT = 5000
_GRID = (
    4  # each request covers a 4 x 4 split of the area, split again if a piece is still too dense
)
_MAX_DEPTH = 3
_WORKERS = 4

METHOD = (
    "Google Open Buildings (building outlines detected from satellite imagery) — "
    "not found by the answering model"
)
CAVEAT = (
    "Outlines come from a Google model run on higher-resolution imagery taken at a different "
    "date, so they can be newer or older than this picture, and small or covered buildings may be "
    "missing. They are not something the answering model found."
)

Bounds = tuple[float, float, float, float]


def _split(bounds: Bounds, n: int) -> list[Bounds]:
    west, south, east, north = bounds
    dx, dy = (east - west) / n, (north - south) / n
    return [
        (west + i * dx, south + j * dy, west + (i + 1) * dx, south + (j + 1) * dy)
        for j in range(n)
        for i in range(n)
    ]


def parse_polygons(features: list[dict[str, Any]]) -> list[MapFeature]:
    """GeoJSON polygons (outer ring only; building outlines have no holes worth cutting) to
    :class:`MapFeature` polygons in (longitude, latitude)."""
    result: list[MapFeature] = []
    for feature in features:
        geometry = feature.get("geometry") or {}
        kind = geometry.get("type")
        if kind == "Polygon":
            rings = [geometry["coordinates"][0]]
        elif kind == "MultiPolygon":
            rings = [polygon[0] for polygon in geometry["coordinates"]]
        else:
            continue
        for ring in rings:
            points = tuple((float(x), float(y)) for x, y, *_ in ring)
            if len(points) >= 4:
                result.append(MapFeature("polygon", points))
    return result


class OpenBuildingsMapData:
    """Building outlines from Earth Engine; ``earth_engine`` returns the signed-in module."""

    def __init__(
        self,
        *,
        earth_engine: Any,
        cache_dir: Path | None = None,
        min_confidence: float = MIN_CONFIDENCE,
    ) -> None:
        self._earth_engine = earth_engine
        self._cache_dir = cache_dir
        self._min_confidence = min_confidence
        self._memory: dict[tuple[float, ...], list[MapFeature]] = {}
        self._lock = asyncio.Lock()

    def describe(self, concept: HighlightConcept, bounds_wgs84: Bounds) -> tuple[str, str]:
        return METHOD, CAVEAT

    async def fetch(self, concept: HighlightConcept, bounds_wgs84: Bounds) -> list[MapFeature]:
        if concept is not HighlightConcept.BUILDINGS:
            raise ImageryUnavailableError(
                f"Open Buildings only has buildings, not '{concept.value}'.",
                concept=concept.value,
            )
        key = tuple(round(v, 4) for v in bounds_wgs84)
        async with self._lock:
            if key in self._memory:
                return self._memory[key]
        stored = await asyncio.to_thread(self._read_disk, key)
        if stored is None:
            try:
                stored = await asyncio.to_thread(self._fetch_sync, bounds_wgs84)
            except ImageryUnavailableError:
                raise
            except Exception as exc:
                logger.warning("open_buildings_failed", error=str(exc))
                raise ImageryUnavailableError(
                    "Google Open Buildings couldn't be reached through Earth Engine right now.",
                    concept=concept.value,
                ) from exc
            await asyncio.to_thread(self._write_disk, key, stored)
            logger.info("open_buildings_fetched", buildings=len(stored))
        async with self._lock:
            self._memory[key] = stored
        return stored

    # --- Earth Engine (blocking) ------------------------------------------------------------

    def _fetch_sync(self, bounds: Bounds) -> list[MapFeature]:
        ee = self._earth_engine() if callable(self._earth_engine) else self._earth_engine
        pieces = _split(bounds, _GRID)
        # The pieces are independent requests, mostly waiting on Google: a few at a time.
        with ThreadPoolExecutor(max_workers=_WORKERS) as pool:
            parts = list(pool.map(lambda piece: self._piece(ee, piece, depth=1), pieces))
        return [feature for part in parts for feature in part]

    def _piece(self, ee: Any, piece: Bounds, depth: int) -> list[MapFeature]:
        region = ee.Geometry.Rectangle(list(piece), "EPSG:4326", False)
        collection = (
            ee.FeatureCollection(COLLECTION)
            .filterBounds(region)
            .filter(ee.Filter.gte("confidence", self._min_confidence))
            .select([])  # outlines only: no properties to download
        )
        try:
            return parse_polygons(collection.getInfo()["features"])
        except Exception as exc:
            if "5000" in str(exc) and depth < _MAX_DEPTH:  # too dense for one request: split it
                return [f for sub in _split(piece, 2) for f in self._piece(ee, sub, depth + 1)]
            raise

    # --- disk cache -------------------------------------------------------------------------

    def _disk_path(self, key: tuple[float, ...]) -> Path | None:
        if self._cache_dir is None:
            return None
        digest = hashlib.sha1(repr(key).encode("utf-8")).hexdigest()[:20]
        return self._cache_dir / f"open_buildings-{digest}.json"

    def _read_disk(self, key: tuple[float, ...]) -> list[MapFeature] | None:
        path = self._disk_path(key)
        if path is None or not path.exists():
            return None
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
            return [MapFeature("polygon", tuple((float(x), float(y)) for x, y in p)) for p in raw]
        except (OSError, ValueError, TypeError):
            return None

    def _write_disk(self, key: tuple[float, ...], features: list[MapFeature]) -> None:
        path = self._disk_path(key)
        if path is None:
            return
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps([[list(p) for p in f.points] for f in features]), "utf-8")
        except OSError:
            logger.warning("open_buildings_cache_unwritable", path=str(path))
