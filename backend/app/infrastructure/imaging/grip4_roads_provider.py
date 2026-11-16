"""Roads for India from the Global Roads Inventory (GRIP4), served through Earth Engine.

Implements :class:`~app.domain.interfaces.MapDataPort` for **roads only**. Why it exists: the
"highlight the roads" answer drew OpenStreetMap through the free public Overpass servers, which are
often at capacity (a tiny query took 39-47 s on 16 Nov 2026, and a busy server answers "at
capacity"), so for a place picked at random nothing was cached and the highlight failed with "map
data couldn't be reached". GRIP4 sits in Earth Engine, which the app already uses and which
answered in about a second, and it covers India.

GRIP4 (Meijer et al., 2018) is a global road inventory compiled around 2018 from national road
datasets and OpenStreetMap. It is road lines only, so newer roads can be missing, and it is not
something the answering model found. The description attached to the highlight says so.
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
from app.infrastructure.imaging.open_buildings_provider import Bounds, split_bounds

logger = get_logger(__name__)

# The inventory is split by world region; this one holds India (checked at Ahmedabad: 2,039 road
# segments in a 7 x 8 km box).
COLLECTION = "projects/sat-io/open-datasets/GRIP4/South-East-Asia"
_ELEMENT_LIMIT_MESSAGE = "5000"
_GRID = 2
_MAX_DEPTH = 3
_WORKERS = 4

METHOD = (
    "GRIP4 Global Roads Inventory (road lines compiled from national datasets and OpenStreetMap) "
    "— not found by the answering model"
)
CAVEAT = (
    "Road lines come from a global inventory compiled around 2018, so newer roads can be missing, "
    "and they are not something the answering model found."
)


def parse_lines(features: list[dict[str, Any]]) -> list[MapFeature]:
    """GeoJSON line features to :class:`MapFeature` lines in (longitude, latitude)."""
    result: list[MapFeature] = []
    for feature in features:
        geometry = feature.get("geometry") or {}
        kind = geometry.get("type")
        if kind == "LineString":
            parts = [geometry["coordinates"]]
        elif kind == "MultiLineString":
            parts = list(geometry["coordinates"])
        else:
            continue
        for part in parts:
            points = tuple((float(x), float(y)) for x, y, *_ in part)
            if len(points) >= 2:
                result.append(MapFeature("line", points))
    return result


class Grip4RoadsMapData:
    """Road lines from Earth Engine; ``earth_engine`` returns the signed-in module."""

    def __init__(self, *, earth_engine: Any, cache_dir: Path | None = None) -> None:
        self._earth_engine = earth_engine
        self._cache_dir = cache_dir
        self._memory: dict[tuple[float, ...], list[MapFeature]] = {}
        self._lock = asyncio.Lock()

    def describe(self, concept: HighlightConcept, bounds_wgs84: Bounds) -> tuple[str, str]:
        return METHOD, CAVEAT

    async def fetch(self, concept: HighlightConcept, bounds_wgs84: Bounds) -> list[MapFeature]:
        if concept is not HighlightConcept.ROADS:
            raise ImageryUnavailableError(
                f"The roads inventory only has roads, not '{concept.value}'.", concept=concept.value
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
                logger.warning("grip4_roads_failed", error=str(exc))
                raise ImageryUnavailableError(
                    "The roads inventory couldn't be reached through Earth Engine right now.",
                    concept=concept.value,
                ) from exc
            await asyncio.to_thread(self._write_disk, key, stored)
            logger.info("grip4_roads_fetched", segments=len(stored))
        async with self._lock:
            self._memory[key] = stored
        return stored

    # --- Earth Engine (blocking) ----------------------------------------------------------

    def _fetch_sync(self, bounds: Bounds) -> list[MapFeature]:
        ee = self._earth_engine() if callable(self._earth_engine) else self._earth_engine
        pieces = split_bounds(bounds, _GRID)
        with ThreadPoolExecutor(max_workers=_WORKERS) as pool:
            parts = list(pool.map(lambda piece: self._piece(ee, piece, depth=1), pieces))
        return [feature for part in parts for feature in part]

    def _piece(self, ee: Any, piece: Bounds, depth: int) -> list[MapFeature]:
        region = ee.Geometry.Rectangle(list(piece), "EPSG:4326", False)
        collection = ee.FeatureCollection(COLLECTION).filterBounds(region).select([])
        try:
            return parse_lines(collection.getInfo()["features"])
        except Exception as exc:
            if _ELEMENT_LIMIT_MESSAGE in str(exc) and depth < _MAX_DEPTH:
                return [
                    f for sub in split_bounds(piece, 2) for f in self._piece(ee, sub, depth + 1)
                ]
            raise

    # --- disk cache -----------------------------------------------------------------------

    def _disk_path(self, key: tuple[float, ...]) -> Path | None:
        if self._cache_dir is None:
            return None
        digest = hashlib.sha1(repr(key).encode("utf-8")).hexdigest()[:20]
        return self._cache_dir / f"grip4_roads-{digest}.json"

    def _read_disk(self, key: tuple[float, ...]) -> list[MapFeature] | None:
        path = self._disk_path(key)
        if path is None or not path.exists():
            return None
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
            return [MapFeature("line", tuple((float(x), float(y)) for x, y in p)) for p in raw]
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
            logger.warning("grip4_roads_cache_unwritable", path=str(path))
