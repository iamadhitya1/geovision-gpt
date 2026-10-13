"""Roads, buildings, forests and more, from OpenStreetMap, for a stretch of the Earth.

Why this exists: at 10 m per pixel a road is about one pixel wide and a forest looks like any
other green field, so no image-only method here can honestly point at them. OpenStreetMap
already knows where they are — and RSVQA's own questions and answers were built from it — so
for a georeferenced image the app can draw the real thing. It is **map data, not something
detected in the image**: it can be newer or older than the imagery, and anything nobody has
mapped is missing. The highlight says so.

Uses the public Overpass API (one request per concept per scene, results cached). Map data ©
OpenStreetMap contributors, ODbL — the Map page and the highlight panel carry the attribution.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
from collections import OrderedDict
from pathlib import Path
from typing import Any

import httpx

from app.core.logging import get_logger
from app.domain.concepts import HighlightConcept
from app.domain.entities import MapFeature
from app.domain.errors import ImageryUnavailableError

logger = get_logger(__name__)

# One public server. Three others (kumi.systems, private.coffee, maps.mail.ru) were tried and
# each timed out on every request, so listing them only added 20 s of waiting to each failure.
# A tuple so another reliable mirror can be added (or one self-hosted) without changing code.
DEFAULT_ENDPOINTS = ("https://overpass-api.de/api/interpreter",)

_ROAD_CLASSES = (
    "motorway|trunk|primary|secondary|tertiary|unclassified|residential|service|living_street"
    "|motorway_link|trunk_link|primary_link|secondary_link|tertiary_link"
)

# Overpass filters per concept. Each is a bare selector; the bounding box is added on the way out.
_SELECTORS: dict[HighlightConcept, list[str]] = {
    HighlightConcept.ROADS: [f'way["highway"~"^({_ROAD_CLASSES})$"]'],
    # Ways only: also asking for building *relations* made the public server time out on a
    # dense town, and the few multipolygon buildings aren't worth that.
    HighlightConcept.BUILDINGS: ['way["building"]'],
    HighlightConcept.FOREST: [
        'way["natural"="wood"]',
        'way["landuse"="forest"]',
        'relation["natural"="wood"]',
        'relation["landuse"="forest"]',
    ],
    HighlightConcept.GRASS: [
        'way["landuse"~"^(grass|meadow|recreation_ground|village_green)$"]',
        'way["natural"="grassland"]',
        'way["leisure"~"^(park|garden)$"]',
    ],
    HighlightConcept.FARMLAND: [
        'way["landuse"~"^(farmland|farmyard|orchard|vineyard|greenhouse_horticulture)$"]',
        'relation["landuse"="farmland"]',
    ],
    HighlightConcept.CEMETERY: [
        'way["landuse"="cemetery"]',
        'way["amenity"="grave_yard"]',
        'relation["landuse"="cemetery"]',
    ],
}

SUPPORTED = frozenset(_SELECTORS)

_CACHE_SIZE = 64

# A dense town has thousands of buildings in one scene, and asking for them all at once makes
# the free servers time out. These are asked for in n x n smaller pieces instead, one after
# another, and joined. If any piece fails the whole layer fails: half a map would mislead.
_TILES: dict[HighlightConcept, int] = {HighlightConcept.BUILDINGS: 2}


def _split(
    bounds: tuple[float, float, float, float], n: int
) -> list[tuple[float, float, float, float]]:
    """``bounds`` (west, south, east, north) cut into ``n`` x ``n`` equal pieces."""
    west, south, east, north = bounds
    dx, dy = (east - west) / n, (north - south) / n
    return [
        (west + i * dx, south + j * dy, west + (i + 1) * dx, south + (j + 1) * dy)
        for j in range(n)
        for i in range(n)
    ]


# "Slow down" / "try again" answers from a busy public server. Worth one more try after a short
# wait before giving up on that server.
_TRY_AGAIN = frozenset({429, 502, 503, 504})
_ATTEMPTS = 2
_RETRY_WAIT_SECONDS = 3.0

# The longest a person is made to wait for one layer. Past this the fetch gives up (keeping
# whatever pieces it finished) and the highlight falls back or says so, rather than leaving
# the button on "locating…" for minutes while the free server struggles.
_DEADLINE_SECONDS = 30.0


def build_query(concept: HighlightConcept, bounds_wgs84: tuple[float, float, float, float]) -> str:
    """The Overpass QL for ``concept`` inside ``bounds_wgs84`` = (west, south, east, north)."""
    west, south, east, north = bounds_wgs84
    bbox = f"({south},{west},{north},{east})"
    body = "".join(f"{selector}{bbox};" for selector in _SELECTORS[concept])
    return f"[out:json][timeout:25];({body});out geom;"


def _points(geometry: list[dict[str, float]]) -> tuple[tuple[float, float], ...]:
    return tuple((float(p["lon"]), float(p["lat"])) for p in geometry)


def parse_features(payload: dict[str, Any], concept: HighlightConcept) -> list[MapFeature]:
    """Turn an Overpass ``out geom`` response into lines and polygons.

    Roads are lines. Everything else is an area: a closed way is a polygon, and so is each
    outer member of a multipolygon relation (a simplification — holes are not cut out).
    """
    features: list[MapFeature] = []
    for element in payload.get("elements", []):
        if element.get("type") == "way" and element.get("geometry"):
            points = _points(element["geometry"])
            if len(points) < 2:
                continue
            closed = points[0] == points[-1] and len(points) >= 4
            is_area = concept is not HighlightConcept.ROADS
            features.append(MapFeature("polygon" if is_area and closed else "line", points))
        elif element.get("type") == "relation":
            for member in element.get("members", []):
                if member.get("role") == "outer" and member.get("geometry"):
                    points = _points(member["geometry"])
                    if len(points) >= 4 and points[0] == points[-1]:
                        features.append(MapFeature("polygon", points))
                    elif len(points) >= 2:
                        features.append(MapFeature("line", points))
    return features


class OverpassMapData:
    """Implements :class:`~app.domain.interfaces.MapDataPort`.

    ``client`` is injectable so tests can serve canned responses without a network.
    """

    def __init__(
        self,
        *,
        user_agent: str,
        endpoints: tuple[str, ...] = DEFAULT_ENDPOINTS,
        timeout_seconds: float = 20.0,
        client: httpx.AsyncClient | None = None,
        cache_dir: Path | None = None,
        deadline_seconds: float | None = None,
    ) -> None:
        self._endpoints = endpoints
        self._deadline_seconds = deadline_seconds
        self._cache_dir = cache_dir
        self._client = client or httpx.AsyncClient(
            timeout=timeout_seconds, headers={"User-Agent": user_agent}
        )
        self._cache: OrderedDict[tuple[str, tuple[float, ...]], list[MapFeature]] = OrderedDict()
        self._lock = asyncio.Lock()
        self._gate = asyncio.Semaphore(1)

    async def fetch(
        self, concept: HighlightConcept, bounds_wgs84: tuple[float, float, float, float]
    ) -> list[MapFeature]:
        """All features of ``concept`` inside ``bounds_wgs84``.

        A dense layer is asked for in several smaller pieces (see ``_TILES``), each cached on
        its own — so if the free server is slow and the time limit runs out part-way, the
        pieces already fetched are kept and the next attempt carries on from there instead of
        starting over. If any piece is still missing the layer fails: half a map would mislead.
        """
        if concept not in SUPPORTED:
            raise ImageryUnavailableError(f"No map data is available for '{concept.value}'.")

        pieces = _split(bounds_wgs84, _TILES.get(concept, 1))
        try:
            async with asyncio.timeout(self._deadline_seconds or _DEADLINE_SECONDS):
                parts = [await self._piece(concept, piece) for piece in pieces]
        except TimeoutError as exc:
            raise ImageryUnavailableError(
                "OpenStreetMap map data is too slow right now; try again in a moment.",
                concept=concept.value,
            ) from exc
        return [feature for part in parts for feature in part]

    async def _piece(
        self, concept: HighlightConcept, bounds_wgs84: tuple[float, float, float, float]
    ) -> list[MapFeature]:
        # ~11 m precision: bounds re-derived from a GeoTIFF differ from the ones a place was
        # warmed up with by a hair, and should still find the stored entry.
        key = (concept.value, tuple(round(v, 4) for v in bounds_wgs84))
        async with self._lock:
            if key in self._cache:
                self._cache.move_to_end(key)
                return self._cache[key]
        stored = await asyncio.to_thread(self._read_disk, key)
        if stored is not None:
            async with self._lock:
                self._remember(key, stored)
            return stored

        # One request at a time: the public servers rate-limit a client that fires several at
        # once (a comparison asks for two things), and being polite is part of using them.
        async with self._gate:
            features = await self._fetch_piece(concept, bounds_wgs84)
        async with self._lock:
            self._remember(key, features)
        await asyncio.to_thread(self._write_disk, key, features)
        logger.info("map_data_fetched", concept=concept.value, features=len(features))
        return features

    async def _fetch_piece(
        self, concept: HighlightConcept, bounds_wgs84: tuple[float, float, float, float]
    ) -> list[MapFeature]:
        """One Overpass request, over each server in turn, retrying a busy one once."""
        query = build_query(concept, bounds_wgs84)
        last_error: Exception | None = None
        for endpoint in self._endpoints:
            for attempt in range(_ATTEMPTS):
                try:
                    response = await self._client.post(endpoint, data={"data": query})
                    if response.status_code in _TRY_AGAIN and attempt + 1 < _ATTEMPTS:
                        logger.info("map_data_busy", endpoint=endpoint, status=response.status_code)
                        await asyncio.sleep(_RETRY_WAIT_SECONDS)
                        continue
                    response.raise_for_status()
                    return parse_features(response.json(), concept)
                except (httpx.HTTPError, ValueError) as exc:
                    logger.warning("map_data_failed", endpoint=endpoint, error=str(exc))
                    last_error = exc
                    break  # next server

        raise ImageryUnavailableError(
            "OpenStreetMap map data couldn't be reached right now.", concept=concept.value
        ) from last_error

    # --- caching -------------------------------------------------------------------------
    #
    # In memory for repeat clicks, and on disk so a place rehearsed once while online still
    # works instantly (and offline) at the demo. Map data for a fixed stretch of the Earth is
    # slow to change, and the public servers are the least reliable part of this feature.

    def _remember(self, key: tuple[str, tuple[float, ...]], features: list[MapFeature]) -> None:
        self._cache[key] = features
        self._cache.move_to_end(key)
        while len(self._cache) > _CACHE_SIZE:
            self._cache.popitem(last=False)

    def _disk_path(self, key: tuple[str, tuple[float, ...]]) -> Path | None:
        if self._cache_dir is None:
            return None
        digest = hashlib.sha1(repr(key).encode("utf-8")).hexdigest()[:20]
        return self._cache_dir / f"{key[0]}-{digest}.json"

    def _read_disk(self, key: tuple[str, tuple[float, ...]]) -> list[MapFeature] | None:
        path = self._disk_path(key)
        if path is None or not path.exists():
            return None
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
            return [
                MapFeature(item["kind"], tuple((float(x), float(y)) for x, y in item["points"]))
                for item in raw
            ]
        except (OSError, ValueError, KeyError, TypeError):
            logger.warning("map_data_cache_unreadable", path=str(path))
            return None

    def _write_disk(self, key: tuple[str, tuple[float, ...]], features: list[MapFeature]) -> None:
        path = self._disk_path(key)
        if path is None:
            return
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            payload = [{"kind": f.kind, "points": [list(p) for p in f.points]} for f in features]
            path.write_text(json.dumps(payload), encoding="utf-8")
        except OSError:
            logger.warning("map_data_cache_unwritable", path=str(path))

    async def aclose(self) -> None:
        await self._client.aclose()
