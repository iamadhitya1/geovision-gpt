"""Choose the map-data source by place: a regional source where it is good, OpenStreetMap elsewhere.

Implements :class:`~app.domain.interfaces.MapDataPort`. The regional source (Google Open Buildings)
covers South Asia, where OpenStreetMap barely maps buildings; outside that region Open Buildings
has nothing, so a picture of the Netherlands keeps using OpenStreetMap. A source that is *chosen*
and then fails is not silently swapped for the other: the highlight reports that map data couldn't
be reached and falls back to the image-based method, so the label on screen is always the truth.
"""

from __future__ import annotations

from app.domain.concepts import HighlightConcept
from app.domain.entities import MapFeature
from app.domain.interfaces import MapDataPort

Bounds = tuple[float, float, float, float]

# (west, south, east, north): the Indian subcontinent, which Open Buildings covers.
SOUTH_ASIA: Bounds = (67.0, 6.0, 98.0, 37.5)


def _inside(bounds: Bounds, region: Bounds) -> bool:
    west, south, east, north = bounds
    lon, lat = (west + east) / 2, (south + north) / 2
    return region[0] <= lon <= region[2] and region[1] <= lat <= region[3]


class RegionalMapData:
    def __init__(
        self,
        *,
        regional: MapDataPort,
        fallback: MapDataPort,
        concepts: frozenset[HighlightConcept],
        region: Bounds = SOUTH_ASIA,
    ) -> None:
        self._regional = regional
        self._fallback = fallback
        self._concepts = concepts
        self._region = region

    def _source(self, concept: HighlightConcept, bounds_wgs84: Bounds) -> MapDataPort:
        if concept in self._concepts and _inside(bounds_wgs84, self._region):
            return self._regional
        return self._fallback

    async def aclose(self) -> None:
        close = getattr(self._fallback, "aclose", None)
        if close is not None:
            await close()

    def describe(self, concept: HighlightConcept, bounds_wgs84: Bounds) -> tuple[str, str]:
        return self._source(concept, bounds_wgs84).describe(concept, bounds_wgs84)

    async def fetch(self, concept: HighlightConcept, bounds_wgs84: Bounds) -> list[MapFeature]:
        return await self._source(concept, bounds_wgs84).fetch(concept, bounds_wgs84)
