"""Building outlines from Google Open Buildings, and choosing between them and OpenStreetMap.

No network and no Earth Engine account: a fake ``ee`` module serves canned outlines.
"""

from __future__ import annotations

import io
from pathlib import Path

import pytest
from PIL import Image

from app.domain.concepts import HighlightConcept
from app.domain.entities import MapFeature
from app.domain.errors import ImageryUnavailableError
from app.infrastructure.imaging.open_buildings_provider import (
    CAVEAT,
    METHOD,
    OpenBuildingsMapData,
    parse_polygons,
)
from app.infrastructure.imaging.osm_provider import OSM_CAVEAT, OSM_METHOD
from app.infrastructure.imaging.regional_map_data import RegionalMapData
from app.infrastructure.vqa.segmenter import ZeroShotSegmenter

C = HighlightConcept
GUJARAT = (72.50, 23.00, 72.52, 23.02)  # west, south, east, north: Ahmedabad
NETHERLANDS = (5.70, 52.90, 5.72, 52.92)

SQUARE = [[72.505, 23.005], [72.510, 23.005], [72.510, 23.010], [72.505, 23.010], [72.505, 23.005]]


def _polygon(ring=SQUARE) -> dict:
    return {"type": "Feature", "geometry": {"type": "Polygon", "coordinates": [ring]}}


# --- Reading Earth Engine's answer -------------------------------------------------------


def test_polygons_become_map_features_in_longitude_latitude() -> None:
    (feature,) = parse_polygons([_polygon()])

    assert feature.kind == "polygon"
    assert feature.points[0] == (72.505, 23.005)
    assert len(feature.points) == 5


def test_a_multipolygon_gives_one_outline_per_part() -> None:
    multi = {
        "type": "Feature",
        "geometry": {"type": "MultiPolygon", "coordinates": [[SQUARE], [SQUARE]]},
    }

    assert len(parse_polygons([multi])) == 2


def test_anything_that_is_not_a_closed_outline_is_skipped() -> None:
    point = {"type": "Feature", "geometry": {"type": "Point", "coordinates": [72.5, 23.0]}}
    sliver = _polygon([[72.5, 23.0], [72.6, 23.0]])
    nothing = {"type": "Feature", "geometry": None}

    assert parse_polygons([point, sliver, nothing]) == []


# --- The source ----------------------------------------------------------------------------


class _Collection:
    def __init__(self, ee: FakeEarthEngine) -> None:
        self._ee = ee

    def filterBounds(self, region):  # noqa: N802 - Earth Engine's own name
        self._ee.regions.append(region)
        return self

    def filter(self, condition):
        self._ee.conditions.append(condition)
        return self

    def select(self, names):
        return self

    def getInfo(self):  # noqa: N802
        self._ee.requests += 1
        if self._ee.too_dense and self._ee.requests <= self._ee.too_dense:
            raise RuntimeError("Collection query aborted after accumulating over 5000 elements.")
        return {"features": [_polygon()]}


class FakeEarthEngine:
    def __init__(self, *, too_dense: int = 0) -> None:
        self.regions: list = []
        self.conditions: list = []
        self.requests = 0
        self.too_dense = too_dense  # the first N requests fail as "too many elements"
        self.Geometry = self
        self.Filter = self

    def Rectangle(self, coords, proj=None, geodesic=None):  # noqa: N802
        return ("rectangle", tuple(coords))

    def gte(self, name, value):
        return ("gte", name, value)

    def FeatureCollection(self, name):  # noqa: N802
        self.collection = name
        return _Collection(self)


async def test_buildings_come_back_as_outlines_for_the_whole_area() -> None:
    ee = FakeEarthEngine()
    source = OpenBuildingsMapData(earth_engine=lambda: ee)

    features = await source.fetch(C.BUILDINGS, GUJARAT)

    assert ee.collection == "GOOGLE/Research/open-buildings/v3/polygons"
    assert ee.requests == 16  # the area is asked for as a 4 x 4 split
    assert len(features) == 16
    assert all(f.kind == "polygon" for f in features)
    assert ("gte", "confidence", 0.7) in ee.conditions  # weak detections are left out


async def test_a_piece_too_dense_for_one_request_is_split_and_asked_again() -> None:
    ee = FakeEarthEngine(too_dense=1)
    source = OpenBuildingsMapData(earth_engine=lambda: ee)

    features = await source.fetch(C.BUILDINGS, GUJARAT)

    assert ee.requests == 16 + 4  # one piece failed once and was split in four
    assert len(features) == 16 - 1 + 4


async def test_only_buildings_are_offered() -> None:
    source = OpenBuildingsMapData(earth_engine=lambda: FakeEarthEngine())

    with pytest.raises(ImageryUnavailableError, match="only has buildings"):
        await source.fetch(C.ROADS, GUJARAT)


async def test_a_place_fetched_once_is_not_fetched_again_even_after_a_restart(
    tmp_path: Path,
) -> None:
    ee = FakeEarthEngine()
    first = OpenBuildingsMapData(earth_engine=lambda: ee, cache_dir=tmp_path)
    await first.fetch(C.BUILDINGS, GUJARAT)
    requests_after_first = ee.requests

    again = await first.fetch(C.BUILDINGS, GUJARAT)  # in memory
    after_restart = OpenBuildingsMapData(earth_engine=lambda: ee, cache_dir=tmp_path)
    from_disk = await after_restart.fetch(C.BUILDINGS, GUJARAT)  # from the file

    assert ee.requests == requests_after_first
    assert len(again) == len(from_disk) == 16


async def test_an_earth_engine_failure_is_reported_not_swallowed() -> None:
    class Down(FakeEarthEngine):
        def FeatureCollection(self, name):  # noqa: N802
            raise RuntimeError("not signed in")

    source = OpenBuildingsMapData(earth_engine=lambda: Down())

    with pytest.raises(ImageryUnavailableError, match="Earth Engine"):
        await source.fetch(C.BUILDINGS, GUJARAT)


def test_it_says_where_the_outlines_came_from() -> None:
    method, caveat = OpenBuildingsMapData(earth_engine=lambda: None).describe(C.BUILDINGS, GUJARAT)

    assert "Open Buildings" in method and "not found by the answering model" in method
    assert (method, caveat) == (METHOD, CAVEAT)
    assert "OpenStreetMap" not in method


# --- Choosing between the two sources ------------------------------------------------------


class _Source:
    def __init__(self, name: str) -> None:
        self.name = name
        self.asked: list[tuple[HighlightConcept, tuple]] = []

    def describe(self, concept, bounds_wgs84):
        return f"{self.name} method", f"{self.name} caveat"

    async def fetch(self, concept, bounds_wgs84):
        self.asked.append((concept, bounds_wgs84))
        return [MapFeature("polygon", tuple(map(tuple, SQUARE)))]


def _router() -> tuple[RegionalMapData, _Source, _Source]:
    regional, fallback = _Source("regional"), _Source("fallback")
    router = RegionalMapData(
        regional=regional, fallback=fallback, concepts=frozenset({C.BUILDINGS})
    )
    return router, regional, fallback


async def test_buildings_in_india_use_the_regional_source() -> None:
    router, regional, fallback = _router()

    await router.fetch(C.BUILDINGS, GUJARAT)

    assert [c for c, _ in regional.asked] == [C.BUILDINGS]
    assert fallback.asked == []
    assert router.describe(C.BUILDINGS, GUJARAT) == ("regional method", "regional caveat")


async def test_buildings_elsewhere_still_use_openstreetmap() -> None:
    router, regional, fallback = _router()

    await router.fetch(C.BUILDINGS, NETHERLANDS)

    assert regional.asked == []
    assert len(fallback.asked) == 1
    assert router.describe(C.BUILDINGS, NETHERLANDS) == ("fallback method", "fallback caveat")


async def test_other_things_in_india_still_use_openstreetmap() -> None:
    router, regional, fallback = _router()

    await router.fetch(C.ROADS, GUJARAT)

    assert regional.asked == []
    assert [c for c, _ in fallback.asked] == [C.ROADS]


# --- What the highlight says on screen -------------------------------------------------------


def _png(size: int = 64) -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", (size, size), (90, 90, 90)).save(buffer, format="PNG")
    return buffer.getvalue()


class _Fixed:
    """A source that returns one building and describes itself as given."""

    def __init__(self, method: str, caveat: str) -> None:
        self._described = (method, caveat)

    def describe(self, concept, bounds_wgs84):
        return self._described

    async def fetch(self, concept, bounds_wgs84):
        return [MapFeature("polygon", tuple(map(tuple, SQUARE)))]


async def test_a_building_highlight_in_gujarat_names_open_buildings_not_openstreetmap() -> None:
    router = RegionalMapData(
        regional=_Fixed(METHOD, CAVEAT),
        fallback=_Fixed(OSM_METHOD, OSM_CAVEAT),
        concepts=frozenset({C.BUILDINGS}),
    )
    segmenter = ZeroShotSegmenter(clipseg_model_id="not-loaded-in-these-tests", map_data=router)

    result = await segmenter.highlight(_png(), [C.BUILDINGS], bounds_wgs84=GUJARAT)

    (layer,) = result.layers
    assert "Open Buildings" in layer.method
    assert "OpenStreetMap" not in layer.method
    assert layer.caveat is not None and "answering model" in layer.caveat
    assert layer.coverage > 0


async def test_a_building_highlight_in_the_netherlands_still_names_openstreetmap() -> None:
    router = RegionalMapData(
        regional=_Fixed(METHOD, CAVEAT),
        fallback=_Fixed(OSM_METHOD, OSM_CAVEAT),
        concepts=frozenset({C.BUILDINGS}),
    )
    segmenter = ZeroShotSegmenter(clipseg_model_id="not-loaded-in-these-tests", map_data=router)

    result = await segmenter.highlight(_png(), [C.BUILDINGS], bounds_wgs84=NETHERLANDS)

    (layer,) = result.layers
    assert "OpenStreetMap" in layer.method
