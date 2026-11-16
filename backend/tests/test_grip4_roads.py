"""Road lines for India from the GRIP4 inventory, and the routing that uses them.

No network and no Earth Engine account: a fake ``ee`` module serves canned road lines.
"""

from __future__ import annotations

import io
from pathlib import Path

import pytest
from PIL import Image

from app.domain.concepts import HighlightConcept
from app.domain.entities import MapFeature
from app.domain.errors import ImageryUnavailableError
from app.infrastructure.imaging.grip4_roads_provider import (
    CAVEAT,
    METHOD,
    Grip4RoadsMapData,
    parse_lines,
)
from app.infrastructure.imaging.open_buildings_provider import CAVEAT as OB_CAVEAT
from app.infrastructure.imaging.open_buildings_provider import METHOD as OB_METHOD
from app.infrastructure.imaging.osm_provider import OSM_CAVEAT, OSM_METHOD
from app.infrastructure.imaging.regional_map_data import RegionalMapData
from app.infrastructure.vqa.segmenter import ZeroShotSegmenter

C = HighlightConcept
GUJARAT = (72.50, 23.00, 72.52, 23.02)
NETHERLANDS = (5.70, 52.90, 5.72, 52.92)
LINE = [[72.505, 23.010], [72.515, 23.010]]


def _line(coords=LINE) -> dict:
    return {"type": "Feature", "geometry": {"type": "LineString", "coordinates": coords}}


# --- Reading Earth Engine's answer -----------------------------------------------------------


def test_a_line_becomes_a_road_in_longitude_latitude() -> None:
    (feature,) = parse_lines([_line()])

    assert feature.kind == "line"
    assert feature.points == ((72.505, 23.010), (72.515, 23.010))


def test_a_multiline_gives_one_road_per_part() -> None:
    multi = {
        "type": "Feature",
        "geometry": {"type": "MultiLineString", "coordinates": [LINE, LINE]},
    }

    assert len(parse_lines([multi])) == 2


def test_anything_that_is_not_a_line_is_skipped() -> None:
    point = {"type": "Feature", "geometry": {"type": "Point", "coordinates": [72.5, 23.0]}}
    stub = _line([[72.5, 23.0]])

    assert parse_lines([point, stub, {"type": "Feature", "geometry": None}]) == []


# --- The source --------------------------------------------------------------------------------


class _Collection:
    def __init__(self, ee: FakeEarthEngine) -> None:
        self._ee = ee

    def filterBounds(self, region):  # noqa: N802 - Earth Engine's own name
        return self

    def select(self, names):
        return self

    def getInfo(self):  # noqa: N802
        self._ee.requests += 1
        if self._ee.too_dense and self._ee.requests <= self._ee.too_dense:
            raise RuntimeError("Collection query aborted after accumulating over 5000 elements.")
        return {"features": [_line()]}


class FakeEarthEngine:
    def __init__(self, *, too_dense: int = 0) -> None:
        self.requests = 0
        self.too_dense = too_dense
        self.Geometry = self

    def Rectangle(self, coords, proj=None, geodesic=None):  # noqa: N802
        return ("rectangle", tuple(coords))

    def FeatureCollection(self, name):  # noqa: N802
        self.collection = name
        return _Collection(self)


async def test_roads_come_back_as_lines_for_the_whole_area() -> None:
    ee = FakeEarthEngine()

    features = await Grip4RoadsMapData(earth_engine=lambda: ee).fetch(C.ROADS, GUJARAT)

    assert ee.collection == "projects/sat-io/open-datasets/GRIP4/South-East-Asia"
    assert ee.requests == 4  # asked for as a 2 x 2 split
    assert len(features) == 4
    assert all(f.kind == "line" for f in features)


async def test_a_piece_too_dense_for_one_request_is_split_and_asked_again() -> None:
    ee = FakeEarthEngine(too_dense=1)

    features = await Grip4RoadsMapData(earth_engine=lambda: ee).fetch(C.ROADS, GUJARAT)

    assert ee.requests == 4 + 4  # one piece failed once and was split in four
    assert len(features) == 3 + 4


async def test_only_roads_are_offered() -> None:
    with pytest.raises(ImageryUnavailableError, match="only has roads"):
        await Grip4RoadsMapData(earth_engine=lambda: FakeEarthEngine()).fetch(C.BUILDINGS, GUJARAT)


async def test_a_place_fetched_once_is_not_fetched_again_even_after_a_restart(
    tmp_path: Path,
) -> None:
    ee = FakeEarthEngine()
    first = Grip4RoadsMapData(earth_engine=lambda: ee, cache_dir=tmp_path)
    await first.fetch(C.ROADS, GUJARAT)
    after_first = ee.requests

    again = await first.fetch(C.ROADS, GUJARAT)
    restarted = Grip4RoadsMapData(earth_engine=lambda: ee, cache_dir=tmp_path)
    from_disk = await restarted.fetch(C.ROADS, GUJARAT)

    assert ee.requests == after_first
    assert len(again) == len(from_disk) == 4


async def test_an_earth_engine_failure_is_reported_not_swallowed() -> None:
    class Down(FakeEarthEngine):
        def FeatureCollection(self, name):  # noqa: N802
            raise RuntimeError("not signed in")

    with pytest.raises(ImageryUnavailableError, match="Earth Engine"):
        await Grip4RoadsMapData(earth_engine=lambda: Down()).fetch(C.ROADS, GUJARAT)


def test_it_says_where_the_roads_came_from() -> None:
    method, caveat = Grip4RoadsMapData(earth_engine=lambda: None).describe(C.ROADS, GUJARAT)

    assert (method, caveat) == (METHOD, CAVEAT)
    assert "GRIP4" in method and "not found by the answering model" in method
    assert "OpenStreetMap" not in method.split("(")[0]


# --- The routing the app builds: roads and buildings regional, OpenStreetMap elsewhere ---------


class _Source:
    def __init__(self, method: str, caveat: str) -> None:
        self._described = (method, caveat)
        self.asked: list[HighlightConcept] = []

    def describe(self, concept, bounds_wgs84):
        return self._described

    async def fetch(self, concept, bounds_wgs84):
        self.asked.append(concept)
        return [MapFeature("line", tuple(map(tuple, LINE)))]


def _app_routing() -> tuple[RegionalMapData, _Source, _Source, _Source]:
    roads = _Source(METHOD, CAVEAT)
    buildings = _Source(OB_METHOD, OB_CAVEAT)
    osm = _Source(OSM_METHOD, OSM_CAVEAT)
    router = RegionalMapData(
        regional=roads,
        fallback=RegionalMapData(
            regional=buildings, fallback=osm, concepts=frozenset({C.BUILDINGS})
        ),
        concepts=frozenset({C.ROADS}),
    )
    return router, roads, buildings, osm


async def test_in_india_roads_come_from_grip4_and_buildings_from_open_buildings() -> None:
    router, roads, buildings, osm = _app_routing()

    await router.fetch(C.ROADS, GUJARAT)
    await router.fetch(C.BUILDINGS, GUJARAT)

    assert roads.asked == [C.ROADS]
    assert buildings.asked == [C.BUILDINGS]
    assert osm.asked == []
    assert router.describe(C.ROADS, GUJARAT) == (METHOD, CAVEAT)
    assert router.describe(C.BUILDINGS, GUJARAT) == (OB_METHOD, OB_CAVEAT)


async def test_everything_else_in_india_and_everything_outside_it_stays_on_openstreetmap() -> None:
    router, roads, buildings, osm = _app_routing()

    await router.fetch(C.FOREST, GUJARAT)
    await router.fetch(C.ROADS, NETHERLANDS)
    await router.fetch(C.BUILDINGS, NETHERLANDS)

    assert roads.asked == [] and buildings.asked == []
    assert osm.asked == [C.FOREST, C.ROADS, C.BUILDINGS]


# --- What the highlight says on screen -------------------------------------------------------


def _png(size: int = 64) -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", (size, size), (90, 90, 90)).save(buffer, format="PNG")
    return buffer.getvalue()


async def test_a_road_highlight_in_gujarat_names_the_inventory_not_openstreetmap() -> None:
    router, *_ = _app_routing()
    segmenter = ZeroShotSegmenter(clipseg_model_id="not-loaded-in-these-tests", map_data=router)

    result = await segmenter.highlight(_png(), [C.ROADS], bounds_wgs84=GUJARAT)

    (layer,) = result.layers
    assert "GRIP4" in layer.method
    assert layer.caveat is not None and "answering model" in layer.caveat
    assert layer.coverage > 0
