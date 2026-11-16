"""Forest, grass and farmland from the WorldCover land-cover map, and what the highlight says.

No network and no Earth Engine account: a fake ``ee`` module serves a canned mask.
"""

from __future__ import annotations

import io
from pathlib import Path

import pytest
from PIL import Image

from app.domain.concepts import HighlightConcept
from app.domain.entities import MapFeature
from app.domain.errors import ImageryUnavailableError
from app.infrastructure.imaging.osm_provider import OSM_CAVEAT, OSM_METHOD
from app.infrastructure.imaging.worldcover_provider import METHOD, WorldCoverLandCover
from app.infrastructure.vqa.segmenter import ZeroShotSegmenter

C = HighlightConcept
BOUNDS = (72.50, 23.00, 72.52, 23.02)  # west, south, east, north


def _mask_png(size: int = 64, present: bool = True) -> bytes:
    """Left half 'present' (255), right half not; or all-absent."""
    picture = Image.new("L", (size, size), 0)
    if present:
        picture.paste(Image.new("L", (size // 2, size), 255), (0, 0))
    buffer = io.BytesIO()
    picture.save(buffer, format="PNG")
    return buffer.getvalue()


def _photo(size: int = 64) -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", (size, size), (90, 90, 90)).save(buffer, format="PNG")
    return buffer.getvalue()


class _Chain:
    """Any Earth Engine image: every method returns it again, and the class code is recorded."""

    def __init__(self, ee: FakeEarthEngine) -> None:
        self._ee = ee

    def first(self):
        return self

    def select(self, name):
        return self

    def eq(self, code):
        self._ee.codes.append(code)
        return self

    def multiply(self, value):
        return self

    def toUint8(self):  # noqa: N802 - Earth Engine's own name
        return self


class FakeEarthEngine:
    def __init__(self, png: bytes | None = None) -> None:
        self.codes: list[int] = []
        self.requests: list[dict] = []
        self._png = png if png is not None else _mask_png()
        self.data = self

    def ImageCollection(self, name):  # noqa: N802
        self.collection = name
        return _Chain(self)

    def computePixels(self, request):  # noqa: N802
        self.requests.append(request)
        return self._png


# --- the source ----------------------------------------------------------------------------


@pytest.mark.parametrize(("concept", "code"), [(C.FOREST, 10), (C.GRASS, 30), (C.FARMLAND, 40)])
async def test_each_thing_asks_for_its_own_worldcover_class(concept, code) -> None:
    ee = FakeEarthEngine()

    await WorldCoverLandCover(earth_engine=lambda: ee).mask(concept, BOUNDS, (64, 64))

    assert ee.collection == "ESA/WorldCover/v200"
    assert ee.codes == [code]


async def test_the_request_covers_exactly_the_picture_with_north_at_the_top() -> None:
    ee = FakeEarthEngine()

    await WorldCoverLandCover(earth_engine=lambda: ee).mask(C.FOREST, BOUNDS, (128, 64))

    grid = ee.requests[0]["grid"]
    assert ee.requests[0]["fileFormat"] == "PNG"
    assert grid["dimensions"] == {"width": 128, "height": 64}
    assert grid["crsCode"] == "EPSG:4326"
    transform = grid["affineTransform"]
    assert transform["translateX"] == pytest.approx(72.50)  # west edge
    assert transform["translateY"] == pytest.approx(23.02)  # north edge: row 0
    assert transform["scaleX"] == pytest.approx(0.02 / 128)
    assert transform["scaleY"] == pytest.approx(-0.02 / 64)  # negative: rows run southwards


async def test_roads_and_other_things_are_refused() -> None:
    source = WorldCoverLandCover(earth_engine=lambda: FakeEarthEngine())

    with pytest.raises(ImageryUnavailableError, match="no 'roads' class"):
        await source.mask(C.ROADS, BOUNDS, (64, 64))


async def test_a_place_fetched_once_is_not_fetched_again(tmp_path: Path) -> None:
    ee = FakeEarthEngine()
    first = WorldCoverLandCover(earth_engine=lambda: ee, cache_dir=tmp_path)
    await first.mask(C.FOREST, BOUNDS, (64, 64))

    again = WorldCoverLandCover(earth_engine=lambda: ee, cache_dir=tmp_path)
    await again.mask(C.FOREST, BOUNDS, (64, 64))

    assert len(ee.requests) == 1


async def test_an_earth_engine_failure_is_reported_not_swallowed() -> None:
    class Down(FakeEarthEngine):
        def ImageCollection(self, name):  # noqa: N802
            raise RuntimeError("not signed in")

    with pytest.raises(ImageryUnavailableError, match="Earth Engine"):
        await WorldCoverLandCover(earth_engine=lambda: Down()).mask(C.FOREST, BOUNDS, (64, 64))


def test_it_says_where_the_mask_came_from() -> None:
    source = WorldCoverLandCover(earth_engine=lambda: None)

    method, forest_caveat = source.describe(C.FOREST)
    _, grass_caveat = source.describe(C.GRASS)

    assert method == METHOD and "WorldCover" in method
    assert "not found by the answering model" in method
    assert "trees outside forests" in forest_caveat  # tree cover is not the same as forest
    assert "trees outside forests" not in grass_caveat
    assert "grassland" in grass_caveat


# --- the highlight ----------------------------------------------------------------------------


class _Source:
    def __init__(self, *, fail: bool = False, png: bytes | None = None) -> None:
        self.fail = fail
        self.png = png if png is not None else _mask_png()
        self.asked: list[HighlightConcept] = []

    def describe(self, concept, bounds_wgs84=None):
        return OSM_METHOD, OSM_CAVEAT

    async def mask(self, concept, bounds_wgs84, size):
        self.asked.append(concept)
        if self.fail:
            raise ImageryUnavailableError("down")
        return self.png


class _Osm:
    def __init__(self, features=None) -> None:
        self._features = features if features is not None else []
        self.asked: list[HighlightConcept] = []

    def describe(self, concept, bounds_wgs84):
        return OSM_METHOD, OSM_CAVEAT

    async def fetch(self, concept, bounds_wgs84):
        self.asked.append(concept)
        return self._features


class _LandCover(WorldCoverLandCover):
    """The real describe() text, with a canned mask instead of Earth Engine."""

    def __init__(self, *, fail: bool = False) -> None:
        super().__init__(earth_engine=lambda: None)
        self.fail = fail
        self.asked: list[HighlightConcept] = []

    async def mask(self, concept, bounds_wgs84, size):
        self.asked.append(concept)
        if self.fail:
            raise ImageryUnavailableError("down")
        return _mask_png(size[0])


def _segmenter(land_cover, osm=None) -> ZeroShotSegmenter:
    return ZeroShotSegmenter(
        clipseg_model_id="not-loaded-in-these-tests", map_data=osm, land_cover=land_cover
    )


async def test_forest_is_drawn_from_worldcover_and_says_so() -> None:
    land_cover, osm = _LandCover(), _Osm()

    result = await _segmenter(land_cover, osm).highlight(_photo(), [C.FOREST], bounds_wgs84=BOUNDS)

    (layer,) = result.layers
    assert "WorldCover" in layer.method and "OpenStreetMap" not in layer.method
    assert layer.caveat is not None and "answering model" in layer.caveat
    assert layer.coverage == pytest.approx(0.5, abs=0.05)  # the left half of the canned mask
    assert land_cover.asked == [C.FOREST]
    assert osm.asked == []  # the public server is never needed


async def test_when_earth_engine_fails_forest_falls_back_to_openstreetmap() -> None:
    square = MapFeature(
        "polygon",
        (
            (72.505, 23.005),
            (72.515, 23.005),
            (72.515, 23.015),
            (72.505, 23.015),
            (72.505, 23.005),
        ),
    )
    land_cover, osm = _LandCover(fail=True), _Osm([square])

    result = await _segmenter(land_cover, osm).highlight(_photo(), [C.FOREST], bounds_wgs84=BOUNDS)

    (layer,) = result.layers
    assert "OpenStreetMap" in layer.method
    assert osm.asked == [C.FOREST]


async def test_when_both_fail_forest_falls_back_to_the_picture_and_says_so() -> None:
    class DownOsm(_Osm):
        async def fetch(self, concept, bounds_wgs84):
            raise ImageryUnavailableError("down")

    result = await _segmenter(_LandCover(fail=True), DownOsm()).highlight(
        _photo(), [C.FOREST], bounds_wgs84=BOUNDS
    )

    (layer,) = result.layers
    assert "colour index" in layer.method
    assert layer.caveat is not None
    assert "map data couldn't be reached" in layer.caveat.lower()


async def test_a_picture_with_no_location_skips_the_land_cover_map() -> None:
    land_cover = _LandCover()

    result = await _segmenter(land_cover, _Osm()).highlight(_photo(), [C.FOREST])

    assert land_cover.asked == []
    assert "colour index" in result.layers[0].method


async def test_vegetation_roads_and_buildings_never_ask_the_land_cover_map() -> None:
    land_cover, osm = _LandCover(), _Osm()
    segmenter = _segmenter(land_cover, osm)

    await segmenter.highlight(_photo(), [C.VEGETATION], bounds_wgs84=BOUNDS)
    await segmenter.highlight(_photo(), [C.ROADS], bounds_wgs84=BOUNDS)

    assert land_cover.asked == []
    assert osm.asked == [C.ROADS]
