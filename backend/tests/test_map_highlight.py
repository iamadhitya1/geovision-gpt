"""Highlighting what a question names: map data from OpenStreetMap, the colour index, and how
the pieces are combined into one overlay — see ``app/infrastructure/imaging/osm_provider.py``
and ``app/infrastructure/vqa/segmenter.py``.

No network and no model download: OpenStreetMap is served from canned responses, and the
CLIPSeg-backed concepts (water, and buildings without map data) are covered elsewhere.
"""

from __future__ import annotations

import io

import httpx
import numpy as np
import pytest
from PIL import Image

from app.domain.concepts import HighlightConcept
from app.domain.entities import MapFeature
from app.domain.errors import ImageryUnavailableError
from app.infrastructure.imaging.osm_provider import (
    OverpassMapData,
    build_query,
    parse_features,
)
from app.infrastructure.vqa.segmenter import ZeroShotSegmenter, rasterise

C = HighlightConcept
BOUNDS = (72.50, 23.00, 72.52, 23.02)  # west, south, east, north

SQUARE = [
    {"lat": 23.008, "lon": 72.508},
    {"lat": 23.008, "lon": 72.512},
    {"lat": 23.012, "lon": 72.512},
    {"lat": 23.012, "lon": 72.508},
    {"lat": 23.008, "lon": 72.508},
]


# --- Asking OpenStreetMap -----------------------------------------------------------------


def test_the_query_asks_only_for_the_requested_kind_inside_the_boxed_area() -> None:
    query = build_query(C.ROADS, BOUNDS)

    assert '["highway"' in query
    assert "(23.0,72.5,23.02,72.52)" in query  # south, west, north, east
    assert "out geom" in query
    assert "building" not in query


@pytest.mark.parametrize(
    "concept", [C.ROADS, C.BUILDINGS, C.FOREST, C.GRASS, C.FARMLAND, C.CEMETERY]
)
def test_every_map_data_concept_has_a_query(concept: HighlightConcept) -> None:
    assert build_query(concept, BOUNDS).startswith("[out:json]")


def test_a_road_is_a_line() -> None:
    payload = {
        "elements": [
            {"type": "way", "geometry": [{"lat": 23.0, "lon": 72.5}, {"lat": 23.01, "lon": 72.51}]}
        ]
    }

    (feature,) = parse_features(payload, C.ROADS)

    assert feature.kind == "line"
    assert feature.points[0] == (72.5, 23.0)  # (longitude, latitude)


def test_a_closed_building_outline_is_a_polygon() -> None:
    (feature,) = parse_features({"elements": [{"type": "way", "geometry": SQUARE}]}, C.BUILDINGS)

    assert feature.kind == "polygon"


def test_a_closed_road_loop_is_still_a_line() -> None:
    (feature,) = parse_features({"elements": [{"type": "way", "geometry": SQUARE}]}, C.ROADS)

    assert feature.kind == "line"


def test_only_the_outer_ring_of_a_multipolygon_is_drawn() -> None:
    relation = {
        "type": "relation",
        "members": [
            {"role": "outer", "geometry": SQUARE},
            {"role": "inner", "geometry": SQUARE},
        ],
    }

    features = parse_features({"elements": [relation]}, C.FOREST)

    assert [f.kind for f in features] == ["polygon"]


def test_elements_without_geometry_are_skipped() -> None:
    assert parse_features({"elements": [{"type": "way"}, {"type": "node"}]}, C.BUILDINGS) == []


def _overpass(handler) -> OverpassMapData:
    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    return OverpassMapData(
        user_agent="test", endpoints=("https://one.test/api", "https://two.test/api"), client=client
    )


async def test_features_come_back_parsed_and_the_second_ask_is_served_from_cache() -> None:
    calls: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(str(request.url))
        return httpx.Response(200, json={"elements": [{"type": "way", "geometry": SQUARE}]})

    provider = _overpass(handler)

    first = await provider.fetch(C.FOREST, BOUNDS)
    second = await provider.fetch(C.FOREST, BOUNDS)

    assert len(first) == 1
    assert second == first
    assert len(calls) == 1


async def test_the_second_server_is_tried_when_the_first_fails() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if "one.test" in str(request.url):
            return httpx.Response(504)
        return httpx.Response(200, json={"elements": [{"type": "way", "geometry": SQUARE}]})

    features = await _overpass(handler).fetch(C.FOREST, BOUNDS)

    assert len(features) == 1


async def test_every_server_failing_is_reported_not_swallowed() -> None:
    with pytest.raises(ImageryUnavailableError):
        await _overpass(lambda _request: httpx.Response(500)).fetch(C.ROADS, BOUNDS)


async def test_a_thing_with_no_map_data_is_refused_without_calling_out() -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        raise AssertionError("must not be called")

    with pytest.raises(ImageryUnavailableError):
        await _overpass(handler).fetch(C.WATER, BOUNDS)


# --- Drawing it -----------------------------------------------------------------------------


def test_a_road_is_drawn_as_a_line_through_the_right_row() -> None:
    mid_lat = (BOUNDS[1] + BOUNDS[3]) / 2
    road = MapFeature("line", ((BOUNDS[0], mid_lat), (BOUNDS[2], mid_lat)))

    mask = rasterise([road], BOUNDS, (100, 100), line_width=2)

    assert mask[48:52, :].any()
    assert not mask[5, :].any()
    assert not mask[95, :].any()


def test_a_building_is_filled_in() -> None:
    mid_lon, mid_lat = (BOUNDS[0] + BOUNDS[2]) / 2, (BOUNDS[1] + BOUNDS[3]) / 2
    d = 0.003
    building = MapFeature(
        "polygon",
        (
            (mid_lon - d, mid_lat - d),
            (mid_lon + d, mid_lat - d),
            (mid_lon + d, mid_lat + d),
            (mid_lon - d, mid_lat + d),
            (mid_lon - d, mid_lat - d),
        ),
    )

    mask = rasterise([building], BOUNDS, (100, 100), line_width=2)

    assert mask[50, 50]
    assert not mask[2, 2]


def test_nothing_to_draw_gives_an_empty_mask() -> None:
    assert not rasterise([], BOUNDS, (64, 64), line_width=2).any()


# --- Choosing the method per thing ------------------------------------------------------------


class FakeMapData:
    def __init__(self, features: dict[HighlightConcept, list[MapFeature]] | None = None) -> None:
        self._features = features or {}
        self.asked: list[HighlightConcept] = []
        self.fail = False

    async def fetch(self, concept, bounds_wgs84):
        self.asked.append(concept)
        if self.fail:
            raise ImageryUnavailableError("down")
        return self._features.get(concept, [])


def _png(colour: tuple[int, int, int], size: int = 64) -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", (size, size), colour).save(buffer, format="PNG")
    return buffer.getvalue()


def _segmenter(map_data: FakeMapData | None) -> ZeroShotSegmenter:
    return ZeroShotSegmenter(clipseg_model_id="not-loaded-in-these-tests", map_data=map_data)


MID_LAT = (BOUNDS[1] + BOUNDS[3]) / 2
ROAD = MapFeature("line", ((BOUNDS[0], MID_LAT), (BOUNDS[2], MID_LAT)))
FOREST_BLOCK = MapFeature(
    "polygon",
    (
        (72.505, 23.005),
        (72.515, 23.005),
        (72.515, 23.015),
        (72.505, 23.015),
        (72.505, 23.005),
    ),
)


async def test_roads_come_from_map_data_and_say_so() -> None:
    fake = FakeMapData({C.ROADS: [ROAD]})

    result = await _segmenter(fake).highlight(_png((90, 90, 90)), [C.ROADS], bounds_wgs84=BOUNDS)

    (layer,) = result.layers
    assert layer.concept == "roads"
    assert "OpenStreetMap" in layer.method
    assert "not detected from the image" in layer.method
    assert layer.caveat is not None and "drawn wider" in layer.caveat
    assert layer.coverage > 0
    assert fake.asked == [C.ROADS]


async def test_roads_cannot_be_shown_without_a_location() -> None:
    with pytest.raises(ImageryUnavailableError, match="no location"):
        await _segmenter(FakeMapData()).highlight(_png((90, 90, 90)), [C.ROADS])


async def test_roads_cannot_be_shown_when_map_data_is_unreachable() -> None:
    fake = FakeMapData()
    fake.fail = True

    with pytest.raises(ImageryUnavailableError, match="couldn't be reached"):
        await _segmenter(fake).highlight(_png((90, 90, 90)), [C.ROADS], bounds_wgs84=BOUNDS)


async def test_roads_cannot_be_shown_when_map_data_is_switched_off() -> None:
    with pytest.raises(ImageryUnavailableError, match="switched off"):
        await _segmenter(None).highlight(_png((90, 90, 90)), [C.ROADS], bounds_wgs84=BOUNDS)


async def test_forest_falls_back_to_the_colour_index_and_admits_it_cannot_tell_types_apart() -> (
    None
):
    fake = FakeMapData()
    fake.fail = True

    result = await _segmenter(fake).highlight(_png((30, 160, 40)), [C.FOREST], bounds_wgs84=BOUNDS)

    (layer,) = result.layers
    assert "colour index" in layer.method
    assert layer.coverage > 0.9  # an all-green image
    assert layer.caveat is not None
    assert "couldn't be reached" in layer.caveat
    assert "forest from grass" in layer.caveat


async def test_vegetation_never_asks_map_data() -> None:
    fake = FakeMapData()

    result = await _segmenter(fake).highlight(
        _png((30, 160, 40)), [C.VEGETATION], bounds_wgs84=BOUNDS
    )

    assert fake.asked == []
    assert result.layers[0].caveat is None


async def test_forest_uses_map_data_when_it_has_some() -> None:
    fake = FakeMapData({C.FOREST: [FOREST_BLOCK]})

    result = await _segmenter(fake).highlight(_png((90, 90, 90)), [C.FOREST], bounds_wgs84=BOUNDS)

    (layer,) = result.layers
    assert "OpenStreetMap" in layer.method
    assert 0.2 < layer.coverage < 0.6  # the block is about a quarter of the scene


async def test_forest_the_map_does_not_know_is_reported_as_absent_not_guessed_from_colour() -> None:
    """No mapped forest is a real answer ('none here'); it must not turn into a green guess."""
    result = await _segmenter(FakeMapData()).highlight(
        _png((30, 160, 40)), [C.FOREST], bounds_wgs84=BOUNDS
    )

    assert result.layers[0].coverage == 0.0


async def test_a_comparison_gets_two_layers_in_two_colours_on_one_overlay() -> None:
    fake = FakeMapData({C.ROADS: [ROAD], C.FOREST: [FOREST_BLOCK]})

    result = await _segmenter(fake).highlight(
        _png((90, 90, 90), size=100), [C.ROADS, C.FOREST], bounds_wgs84=BOUNDS
    )

    assert [layer.concept for layer in result.layers] == ["roads", "forest"]
    assert result.layers[0].color != result.layers[1].color
    overlay = Image.open(io.BytesIO(result.overlay_png))
    assert overlay.mode == "RGBA"
    assert overlay.size == (100, 100)


async def test_everything_outside_the_things_asked_about_is_dimmed() -> None:
    fake = FakeMapData({C.FOREST: [FOREST_BLOCK]})

    result = await _segmenter(fake).highlight(
        _png((90, 90, 90), size=100), [C.FOREST], bounds_wgs84=BOUNDS
    )

    pixels = np.asarray(Image.open(io.BytesIO(result.overlay_png)))
    assert pixels[1, 1, 3] > 100  # a corner, outside the block: darkened
    assert pixels[50, 50, 3] < 100  # the middle of the block: left clear


# --- The on-disk cache ------------------------------------------------------------------------


async def test_features_fetched_once_are_available_later_with_no_network(tmp_path) -> None:
    online = OverpassMapData(
        user_agent="test",
        endpoints=("https://one.test/api",),
        client=httpx.AsyncClient(
            transport=httpx.MockTransport(
                lambda _r: httpx.Response(
                    200, json={"elements": [{"type": "way", "geometry": SQUARE}]}
                )
            )
        ),
        cache_dir=tmp_path,
    )
    fetched = await online.fetch(C.BUILDINGS, BOUNDS)

    def dead(_request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("no network")

    offline = OverpassMapData(
        user_agent="test",
        endpoints=("https://one.test/api",),
        client=httpx.AsyncClient(transport=httpx.MockTransport(dead)),
        cache_dir=tmp_path,
    )  # a fresh server process: nothing in memory, only the files on disk

    assert await offline.fetch(C.BUILDINGS, BOUNDS) == fetched
    assert list(tmp_path.glob("buildings-*.json"))


async def test_a_different_place_is_not_served_from_another_places_cache(tmp_path) -> None:
    calls: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(str(request.url))
        return httpx.Response(200, json={"elements": []})

    provider = OverpassMapData(
        user_agent="test",
        endpoints=("https://one.test/api",),
        client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
        cache_dir=tmp_path,
    )

    await provider.fetch(C.ROADS, BOUNDS)
    await provider.fetch(C.ROADS, (73.0, 24.0, 73.02, 24.02))

    assert len(calls) == 2


async def test_a_corrupt_cache_file_is_ignored_and_refetched(tmp_path) -> None:
    provider = OverpassMapData(
        user_agent="test",
        endpoints=("https://one.test/api",),
        client=httpx.AsyncClient(
            transport=httpx.MockTransport(
                lambda _r: httpx.Response(
                    200, json={"elements": [{"type": "way", "geometry": SQUARE}]}
                )
            )
        ),
        cache_dir=tmp_path,
    )
    await provider.fetch(C.FOREST, BOUNDS)
    for f in tmp_path.glob("*.json"):
        f.write_text("{not json", encoding="utf-8")
    fresh = OverpassMapData(
        user_agent="test",
        endpoints=("https://one.test/api",),
        client=httpx.AsyncClient(
            transport=httpx.MockTransport(
                lambda _r: httpx.Response(
                    200, json={"elements": [{"type": "way", "geometry": SQUARE}]}
                )
            )
        ),
        cache_dir=tmp_path,
    )

    assert len(await fresh.fetch(C.FOREST, BOUNDS)) == 1


# --- Being polite to a busy public server -------------------------------------------------------


async def test_a_busy_server_is_retried_once_before_giving_up_on_it(monkeypatch) -> None:
    monkeypatch.setattr("app.infrastructure.imaging.osm_provider._RETRY_WAIT_SECONDS", 0.0)
    answers = iter([httpx.Response(429), httpx.Response(200, json={"elements": []})])
    calls: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(str(request.url))
        return next(answers)

    provider = OverpassMapData(
        user_agent="test",
        endpoints=("https://one.test/api",),
        client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
    )

    assert await provider.fetch(C.ROADS, BOUNDS) == []
    assert len(calls) == 2


async def test_a_hard_failure_is_not_retried_on_the_same_server(monkeypatch) -> None:
    monkeypatch.setattr("app.infrastructure.imaging.osm_provider._RETRY_WAIT_SECONDS", 0.0)
    calls: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(str(request.url))
        return httpx.Response(400)

    provider = OverpassMapData(
        user_agent="test",
        endpoints=("https://one.test/api",),
        client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
    )

    with pytest.raises(ImageryUnavailableError):
        await provider.fetch(C.ROADS, BOUNDS)
    assert len(calls) == 1


async def test_two_things_asked_at_once_reach_the_server_one_at_a_time() -> None:
    import asyncio

    in_flight = 0
    peak = 0

    async def slow_handler(request: httpx.Request) -> httpx.Response:
        nonlocal in_flight, peak
        in_flight += 1
        peak = max(peak, in_flight)
        await asyncio.sleep(0.05)
        in_flight -= 1
        return httpx.Response(200, json={"elements": []})

    provider = OverpassMapData(
        user_agent="test",
        endpoints=("https://one.test/api",),
        client=httpx.AsyncClient(transport=httpx.MockTransport(slow_handler)),
    )

    await asyncio.gather(provider.fetch(C.ROADS, BOUNDS), provider.fetch(C.BUILDINGS, BOUNDS))

    assert peak == 1


# --- Dense layers are asked for in smaller pieces ------------------------------------------------


async def test_buildings_are_fetched_in_four_pieces_and_joined() -> None:
    seen_boxes: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        body = request.content.decode()
        seen_boxes.append(body)
        return httpx.Response(200, json={"elements": [{"type": "way", "geometry": SQUARE}]})

    provider = OverpassMapData(
        user_agent="test",
        endpoints=("https://one.test/api",),
        client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
    )

    features = await provider.fetch(C.BUILDINGS, BOUNDS)

    assert len(seen_boxes) == 4
    assert len(set(seen_boxes)) == 4  # four different boxes, not the same one four times
    assert len(features) == 4  # one per piece, all joined


async def test_roads_are_still_fetched_in_one_request() -> None:
    calls: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(str(request.url))
        return httpx.Response(200, json={"elements": []})

    provider = OverpassMapData(
        user_agent="test",
        endpoints=("https://one.test/api",),
        client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
    )

    await provider.fetch(C.ROADS, BOUNDS)

    assert len(calls) == 1


async def test_if_one_piece_fails_the_layer_fails_but_finished_pieces_are_kept(tmp_path) -> None:
    """Half a map would mislead, so the layer fails — but the pieces that did arrive are kept,
    so the next attempt only fetches what's missing instead of starting from nothing."""
    served = {"n": 0}
    fail_third = {"on": True}

    def handler(_request: httpx.Request) -> httpx.Response:
        served["n"] += 1
        if fail_third["on"] and served["n"] == 3:
            return httpx.Response(400)
        return httpx.Response(200, json={"elements": []})

    provider = OverpassMapData(
        user_agent="test",
        endpoints=("https://one.test/api",),
        client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
        cache_dir=tmp_path,
    )

    with pytest.raises(ImageryUnavailableError):
        await provider.fetch(C.BUILDINGS, BOUNDS)
    assert served["n"] == 3  # two pieces done, the third failed, the fourth never asked
    assert len(list(tmp_path.glob("buildings-*.json"))) == 2

    fail_third["on"] = False
    assert await provider.fetch(C.BUILDINGS, BOUNDS) == []
    assert served["n"] == 5  # only the two missing pieces were fetched on the retry


async def test_a_layer_that_takes_too_long_is_given_up_on_not_waited_for(monkeypatch) -> None:
    import asyncio

    monkeypatch.setattr("app.infrastructure.imaging.osm_provider._DEADLINE_SECONDS", 0.2)

    async def very_slow(_request: httpx.Request) -> httpx.Response:
        await asyncio.sleep(5)
        return httpx.Response(200, json={"elements": []})

    provider = OverpassMapData(
        user_agent="test",
        endpoints=("https://one.test/api",),
        client=httpx.AsyncClient(transport=httpx.MockTransport(very_slow)),
    )

    with pytest.raises(ImageryUnavailableError, match="too slow"):
        await provider.fetch(C.ROADS, BOUNDS)
