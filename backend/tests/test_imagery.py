"""Fetching a satellite scene for a point on the map — see
``app/infrastructure/imaging/sentinel_provider.py`` and ``POST /images/capture``.

No network: the provider is exercised against an ``httpx.MockTransport`` serving synthetic
tiles, and the endpoint against a fake provider.
"""

from __future__ import annotations

import io
import math

import httpx
import pytest
import rasterio
from httpx import AsyncClient
from PIL import Image
from rasterio.io import MemoryFile

from app.domain.entities import User
from app.domain.errors import ImageryUnavailableError
from app.infrastructure.imaging.sentinel_provider import (
    EoxSentinel2Provider,
    choose_zoom,
    ground_resolution_m,
    mercator_pixel,
)
from tests.conftest import USER_PASSWORD, auth_headers, make_geotiff_bytes

TILE_URL = "https://tiles.example.test/{z}/{y}/{x}.jpg"


def _jpeg_tile(colour: tuple[int, int, int]) -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", (256, 256), colour).save(buffer, format="JPEG")
    return buffer.getvalue()


def _provider(handler) -> EoxSentinel2Provider:
    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    return EoxSentinel2Provider(tile_url_template=TILE_URL, user_agent="test", client=client)


# --- The tile arithmetic -----------------------------------------------------------------


def test_one_pixel_at_the_equator_on_zoom_zero_is_about_156_km() -> None:
    assert ground_resolution_m(0.0, 0) == pytest.approx(156_543.03, rel=1e-6)


def test_pixels_get_smaller_going_north_because_of_web_mercator() -> None:
    assert ground_resolution_m(60.0, 10) == pytest.approx(
        ground_resolution_m(0.0, 10) / 2, rel=1e-3
    )


@pytest.mark.parametrize(
    ("latitude", "expected"),
    [(0.0, 14), (23.0, 14), (53.0, 14), (70.0, 13)],
)
def test_zoom_is_the_coarsest_one_no_coarser_than_ten_metres_per_pixel(
    latitude: float, expected: int
) -> None:
    assert choose_zoom(latitude, 10.0, max_zoom=14) == expected
    assert ground_resolution_m(latitude, expected) <= 10.0 + 1e-9


def test_the_origin_of_the_map_is_the_centre_of_the_world_pixel_grid() -> None:
    assert mercator_pixel(0.0, 0.0, 1) == pytest.approx((256.0, 256.0))


# --- Building a scene ---------------------------------------------------------------------


async def test_scene_is_a_256_pixel_georeferenced_geotiff_centred_on_the_point() -> None:
    requested: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requested.append(str(request.url))
        return httpx.Response(200, content=_jpeg_tile((20, 120, 40)))

    provider = _provider(handler)
    latitude, longitude = 52.9554, 5.7743

    data = await provider.fetch_scene(latitude, longitude)

    with MemoryFile(data) as memory, memory.open() as scene:
        assert (scene.width, scene.height, scene.count) == (256, 256, 3)
        assert scene.crs == rasterio.crs.CRS.from_epsg(3857)
        left, bottom, right, top = scene.bounds
        centre_x = math.radians(longitude) * 6_378_137.0
        centre_y = math.log(math.tan(math.pi / 4 + math.radians(latitude) / 2)) * 6_378_137.0
        assert (left + right) / 2 == pytest.approx(centre_x, abs=1.0)
        assert (bottom + top) / 2 == pytest.approx(centre_y, abs=1.0)
        # 2.56 km on the ground is 2.56 km / cos(lat) wide in Web Mercator units.
        assert right - left == pytest.approx(2560 / math.cos(math.radians(latitude)), rel=1e-6)
        assert scene.read().mean() > 0  # real pixels, not an empty raster
    assert 1 <= len(requested) <= 9
    assert all("/14/" in url for url in requested)  # latitude 53 -> zoom 14


async def test_a_dead_imagery_service_is_reported_not_swallowed() -> None:
    provider = _provider(lambda _request: httpx.Response(500))

    with pytest.raises(ImageryUnavailableError):
        await provider.fetch_scene(23.0, 72.5)


async def test_a_network_failure_is_reported_not_swallowed() -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("no route to host")

    with pytest.raises(ImageryUnavailableError):
        await _provider(handler).fetch_scene(23.0, 72.5)


@pytest.mark.parametrize(("latitude", "longitude"), [(85.0, 0.0), (-81.0, 10.0), (10.0, 200.0)])
async def test_places_outside_coverage_are_refused_without_calling_the_service(
    latitude: float, longitude: float
) -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        raise AssertionError("the imagery service must not be called")

    with pytest.raises(ImageryUnavailableError):
        await _provider(handler).fetch_scene(latitude, longitude)


# --- The endpoint --------------------------------------------------------------------------


class FakeImageryProvider:
    def __init__(self) -> None:
        self.requested: list[tuple[float, float]] = []

    async def fetch_scene(self, latitude: float, longitude: float) -> bytes:
        self.requested.append((latitude, longitude))
        return make_geotiff_bytes(width=64, height=64, band_count=3)


async def test_capture_stores_a_scene_as_the_callers_own_image(
    app, client: AsyncClient, researcher: User
) -> None:
    fake = FakeImageryProvider()
    app.state.imagery = fake
    headers = await auth_headers(client, researcher.email, USER_PASSWORD)

    response = await client.post(
        "/images/capture", headers=headers, json={"latitude": 52.9554, "longitude": 5.7743}
    )

    assert response.status_code == 201, response.text
    body = response.json()
    assert fake.requested == [(52.9554, 5.7743)]
    assert body["original_filename"] == "sentinel2_52.9554N_5.7743E.tif"
    assert body["status"] == "ready"  # went through the normal upload pipeline
    listed = await client.get("/images", headers=headers)
    assert [i["id"] for i in listed.json()["items"]] == [body["id"]]


async def test_capture_names_southern_and_western_points_correctly(
    app, client: AsyncClient, researcher: User
) -> None:
    app.state.imagery = FakeImageryProvider()
    headers = await auth_headers(client, researcher.email, USER_PASSWORD)

    response = await client.post(
        "/images/capture", headers=headers, json={"latitude": -33.8688, "longitude": -70.6693}
    )

    assert response.json()["original_filename"] == "sentinel2_33.8688S_70.6693W.tif"


async def test_capture_is_unavailable_when_imagery_is_switched_off(
    app, client: AsyncClient, researcher: User
) -> None:
    app.state.imagery = None
    headers = await auth_headers(client, researcher.email, USER_PASSWORD)

    response = await client.post(
        "/images/capture", headers=headers, json={"latitude": 10.0, "longitude": 10.0}
    )

    assert response.status_code == 502
    assert response.json()["code"] == "imagery_unavailable"


@pytest.mark.parametrize(
    "payload",
    [
        {"latitude": 91.0, "longitude": 0.0},
        {"latitude": 85.0, "longitude": 0.0},
        {"latitude": 0.0, "longitude": 181.0},
        {"latitude": 0.0},
    ],
)
async def test_capture_rejects_impossible_coordinates(
    app, client: AsyncClient, researcher: User, payload: dict
) -> None:
    app.state.imagery = FakeImageryProvider()
    headers = await auth_headers(client, researcher.email, USER_PASSWORD)

    response = await client.post("/images/capture", headers=headers, json=payload)

    assert response.status_code == 422


async def test_capture_requires_a_signed_in_user(app, client: AsyncClient) -> None:
    app.state.imagery = FakeImageryProvider()

    response = await client.post("/images/capture", json={"latitude": 10.0, "longitude": 10.0})

    assert response.status_code == 401


# --- Where a captured scene sits, worked out before it exists --------------------------------


@pytest.mark.parametrize(
    ("latitude", "longitude"), [(52.9554, 5.7743), (23.0225, 72.5714), (-33.8688, -70.6693)]
)
async def test_the_predicted_footprint_matches_the_scene_that_is_actually_built(
    latitude: float, longitude: float
) -> None:
    from rasterio.warp import transform_bounds

    from app.infrastructure.imaging.sentinel_provider import scene_bounds_wgs84

    provider = _provider(lambda _r: httpx.Response(200, content=_jpeg_tile((20, 120, 40))))
    data = await provider.fetch_scene(latitude, longitude)

    with MemoryFile(data) as memory, memory.open() as scene:
        actual = transform_bounds(scene.crs, "EPSG:4326", *scene.bounds)

    assert scene_bounds_wgs84(latitude, longitude) == pytest.approx(actual, abs=1e-6)
