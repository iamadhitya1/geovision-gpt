"""Sliding-window area analysis: the use case, the endpoint and the imagery behind it.

No network, no model and no Google account: the picture is synthetic (land on the left, water
on the right), the "model" answers by colour, and Earth Engine is a fake module that records the
request it is given.
"""

from __future__ import annotations

import io
import math
from datetime import date

import httpx
import pytest
from httpx import AsyncClient
from PIL import Image

from app.application.area_service import MIN_SIZE_KM, AreaAnalysisService
from app.domain.entities import AnswerAlternative, User, VqaAnswer
from app.domain.enums import FusionType, ImagerySource
from app.domain.errors import (
    ImageryUnavailableError,
    ModelUnavailableError,
    UnsupportedQuestionError,
    ValidationError,
)
from app.infrastructure.imaging.area_scene import MercatorAreaScene
from app.infrastructure.imaging.earth_engine_provider import EarthEngineProvider, build_grid
from app.infrastructure.imaging.sentinel_provider import EoxSentinel2Provider
from tests.conftest import USER_PASSWORD, auth_headers

LAT, LON = 23.0, 72.5
LAND, WATER = (60, 140, 50), (20, 40, 160)


def split_picture(size_px: int) -> Image.Image:
    """Land on the left half, water on the right half."""
    picture = Image.new("RGB", (size_px, size_px), LAND)
    picture.paste(Image.new("RGB", (size_px - size_px // 2, size_px), WATER), (size_px // 2, 0))
    return picture


def scene_of(size_px: int, source: ImagerySource = ImagerySource.SENTINEL_CLOUDLESS):
    return MercatorAreaScene(
        split_picture(size_px),
        latitude=LAT,
        longitude=LON,
        ground_m=10.0,
        source=source,
        captured="a synthetic test picture",
    )


class FakeProvider:
    def __init__(self, source: ImagerySource = ImagerySource.SENTINEL_CLOUDLESS) -> None:
        self.source = source
        self.calls: list[dict] = []

    async def fetch_area(self, latitude, longitude, *, size_px, date_from=None, date_to=None):
        self.calls.append(
            {"size_px": size_px, "date_from": date_from, "date_to": date_to, "lat": latitude}
        )
        return scene_of(size_px, self.source)


class FakeModel:
    """Says "yes" when most of a window is water-coloured, "no" otherwise."""

    def __init__(self, fusions=(FusionType.ELEMENTWISE, FusionType.CROSS_ATTENTION)) -> None:
        self._fusions = frozenset(fusions)
        self.asked: list[str] = []
        self.fail_for: FusionType | None = None

    def available_fusions(self):
        return self._fusions

    async def predict(self, image_data, question, *, fusion, top_k=3):
        self.asked.append(question)
        if fusion is self.fail_for:
            raise RuntimeError("model blew up")
        pixels = Image.open(io.BytesIO(image_data)).convert("RGB")
        water = sum(1 for p in pixels.getdata() if p == WATER) / (pixels.width * pixels.height)
        answer = "yes" if water > 0.5 else "no"
        return VqaAnswer(
            fusion=fusion,
            answer=answer,
            confidence=0.9,
            alternatives=(AnswerAlternative(answer=answer, confidence=0.9),),
        )

    async def explain(self, image_data, question):
        raise NotImplementedError


def service(*providers: FakeProvider, model=None, max_size_km=20.0, max_windows=256):
    return AreaAnalysisService(
        model=model or FakeModel(),
        providers={p.source: p for p in providers},
        max_size_km=max_size_km,
        max_windows=max_windows,
    )


# --- The use case ------------------------------------------------------------------------


async def test_water_on_the_right_is_found_on_the_right_and_not_on_the_left() -> None:
    result = await service(FakeProvider()).analyse(
        question="Is there a water area?", latitude=LAT, longitude=LON, size_km=5.12
    )

    elementwise = next(r for r in result.results if r.fusion is FusionType.ELEMENTWISE)
    by_cell = {(c.row, c.col): c for c in elementwise.cells}
    # 512 px, stride 128: a 4 x 4 grid of cells; the picture is split at pixel 256.
    assert result.window_count == 9
    assert {c.answer for (_, col), c in by_cell.items() if col == 0} == {"no"}
    assert {c.answer for (_, col), c in by_cell.items() if col == 3} == {"yes"}
    # The window starting at pixel 128 straddles the split, so the cells beside it are split votes.
    assert any(c.agreement < 1.0 for c in by_cell.values())
    shares = dict(elementwise.share_by_answer)
    assert shares["no"] >= 0.5
    assert shares["yes"] >= 0.25
    assert elementwise.failed_windows == 0


async def test_every_loaded_model_is_asked_and_reported_separately() -> None:
    result = await service(FakeProvider()).analyse(
        question="Is there a water area?", latitude=LAT, longitude=LON, size_km=2.56
    )

    assert [r.fusion for r in result.results] == [
        FusionType.CROSS_ATTENTION,
        FusionType.ELEMENTWISE,
    ]
    assert result.window_count == 1


async def test_cells_are_placed_on_the_map_inside_the_scene_and_tile_it() -> None:
    result = await service(FakeProvider()).analyse(
        question="Is there a water area?", latitude=LAT, longitude=LON, size_km=5.12
    )

    west, south, east, north = result.bounds_wgs84
    cells = result.results[0].cells
    assert len(cells) == 16
    assert min(c.bounds_wgs84[0] for c in cells) == pytest.approx(west)
    assert max(c.bounds_wgs84[2] for c in cells) == pytest.approx(east)
    assert min(c.bounds_wgs84[1] for c in cells) == pytest.approx(south)
    assert max(c.bounds_wgs84[3] for c in cells) == pytest.approx(north)
    assert all(
        c.bounds_wgs84[0] < c.bounds_wgs84[2] and c.bounds_wgs84[1] < c.bounds_wgs84[3]
        for c in cells
    )


async def test_the_scene_is_centred_on_the_point_and_5_km_is_about_5_km_wide() -> None:
    result = await service(FakeProvider()).analyse(
        question="Is there a water area?", latitude=LAT, longitude=LON, size_km=5.12
    )

    west, south, east, north = result.bounds_wgs84
    assert (west + east) / 2 == pytest.approx(LON, abs=1e-6)
    assert (south + north) / 2 == pytest.approx(LAT, abs=1e-3)
    width_m = math.radians(east - west) * 6_378_137.0 * math.cos(math.radians(LAT))
    assert width_m == pytest.approx(5120, rel=0.01)


async def test_a_loosely_worded_question_is_reworded_and_the_model_is_asked_the_template() -> None:
    model = FakeModel()
    result = await service(FakeProvider(), model=model).analyse(
        question="any water here?", latitude=LAT, longitude=LON, size_km=2.56
    )

    assert result.question == "any water here?"
    assert result.question_used == "Is there a water area?"
    assert result.note is not None
    assert set(model.asked) == {"Is there a water area?"}


async def test_a_question_the_models_cannot_answer_is_refused_before_any_imagery_is_fetched() -> (
    None
):
    provider = FakeProvider()

    with pytest.raises(UnsupportedQuestionError):
        await service(provider).analyse(
            question="What colour is the sky?", latitude=LAT, longitude=LON, size_km=5
        )

    assert provider.calls == []


@pytest.mark.parametrize("size_km", [0.5, MIN_SIZE_KM - 0.01, 20.5])
async def test_areas_outside_the_allowed_sizes_are_refused(size_km: float) -> None:
    provider = FakeProvider()

    with pytest.raises(ValidationError):
        await service(provider).analyse(
            question="Is there a water area?", latitude=LAT, longitude=LON, size_km=size_km
        )

    assert provider.calls == []


async def test_an_area_needing_too_many_windows_is_refused() -> None:
    with pytest.raises(ValidationError, match="windows"):
        await service(FakeProvider(), max_windows=8).analyse(
            question="Is there a water area?", latitude=LAT, longitude=LON, size_km=5.12
        )


async def test_dates_in_the_wrong_order_are_refused() -> None:
    with pytest.raises(ValidationError, match="start date"):
        await service(FakeProvider()).analyse(
            question="Is there a water area?",
            latitude=LAT,
            longitude=LON,
            size_km=5,
            date_from=date(2026, 10, 1),
            date_to=date(2026, 9, 1),
        )


async def test_earth_engine_is_preferred_when_available_and_dates_are_passed_through() -> None:
    eox = FakeProvider(ImagerySource.SENTINEL_CLOUDLESS)
    earth_engine = FakeProvider(ImagerySource.EARTH_ENGINE)
    the_service = service(eox, earth_engine)

    result = await the_service.analyse(
        question="Is there a water area?",
        latitude=LAT,
        longitude=LON,
        size_km=2.56,
        date_from=date(2026, 8, 1),
        date_to=date(2026, 10, 1),
    )

    assert the_service.available_sources() == [
        ImagerySource.EARTH_ENGINE,
        ImagerySource.SENTINEL_CLOUDLESS,
    ]
    assert result.source is ImagerySource.EARTH_ENGINE
    assert earth_engine.calls[0]["date_from"] == date(2026, 8, 1)
    assert eox.calls == []


async def test_a_source_can_be_chosen_explicitly() -> None:
    eox = FakeProvider(ImagerySource.SENTINEL_CLOUDLESS)
    earth_engine = FakeProvider(ImagerySource.EARTH_ENGINE)

    result = await service(eox, earth_engine).analyse(
        question="Is there a water area?",
        latitude=LAT,
        longitude=LON,
        size_km=2.56,
        source=ImagerySource.SENTINEL_CLOUDLESS,
    )

    assert result.source is ImagerySource.SENTINEL_CLOUDLESS
    assert earth_engine.calls == []


async def test_asking_for_a_source_that_is_not_set_up_is_reported() -> None:
    with pytest.raises(ImageryUnavailableError):
        await service(FakeProvider()).analyse(
            question="Is there a water area?",
            latitude=LAT,
            longitude=LON,
            size_km=2.56,
            source=ImagerySource.EARTH_ENGINE,
        )


async def test_no_sources_at_all_is_reported() -> None:
    with pytest.raises(ImageryUnavailableError):
        await service().analyse(
            question="Is there a water area?", latitude=LAT, longitude=LON, size_km=2.56
        )


async def test_one_model_failing_still_returns_the_other_and_counts_the_failures() -> None:
    model = FakeModel()
    model.fail_for = FusionType.CROSS_ATTENTION

    result = await service(FakeProvider(), model=model).analyse(
        question="Is there a water area?", latitude=LAT, longitude=LON, size_km=5.12
    )

    failed = next(r for r in result.results if r.fusion is FusionType.CROSS_ATTENTION)
    working = next(r for r in result.results if r.fusion is FusionType.ELEMENTWISE)
    assert failed.failed_windows == 9
    assert failed.cells == ()
    assert len(working.cells) == 16


async def test_every_model_failing_is_an_error_not_an_empty_map() -> None:
    class Broken(FakeModel):
        async def predict(self, image_data, question, *, fusion, top_k=3):
            raise RuntimeError("down")

    with pytest.raises(ModelUnavailableError):
        await service(FakeProvider(), model=Broken()).analyse(
            question="Is there a water area?", latitude=LAT, longitude=LON, size_km=2.56
        )


async def test_no_loaded_model_is_reported() -> None:
    with pytest.raises(ModelUnavailableError):
        await service(FakeProvider(), model=FakeModel(fusions=())).analyse(
            question="Is there a water area?", latitude=LAT, longitude=LON, size_km=2.56
        )


# --- The scene ---------------------------------------------------------------------------


def test_a_window_is_cut_out_at_the_right_place_and_size() -> None:
    scene = scene_of(512)

    left = Image.open(io.BytesIO(scene.crop_png(0, 0, 256)))
    right = Image.open(io.BytesIO(scene.crop_png(256, 0, 256)))

    assert left.size == (256, 256)
    assert left.getpixel((10, 10)) == LAND
    assert right.getpixel((10, 10)) == WATER


# --- The EOX mosaic as an area source -----------------------------------------------------


def _jpeg_tile(colour: tuple[int, int, int]) -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", (256, 256), colour).save(buffer, format="JPEG")
    return buffer.getvalue()


async def test_the_eox_mosaic_builds_a_bigger_scene_on_the_same_grid() -> None:
    requested: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requested.append(str(request.url))
        return httpx.Response(200, content=_jpeg_tile((20, 120, 40)))

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    provider = EoxSentinel2Provider(
        tile_url_template="https://tiles.example.test/{z}/{y}/{x}.jpg",
        user_agent="test",
        client=client,
    )

    scene = await provider.fetch_area(52.9554, 5.7743, size_px=512)

    assert (scene.width_px, scene.height_px) == (512, 512)
    assert scene.source is ImagerySource.SENTINEL_CLOUDLESS
    assert scene.captured is not None and "2020" in scene.captured
    west, _south, east, _north = scene.bounds_wgs84
    assert (west + east) / 2 == pytest.approx(5.7743, abs=1e-6)
    assert Image.open(io.BytesIO(scene.crop_png(100, 100, 256))).size == (256, 256)
    assert 4 <= len(requested) <= 25  # a 5 km area is a handful of tiles, not hundreds


# --- Google Earth Engine ------------------------------------------------------------------


class _Anything:
    """Stands in for any Earth Engine image or number: every method returns it again."""

    def __getattr__(self, _name):
        return lambda *args, **kwargs: self


class _Value:
    def __init__(self, value) -> None:
        self._value = value

    def getInfo(self):  # noqa: N802 - Earth Engine's own name
        return self._value


class _Collection:
    def __init__(self, ee: FakeEarthEngine) -> None:
        self._ee = ee

    def filterBounds(self, region):  # noqa: N802
        self._ee.calls.append(("filterBounds", region))
        return self

    def filterDate(self, start, end):  # noqa: N802
        self._ee.calls.append(("filterDate", start, end))
        return self

    def filter(self, condition):
        self._ee.calls.append(("filter", condition))
        return self

    def size(self):
        return _Value(self._ee.count)

    def aggregate_max(self, prop):
        self._ee.calls.append(("aggregate_max", prop))
        return _Value(self._ee.newest_ms)

    def map(self, function):
        function(_Anything())  # the cloud-mask function must at least run
        self._ee.calls.append(("map",))
        return self

    def sort(self, prop, ascending=True):
        self._ee.calls.append(("sort", prop, ascending))
        return self

    def mosaic(self):
        self._ee.calls.append(("mosaic",))
        return _Anything()


class FakeEarthEngine:
    """Just enough of the ``ee`` module to run the provider and record what it asked for."""

    def __init__(self, *, count=3, png=None, newest_ms=1_760_000_000_000) -> None:
        self.calls: list[tuple] = []
        self.count = count
        self.newest_ms = newest_ms
        self.initialised_with: str | None = None
        self.init_error: Exception | None = None
        self._png = png
        self.requests: list[dict] = []
        self.Geometry = self
        self.Filter = self
        self.data = self

    def Initialize(self, project=None):  # noqa: N802
        if self.init_error is not None:
            raise self.init_error
        self.initialised_with = project

    def Rectangle(self, coords, proj=None, geodesic=None):  # noqa: N802
        return ("rectangle", coords)

    def lt(self, name, value):
        return ("lt", name, value)

    def ImageCollection(self, name):  # noqa: N802
        self.calls.append(("ImageCollection", name))
        return _Collection(self)

    def computePixels(self, request):  # noqa: N802
        self.requests.append(request)
        return self._png if self._png is not None else _png(256, alpha=255)


def _png(size: int, *, alpha: int, empty_columns: int = 0) -> bytes:
    """An RGBA picture; the first ``empty_columns`` columns are fully transparent."""
    picture = Image.new("RGBA", (size, size), (90, 130, 70, alpha))
    if empty_columns:
        picture.paste(Image.new("RGBA", (empty_columns, size), (0, 0, 0, 0)), (0, 0))
    buffer = io.BytesIO()
    picture.save(buffer, format="PNG")
    return buffer.getvalue()


def test_the_earth_engine_grid_is_centred_and_in_web_mercator() -> None:
    grid = build_grid(52.0, 5.0, 512, 10.0)

    transform = grid["affineTransform"]
    pixel_m = 10.0 / math.cos(math.radians(52.0))
    assert grid["crsCode"] == "EPSG:3857"
    assert grid["dimensions"] == {"width": 512, "height": 512}
    assert transform["scaleX"] == pytest.approx(pixel_m)
    assert transform["scaleY"] == pytest.approx(-pixel_m)
    centre_x = math.radians(5.0) * 6_378_137.0
    assert transform["translateX"] + 256 * pixel_m == pytest.approx(centre_x)


async def test_earth_engine_asks_for_the_newest_clear_sentinel_2_pixels_in_the_period() -> None:
    fake = FakeEarthEngine(png=_png(512, alpha=255))
    provider = EarthEngineProvider(project="my-project", ee_module=fake)

    scene = await provider.fetch_area(
        LAT, LON, size_px=512, date_from=date(2026, 8, 1), date_to=date(2026, 10, 1)
    )

    assert ("ImageCollection", "COPERNICUS/S2_HARMONIZED") in fake.calls
    assert ("filterDate", "2026-08-01", "2026-10-02") in fake.calls  # end date is inclusive
    assert ("sort", "system:time_start", True) in fake.calls  # oldest first -> newest on top
    assert ("mosaic",) in fake.calls
    request = fake.requests[0]
    assert request["fileFormat"] == "PNG"
    assert request["grid"]["dimensions"] == {"width": 512, "height": 512}
    assert (scene.width_px, scene.height_px) == (512, 512)
    assert scene.source is ImagerySource.EARTH_ENGINE
    assert scene.captured is not None
    assert "2026-08-01" in scene.captured
    assert "newest scene in the picture 2025-" in scene.captured  # from newest_ms


async def test_earth_engine_defaults_to_the_recent_past_when_no_dates_are_given() -> None:
    fake = FakeEarthEngine(png=_png(256, alpha=255))
    provider = EarthEngineProvider(project="p", default_window_days=30, ee_module=fake)

    await provider.fetch_area(LAT, LON, size_px=256)

    start = next(c for c in fake.calls if c[0] == "filterDate")
    first, last = date.fromisoformat(start[1]), date.fromisoformat(start[2])
    assert (last - first).days == 31  # 30 days, plus the inclusive end date


async def test_a_period_with_no_scenes_says_so() -> None:
    provider = EarthEngineProvider(project="p", ee_module=FakeEarthEngine(count=0))

    with pytest.raises(ImageryUnavailableError, match="no clear Sentinel-2 scene"):
        await provider.fetch_area(LAT, LON, size_px=256)


async def test_a_picture_that_is_mostly_cloud_is_refused_rather_than_analysed() -> None:
    fake = FakeEarthEngine(png=_png(256, alpha=255, empty_columns=200))
    provider = EarthEngineProvider(project="p", ee_module=fake)

    with pytest.raises(ImageryUnavailableError, match="cloud"):
        await provider.fetch_area(LAT, LON, size_px=256)


async def test_a_little_cloud_is_painted_grey_and_mentioned() -> None:
    fake = FakeEarthEngine(png=_png(256, alpha=255, empty_columns=40))
    provider = EarthEngineProvider(project="p", ee_module=fake)

    scene = await provider.fetch_area(LAT, LON, size_px=256)

    assert scene.captured is not None and "grey" in scene.captured
    assert Image.open(io.BytesIO(scene.crop_png(0, 0, 256))).getpixel((5, 5)) == (128, 128, 128)


async def test_a_machine_that_is_not_signed_in_is_told_how_to_sign_in() -> None:
    fake = FakeEarthEngine()
    fake.init_error = RuntimeError("no credentials")
    provider = EarthEngineProvider(project="my-project", ee_module=None)
    provider._ee = fake  # a signed-out client, without the real library

    expected = "earthengine --project my-project authenticate"
    with pytest.raises(ImageryUnavailableError, match=expected):
        await provider.fetch_area(LAT, LON, size_px=256)


async def test_earth_engine_is_initialised_once_with_the_configured_project() -> None:
    fake = FakeEarthEngine(png=_png(256, alpha=255))
    provider = EarthEngineProvider(project="my-project", ee_module=None)
    provider._ee = fake

    await provider.fetch_area(LAT, LON, size_px=256)
    await provider.fetch_area(LAT, LON, size_px=256)

    assert fake.initialised_with == "my-project"


async def test_an_earth_engine_failure_is_reported_not_swallowed() -> None:
    class Failing(FakeEarthEngine):
        def computePixels(self, request):  # noqa: N802
            raise RuntimeError("Too many pixels requested")

    provider = EarthEngineProvider(project="p", ee_module=Failing())

    with pytest.raises(ImageryUnavailableError, match="couldn't build"):
        await provider.fetch_area(LAT, LON, size_px=256)


# --- The endpoint ------------------------------------------------------------------------


async def test_the_endpoint_returns_shares_and_cells_per_model(
    app, client: AsyncClient, researcher: User
) -> None:
    app.state.vqa_model = FakeModel()
    app.state.area_providers = {ImagerySource.SENTINEL_CLOUDLESS: FakeProvider()}
    headers = await auth_headers(client, researcher.email, USER_PASSWORD)

    response = await client.post(
        "/area/analyse",
        headers=headers,
        json={
            "question": "Is there a water area?",
            "latitude": LAT,
            "longitude": LON,
            "size_km": 5.12,
        },
    )

    assert response.status_code == 200, response.text
    body = response.json()
    assert body["window_count"] == 9
    assert body["source"] == "sentinel_cloudless"
    assert len(body["bounds_wgs84"]) == 4
    first = body["results"][0]
    assert {s["answer"] for s in first["shares"]} == {"yes", "no"}
    assert len(first["cells"]) == 16
    assert set(first["cells"][0]) >= {"row", "col", "bounds_wgs84", "answer", "agreement"}


async def test_the_endpoint_refuses_a_question_the_models_were_not_trained_on(
    app, client: AsyncClient, researcher: User
) -> None:
    app.state.vqa_model = FakeModel()
    app.state.area_providers = {ImagerySource.SENTINEL_CLOUDLESS: FakeProvider()}
    headers = await auth_headers(client, researcher.email, USER_PASSWORD)

    response = await client.post(
        "/area/analyse",
        headers=headers,
        json={
            "question": "What colour is the sky?",
            "latitude": LAT,
            "longitude": LON,
            "size_km": 5,
        },
    )

    assert response.status_code == 422
    assert response.json()["code"] == "unsupported_question"


async def test_the_endpoint_refuses_an_oversized_area(
    app, client: AsyncClient, researcher: User
) -> None:
    app.state.vqa_model = FakeModel()
    app.state.area_providers = {ImagerySource.SENTINEL_CLOUDLESS: FakeProvider()}
    headers = await auth_headers(client, researcher.email, USER_PASSWORD)

    response = await client.post(
        "/area/analyse",
        headers=headers,
        json={
            "question": "Is there a water area?",
            "latitude": LAT,
            "longitude": LON,
            "size_km": 60,
        },
    )

    assert response.status_code == 422


async def test_capabilities_say_which_sources_and_sizes_this_server_offers(
    app, client: AsyncClient, researcher: User
) -> None:
    app.state.vqa_model = FakeModel()
    app.state.area_providers = {
        ImagerySource.SENTINEL_CLOUDLESS: FakeProvider(),
        ImagerySource.EARTH_ENGINE: FakeProvider(ImagerySource.EARTH_ENGINE),
    }
    headers = await auth_headers(client, researcher.email, USER_PASSWORD)

    response = await client.get("/area/capabilities", headers=headers)

    assert response.status_code == 200
    body = response.json()
    assert body["sources"] == ["earth_engine", "sentinel_cloudless"]
    assert body["window_km"] == pytest.approx(2.56)
    assert body["min_size_km"] == pytest.approx(2.56)
    assert body["fusions"] == ["cross_attention", "elementwise"]


async def test_the_area_endpoints_need_a_signed_in_user(app, client: AsyncClient) -> None:
    assert (await client.get("/area/capabilities")).status_code == 401
    response = await client.post(
        "/area/analyse",
        json={
            "question": "Is there a water area?",
            "latitude": LAT,
            "longitude": LON,
            "size_km": 5,
        },
    )
    assert response.status_code == 401
