"""Recent satellite imagery for any place, from Google Earth Engine.

Earth Engine is Google's satellite-data service and the route it offers to researchers: free for
non-commercial research once you register (https://code.earthengine.google.com/register) and
create a Google Cloud project. Its terms limit use to development, research and education.
This is deliberately *not* Google Maps or Google Earth's own basemap pictures, whose terms
restrict using them for machine learning.

What it fetches: Sentinel-2 top-of-atmosphere imagery (``COPERNICUS/S2_HARMONIZED``), the same
satellite the models were trained on, for the date range asked for. Cloud and cirrus pixels are
masked, and the remaining clear pixels are laid over each other oldest first, so each spot shows
its **newest clear view** in that period.

Why top-of-atmosphere, bilinear, and a 4250 stretch: the first version used surface reflectance
with nearest-neighbour sampling and the usual 3000 stretch, and the models answered about 3.5
points worse on those pictures than on the dataset's own (and said "yes" far less often). The
dataset's pictures are hazy, lower in contrast and smoother than that rendering. On 12 validation
images (no labels used) the top-of-atmosphere product's contrast and sharpness were much closer to
the dataset's (distance 0.28 against 0.60), and the stretch was set so average brightness matches.
``scripts/ee_vs_dataset.py`` reruns the accuracy comparison against the dataset's own images.

Sentinel-2 passes over a place about every five days, so "latest" means days to weeks old, never
live.

Set-up, once per machine (needs a browser sign-in as the registered Google account)::

    uv pip install earthengine-api
    earthengine --project YOUR_CLOUD_PROJECT_ID authenticate

then set ``GVG_EARTH_ENGINE_PROJECT=YOUR_CLOUD_PROJECT_ID``. The credentials stay on this
machine, in the user's profile; this code never sees or stores a password.
"""

from __future__ import annotations

import asyncio
import io
import math
from datetime import UTC, date, datetime, timedelta
from typing import Any

from PIL import Image

from app.core.logging import get_logger
from app.domain.entities import CapturedScene
from app.domain.enums import ImagerySource
from app.domain.errors import ImageryUnavailableError
from app.domain.interfaces import AreaScenePort
from app.infrastructure.imaging.area_scene import MercatorAreaScene
from app.infrastructure.imaging.sentinel_provider import to_geotiff

logger = get_logger(__name__)

# Surface reflectance (atmosphere corrected, has a scene-classification layer) and top-of-atmosphere
# (hazier, has only a coarse cloud flag). Which one looks like the training pictures is checked, not
# assumed: see scripts/ee_vs_dataset.py.
COLLECTION_SURFACE = "COPERNICUS/S2_SR_HARMONIZED"
COLLECTION_TOP_OF_ATMOSPHERE = "COPERNICUS/S2_HARMONIZED"
_MERCATOR_RADIUS_M = 6_378_137.0
# Sentinel-2 scene-classification values to drop: 1 saturated, 3 cloud shadow, 8 and 9 medium and
# high-probability cloud, 10 thin cirrus.
_UNUSABLE_SCL = (1, 3, 8, 9, 10)
# Reflectance (scaled by 10000) shown as black..white. The usual true-colour stretch is 0-3000, but
# that rendered top-of-atmosphere pictures about 30% brighter than the dataset's, so the stretch is
# 4250: chosen so the average brightness matches the dataset's on validation images (no labels
# used).
DEFAULT_REFLECTANCE_MAX = 4250
DEFAULT_RESAMPLING = "bilinear"
# Sentinel-2's QA60 flags: bit 10 opaque cloud, bit 11 cirrus.
_QA60_CLOUD_BITS = (1 << 10, 1 << 11)
# Above this share of "no clear view anywhere in the period" the picture is mostly black holes.
_MAX_EMPTY_SHARE = 0.5
# Where a pixel has no clear view the picture is transparent; those pixels are painted this grey,
# not black, which is less likely to be read as water.
_EMPTY_FILL = (128, 128, 128)


def build_grid(latitude: float, longitude: float, size_px: int, ground_m: float) -> dict[str, Any]:
    """The Earth Engine ``grid`` for a ``size_px`` square on the Web Mercator grid, centred on
    the point, with ``ground_m`` of ground per pixel (see :mod:`.area_scene` for why the
    Mercator pixel is wider than that)."""
    pixel_m = ground_m / math.cos(math.radians(latitude))
    centre_x = math.radians(longitude) * _MERCATOR_RADIUS_M
    centre_y = math.log(math.tan(math.pi / 4 + math.radians(latitude) / 2)) * _MERCATOR_RADIUS_M
    return {
        "dimensions": {"width": size_px, "height": size_px},
        "affineTransform": {
            "scaleX": pixel_m,
            "shearX": 0,
            "translateX": centre_x - size_px * pixel_m / 2,
            "shearY": 0,
            "scaleY": -pixel_m,
            "translateY": centre_y + size_px * pixel_m / 2,
        },
        "crsCode": "EPSG:3857",
    }


class EarthEngineProvider:
    """Implements :class:`~app.domain.interfaces.AreaImageryProvider`.

    ``ee_module`` is injectable so tests can run without the Earth Engine library, an account
    or a network.
    """

    def __init__(
        self,
        *,
        project: str,
        ground_resolution_m: float = 10.0,
        default_window_days: int = 90,
        max_cloudy_percent: int = 60,
        collection: str = COLLECTION_TOP_OF_ATMOSPHERE,
        reflectance_max: int = DEFAULT_REFLECTANCE_MAX,
        resampling: str | None = DEFAULT_RESAMPLING,
        ee_module: Any | None = None,
    ) -> None:
        self._project = project
        self._ground_m = ground_resolution_m
        self._default_days = default_window_days
        self._max_cloudy = max_cloudy_percent
        self._collection = collection
        self._reflectance_max = reflectance_max
        self._resampling = resampling  # None = Earth Engine's default (nearest neighbour)
        self._ee: Any | None = ee_module
        self._initialised = ee_module is not None

    async def fetch_area(
        self,
        latitude: float,
        longitude: float,
        *,
        size_px: int,
        date_from: date | None = None,
        date_to: date | None = None,
    ) -> AreaScenePort:
        end = date_to or datetime.now(UTC).date()
        start = date_from or end - timedelta(days=self._default_days)
        return await asyncio.to_thread(self._fetch_sync, latitude, longitude, size_px, start, end)

    async def fetch_scene(self, latitude: float, longitude: float) -> CapturedScene:
        """One 256 px scene (the models' input size) as a GeoTIFF, for "Analyse this place": the
        same newest-clear-view picture and rendering as the larger-area analysis."""
        scene = await self.fetch_area(latitude, longitude, size_px=256)
        data = await asyncio.to_thread(self._geotiff, scene, latitude, longitude)
        return CapturedScene(data, ImagerySource.EARTH_ENGINE)

    def _geotiff(self, scene: AreaScenePort, latitude: float, longitude: float) -> bytes:
        chip = Image.open(io.BytesIO(scene.crop_png(0, 0, 256)))
        return to_geotiff(chip, latitude, longitude, 256, self._ground_m)

    async def aclose(self) -> None:
        """Nothing to close: the Earth Engine client holds no connection of ours."""

    def client(self) -> Any:
        """The signed-in ``ee`` module, for other Earth Engine-backed sources to share (blocking
        on first use, while it signs in)."""
        return self._earth_engine()

    # --- everything below blocks (Earth Engine's client is synchronous) -----------------

    def _earth_engine(self) -> Any:
        if self._ee is None:
            try:
                import ee  # optional dependency, only needed when Earth Engine is enabled
            except ImportError as exc:
                raise ImageryUnavailableError(
                    "Google Earth Engine is switched on but its Python library isn't installed. "
                    "Run: uv pip install earthengine-api"
                ) from exc
            self._ee = ee
        if not self._initialised:
            try:
                self._ee.Initialize(project=self._project)
            except Exception as exc:
                logger.warning("earth_engine_init_failed", error=str(exc))
                raise ImageryUnavailableError(
                    "Google Earth Engine isn't signed in on this server. Run: "
                    f"earthengine --project {self._project} authenticate"
                ) from exc
            self._initialised = True
        return self._ee

    def _fetch_sync(
        self, latitude: float, longitude: float, size_px: int, start: date, end: date
    ) -> AreaScenePort:
        ee = self._earth_engine()
        grid = build_grid(latitude, longitude, size_px, self._ground_m)
        scene_for_bounds = MercatorAreaScene(
            Image.new("RGB", (size_px, size_px)),
            latitude=latitude,
            longitude=longitude,
            ground_m=self._ground_m,
            source=ImagerySource.EARTH_ENGINE,
            captured=None,
        )
        west, south, east, north = scene_for_bounds.bounds_wgs84

        try:
            region = ee.Geometry.Rectangle([west, south, east, north], "EPSG:4326", False)
            collection = (
                ee.ImageCollection(self._collection)
                .filterBounds(region)
                .filterDate(start.isoformat(), (end + timedelta(days=1)).isoformat())
                .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", self._max_cloudy))
            )
            if int(collection.size().getInfo()) == 0:
                raise ImageryUnavailableError(
                    f"Google Earth Engine has no clear Sentinel-2 scene of this place between "
                    f"{start} and {end}. Try a longer date range.",
                    latitude=latitude,
                    longitude=longitude,
                )
            newest_ms = collection.aggregate_max("system:time_start").getInfo()

            def keep_clear(image: Any) -> Any:
                if self._collection == COLLECTION_SURFACE:
                    scene_class = image.select("SCL")
                    usable = scene_class.neq(_UNUSABLE_SCL[0])
                    for value in _UNUSABLE_SCL[1:]:
                        usable = usable.And(scene_class.neq(value))
                else:
                    flags = image.select("QA60")
                    usable = flags.bitwiseAnd(_QA60_CLOUD_BITS[0]).eq(0)
                    usable = usable.And(flags.bitwiseAnd(_QA60_CLOUD_BITS[1]).eq(0))
                image = image.updateMask(usable)
                return image.resample(self._resampling) if self._resampling else image

            # Oldest first, so on top of the pile the newest clear pixel wins at every spot.
            picture = (
                collection.map(keep_clear)
                .sort("system:time_start")
                .mosaic()
                .select(["B4", "B3", "B2"])
                .visualize(min=0, max=self._reflectance_max)
            )
            png: bytes = ee.data.computePixels(
                {"expression": picture, "fileFormat": "PNG", "grid": grid}
            )
        except ImageryUnavailableError:
            raise
        except Exception as exc:
            logger.warning("earth_engine_fetch_failed", error=str(exc))
            raise ImageryUnavailableError(
                "Google Earth Engine couldn't build that picture. If the area is large, try a "
                "smaller one; otherwise try again in a moment.",
                latitude=latitude,
                longitude=longitude,
            ) from exc

        image, empty_share = _paint_empty(png)
        if empty_share > _MAX_EMPTY_SHARE:
            raise ImageryUnavailableError(
                f"Between {start} and {end} more than half of this area was cloud-covered in "
                "every Sentinel-2 pass. Try a longer date range.",
                latitude=latitude,
                longitude=longitude,
            )

        newest = datetime.fromtimestamp(float(newest_ms) / 1000.0, UTC).date()
        note = f"Sentinel-2 from {start} to {end}; newest scene in the picture {newest}"
        if empty_share > 0.02:
            note += f"; {empty_share:.0%} had no clear view in that period and is shown grey"
        logger.info(
            "earth_engine_area_built",
            latitude=latitude,
            longitude=longitude,
            size_px=size_px,
            newest=str(newest),
            empty_share=round(empty_share, 3),
        )
        return MercatorAreaScene(
            image,
            latitude=latitude,
            longitude=longitude,
            ground_m=self._ground_m,
            source=ImagerySource.EARTH_ENGINE,
            captured=note,
        )


def _paint_empty(png: bytes) -> tuple[Image.Image, float]:
    """Decode Earth Engine's PNG. Pixels with no clear view come back transparent; paint those
    grey and report what share of the picture they were."""
    decoded = Image.open(io.BytesIO(png)).convert("RGBA")
    alpha = decoded.getchannel("A")
    histogram = alpha.histogram()
    total = decoded.width * decoded.height
    empty_share = histogram[0] / total if total else 0.0
    background = Image.new("RGBA", decoded.size, (*_EMPTY_FILL, 255))
    background.alpha_composite(decoded)
    return background.convert("RGB"), empty_share
