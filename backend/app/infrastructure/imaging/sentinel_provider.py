"""A satellite scene for any point on Earth, shaped like the data the models were trained on.

The map shows live imagery; to ask the models about "this place" the app needs a scene the
models can fairly answer questions about. RSVQA-LR scenes are 256 x 256 px at about 10 m per
pixel, from Sentinel-2. A screenshot of a very high-resolution basemap is the wrong input —
the robustness study in ``ml/README.md`` shows the models are fragile to a change of scale —
so this fetches the Sentinel-2 cloudless mosaic (EOX IT Services GmbH) and cuts a 256 x 256
scene covering 2.56 km on the ground, whatever the zoom or latitude.

Imagery: "Sentinel-2 cloudless - https://s2maps.eu by EOX IT Services GmbH (Contains modified
Copernicus Sentinel data)". Free for non-commercial use with attribution — the Map page shows
that attribution wherever the imagery is displayed.
"""

from __future__ import annotations

import asyncio
import io
import math

import httpx
import numpy as np
from PIL import Image
from rasterio.io import MemoryFile
from rasterio.transform import from_bounds

from app.core.logging import get_logger
from app.domain.errors import ImageryUnavailableError

logger = get_logger(__name__)

_TILE_PX = 256
_EARTH_CIRCUMFERENCE_M = 40_075_016.686
_MERCATOR_RADIUS_M = 6_378_137.0

# Beyond this the Web Mercator tile grid and the Sentinel-2 mosaic both thin out.
MAX_LATITUDE = 80.0


def ground_resolution_m(latitude: float, zoom: int) -> float:
    """Metres of ground per pixel of a 256 px Web Mercator tile at ``zoom`` and ``latitude``."""
    return float(_EARTH_CIRCUMFERENCE_M * math.cos(math.radians(latitude)) / (_TILE_PX * 2**zoom))


def choose_zoom(latitude: float, target_m: float, max_zoom: int) -> int:
    """The coarsest zoom whose pixels are no larger than ``target_m`` — so the scene is built
    by shrinking a slightly finer image down to the target, never by stretching a coarser one
    up. Capped at ``max_zoom``, the finest level the imagery service serves."""
    wanted = math.ceil(
        math.log2(_EARTH_CIRCUMFERENCE_M * math.cos(math.radians(latitude)) / (_TILE_PX * target_m))
    )
    return max(0, min(max_zoom, wanted))


def mercator_pixel(latitude: float, longitude: float, zoom: int) -> tuple[float, float]:
    """Global pixel coordinates (origin top-left of the world) of a point at ``zoom``."""
    world = _TILE_PX * 2**zoom
    x = (longitude + 180.0) / 360.0 * world
    y = (1.0 - math.asinh(math.tan(math.radians(latitude))) / math.pi) / 2.0 * world
    return x, y


def scene_bounds_wgs84(
    latitude: float, longitude: float, size_px: int = 256, ground_m: float = 10.0
) -> tuple[float, float, float, float]:
    """(west, south, east, north) of the scene ``fetch_scene`` cuts around a point — the same
    square, in degrees. Lets the map-data cache be filled for a place before it is ever captured."""
    half = size_px * ground_m / math.cos(math.radians(latitude)) / 2.0  # Web Mercator metres
    centre_x = math.radians(longitude) * _MERCATOR_RADIUS_M
    centre_y = math.log(math.tan(math.pi / 4 + math.radians(latitude) / 2)) * _MERCATOR_RADIUS_M

    def to_latitude(y: float) -> float:
        return math.degrees(2 * math.atan(math.exp(y / _MERCATOR_RADIUS_M)) - math.pi / 2)

    return (
        math.degrees((centre_x - half) / _MERCATOR_RADIUS_M),
        to_latitude(centre_y - half),
        math.degrees((centre_x + half) / _MERCATOR_RADIUS_M),
        to_latitude(centre_y + half),
    )


def _to_geotiff(
    chip: Image.Image, latitude: float, longitude: float, size_px: int, ground_m: float
) -> bytes:
    """EPSG:3857 GeoTIFF whose extent is the scene's real footprint. Web Mercator stretches
    distances by 1/cos(latitude), so the scene's width in Mercator metres is larger than its
    width on the ground — the same reason RSVQA-LR's own scenes report ~16 m per pixel at
    52 degrees north."""
    half = size_px * ground_m / math.cos(math.radians(latitude)) / 2.0
    centre_x = math.radians(longitude) * _MERCATOR_RADIUS_M
    centre_y = math.log(math.tan(math.pi / 4 + math.radians(latitude) / 2)) * _MERCATOR_RADIUS_M
    transform = from_bounds(
        centre_x - half, centre_y - half, centre_x + half, centre_y + half, size_px, size_px
    )
    bands = np.asarray(chip.convert("RGB"), dtype=np.uint8).transpose(2, 0, 1)
    with MemoryFile() as memory:
        with memory.open(
            driver="GTiff",
            width=size_px,
            height=size_px,
            count=3,
            dtype="uint8",
            crs="EPSG:3857",
            transform=transform,
        ) as dataset:
            dataset.write(bands)
        data: bytes = memory.read()
    return data


class EoxSentinel2Provider:
    """Implements :class:`~app.domain.interfaces.ImageryProvider`.

    ``client`` is injectable so tests can serve synthetic tiles without a network.
    """

    def __init__(
        self,
        *,
        tile_url_template: str,
        user_agent: str,
        size_px: int = 256,
        ground_resolution_target_m: float = 10.0,
        max_zoom: int = 14,
        timeout_seconds: float = 15.0,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        self._tile_url = tile_url_template
        self._size_px = size_px
        self._target_m = ground_resolution_target_m
        self._max_zoom = max_zoom
        self._client = client or httpx.AsyncClient(
            timeout=timeout_seconds, headers={"User-Agent": user_agent}
        )

    async def fetch_scene(self, latitude: float, longitude: float) -> bytes:
        if abs(latitude) > MAX_LATITUDE or not -180.0 <= longitude <= 180.0:
            raise ImageryUnavailableError(
                f"No imagery for ({latitude}, {longitude}): the imagery covers latitudes up to "
                f"{MAX_LATITUDE:.0f} degrees north or south.",
                latitude=latitude,
                longitude=longitude,
            )

        zoom = choose_zoom(latitude, self._target_m, self._max_zoom)
        # How many pixels of this zoom make up the scene's ground footprint.
        window = self._size_px * self._target_m / ground_resolution_m(latitude, zoom)
        centre_x, centre_y = mercator_pixel(latitude, longitude, zoom)
        left, top = centre_x - window / 2, centre_y - window / 2

        tiles_per_side = 2**zoom
        first_x, last_x = math.floor(left / _TILE_PX), math.floor((left + window) / _TILE_PX)
        first_y, last_y = math.floor(top / _TILE_PX), math.floor((top + window) / _TILE_PX)
        if first_y < 0 or last_y >= tiles_per_side:
            raise ImageryUnavailableError(
                "That location is too close to the edge of the map.",
                latitude=latitude,
                longitude=longitude,
            )

        wanted = [(x, y) for y in range(first_y, last_y + 1) for x in range(first_x, last_x + 1)]
        images = await asyncio.gather(
            *(self._fetch_tile(zoom, x % tiles_per_side, y) for x, y in wanted)
        )

        mosaic = Image.new(
            "RGB", ((last_x - first_x + 1) * _TILE_PX, (last_y - first_y + 1) * _TILE_PX)
        )
        for (x, y), tile in zip(wanted, images, strict=True):
            mosaic.paste(tile, ((x - first_x) * _TILE_PX, (y - first_y) * _TILE_PX))

        box = (
            left - first_x * _TILE_PX,
            top - first_y * _TILE_PX,
            left - first_x * _TILE_PX + window,
            top - first_y * _TILE_PX + window,
        )
        chip = mosaic.resize((self._size_px, self._size_px), Image.Resampling.LANCZOS, box=box)

        logger.info(
            "imagery_scene_built",
            latitude=latitude,
            longitude=longitude,
            zoom=zoom,
            tiles=len(wanted),
        )
        return await asyncio.to_thread(
            _to_geotiff, chip, latitude, longitude, self._size_px, self._target_m
        )

    async def _fetch_tile(self, zoom: int, x: int, y: int) -> Image.Image:
        url = self._tile_url.format(z=zoom, x=x, y=y)
        try:
            response = await self._client.get(url)
            response.raise_for_status()
            return Image.open(io.BytesIO(response.content)).convert("RGB")
        except (httpx.HTTPError, OSError) as exc:
            logger.warning("imagery_tile_failed", url=url, error=str(exc))
            raise ImageryUnavailableError(
                "The imagery service didn't return a tile for that location.", url=url
            ) from exc

    async def aclose(self) -> None:
        await self._client.aclose()
