"""A satellite picture of an area, held in memory and handed out one window at a time.

Implements :class:`~app.domain.interfaces.AreaScenePort`. Both imagery sources (the EOX Sentinel-2
mosaic and Google Earth Engine) build their picture on the same grid: Web Mercator, with a ground
size per pixel of 10 m. Web Mercator stretches distances by 1/cos(latitude), so a pixel that covers
10 m of ground is ``10 / cos(latitude)`` Mercator metres wide; that is the same reason RSVQA-LR's
own scenes report about 16 m per pixel at 52 degrees north. Keeping one class for both sources
means the cell-to-map conversion is written, and tested, once.
"""

from __future__ import annotations

import io
import math

from PIL import Image

from app.domain.enums import ImagerySource

_MERCATOR_RADIUS_M = 6_378_137.0


def _latitude_of(mercator_y: float) -> float:
    return math.degrees(2 * math.atan(math.exp(mercator_y / _MERCATOR_RADIUS_M)) - math.pi / 2)


class MercatorAreaScene:
    """An RGB picture centred on a point, ``image.width`` x ``image.height`` px at
    ``ground_m`` metres of ground per pixel."""

    def __init__(
        self,
        image: Image.Image,
        *,
        latitude: float,
        longitude: float,
        ground_m: float,
        source: ImagerySource,
        captured: str | None,
    ) -> None:
        self._image = image.convert("RGB")
        self.width_px = self._image.width
        self.height_px = self._image.height
        self.source = source
        self.captured = captured
        # Mercator metres per pixel at this latitude, and the picture's top-left corner.
        self._pixel_m = ground_m / math.cos(math.radians(latitude))
        centre_x = math.radians(longitude) * _MERCATOR_RADIUS_M
        centre_y = math.log(math.tan(math.pi / 4 + math.radians(latitude) / 2)) * _MERCATOR_RADIUS_M
        self._left_m = centre_x - self.width_px * self._pixel_m / 2
        self._top_m = centre_y + self.height_px * self._pixel_m / 2
        self.bounds_wgs84 = self.box_wgs84(0, 0, self.width_px, self.height_px)

    def crop_png(self, x: int, y: int, size: int) -> bytes:
        buffer = io.BytesIO()
        self._image.crop((x, y, x + size, y + size)).save(buffer, format="PNG")
        return buffer.getvalue()

    def box_wgs84(
        self, left: int, top: int, right: int, bottom: int
    ) -> tuple[float, float, float, float]:
        west = math.degrees((self._left_m + left * self._pixel_m) / _MERCATOR_RADIUS_M)
        east = math.degrees((self._left_m + right * self._pixel_m) / _MERCATOR_RADIUS_M)
        north = _latitude_of(self._top_m - top * self._pixel_m)
        south = _latitude_of(self._top_m - bottom * self._pixel_m)
        return (west, south, east, north)
