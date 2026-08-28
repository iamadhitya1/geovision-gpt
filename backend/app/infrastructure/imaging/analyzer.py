"""Raster inspection: content sniffing, metadata extraction, thumbnailing.

This is the one module in the project allowed to import ``rasterio`` and ``PIL`` directly
— every other layer reaches this functionality through the
:class:`~app.domain.interfaces.ImageAnalyzer` port, exactly the same discipline Module 1
applied to ``argon2``/``pyjwt`` behind their own ports.

Every public method offloads its actual work to a worker thread via ``asyncio.to_thread``:
rasterio, PIL and numpy are all synchronous, CPU-bound libraries, and calling them directly
from an async def would block the event loop for the duration of every upload.
"""

from __future__ import annotations

import asyncio
import io
import math
from typing import Any

import numpy as np
import rasterio
from PIL import Image, UnidentifiedImageError
from rasterio.errors import RasterioIOError
from rasterio.io import MemoryFile
from rasterio.warp import transform_bounds

from app.core.logging import get_logger
from app.domain.entities import ImageMetadata
from app.domain.enums import IMAGE_FORMAT_EXTENSIONS, ImageFormat
from app.domain.errors import CorruptImageError, UnsupportedFileTypeError
from app.domain.interfaces import ImageAnalyzer

logger = get_logger(__name__)

# Pixels at or above this brightness in every one of R/G/B are counted as "cloud-like" by
# the heuristic below. 200/255 is a common quick-and-dirty threshold in EO tooling for
# "bright enough to plausibly be cloud rather than ground" — it is not a validated cloud
# mask and is documented as such everywhere it's surfaced to a caller.
_BRIGHT_PIXEL_THRESHOLD = 200
_CLOUD_HEURISTIC_SAMPLE_SIZE = 256  # downsample to at most this many px/side before scoring

# Rough metres-per-degree of latitude; longitude is then scaled by cos(latitude). Accurate
# to within ~0.5% away from the poles, which is more than sufficient for an "approximate
# GSD" figure — this is not used for any measurement the SGV verification loop relies on.
_METRES_PER_DEGREE_LATITUDE = 111_320.0


def _extension_to_format(claimed_extension: str) -> ImageFormat:
    ext = claimed_extension.lower().lstrip(".")
    for image_format, extensions in IMAGE_FORMAT_EXTENSIONS.items():
        if ext in extensions:
            return image_format
    raise UnsupportedFileTypeError(f"'.{ext}' is not a supported extension.", extension=ext)


class RasterImageAnalyzer(ImageAnalyzer):
    """rasterio for GeoTIFF, Pillow for PNG/JPEG — dispatched by detected format."""

    async def sniff_format(self, data: bytes, claimed_extension: str) -> ImageFormat:
        claimed_format = _extension_to_format(claimed_extension)
        return await asyncio.to_thread(self._sniff_format_sync, data, claimed_format)

    async def inspect(self, data: bytes, image_format: ImageFormat) -> ImageMetadata:
        return await asyncio.to_thread(self._inspect_sync, data, image_format)

    async def generate_thumbnail(
        self, data: bytes, image_format: ImageFormat, *, max_size: int
    ) -> bytes:
        return await asyncio.to_thread(self._generate_thumbnail_sync, data, image_format, max_size)

    # --- sniffing --------------------------------------------------------------------

    @staticmethod
    def _sniff_format_sync(data: bytes, claimed_format: ImageFormat) -> ImageFormat:
        if claimed_format is ImageFormat.GEOTIFF:
            try:
                with MemoryFile(data) as memfile, memfile.open() as src:
                    if src.driver != "GTiff":
                        raise CorruptImageError(
                            f"Claimed a GeoTIFF but rasterio opened it with driver "
                            f"'{src.driver}', not 'GTiff'."
                        )
            except RasterioIOError as exc:
                raise CorruptImageError(
                    "Claimed a GeoTIFF but the content isn't a readable raster."
                ) from exc
            return ImageFormat.GEOTIFF

        # PNG / JPEG
        try:
            with Image.open(io.BytesIO(data)) as img:
                img.verify()
                detected = img.format  # "PNG" or "JPEG", set by verify()
        except (UnidentifiedImageError, OSError) as exc:
            raise CorruptImageError(
                f"Claimed a {claimed_format.value.upper()} but the content isn't a readable image."
            ) from exc

        expected = "PNG" if claimed_format is ImageFormat.PNG else "JPEG"
        if detected != expected:
            raise CorruptImageError(
                f"Claimed a {expected} but the content is actually {detected}.",
                claimed=expected,
                detected=str(detected),
            )
        return claimed_format

    # --- inspection --------------------------------------------------------------------

    @staticmethod
    def _inspect_sync(data: bytes, image_format: ImageFormat) -> ImageMetadata:
        if image_format is ImageFormat.GEOTIFF:
            return RasterImageAnalyzer._inspect_geotiff(data)
        return RasterImageAnalyzer._inspect_plain_image(data)

    @staticmethod
    def _inspect_geotiff(data: bytes) -> ImageMetadata:
        with MemoryFile(data) as memfile, memfile.open() as src:
            width, height, band_count = src.width, src.height, src.count
            dtype = src.dtypes[0] if src.count else "uint8"

            crs = pixel_size_x = pixel_size_y = gsd_meters = None
            bounds_native = bounds_wgs84 = None

            if src.crs is not None:
                crs = src.crs.to_string()
                pixel_size_x, pixel_size_y = abs(src.res[0]), abs(src.res[1])
                bounds_native = (
                    src.bounds.left,
                    src.bounds.bottom,
                    src.bounds.right,
                    src.bounds.top,
                )
                try:
                    bounds_wgs84 = transform_bounds(src.crs, "EPSG:4326", *src.bounds)
                except Exception:
                    logger.warning("bounds_reprojection_failed", crs=crs)
                gsd_meters = _approximate_gsd_meters(
                    src.crs, pixel_size_x, pixel_size_y, bounds_wgs84
                )

            cloud_cover = None
            if band_count >= 3:
                cloud_cover = _estimate_cloud_cover_from_dataset(src)

        return ImageMetadata(
            width_px=width,
            height_px=height,
            band_count=band_count,
            dtype=str(dtype),
            crs=crs,
            pixel_size_x=pixel_size_x,
            pixel_size_y=pixel_size_y,
            gsd_meters_approx=gsd_meters,
            bounds_native=bounds_native,
            bounds_wgs84=bounds_wgs84,
            cloud_cover_percent=cloud_cover,
        )

    @staticmethod
    def _inspect_plain_image(data: bytes) -> ImageMetadata:
        with Image.open(io.BytesIO(data)) as img:
            width, height = img.size
            band_count = len(img.getbands())
            cloud_cover = None
            if band_count >= 3:
                sample = img.convert("RGB")
                sample.thumbnail((_CLOUD_HEURISTIC_SAMPLE_SIZE, _CLOUD_HEURISTIC_SAMPLE_SIZE))
                cloud_cover = _bright_pixel_ratio(np.asarray(sample))

        return ImageMetadata(
            width_px=width,
            height_px=height,
            band_count=band_count,
            dtype="uint8",  # PNG/JPEG are decoded to 8-bit-per-channel by Pillow
            cloud_cover_percent=cloud_cover,
        )

    # --- thumbnailing --------------------------------------------------------------------

    @staticmethod
    def _generate_thumbnail_sync(data: bytes, image_format: ImageFormat, max_size: int) -> bytes:
        if image_format is ImageFormat.GEOTIFF:
            image = RasterImageAnalyzer._geotiff_to_pil_image(data, max_size)
        else:
            image = Image.open(io.BytesIO(data)).convert("RGB")

        image.thumbnail((max_size, max_size))
        buffer = io.BytesIO()
        image.convert("RGB").save(buffer, format="JPEG", quality=85)
        return buffer.getvalue()

    @staticmethod
    def _geotiff_to_pil_image(data: bytes, max_size: int) -> Image.Image:
        """Read a downsampled preview directly from the raster rather than decoding it at
        full resolution first — a multi-band, multi-megapixel scene thumbnailed the naive
        way would mean reading and normalising far more data than the preview needs."""
        with MemoryFile(data) as memfile, memfile.open() as src:
            band_indices = (1, 2, 3) if src.count >= 3 else (1,)
            scale = min(1.0, max_size / max(src.width, src.height))
            out_height = max(1, round(src.height * scale))
            out_width = max(1, round(src.width * scale))

            array = src.read(band_indices, out_shape=(len(band_indices), out_height, out_width))
            normalized = _normalize_to_uint8(array)

        if len(band_indices) == 1:
            return Image.fromarray(normalized[0], mode="L").convert("RGB")
        return Image.fromarray(np.moveaxis(normalized, 0, -1), mode="RGB")


def _normalize_to_uint8(array: np.ndarray) -> np.ndarray:
    """Per-band min-max stretch to 0-255 — GeoTIFFs are frequently uint16 or float32 with
    a dynamic range far outside 0-255, so a naive cast would produce a near-black or
    blown-out thumbnail instead of a usable preview."""
    result = np.empty(array.shape, dtype=np.uint8)
    for band_index in range(array.shape[0]):
        band = array[band_index].astype(np.float64)
        band_min, band_max = float(np.nanmin(band)), float(np.nanmax(band))
        if band_max - band_min < 1e-9:
            # A perfectly uniform band (every pixel identical — a solid white or black
            # region, not just an edge case in synthetic test data) has no range to
            # stretch. Preserving its actual value is what's correct here: defaulting to
            # 0 would silently turn a uniformly *white* band black, which is exactly
            # backwards for a brightness-based heuristic like cloud-cover scoring.
            result[band_index] = int(np.clip(round(band_min), 0, 255))
            continue
        stretched = (band - band_min) / (band_max - band_min) * 255.0
        result[band_index] = np.clip(stretched, 0, 255).astype(np.uint8)
    return result


def _bright_pixel_ratio(rgb_array: np.ndarray) -> float:
    bright_mask = np.all(rgb_array >= _BRIGHT_PIXEL_THRESHOLD, axis=-1)
    return round(float(np.mean(bright_mask)) * 100.0, 2)


def _estimate_cloud_cover_from_dataset(src: rasterio.io.DatasetReader) -> float:
    scale = min(1.0, _CLOUD_HEURISTIC_SAMPLE_SIZE / max(src.width, src.height))
    out_height = max(1, round(src.height * scale))
    out_width = max(1, round(src.width * scale))
    array = src.read((1, 2, 3), out_shape=(3, out_height, out_width))
    normalized = _normalize_to_uint8(array)
    return _bright_pixel_ratio(np.moveaxis(normalized, 0, -1))


def _approximate_gsd_meters(
    crs: Any, pixel_size_x: float, pixel_size_y: float, bounds_wgs84: tuple[float, ...] | None
) -> float:
    """Ground sample distance in metres. Projected CRSs (the overwhelming majority used in
    remote sensing — UTM zones etc.) already measure in metres, so pixel size is returned
    directly. Geographic CRSs (degrees) are converted using the standard latitude-adjusted
    approximation, evaluated at the scene's centre latitude."""
    if not crs.is_geographic:
        return round((pixel_size_x + pixel_size_y) / 2, 4)

    center_lat = 0.0
    if bounds_wgs84 is not None:
        center_lat = (bounds_wgs84[1] + bounds_wgs84[3]) / 2  # (south + north) / 2

    metres_per_degree_lon = _METRES_PER_DEGREE_LATITUDE * math.cos(math.radians(center_lat))
    gsd_x = pixel_size_x * metres_per_degree_lon
    gsd_y = pixel_size_y * _METRES_PER_DEGREE_LATITUDE
    return round((abs(gsd_x) + abs(gsd_y)) / 2, 4)
