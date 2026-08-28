"""Unit tests for the rasterio/Pillow-backed ImageAnalyzer."""

from __future__ import annotations

import pytest

from app.domain.enums import ImageFormat
from app.domain.errors import CorruptImageError, UnsupportedFileTypeError
from app.infrastructure.imaging.analyzer import RasterImageAnalyzer
from tests.conftest import (
    make_bright_geotiff_bytes,
    make_geotiff_bytes,
    make_jpeg_bytes,
    make_png_bytes,
)


@pytest.fixture
def analyzer() -> RasterImageAnalyzer:
    return RasterImageAnalyzer()


# --- sniff_format ---------------------------------------------------------------------


async def test_sniffs_a_real_geotiff(analyzer: RasterImageAnalyzer) -> None:
    detected = await analyzer.sniff_format(make_geotiff_bytes(), "tif")
    assert detected is ImageFormat.GEOTIFF


async def test_sniffs_a_real_png(analyzer: RasterImageAnalyzer) -> None:
    detected = await analyzer.sniff_format(make_png_bytes(), "png")
    assert detected is ImageFormat.PNG


async def test_sniffs_a_real_jpeg(analyzer: RasterImageAnalyzer) -> None:
    detected = await analyzer.sniff_format(make_jpeg_bytes(), "jpeg")
    assert detected is ImageFormat.JPEG


async def test_rejects_unrecognised_extension(analyzer: RasterImageAnalyzer) -> None:
    with pytest.raises(UnsupportedFileTypeError):
        await analyzer.sniff_format(make_png_bytes(), "exe")


async def test_rejects_content_that_does_not_match_claimed_extension(
    analyzer: RasterImageAnalyzer,
) -> None:
    """A JPEG renamed to .png — the exact spoofing attempt content-sniffing exists to
    catch, as opposed to trusting the client-supplied filename."""
    with pytest.raises(CorruptImageError):
        await analyzer.sniff_format(make_jpeg_bytes(), "png")


async def test_rejects_garbage_claiming_to_be_a_geotiff(analyzer: RasterImageAnalyzer) -> None:
    with pytest.raises(CorruptImageError):
        await analyzer.sniff_format(b"this is not a raster", "tif")


async def test_rejects_garbage_claiming_to_be_a_png(analyzer: RasterImageAnalyzer) -> None:
    with pytest.raises(CorruptImageError):
        await analyzer.sniff_format(b"this is not an image", "png")


# --- inspect ---------------------------------------------------------------------------


async def test_inspects_a_georeferenced_geotiff(analyzer: RasterImageAnalyzer) -> None:
    metadata = await analyzer.inspect(
        make_geotiff_bytes(width=64, height=48, band_count=3), ImageFormat.GEOTIFF
    )

    assert metadata.width_px == 64
    assert metadata.height_px == 48
    assert metadata.band_count == 3
    assert metadata.crs == "EPSG:4326"
    assert metadata.bounds_native is not None
    assert metadata.bounds_wgs84 is not None
    assert metadata.gsd_meters_approx is not None
    assert metadata.gsd_meters_approx > 0
    assert metadata.cloud_cover_percent is not None


async def test_inspects_a_geotiff_without_a_crs(analyzer: RasterImageAnalyzer) -> None:
    """A TIFF with no embedded georeferencing is accepted, not rejected — the geospatial
    fields are simply None, and that's a legitimate, honestly-reported state."""
    metadata = await analyzer.inspect(make_geotiff_bytes(with_crs=False), ImageFormat.GEOTIFF)

    assert metadata.crs is None
    assert metadata.bounds_native is None
    assert metadata.bounds_wgs84 is None
    assert metadata.gsd_meters_approx is None
    assert metadata.width_px == 64  # pixel dimensions are still known regardless


async def test_single_band_geotiff_has_no_cloud_estimate(analyzer: RasterImageAnalyzer) -> None:
    """The heuristic needs RGB-like channels; a single-band raster can't produce one."""
    metadata = await analyzer.inspect(make_geotiff_bytes(band_count=1), ImageFormat.GEOTIFF)
    assert metadata.band_count == 1
    assert metadata.cloud_cover_percent is None


async def test_bright_image_scores_high_on_the_cloud_heuristic(
    analyzer: RasterImageAnalyzer,
) -> None:
    metadata = await analyzer.inspect(make_bright_geotiff_bytes(), ImageFormat.GEOTIFF)
    assert metadata.cloud_cover_percent is not None
    assert metadata.cloud_cover_percent > 95.0


async def test_inspects_a_png(analyzer: RasterImageAnalyzer) -> None:
    metadata = await analyzer.inspect(make_png_bytes(width=32, height=20), ImageFormat.PNG)

    assert metadata.width_px == 32
    assert metadata.height_px == 20
    assert metadata.band_count == 3
    assert metadata.crs is None
    assert metadata.bounds_native is None


# --- generate_thumbnail ------------------------------------------------------------------


async def test_thumbnail_respects_max_size(analyzer: RasterImageAnalyzer) -> None:
    from io import BytesIO

    from PIL import Image

    thumbnail_bytes = await analyzer.generate_thumbnail(
        make_geotiff_bytes(width=200, height=100), ImageFormat.GEOTIFF, max_size=64
    )

    with Image.open(BytesIO(thumbnail_bytes)) as img:
        assert img.format == "JPEG"
        assert max(img.size) <= 64


async def test_thumbnail_from_png(analyzer: RasterImageAnalyzer) -> None:
    from io import BytesIO

    from PIL import Image

    thumbnail_bytes = await analyzer.generate_thumbnail(
        make_png_bytes(width=200, height=200), ImageFormat.PNG, max_size=50
    )

    with Image.open(BytesIO(thumbnail_bytes)) as img:
        assert img.format == "JPEG"
        assert max(img.size) <= 50
