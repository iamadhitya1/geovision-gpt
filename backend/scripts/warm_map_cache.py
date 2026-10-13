"""Fill the map-data cache for the places you'll demo, while you have good internet.

The highlight button draws roads, buildings, forest, grass, farmland and cemeteries from
OpenStreetMap. The public servers are the least reliable part of the app — the same query can
answer in a second and then time out — so run this once at home. Everything it fetches is kept
on disk (``GVG_MAP_DATA_CACHE_DIR``), and from then on those places highlight instantly, with
or without internet.

    python scripts/warm_map_cache.py                           # the three runbook demo images
    python scripts/warm_map_cache.py path/to/a.tif path/to/b.tif
    python scripts/warm_map_cache.py --place 23.0225,72.5714   # a live-map place (Ahmedabad)
    python scripts/warm_map_cache.py --place 23.02,72.57 --place 18.95,72.85

Run from the ``backend`` folder with its virtualenv. Takes about 5-10 seconds per image or
place; it waits between requests to stay polite to the free servers.
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

import rasterio
from rasterio.warp import transform_bounds

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.core.config import get_settings
from app.domain.concepts import HighlightConcept
from app.infrastructure.imaging.osm_provider import SUPPORTED, OverpassMapData
from app.infrastructure.imaging.sentinel_provider import scene_bounds_wgs84

DEFAULT_IMAGES = ["237.tif", "246.tif", "315.tif"]
DEMO_FOLDER = Path.home() / "Desktop" / "GeoVisionGPT Demo Images"
PAUSE_SECONDS = 1.5
RETRY_PAUSE_SECONDS = 20.0


def image_bounds(path: Path) -> tuple[float, float, float, float]:
    with rasterio.open(path) as dataset:
        return tuple(transform_bounds(dataset.crs, "EPSG:4326", *dataset.bounds))  # type: ignore[return-value]


async def warm(targets: dict[str, tuple[float, float, float, float]], *, retries: int) -> int:
    """Fetch every layer for every target, retrying the failures (the free servers answer a
    given query one moment and refuse it the next). Returns how many are still missing."""
    settings = get_settings()
    provider = OverpassMapData(
        user_agent=settings.imagery_user_agent,
        cache_dir=Path(settings.map_data_cache_dir),
        # Patient: there is no person waiting here, and finished pieces are kept between passes.
        deadline_seconds=180.0,
    )
    wanted = [(name, c) for name in targets for c in sorted(SUPPORTED, key=lambda c: c.value)]
    try:
        for attempt in range(retries + 1):
            if attempt:
                print(f"\nRetrying {len(wanted)} that failed (pass {attempt + 1}) ...")
                await asyncio.sleep(RETRY_PAUSE_SECONDS)
            still_missing: list[tuple[str, HighlightConcept]] = []
            current = None
            for name, concept in wanted:
                if name != current:
                    print(f"\n{name}")
                    current = name
                try:
                    features = await provider.fetch(concept, targets[name])
                    print(f"  {concept.value:10} {len(features):5d} features")
                except Exception as exc:
                    still_missing.append((name, concept))
                    print(f"  {concept.value:10} FAILED: {exc}")
                await asyncio.sleep(PAUSE_SECONDS)
            wanted = still_missing
            if not wanted:
                break
    finally:
        await provider.aclose()
    return len(wanted)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        "images", nargs="*", help="GeoTIFFs to warm (default: the 3 runbook images)"
    )
    parser.add_argument(
        "--place", action="append", default=[], metavar="LAT,LON", help="a live-map place to warm"
    )
    parser.add_argument(
        "--retries", type=int, default=3, help="how many times to retry what failed (default 3)"
    )
    args = parser.parse_args()

    targets: dict[str, tuple[float, float, float, float]] = {}
    paths = [Path(p) for p in args.images]
    if not paths and not args.place:
        paths = [DEMO_FOLDER / name for name in DEFAULT_IMAGES]
    for path in paths:
        if not path.exists():
            raise SystemExit(f"Not found: {path}")
        targets[path.name] = image_bounds(path)
    for place in args.place:
        lat, lon = (float(x) for x in place.split(","))
        targets[f"place {lat},{lon}"] = scene_bounds_wgs84(lat, lon)

    failures = asyncio.run(warm(targets, retries=args.retries))
    total = len(targets) * len(SUPPORTED)
    print(
        f"\n{total - failures}/{total} fetched.",
        "Re-run to retry the failures." if failures else "",
    )
    raise SystemExit(1 if failures else 0)


if __name__ == "__main__":
    main()
