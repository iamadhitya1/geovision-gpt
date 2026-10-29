"""Window by window: does "Is there a water area?" agree with OpenStreetMap's water?

A diagnostic, not a benchmark: three places, one question, and OpenStreetMap under-maps small
ponds in rural areas. Run from ``backend/``: ``python scripts/water_vs_osm.py``.
"""

import asyncio
import math
import sys
from pathlib import Path

import httpx
import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, r"C:\Users\adhit\geovision-gpt\backend")
sys.path.insert(0, r"C:\Users\adhit\geovision-gpt\ml")

from app.core.config import get_settings
from app.domain.area import plan_windows
from app.domain.enums import FusionType
from app.infrastructure.imaging.earth_engine_provider import EarthEngineProvider
from app.infrastructure.vqa.model_runner import TorchVqaModelRunner

PLACES = {
    "Ahmedabad, India (city, river)": (23.0225, 72.5714),
    "Amsterdam area, Netherlands (training region)": (52.37, 4.95),
    "Rural Gujarat, India": (22.55, 71.60),
}
SIZE_KM = 10
QUESTION = "Is there a water area?"
ENDPOINTS = [
    "https://overpass-api.de/api/interpreter",
    "https://overpass.kumi.systems/api/interpreter",
]


def merc(lat):
    return math.log(math.tan(math.pi / 4 + math.radians(lat) / 2))


async def osm_water(bounds):
    west, south, east, north = bounds
    box = f"({south},{west},{north},{east})"
    query = (
        "[out:json][timeout:90];("
        f'way["natural"="water"]{box};relation["natural"="water"]{box};'
        f'way["waterway"~"^(river|canal|stream)$"]{box};'
        f'way["landuse"~"^(reservoir|basin)$"]{box};'
        ");out geom;"
    )
    last = None
    async with httpx.AsyncClient(
        timeout=120, headers={"User-Agent": "GeoVisionGPT/0.1 (student research project)"}
    ) as client:
        for url in ENDPOINTS:
            for _ in range(2):
                try:
                    r = await client.post(url, data={"data": query})
                    r.raise_for_status()
                    return r.json()["elements"]
                except Exception as error:  # try the next server / once more
                    last = error
                    await asyncio.sleep(3)
    raise RuntimeError(f"Overpass failed: {last}")


def rasterise(elements, bounds, size):
    west, south, east, north = bounds
    y_top, y_bot = merc(north), merc(south)

    def to_px(lat, lon):
        return ((lon - west) / (east - west) * size, (y_top - merc(lat)) / (y_top - y_bot) * size)

    image = Image.new("L", (size, size), 0)
    draw = ImageDraw.Draw(image)

    def ring(geometry):
        return [to_px(p["lat"], p["lon"]) for p in geometry]

    for el in elements:
        tags = el.get("tags", {})
        if el["type"] == "way" and el.get("geometry"):
            pts = ring(el["geometry"])
            closed = el["geometry"][0] == el["geometry"][-1] and len(pts) > 3
            if closed and (
                tags.get("natural") == "water" or tags.get("landuse") in ("reservoir", "basin")
            ):
                draw.polygon(pts, fill=255)
            else:
                draw.line(pts, fill=255, width=2)
        elif el["type"] == "relation":
            for member in el.get("members", []):
                if member.get("role") == "outer" and member.get("geometry"):
                    draw.polygon(ring(member["geometry"]), fill=255)
    return np.asarray(image) > 0


async def main():
    settings = get_settings()
    provider = EarthEngineProvider(project=settings.earth_engine_project)
    runner = TorchVqaModelRunner(
        checkpoint_paths={
            FusionType.ELEMENTWISE: Path(settings.vqa_checkpoint_elementwise_path),
            FusionType.CROSS_ATTENTION: Path(settings.vqa_checkpoint_cross_attention_path),
        },
        vocab_path=Path(settings.vqa_vocab_path),
        device=settings.vqa_device,
    )
    size_px = SIZE_KM * 100
    plan = plan_windows(size_px, size_px)
    for name, (lat, lon) in PLACES.items():
        scene = await provider.fetch_area(lat, lon, size_px=size_px)
        elements = await osm_water(scene.bounds_wgs84)
        water = rasterise(elements, scene.bounds_wgs84, size_px)
        print(
            f"\n{name}: {len(elements)} OSM water objects; "
            f"water covers {water.mean():.1%} of the area",
            flush=True,
        )
        for fusion in (FusionType.ELEMENTWISE, FusionType.CROSS_ATTENTION):
            tp = fp = fn = tn = 0
            for x, y in plan.origins:
                truth = water[y : y + 256, x : x + 256].sum() >= 1
                answer = (
                    await runner.predict(
                        scene.crop_png(x, y, 256), QUESTION, fusion=fusion, top_k=1
                    )
                ).answer
                said = answer == "yes"
                tp += said and truth
                fp += said and not truth
                fn += (not said) and truth
                tn += (not said) and not truth
            n = tp + fp + fn + tn
            print(
                f"  {fusion.value:16s} windows={n} OSM-water-in-window={tp + fn} | "
                f"said yes={tp + fp} | correct {(tp + tn) / n:.0%} "
                f"(yes&water {tp}, yes&none {fp}, no&water {fn}, no&none {tn})",
                flush=True,
            )


asyncio.run(main())
