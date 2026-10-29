"""Labelled Gujarat tiles: a picture from Earth Engine and labels from Earth Engine's own layers.

``build``    samples random places inside Gujarat, saves the 256 x 256 px, ~10 m/px picture the app
             would analyse (same rendering as the app) and labels it from
             * ESA WorldCover 10 m land cover (water, tree cover, cropland, grass, built-up ...) and
             * Google Open Buildings (a building count),
             both hosted in Earth Engine, so one sign-in covers pictures and labels.
``relabel``  recomputes the labels of existing tiles (e.g. after the label recipe changes).
``evaluate`` asks the deployed models the app's own question templates about those pictures and
             scores them against the labels, including how often they say "no" when the label is no.

The labels are model-made (WorldCover is a classifier, Open Buildings a detector), not surveyed on
the ground: they are wrong in places (WorldCover calls some Kutch terrain cropland), so every
accuracy here is "against WorldCover / Open Buildings", never against ground truth.

Run from ``backend/``::

    python scripts/gujarat_tiles.py build --count 150 --seed 0
    python scripts/gujarat_tiles.py evaluate
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

from dataset.gujarat_dataset import TRUTH as QUESTIONS  # the one definition of "right"

from app.core.config import get_settings
from app.domain.enums import FusionType
from app.infrastructure.imaging.earth_engine_provider import EarthEngineProvider

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "ml" / "dataset" / "gujarat"
WORLDCOVER = {
    10: "tree",
    20: "shrub",
    30: "grass",
    40: "crop",
    50: "built",
    60: "bare",
    70: "snow",
    80: "water",
    90: "wetland",
    95: "mangrove",
    100: "moss",
}
# A thing counts as "present" in a tile when it covers at least this share of it (0.5% is about
# 3 hectares of a 6.5 km2 tile). Judgement calls, written down so the sensitivity can be tested.
MIN_BUILDING_CONFIDENCE = 0.7
# Water: WorldCover's open-water class OR the JRC Global Surface Water map (water seen in at least
# this share of valid Landsat observations since 1984, which finds ponds, tanks and reservoirs the
# 2021 land-cover snapshot can miss, but still not rivers that are dry most of the year).
SURFACE_WATER_OCCURRENCE = 20


def _labels(ee, bounds: tuple[float, float, float, float]) -> dict:
    """One Earth Engine round trip: land cover, surface water, settlement class and a building
    count for the footprint."""
    region = ee.Geometry.Rectangle(list(bounds), "EPSG:4326", False)
    cover = ee.ImageCollection("ESA/WorldCover/v200").first()
    histogram = cover.reduceRegion(ee.Reducer.frequencyHistogram(), region, 10, maxPixels=10**8)
    buildings = (
        ee.FeatureCollection("GOOGLE/Research/open-buildings/v3/polygons")
        .filterBounds(region)
        .filter(ee.Filter.gte("confidence", MIN_BUILDING_CONFIDENCE))
        .size()
    )
    water = (
        ee.Image("JRC/GSW1_4/GlobalSurfaceWater")
        .select("occurrence")
        .unmask(0)
        .gte(SURFACE_WATER_OCCURRENCE)
        .reduceRegion(ee.Reducer.mean(), region, 30)
        .get("occurrence")
    )
    settlement = (
        ee.ImageCollection("JRC/GHSL/P2023A/GHS_SMOD_V2-0")
        .filterDate("2020-01-01", "2021-01-01")
        .first()
        .select("smod_code")
        .reduceRegion(ee.Reducer.mode(), region, 1000)
        .get("smod_code")
    )
    info = ee.Dictionary(
        {
            "hist": histogram.get("Map"),
            "buildings": buildings,
            "water": water,
            "settlement": settlement,
        }
    ).getInfo()
    hist = info["hist"] or {}
    total = sum(hist.values()) or 1
    return {
        "cover": {WORLDCOVER[int(float(k))]: round(v / total, 4) for k, v in hist.items()},
        "buildings": int(info["buildings"]),
        "surface_water": round(float(info["water"] or 0), 4),
        "settlement": int(info["settlement"]) if info["settlement"] is not None else None,
    }


async def build(args: argparse.Namespace) -> int:
    settings = get_settings()
    provider = EarthEngineProvider(project=settings.earth_engine_project)
    (OUT / "images").mkdir(parents=True, exist_ok=True)
    index_path = OUT / "labels.jsonl"
    done = set()
    if index_path.exists():
        done = {json.loads(line)["id"] for line in index_path.read_text().splitlines() if line}

    ee = provider._earth_engine()
    gujarat = ee.FeatureCollection("FAO/GAUL/2015/level1").filter(
        ee.Filter.eq("ADM1_NAME", "Gujarat")
    )
    points = ee.FeatureCollection.randomPoints(gujarat.geometry(), args.count, args.seed).getInfo()
    coords = [f["geometry"]["coordinates"] for f in points["features"]]
    gate = asyncio.Semaphore(args.concurrency)
    written = 0

    async def one(index: int, lon: float, lat: float) -> None:
        nonlocal written
        tile_id = f"s{args.seed}_{index:05d}"
        if tile_id in done:
            return
        async with gate:
            try:
                scene = await provider.fetch_area(lat, lon, size_px=256)
                labels = await asyncio.to_thread(_labels, ee, scene.bounds_wgs84)
            except Exception as error:  # a tile with no clear view is skipped, not fatal
                print(f"{tile_id}: skipped ({error})", flush=True)
                return
        (OUT / "images" / f"{tile_id}.png").write_bytes(scene.crop_png(0, 0, 256))
        record = {
            "id": tile_id,
            "lat": round(lat, 5),
            "lon": round(lon, 5),
            "bounds": scene.bounds_wgs84,
            "captured": scene.captured,
            **labels,
        }
        with index_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record) + "\n")
        written += 1
        print(
            f"{tile_id}: ok ({labels['buildings']} buildings, "
            f"built {labels['cover'].get('built', 0):.0%})",
            flush=True,
        )

    # The first tile alone, so a sign-in or quota problem shows up once, not 150 times.
    await one(0, *coords[0])
    await asyncio.gather(*(one(i, lon, lat) for i, (lon, lat) in enumerate(coords) if i > 0))
    print(f"\nwrote {written} new tiles; {len(done) + written} in {OUT}")
    return 0


async def relabel(args: argparse.Namespace) -> int:
    """Recompute every tile's labels (not its picture), keeping the old file as labels_v1.jsonl."""
    settings = get_settings()
    ee = EarthEngineProvider(project=settings.earth_engine_project)._earth_engine()
    path = OUT / "labels.jsonl"
    old = [json.loads(line) for line in path.read_text().splitlines() if line]
    backup = OUT / "labels_v1.jsonl"
    if not backup.exists():
        backup.write_text(path.read_text())
    gate = asyncio.Semaphore(args.concurrency)

    async def one(tile: dict) -> dict:
        async with gate:
            fresh = await asyncio.to_thread(_labels, ee, tuple(tile["bounds"]))
        return {**tile, **fresh}

    new = await asyncio.gather(*(one(t) for t in old))
    path.write_text("".join(json.dumps(t) + "\n" for t in new))
    print(f"relabelled {len(new)} tiles; old labels kept in {backup.name}")
    return 0


async def evaluate(args: argparse.Namespace) -> int:
    sys.path.insert(0, str(ROOT / "ml"))
    from dataset.count_ranges import RANGE_LABELS, range_index

    from app.infrastructure.vqa.model_runner import TorchVqaModelRunner

    settings = get_settings()
    runner = TorchVqaModelRunner(
        checkpoint_paths={
            FusionType.ELEMENTWISE: Path(settings.vqa_checkpoint_elementwise_path),
            FusionType.CROSS_ATTENTION: Path(settings.vqa_checkpoint_cross_attention_path),
        },
        vocab_path=Path(settings.vqa_vocab_path),
        device=settings.vqa_device,
    )
    tiles = [json.loads(line) for line in (OUT / "labels.jsonl").read_text().splitlines() if line]
    print(f"{len(tiles)} Gujarat tiles")
    questions = dict(QUESTIONS)
    questions["How many buildings are there?"] = lambda t: RANGE_LABELS[range_index(t["buildings"])]

    for fusion in sorted(runner.available_fusions(), key=lambda f: f.value):
        print(f"\n{fusion.value}")
        tally = {q: {"right": 0, "n": 0, "said": {}, "truth": {}, "hit": {}} for q in questions}
        for tile in tiles:
            image = (OUT / "images" / f"{tile['id']}.png").read_bytes()
            for question, truth_of in questions.items():
                truth = truth_of(tile)
                said = (await runner.predict(image, question, fusion=fusion, top_k=1)).answer
                t = tally[question]
                t["n"] += 1
                t["right"] += said == truth
                t["said"][said] = t["said"].get(said, 0) + 1
                t["truth"][truth] = t["truth"].get(truth, 0) + 1
                t["hit"][(truth, said)] = t["hit"].get((truth, said), 0) + 1
        for question, t in tally.items():
            majority = max(t["truth"].values()) / t["n"]
            print(f"  {question}")
            print(
                f"    accuracy {t['right'] / t['n']:.0%} "
                f"(always the commonest label: {majority:.0%})"
            )
            print(
                f"    model said {dict(sorted(t['said'].items()))}   "
                f"label {dict(sorted(t['truth'].items()))}"
            )
            if question in {q for q in QUESTIONS if q != "Is it a rural or an urban area?"}:
                no_total = t["truth"].get("no", 0)
                if no_total:
                    kept = t["hit"].get(("no", "no"), 0)
                    print(
                        f"    of the {no_total} tiles labelled no, the model said no for "
                        f"{kept} ({kept / no_total:.0%})"
                    )
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)
    b = sub.add_parser("build")
    b.add_argument("--count", type=int, default=150)
    b.add_argument("--seed", type=int, default=0)
    b.add_argument("--concurrency", type=int, default=4)
    r = sub.add_parser("relabel")
    r.add_argument("--concurrency", type=int, default=4)
    sub.add_parser("evaluate")
    args = parser.parse_args()
    run = {"build": build, "relabel": relabel, "evaluate": evaluate}[args.command]
    return asyncio.run(run(args))


if __name__ == "__main__":
    raise SystemExit(main())
