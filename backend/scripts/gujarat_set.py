"""The Gujarat training set: a sampling plan, then the pictures and labels for it.

``plan``   chooses where the tiles go and writes ``ml/dataset/gujarat/set/plan.json``. It costs a
           handful of Earth Engine calls and no pictures, so the numbers can be read before the
           expensive part. Rules that matter for an honest result:
           * Gujarat's districts (FAO GAUL level 2) are split **as whole districts** into train /
             validation / test, so the test result is on districts the model never saw.
           * Tiles are drawn half uniformly over a district, a quarter inside urban areas and a
             quarter on places that hold surface water. Uniform sampling alone gives almost no
             urban or water examples to learn from; the ``strategy`` is recorded so that the
             uniform tiles alone can be used to quote honest base rates.
           * Each tile centre is moved a random distance of up to 1 km, so "the interesting thing is
             always in the middle" cannot be learned.
           * Each tile's picture comes from one of two seasons (January to March, or the weeks after
             the monsoon), so the model sees both.
``fetch``  builds the tiles listed in the plan (resumable; ``--limit`` caps one batch, so quota use
           can be watched). Labels are the same as ``gujarat_tiles.py``'s.

Run from ``backend/``::

    python scripts/gujarat_set.py plan --train 3000 --val 500 --test 500
    python scripts/gujarat_set.py fetch --limit 500
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import random

from gujarat_tiles import OUT, SURFACE_WATER_OCCURRENCE, _labels

from app.core.config import get_settings
from app.infrastructure.imaging.earth_engine_provider import EarthEngineProvider

SET = OUT / "set"
# Fractions of a split's tiles drawn each way.
STRATEGIES = {"uniform": 0.5, "urban": 0.25, "water": 0.25}
SEASONS = [("2026-01-01", "2026-03-31"), ("2026-09-15", "2026-10-29")]
SEASON_WEIGHTS = [0.7, 0.3]
JITTER_M = 1000
# The districts holding Gujarat's big cities. They are dealt out across the splits separately, so
# the training split sees real cities too (a plain random split once put all of them in val/test).
CITY_DISTRICTS = {"Ahmadabad", "Surat", "Vadodara", "Rajkot", "Bhavnagar", "Jamnagar"}
CITY_SPLIT = {"train": 3, "val": 1, "test": 2}
OTHER_SPLIT = {"val": 4, "test": 4}  # the rest of the other districts go to train


def _districts(ee) -> list[dict]:
    fc = ee.FeatureCollection("FAO/GAUL/2015/level2").filter(ee.Filter.eq("ADM1_NAME", "Gujarat"))
    info = fc.map(
        lambda f: ee.Feature(None, {"name": f.get("ADM2_NAME"), "area": f.geometry().area(1000)})
    ).getInfo()
    return sorted(
        (
            {"name": f["properties"]["name"], "area": f["properties"]["area"]}
            for f in info["features"]
        ),
        key=lambda d: d["name"],
    )


def _points(ee, geometry, strategy: str, count: int, seed: int) -> list[tuple[float, float]]:
    """(lat, lon) of up to ``count`` places in ``geometry`` drawn the given way."""
    if count <= 0:
        return []
    if strategy == "uniform":
        collection = ee.FeatureCollection.randomPoints(geometry, count, seed)
    elif strategy == "urban":
        # stratifiedSample returns the number of points of a class that is asked for; plain
        # sample() drops most points once a mask is applied and returned only a handful.
        flag = (
            ee.ImageCollection("JRC/GHSL/P2023A/GHS_SMOD_V2-0")
            .filterDate("2020-01-01", "2021-01-01")
            .first()
            .select("smod_code")
            .gte(22)
            .rename("flag")
        )
        collection = flag.stratifiedSample(
            numPoints=0,  # only the class asked for, none of the other
            classBand="flag",
            region=geometry,
            scale=1000,
            seed=seed,
            classValues=[1],
            classPoints=[count],
            geometries=True,
            tileScale=4,
        )
    else:
        flag = (
            ee.Image("JRC/GSW1_4/GlobalSurfaceWater")
            .select("occurrence")
            .unmask(0)
            .gte(SURFACE_WATER_OCCURRENCE)
            .rename("flag")
        )
        collection = flag.stratifiedSample(
            numPoints=0,  # only the class asked for, none of the other
            classBand="flag",
            region=geometry,
            scale=30,
            seed=seed,
            classValues=[1],
            classPoints=[count],
            geometries=True,
            tileScale=4,
        )
    features = collection.getInfo()["features"]
    return [(f["geometry"]["coordinates"][1], f["geometry"]["coordinates"][0]) for f in features]


def plan(args: argparse.Namespace) -> int:
    ee = EarthEngineProvider(project=get_settings().earth_engine_project)._earth_engine()
    rng = random.Random(args.seed)
    districts = _districts(ee)
    split_of: dict[str, str] = {}

    def deal(group: list[str], counts: dict[str, int]) -> None:
        shuffled = sorted(group)
        rng.shuffle(shuffled)
        position = 0
        for split in ("test", "val"):
            for name in shuffled[position : position + counts.get(split, 0)]:
                split_of[name] = split
            position += counts.get(split, 0)
        for name in shuffled[position:]:
            split_of[name] = "train"

    names = [d["name"] for d in districts]
    deal([n for n in names if n in CITY_DISTRICTS], CITY_SPLIT)
    deal([n for n in names if n not in CITY_DISTRICTS], OTHER_SPLIT)
    targets = {"train": args.train, "val": args.val, "test": args.test}
    fc = ee.FeatureCollection("FAO/GAUL/2015/level2").filter(ee.Filter.eq("ADM1_NAME", "Gujarat"))

    tiles = []
    for split, target in targets.items():
        members = [d for d in districts if split_of[d["name"]] == split]
        total_area = sum(d["area"] for d in members)
        for d in members:
            geometry = fc.filter(ee.Filter.eq("ADM2_NAME", d["name"])).geometry()
            share = max(1, round(target * d["area"] / total_area))
            for strategy, fraction in STRATEGIES.items():
                wanted = max(1, round(share * fraction))
                seed = rng.randrange(10**6)
                for lat, lon in _points(ee, geometry, strategy, wanted, seed):
                    # move the centre so the interesting thing is not always in the middle
                    angle, radius = rng.uniform(0, 2 * math.pi), JITTER_M * math.sqrt(rng.random())
                    lat += radius * math.sin(angle) / 111_320
                    lon += radius * math.cos(angle) / (111_320 * math.cos(math.radians(lat)))
                    season = rng.choices(SEASONS, SEASON_WEIGHTS)[0]
                    tiles.append(
                        {
                            "split": split,
                            "district": d["name"],
                            "strategy": strategy,
                            "lat": round(lat, 5),
                            "lon": round(lon, 5),
                            "date_from": season[0],
                            "date_to": season[1],
                        }
                    )
            print(f"{split:5s} {d['name']:18s} {len(tiles)} tiles so far", flush=True)
    for index, tile in enumerate(tiles):
        tile["id"] = f"{tile['split']}_{index:05d}"
    SET.mkdir(parents=True, exist_ok=True)
    (SET / "plan.json").write_text(
        json.dumps({"seed": args.seed, "tiles": tiles}), encoding="utf-8"
    )

    print(f"\n{len(tiles)} tiles planned -> {SET / 'plan.json'}")
    for split in targets:
        in_split = [t for t in tiles if t["split"] == split]
        by_strategy = {s: sum(t["strategy"] == s for t in in_split) for s in STRATEGIES}
        districts_in = sorted({t["district"] for t in in_split})
        print(
            f"  {split}: {len(in_split)} tiles {by_strategy}; districts: {', '.join(districts_in)}"
        )
    return 0


async def fetch(args: argparse.Namespace) -> int:
    provider = EarthEngineProvider(project=get_settings().earth_engine_project)
    ee = provider._earth_engine()
    spec = json.loads((SET / "plan.json").read_text(encoding="utf-8"))["tiles"]
    (SET / "images").mkdir(parents=True, exist_ok=True)
    index = SET / "labels.jsonl"
    done = (
        {json.loads(line)["id"] for line in index.read_text().splitlines() if line}
        if index.exists()
        else set()
    )
    todo = [t for t in spec if t["id"] not in done and (not args.split or t["split"] == args.split)]
    todo = todo[: args.limit] if args.limit else todo
    print(f"{len(done)} done, {len(todo)} to fetch in this batch", flush=True)
    gate = asyncio.Semaphore(args.concurrency)
    written = failed = 0

    async def one(tile: dict) -> None:
        nonlocal written, failed
        async with gate:
            try:
                scene = await provider.fetch_area(
                    tile["lat"],
                    tile["lon"],
                    size_px=256,
                    date_from=_date(tile["date_from"]),
                    date_to=_date(tile["date_to"]),
                )
                labels = await asyncio.to_thread(_labels, ee, scene.bounds_wgs84)
            except Exception as error:  # no clear view, or quota: counted, reported, not fatal
                failed += 1
                print(f"{tile['id']}: skipped ({str(error)[:120]})", flush=True)
                return
        (SET / "images" / f"{tile['id']}.png").write_bytes(scene.crop_png(0, 0, 256))
        record = {**tile, "bounds": scene.bounds_wgs84, "captured": scene.captured, **labels}
        with index.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record) + "\n")
        written += 1
        if written % 25 == 0:
            print(f"  {written} written, {failed} skipped", flush=True)

    if todo:
        await one(todo[0])  # alone first, so a sign-in or quota problem shows up once
        await asyncio.gather(*(one(t) for t in todo[1:]))
    print(
        f"\nbatch done: {written} written, {failed} skipped; {len(done) + written} tiles in total"
    )
    return 0


def _date(text: str):
    from datetime import date

    return date.fromisoformat(text)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("plan")
    p.add_argument("--train", type=int, default=3000)
    p.add_argument("--val", type=int, default=500)
    p.add_argument("--test", type=int, default=500)
    p.add_argument("--seed", type=int, default=0)
    f = sub.add_parser("fetch")
    f.add_argument("--limit", type=int, default=0, help="at most this many tiles in this batch")
    f.add_argument("--split", choices=["train", "val", "test"], default=None)
    f.add_argument("--concurrency", type=int, default=4)
    args = parser.parse_args()
    return plan(args) if args.command == "plan" else asyncio.run(fetch(args))


if __name__ == "__main__":
    raise SystemExit(main())
