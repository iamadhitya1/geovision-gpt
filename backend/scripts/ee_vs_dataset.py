"""Do the models answer as well on Google Earth Engine pictures as on the dataset's own images?

For a sample of RSVQA-LR images (which are georeferenced), rebuild the same 2.56 km footprint from
Earth Engine, then ask the models the dataset's own non-counting questions about BOTH pictures and
score each against the dataset's answers. If the Earth Engine picture is made faithfully, accuracy
should barely move.

Run from ``backend/`` (needs ``earthengine authenticate`` done and ``GVG_EARTH_ENGINE_PROJECT``
set, as for the app)::

    python scripts/ee_vs_dataset.py --split test --images 40 --seed 0

Nothing here is tuned: the picture is made exactly as the app makes it. If a gap shows up and the
way the picture is produced is then adjusted, choose the adjustment on ``--split val`` and measure
the result once on ``--split test``, so the test number is never the one that was fitted.

The imagery is not the same date as the dataset's (the dataset is years old), so a small gap can
be real change on the ground, not only a difference in how the picture is rendered.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import random
import sys
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path

import rasterio
from rasterio.warp import transform_bounds

from app.core.config import get_settings
from app.domain.enums import FusionType
from app.infrastructure.imaging.earth_engine_provider import EarthEngineProvider
from app.infrastructure.vqa.model_runner import TorchVqaModelRunner

DATA_ROOT = Path(__file__).resolve().parents[2] / "ml" / "dataset" / "rsvqa_lr"
QUESTION_TYPES = {"presence", "comp", "rural_urban"}  # counting is scored differently, skipped


def centre_of(tif: Path) -> tuple[float, float]:
    with rasterio.open(tif) as dataset:
        west, south, east, north = transform_bounds(dataset.crs, "EPSG:4326", *dataset.bounds)
    return (south + north) / 2, (west + east) / 2


async def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--split", choices=["train", "val", "test"], default="test")
    parser.add_argument("--images", type=int, default=40)
    parser.add_argument("--questions-per-image", type=int, default=15)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--days", type=int, default=180, help="how far back Earth Engine looks")
    parser.add_argument("--out", type=Path, default=None, help="write per-question results (JSON)")
    args = parser.parse_args()

    sys.path.insert(0, str(DATA_ROOT.parents[1]))
    from dataset.rsvqa_lr_dataset import load_split_examples

    settings = get_settings()
    if not settings.earth_engine_project:
        print("GVG_EARTH_ENGINE_PROJECT is not set.")
        return 1

    examples = [
        e for e in load_split_examples(DATA_ROOT, args.split) if e.question_type in QUESTION_TYPES
    ]
    by_image: dict[Path, list] = defaultdict(list)
    for example in examples:
        by_image[example.image_path].append(example)
    rng = random.Random(args.seed)
    images = rng.sample(sorted(by_image), min(args.images, len(by_image)))

    runner = TorchVqaModelRunner(
        checkpoint_paths={
            FusionType.ELEMENTWISE: Path(settings.vqa_checkpoint_elementwise_path),
            FusionType.CROSS_ATTENTION: Path(settings.vqa_checkpoint_cross_attention_path),
        },
        vocab_path=Path(settings.vqa_vocab_path),
        device=settings.vqa_device,
    )
    fusions = sorted(runner.available_fusions(), key=lambda f: f.value)
    earth_engine = EarthEngineProvider(project=settings.earth_engine_project)

    # correct[(fusion, source)][question_type] = [right, total]
    correct: dict[tuple[str, str], dict[str, list[int]]] = defaultdict(
        lambda: defaultdict(lambda: [0, 0])
    )
    same_answer = defaultdict(lambda: [0, 0])  # fusion -> [agree, total] across the two pictures
    rows: list[dict] = []
    skipped = 0

    for number, tif in enumerate(images, start=1):
        lat, lon = centre_of(tif)
        try:
            scene = await earth_engine.fetch_area(lat, lon, size_px=256)
        except Exception as error:  # a place with no clear scene is skipped, and counted
            skipped += 1
            print(f"[{number}/{len(images)}] {tif.name}: skipped ({error})")
            continue
        earth_png = scene.crop_png(0, 0, 256)
        original = tif.read_bytes()

        questions = by_image[tif]
        rng.shuffle(questions)
        for example in questions[: args.questions_per_image]:
            for fusion in fusions:
                answers = {}
                for source, data in (("dataset", original), ("earth_engine", earth_png)):
                    prediction = await runner.predict(
                        data, example.question, fusion=fusion, top_k=1
                    )
                    answers[source] = prediction.answer
                    tally = correct[(fusion.value, source)][example.question_type]
                    tally[0] += int(prediction.answer == example.answer)
                    tally[1] += 1
                same_answer[fusion.value][0] += int(answers["dataset"] == answers["earth_engine"])
                same_answer[fusion.value][1] += 1
                rows.append(
                    {
                        "image": tif.name,
                        "question": example.question,
                        "type": example.question_type,
                        "truth": example.answer,
                        "fusion": fusion.value,
                        **answers,
                    }
                )
        print(f"[{number}/{len(images)}] {tif.name}: ok ({scene.captured})")

    print()
    print(f"split={args.split} images={len(images) - skipped} (skipped {skipped}) seed={args.seed}")
    for fusion in fusions:
        print(f"\n{fusion.value}")
        for source in ("dataset", "earth_engine"):
            tallies = correct[(fusion.value, source)]
            right = sum(t[0] for t in tallies.values())
            total = sum(t[1] for t in tallies.values())
            detail = ", ".join(
                f"{kind} {t[0] / t[1]:.1%} (n={t[1]})" for kind, t in sorted(tallies.items())
            )
            print(f"  {source:13s} {right / max(total, 1):.1%} of {total}  [{detail}]")
        agree, total = same_answer[fusion.value]
        print(f"  gave the same answer on both pictures: {agree / max(total, 1):.1%}")

    if args.out:
        args.out.write_text(
            json.dumps({"made": datetime.now(UTC).isoformat(), "rows": rows}), encoding="utf-8"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
