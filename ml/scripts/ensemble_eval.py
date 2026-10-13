"""Does averaging several training runs' answers beat any single run?

Three elementwise models were trained from different random seeds (0, 1, 2). They make
different mistakes, so averaging their answer probabilities — a *seed ensemble* — often scores
higher and varies less than any one of them. This script measures that on RSVQA-LR's held-out
test split, with every number computed on exactly the same questions:

* each single model, and the mean of the three (what you'd expect from picking one at random);
* the ensemble (the average of the three models' probabilities, then argmax);
* counting in the paper's five ranges, using the sum-the-probability-per-range decoding.

Each model's probabilities are cached under ``checkpoints/ensemble_cache/`` so a re-run (or
adding a fourth seed) only pays for the new model. Costs 3x the inference of one model, which
is the price of an ensemble — it would be an optional "robust mode", not the default.

    python scripts/ensemble_eval.py
    python scripts/ensemble_eval.py --smoke        # 40 questions, to check it runs
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dataset.count_ranges import RANGE_NAMES, range_index
from dataset.rsvqa_lr_dataset import AnswerVocabulary
from dataset.vqa_examples_dataset import VqaExamplesDataset, in_vocabulary
from models.vqa_model import LitRSVQAModel, build_image_transform, build_tokenizer
from scripts.train import make_collate_fn, resolve_device
from scripts.train_generalist import load_examples

ML_CHECKPOINTS = Path(__file__).resolve().parent.parent / "checkpoints"
BACKEND_CHECKPOINTS = Path(__file__).resolve().parent.parent.parent / "backend" / "checkpoints"
CACHE_DIR = ML_CHECKPOINTS / "ensemble_cache"
VOCAB_PATH = BACKEND_CHECKPOINTS / "answer_vocab_generalist.json"

# Seed 0 is the pair deployed in the app; seeds 1 and 2 are copied down from the cluster.
ELEMENTWISE = {
    "seed0": BACKEND_CHECKPOINTS / "xcit_nano_elementwise_generalist.pt",
    "seed1": ML_CHECKPOINTS / "gen_xcit_s1_best.pt",
    "seed2": ML_CHECKPOINTS / "gen_xcit_s2_best.pt",
}


def run_model(path: Path, examples, vocab, device) -> np.ndarray:
    """Answer probabilities (N x vocab) for every question, as float16 to keep the cache small."""
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    model = LitRSVQAModel(checkpoint["config"], fusion=checkpoint["fusion"]).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    loader = DataLoader(
        VqaExamplesDataset(examples, vocab),
        batch_size=64,
        shuffle=False,
        collate_fn=make_collate_fn(
            build_tokenizer(), build_image_transform(model.config.image_encoder)
        ),
        num_workers=0,  # Windows spawn can't pickle the collate closure
    )
    chunks = []
    with torch.no_grad():
        for batch in loader:
            logits = model(
                batch["pixel_values"].to(device),
                batch["input_ids"].to(device),
                batch["attention_mask"].to(device),
            )
            chunks.append(torch.softmax(logits, dim=-1).cpu().numpy().astype(np.float16))
    return np.concatenate(chunks)


def metrics(
    probs: np.ndarray,
    labels: np.ndarray,
    types: list[str],
    numeric_ids,
    bucket_of: dict,
    truth_bucket,
):
    """Overall, per-type and average accuracy, plus counting in ranges, for one matrix."""
    predicted = probs.argmax(axis=1)
    correct = predicted == labels
    per_type = {
        t: float(correct[[i for i, x in enumerate(types) if x == t]].mean())
        for t in sorted(set(types))
    }

    # Counting in the paper's five ranges, summing probability per range (numeric answers only).
    count_idx = [i for i, t in enumerate(types) if t == "count"]
    p = probs[count_idx].astype(np.float32)[:, numeric_ids]
    p = p / p.sum(axis=1, keepdims=True)
    mass = np.zeros((len(count_idx), len(RANGE_NAMES)), dtype=np.float32)
    for column, answer_id in enumerate(numeric_ids):
        mass[:, bucket_of[answer_id]] += p[:, column]
    in_range = float((mass.argmax(axis=1) == truth_bucket[count_idx]).mean())
    return {
        "overall": float(correct.mean()),
        "average": float(np.mean(list(per_type.values()))),
        "per_type": per_type,
        "count_in_ranges_summed": in_range,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        "--smoke", action="store_true", help="40 questions, to check the script runs"
    )
    args = parser.parse_args()

    device = resolve_device()
    vocab = AnswerVocabulary.from_json(VOCAB_PATH)
    examples = in_vocabulary(load_examples("lr", "test"), vocab)
    if args.smoke:
        examples = examples[:40]
    print(f"{len(examples)} test questions (in vocabulary)")

    labels = np.array([vocab.encode(e.answer) for e in examples])
    types = [e.question_type for e in examples]
    numeric_ids = [i for i in range(len(vocab)) if vocab.decode(i).isdigit()]
    bucket_of = {i: range_index(int(vocab.decode(i))) for i in numeric_ids}
    truth_bucket = np.array(
        [range_index(int(e.answer)) if e.answer.isdigit() else -1 for e in examples]
    )

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    all_probs: dict[str, np.ndarray] = {}
    for name, path in ELEMENTWISE.items():
        cache = CACHE_DIR / f"{name}{'_smoke' if args.smoke else ''}.npy"
        if cache.exists():
            all_probs[name] = np.load(cache)
            print(f"{name}: loaded from cache")
            continue
        t0 = time.time()
        all_probs[name] = run_model(path, examples, vocab, device)
        np.save(cache, all_probs[name])
        print(f"{name}: {time.time() - t0:.0f}s")

    results = {
        name: metrics(p, labels, types, numeric_ids, bucket_of, truth_bucket)
        for name, p in all_probs.items()
    }
    ensemble_probs = np.mean([p.astype(np.float32) for p in all_probs.values()], axis=0)
    results["ensemble"] = metrics(
        ensemble_probs, labels, types, numeric_ids, bucket_of, truth_bucket
    )

    singles = [results[n] for n in ELEMENTWISE]
    mean_single = {
        "overall": float(np.mean([r["overall"] for r in singles])),
        "average": float(np.mean([r["average"] for r in singles])),
        "count_in_ranges_summed": float(np.mean([r["count_in_ranges_summed"] for r in singles])),
    }

    print("\n                 overall  average  count(ranges)")
    for name in [*ELEMENTWISE, "ensemble"]:
        r = results[name]
        print(
            f"  {name:10}  {100 * r['overall']:7.2f}  {100 * r['average']:7.2f}"
            f"  {100 * r['count_in_ranges_summed']:8.2f}"
        )
    print(
        f"  {'mean single':10}  {100 * mean_single['overall']:7.2f}"
        f"  {100 * mean_single['average']:7.2f}"
        f"  {100 * mean_single['count_in_ranges_summed']:8.2f}"
    )

    out = {"n_questions": len(examples), "results": results, "mean_single": mean_single}
    path = ML_CHECKPOINTS / ("ensemble_eval_smoke.json" if args.smoke else "ensemble_eval.json")
    path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\nWritten: {path}")


if __name__ == "__main__":
    main()
