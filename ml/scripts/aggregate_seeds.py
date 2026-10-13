"""Combine several training seeds of each fusion into mean ± spread.

Every widened-model result in this project so far came from a single training run (seed 0), so
a gap of a few tenths of a point between elementwise and cross-attention fusion could be seed
luck. Training each fusion three times (seeds 0, 1, 2) and reporting the spread shows how much
of any gap is real. Each run is evaluated by evaluate_generalist.py, which writes
checkpoints/<tag>_best_eval.json; this script reads those files and summarises them.

    python scripts/aggregate_seeds.py \\
        --elementwise gen_xcit gen_xcit_s1 gen_xcit_s2 \\
        --cross-attention gen_xcit_xattn gen_xcit_xattn_s1 gen_xcit_xattn_s2

Runs that haven't finished yet can simply be left off the lists; the output says how many
seeds each number is based on. With fewer than 2 seeds there is no spread to report.
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

CHECKPOINT_DIR = Path(__file__).resolve().parent.parent / "checkpoints"
TEST_SETS = ["lr:test", "hr:test", "hr:test_phili"]


def load_runs(tags: list[str], which: str) -> list[dict]:
    runs = []
    for tag in tags:
        path = CHECKPOINT_DIR / f"{tag}_{which}_eval.json"
        if not path.exists():
            raise SystemExit(f"Missing {path} — copy it from the cluster's checkpoints/ folder.")
        runs.append(json.loads(path.read_text(encoding="utf-8")))
    return runs


def summarise(values: list[float]) -> dict:
    return {
        "n_seeds": len(values),
        "mean": statistics.fmean(values),
        "std": statistics.stdev(values) if len(values) > 1 else None,
        "min": min(values),
        "max": max(values),
        "values": values,
    }


def summarise_fusion(runs: list[dict]) -> dict:
    out: dict = {}
    for test_set in TEST_SETS:
        metrics = [run[test_set] for run in runs]
        entry = {
            "overall_accuracy": summarise([m["overall_accuracy"] for m in metrics]),
            "average_accuracy": summarise([m["average_accuracy"] for m in metrics]),
            "per_type_accuracy": {},
        }
        for qtype in sorted(metrics[0]["per_type_accuracy"]):
            entry["per_type_accuracy"][qtype] = summarise(
                [m["per_type_accuracy"][qtype] for m in metrics]
            )
        out[test_set] = entry

    # Counting scored in the RSVQA paper's five ranges (written by evaluate_generalist.py);
    # absent from eval files made before that metric existed, so only summarised if present.
    counts = [run["lr:test_count_bucketed"] for run in runs if "lr:test_count_bucketed" in run]
    if counts:
        out["lr:count_bucketed"] = {
            "exact_match": summarise([c["exact_match_in_vocab_numeric_only"] for c in counts]),
            "bucketed": summarise([c["bucketed_numeric_only"] for c in counts]),
        }
        # The "sum the probability of each range" decoding, for the runs evaluated after it
        # was added (older files only have the usual-argmax figure above).
        mass = [c["bucketed_range_mass"] for c in counts if "bucketed_range_mass" in c]
        if mass:
            out["lr:count_bucketed"]["bucketed_range_mass"] = summarise(mass)
    return out


def fmt(stat: dict) -> str:
    spread = f" ± {100 * stat['std']:.2f}" if stat["std"] is not None else ""
    low, high = 100 * stat["min"], 100 * stat["max"]
    return f"{100 * stat['mean']:.2f}{spread}  (range {low:.2f} to {high:.2f})"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--elementwise", nargs="+", required=True, metavar="TAG")
    parser.add_argument("--cross-attention", nargs="+", required=True, metavar="TAG")
    parser.add_argument("--which", choices=["best", "final"], default="best")
    args = parser.parse_args()

    summary = {
        "elementwise": summarise_fusion(load_runs(args.elementwise, args.which)),
        "cross_attention": summarise_fusion(load_runs(args.cross_attention, args.which)),
    }

    for test_set in TEST_SETS:
        print(f"\n=== {test_set} ===")
        for fusion in ("elementwise", "cross_attention"):
            entry = summary[fusion][test_set]
            n = entry["overall_accuracy"]["n_seeds"]
            print(f"  {fusion} ({n} seeds)")
            print(f"    overall: {fmt(entry['overall_accuracy'])}")
            print(f"    average: {fmt(entry['average_accuracy'])}")
            for qtype, stat in entry["per_type_accuracy"].items():
                print(f"    {qtype:<9}: {fmt(stat)}")

    print("\n=== RSVQA-LR counting (exact match vs the paper's five ranges) ===")
    for fusion in ("elementwise", "cross_attention"):
        entry = summary[fusion].get("lr:count_bucketed")
        if entry is None:
            print(f"  {fusion}: no bucketed-count data in these eval files")
            continue
        print(f"  {fusion} ({entry['bucketed']['n_seeds']} seeds)")
        print(f"    exact match: {fmt(entry['exact_match'])}")
        print(f"    bucketed:    {fmt(entry['bucketed'])}")
        if "bucketed_range_mass" in entry:
            mass = entry["bucketed_range_mass"]
            print(f"    bucketed, summed per range ({mass['n_seeds']} seeds): {fmt(mass)}")

    out_path = CHECKPOINT_DIR / "multi_seed_summary.json"
    out_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\nWritten: {out_path}")


if __name__ == "__main__":
    main()
