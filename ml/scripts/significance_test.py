"""Is the elementwise-vs-cross-attention "tie" actually noise, or a real difference?

Every result in this project reports elementwise and cross-attention fusion as "effectively
tied" — e.g. 72.02% vs 71.96% overall on RSVQA-LR's test split for the widened models. That
claim has so far rested on intuition ("a gap this small, with each model trained once,
shouldn't be read as evidence"), not a measurement. This script makes it a measurement.

It runs both of the currently-deployed checkpoints (backend/checkpoints/) over RSVQA-LR's
held-out test split, recording a per-question correct/incorrect outcome for each model in
identical order (both share one answer vocabulary, built jointly before training — see
train_generalist.py's "reuse vocab if it already exists" logic — so the vocabulary filter
produces the exact same question set in the exact same order for both models; this is what
makes a *paired* test valid here).

Two tests, answering slightly different questions:

* McNemar's exact test — of the two fusions, how often does model-A get a question right
  that model-B gets wrong, and vice versa? If that split is close to 50/50, the tie is real;
  if one direction dominates, the "tie" is hiding a real effect.
* Bootstrap confidence interval on the accuracy gap — resamples the paired questions 10,000
  times and reports where 95% of the resampled accuracy differences fall. If that interval
  contains 0, the measured gap is consistent with noise.

    python scripts/significance_test.py
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch
from scipy.stats import binomtest
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dataset.rsvqa_lr_dataset import AnswerVocabulary
from dataset.vqa_examples_dataset import VqaExamplesDataset, in_vocabulary
from models.vqa_model import LitRSVQAModel, build_image_transform, build_tokenizer
from scripts.train import make_collate_fn, resolve_device
from scripts.train_generalist import load_examples

BACKEND_CHECKPOINTS = Path(__file__).resolve().parent.parent.parent / "backend" / "checkpoints"
OUT_DIR = Path(__file__).resolve().parent.parent / "checkpoints"

MODELS = {
    "elementwise": BACKEND_CHECKPOINTS / "xcit_nano_elementwise_generalist.pt",
    "cross_attention": BACKEND_CHECKPOINTS / "xcit_nano_cross_attention_generalist.pt",
}
VOCAB_PATH = BACKEND_CHECKPOINTS / "answer_vocab_generalist.json"


def load_model(checkpoint_path: Path, device: torch.device) -> tuple[LitRSVQAModel, str]:
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = checkpoint["config"]
    fusion = checkpoint["fusion"]
    model = LitRSVQAModel(config, fusion=fusion).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, fusion


def per_question_correctness(
    model: LitRSVQAModel, loader: DataLoader, device: torch.device
) -> tuple[list[bool], list[str]]:
    """Returns (is_correct per question, question_type per question), in loader order."""
    correctness: list[bool] = []
    types: list[str] = []
    with torch.no_grad():
        for batch in loader:
            pixel_values = batch["pixel_values"].to(device)
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            logits = model(pixel_values, input_ids, attention_mask)
            predictions = logits.argmax(dim=-1)
            is_correct = (predictions == labels).tolist()
            correctness.extend(is_correct)
            types.extend(batch["question_types"])
    return correctness, types


def bootstrap_ci(
    a_correct: list[bool], b_correct: list[bool], n_resamples: int = 10000, seed: int = 0
) -> tuple[float, float, float]:
    """Paired bootstrap over questions. Returns (mean_diff, ci_low, ci_high) for
    accuracy(A) - accuracy(B), at the 95% level."""
    import random

    rng = random.Random(seed)
    n = len(a_correct)
    diffs = []
    for _ in range(n_resamples):
        idx = [rng.randrange(n) for _ in range(n)]
        a_acc = sum(a_correct[i] for i in idx) / n
        b_acc = sum(b_correct[i] for i in idx) / n
        diffs.append(a_acc - b_acc)
    diffs.sort()
    mean_diff = sum(diffs) / len(diffs)
    lo = diffs[int(0.025 * n_resamples)]
    hi = diffs[int(0.975 * n_resamples) - 1]
    return mean_diff, lo, hi


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--smoke", action="store_true", help="Tiny subset (40 Qs), to check the script runs."
    )
    args = parser.parse_args()

    device = resolve_device()
    vocab = AnswerVocabulary.from_json(VOCAB_PATH)
    examples = in_vocabulary(load_examples("lr", "test"), vocab)
    if args.smoke:
        examples = examples[:40]
    print(f"Test questions (in vocab): {len(examples)}")

    tokenizer = build_tokenizer()
    results: dict[str, tuple[list[bool], list[str]]] = {}

    for name, path in MODELS.items():
        t0 = time.time()
        model, fusion = load_model(path, device)
        collate = make_collate_fn(tokenizer, build_image_transform(model.config.image_encoder))
        loader = DataLoader(
            VqaExamplesDataset(examples, vocab),
            batch_size=64,
            shuffle=False,
            collate_fn=collate,
            # num_workers=0: this runs locally on Windows, where multiprocessing uses spawn
            # (not the cluster's Linux fork), and spawn can't pickle make_collate_fn's
            # closure. Single-process is fine here — the bottleneck is the model forward
            # pass, not data loading.
            num_workers=0,
        )
        correctness, types = per_question_correctness(model, loader, device)
        acc = sum(correctness) / len(correctness)
        elapsed = time.time() - t0
        print(f"{name} ({fusion}): {acc:.4%} overall, {len(correctness)} Qs, {elapsed:.0f}s")
        results[name] = (correctness, types)
        del model

    a_correct, q_types = results["elementwise"]
    b_correct, _ = results["cross_attention"]
    assert len(a_correct) == len(b_correct) == len(q_types)

    # McNemar's exact test: among questions where the two models disagree, is the split
    # between "A right, B wrong" and "A wrong, B right" close to 50/50?
    a_right_b_wrong = sum(1 for a, b in zip(a_correct, b_correct, strict=True) if a and not b)
    a_wrong_b_right = sum(1 for a, b in zip(a_correct, b_correct, strict=True) if not a and b)
    discordant = a_right_b_wrong + a_wrong_b_right
    mcnemar_p = (
        binomtest(a_right_b_wrong, discordant, p=0.5, alternative="two-sided").pvalue
        if discordant > 0
        else 1.0
    )

    mean_diff, ci_lo, ci_hi = bootstrap_ci(a_correct, b_correct)

    overall_a = sum(a_correct) / len(a_correct)
    overall_b = sum(b_correct) / len(b_correct)

    summary = {
        "n_questions": len(a_correct),
        "elementwise_overall_accuracy": overall_a,
        "cross_attention_overall_accuracy": overall_b,
        "measured_gap_elementwise_minus_cross_attention": overall_a - overall_b,
        "mcnemar": {
            "elementwise_right_cross_attention_wrong": a_right_b_wrong,
            "elementwise_wrong_cross_attention_right": a_wrong_b_right,
            "discordant_pairs": discordant,
            "p_value": mcnemar_p,
            "verdict": (
                "significant difference (p < 0.05)"
                if mcnemar_p < 0.05
                else "no significant difference (p >= 0.05) — consistent with a real tie"
            ),
        },
        "bootstrap_95ci": {
            "n_resamples": 10000,
            "mean_diff": mean_diff,
            "ci_low": ci_lo,
            "ci_high": ci_hi,
            "contains_zero": ci_lo <= 0 <= ci_hi,
        },
    }

    # Same two tests, restricted to each question type, for a more granular read.
    by_type: dict[str, dict] = {}
    for qtype in sorted(set(q_types)):
        idx = [i for i, t in enumerate(q_types) if t == qtype]
        a_t = [a_correct[i] for i in idx]
        b_t = [b_correct[i] for i in idx]
        rbw = sum(1 for a, b in zip(a_t, b_t, strict=True) if a and not b)
        wbr = sum(1 for a, b in zip(a_t, b_t, strict=True) if not a and b)
        disc = rbw + wbr
        p = binomtest(rbw, disc, p=0.5, alternative="two-sided").pvalue if disc > 0 else 1.0
        by_type[qtype] = {
            "n": len(idx),
            "elementwise_accuracy": sum(a_t) / len(a_t),
            "cross_attention_accuracy": sum(b_t) / len(b_t),
            "mcnemar_p_value": p,
        }
    summary["by_question_type"] = by_type

    print("\n=== Significance test: elementwise vs cross-attention (widened models) ===")
    print(json.dumps(summary, indent=2))

    out_name = (
        "significance_test_widened_smoke.json" if args.smoke else "significance_test_widened.json"
    )
    out_path = OUT_DIR / out_name
    out_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\nWritten: {out_path}")


if __name__ == "__main__":
    main()
