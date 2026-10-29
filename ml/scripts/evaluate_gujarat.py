"""Score a model on the Gujarat tiles of one split (use ``test``: the held-out districts, once).

    python scripts/evaluate_gujarat.py --tag gen_guj --split test
    # the old model, for a before/after on the same tiles:
    python scripts/evaluate_gujarat.py --split test \\
        --checkpoint ../backend/checkpoints/xcit_nano_elementwise_generalist.pt \\
        --vocab ../backend/checkpoints/answer_vocab_generalist.json

The pass rule was written before any Gujarat model existed, so it cannot be bent to the result:

* water, forest and rural/urban: **balanced accuracy of at least 80%**: the mean of the rate of
  correct "yes" (or "urban") and the rate of correct "no" (or "rural"), so a model that always
  says the same thing scores 50% however common that answer is;
* building count, as one of the five ranges: above the accuracy of always naming the commonest
  range.

Presence of a building is reported but not part of the rule (93% of tiles have one).

Everything is scored against WorldCover, JRC surface water, GHSL and Google Open Buildings, which
are model-made labels, not ground truth. The tiles drawn uniformly over a district are reported
separately, because the oversampled urban and water tiles change how common each answer is.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dataset.count_ranges import RANGE_LABELS, range_index
from dataset.gujarat_dataset import (
    BUILDING,
    COUNT,
    FOREST,
    GUJARAT_ROOT,
    RURAL_URBAN,
    TRUTH,
    WATER,
    read_tiles,
)
from dataset.rsvqa_lr_dataset import AnswerVocabulary
from models.vqa_model import LitRSVQAModel, build_image_transform, build_tokenizer
from scripts.train import make_collate_fn, resolve_device

CHECKPOINT_DIR = Path(__file__).resolve().parent.parent / "checkpoints"
BALANCED_TARGET = 0.80
# question -> (the answer that counts as "positive", is it part of the pass rule)
BINARY = {
    WATER: ("yes", True),
    FOREST: ("yes", True),
    RURAL_URBAN: ("urban", True),
    BUILDING: ("yes", False),
}


class Questions(Dataset):
    """One (tile, question) per item; the label field is unused here but the collate needs it."""

    def __init__(self, items: list[tuple[dict, str]], root: Path) -> None:
        self._items, self._root = items, root

    def __len__(self) -> int:
        return len(self._items)

    def __getitem__(self, index: int) -> dict:
        tile, question = self._items[index]
        with Image.open(self._root / "images" / f"{tile['id']}.png") as opened:
            image = opened.convert("RGB")
        return {"image": image, "question": question, "label": 0, "question_type": "x"}


def predict(model, vocab, tokenizer, device, items, root) -> list[torch.Tensor]:
    """Logits for every (tile, question), in order."""
    loader = DataLoader(
        Questions(items, root),
        batch_size=64,
        shuffle=False,
        collate_fn=make_collate_fn(tokenizer, build_image_transform(model.config.image_encoder)),
        num_workers=0,
    )
    out: list[torch.Tensor] = []
    model.eval()
    with torch.no_grad():
        for batch in loader:
            logits = model(
                batch["pixel_values"].to(device),
                batch["input_ids"].to(device),
                batch["attention_mask"].to(device),
            ).cpu()
            out.extend(logits)
    return out


def binary_report(truth: list[str], said: list[str], positive: str) -> dict:
    tp = sum(t == positive and s == positive for t, s in zip(truth, said, strict=True))
    fn = sum(t == positive and s != positive for t, s in zip(truth, said, strict=True))
    tn = sum(t != positive and s != positive for t, s in zip(truth, said, strict=True))
    fp = sum(t != positive and s == positive for t, s in zip(truth, said, strict=True))
    sensitivity = tp / (tp + fn) if tp + fn else float("nan")
    specificity = tn / (tn + fp) if tn + fp else float("nan")
    n = len(truth)
    return {
        "n": n,
        "accuracy": (tp + tn) / n,
        "sensitivity": sensitivity,
        "specificity": specificity,
        "balanced_accuracy": (sensitivity + specificity) / 2,
        "commonest_label_accuracy": max(Counter(truth).values()) / n,
        "said_positive": (tp + fp) / n,
        "label_positive": (tp + fn) / n,
    }


def evaluate(model, vocab, tokenizer, device, tiles, root) -> dict:
    numeric_ids = [i for i in range(len(vocab)) if vocab.decode(i).isdigit()]
    ids_in_range = [
        torch.tensor([i for i in numeric_ids if range_index(int(vocab.decode(i))) == r])
        for r in range(5)
    ]
    mask = torch.full((len(vocab),), float("-inf"))
    mask[numeric_ids] = 0.0

    items = [(t, q) for t in tiles for q in (*BINARY, COUNT)]
    logits = predict(model, vocab, tokenizer, device, items, root)
    said: dict[str, list[str]] = {q: [] for q in (*BINARY, COUNT)}
    truth: dict[str, list[str]] = {q: [] for q in (*BINARY, COUNT)}
    for (tile, question), row in zip(items, logits, strict=True):
        if question == COUNT:
            probs = torch.softmax(row + mask, dim=-1)  # the app's range-mass decoding
            mass = torch.stack([probs[ids].sum() for ids in ids_in_range])
            said[question].append(RANGE_LABELS[int(mass.argmax())])
            truth[question].append(RANGE_LABELS[range_index(int(tile["buildings"]))])
        else:
            said[question].append(vocab.decode(int(row.argmax())))
            truth[question].append(TRUTH[question](tile))

    report = {q: binary_report(truth[q], said[q], BINARY[q][0]) for q in BINARY}
    n = len(truth[COUNT])
    report[COUNT] = {
        "n": n,
        "accuracy": sum(a == b for a, b in zip(said[COUNT], truth[COUNT], strict=True)) / n,
        "commonest_label_accuracy": max(Counter(truth[COUNT]).values()) / n,
        "said": dict(Counter(said[COUNT])),
        "label": dict(Counter(truth[COUNT])),
    }
    return report


def show(title: str, report: dict) -> bool:
    print(f"\n{title}")
    passed = True
    for question, (_, in_rule) in BINARY.items():
        r = report[question]
        ok = r["balanced_accuracy"] >= BALANCED_TARGET
        flag = ("PASS" if ok else "FAIL") if in_rule else "(not in the rule)"
        passed &= ok or not in_rule
        print(
            f"  {question:34s} balanced {r['balanced_accuracy']:.0%}  "
            f"[yes-rate right {r['sensitivity']:.0%}, no-rate right {r['specificity']:.0%}]  "
            f"accuracy {r['accuracy']:.0%} vs commonest {r['commonest_label_accuracy']:.0%}  "
            f"said-positive {r['said_positive']:.0%} / label {r['label_positive']:.0%}  {flag}"
        )
    c = report[COUNT]
    ok = c["accuracy"] > c["commonest_label_accuracy"]
    passed &= ok
    print(
        f"  {COUNT:34s} in-range {c['accuracy']:.0%} vs always-commonest "
        f"{c['commonest_label_accuracy']:.0%}  {'PASS' if ok else 'FAIL'}"
    )
    print(f"    said {c['said']}\n    label {c['label']}")
    return passed


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--tag", help="a run's tag: uses checkpoints/<tag>_best.pt")
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--vocab", type=Path)
    parser.add_argument("--split", choices=["train", "val", "test"], default="test")
    parser.add_argument("--root", type=Path, default=GUJARAT_ROOT)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()
    checkpoint = args.checkpoint or CHECKPOINT_DIR / f"{args.tag}_best.pt"
    vocab_path = args.vocab or CHECKPOINT_DIR / f"{args.tag}_answer_vocab.json"

    device = resolve_device()
    saved = torch.load(checkpoint, map_location=device, weights_only=False)
    model = LitRSVQAModel(saved["config"], fusion=saved["fusion"]).to(device)
    model.load_state_dict(saved["model_state_dict"])
    vocab = AnswerVocabulary.from_json(vocab_path)
    tokenizer = build_tokenizer()

    tiles = read_tiles(args.root, args.split)
    print(f"{checkpoint.name} ({saved['fusion']}) on {len(tiles)} {args.split} tiles")
    everything = evaluate(model, vocab, tokenizer, device, tiles, args.root)
    uniform = [t for t in tiles if t["strategy"] == "uniform"]
    base_rates = evaluate(model, vocab, tokenizer, device, uniform, args.root)

    passed = show(f"All {args.split} tiles (urban and water oversampled)", everything)
    show(f"Uniform tiles only ({len(uniform)}): true base rates", base_rates)
    print(f"\nPASS RULE (all {args.split} tiles): {'MET' if passed else 'NOT MET'}")
    if args.out:
        args.out.write_text(
            json.dumps({"all": everything, "uniform": base_rates, "passed": passed}, indent=2),
            encoding="utf-8",
        )


if __name__ == "__main__":
    main()
