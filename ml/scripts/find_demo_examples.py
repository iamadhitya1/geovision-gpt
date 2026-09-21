"""Find test-split images and questions a trained checkpoint answers correctly and
confidently — the safe material for a live demo.

Only the held-out *test* split is used, so nothing shown was seen in training. The script
also lists confident WRONG answers per image: knowing those in advance is as useful as the
good ones (you avoid them on stage, and can explain them if asked).

    python scripts/find_demo_examples.py --checkpoint ../backend/checkpoints/xcit_nano_elementwise_epoch9.pt
"""

from __future__ import annotations

import argparse
import random
import sys
from collections import defaultdict
from pathlib import Path

import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dataset.rsvqa_lr_dataset import AnswerVocabulary, load_split_examples
from models.vqa_model import (
    LitRSVQAModel,
    build_image_transform,
    build_tokenizer,
)

DATA_ROOT = Path(__file__).resolve().parent.parent / "dataset" / "rsvqa_lr"
VOCAB_PATH = DATA_ROOT / "answer_vocab.json"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--fusion", choices=["elementwise", "cross_attention"], default="elementwise")
    parser.add_argument("--images", type=int, default=40, help="how many test images to sample")
    parser.add_argument("--per-image", type=int, default=16, help="questions sampled per image")
    parser.add_argument("--min-confidence", type=float, default=0.6)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    vocab = AnswerVocabulary.from_json(VOCAB_PATH)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    config = checkpoint["config"]
    model = LitRSVQAModel(config, fusion=args.fusion)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    tokenizer = build_tokenizer()
    transform = build_image_transform(config.image_encoder)

    by_image: dict[Path, list] = defaultdict(list)
    for example in load_split_examples(DATA_ROOT, "test"):
        by_image[example.image_path].append(example)

    rng = random.Random(args.seed)
    images = rng.sample(sorted(by_image), k=min(args.images, len(by_image)))

    report = []
    for path in images:
        questions = rng.sample(by_image[path], k=min(args.per_image, len(by_image[path])))
        with Image.open(path) as img:
            pixels = transform(img.convert("RGB")).unsqueeze(0).repeat(len(questions), 1, 1, 1)
        encoded = tokenizer(
            [q.question for q in questions], padding=True, truncation=True, return_tensors="pt"
        )
        with torch.no_grad():
            probs = torch.softmax(
                model(pixels, encoded["input_ids"], encoded["attention_mask"]), dim=-1
            )
        confidence, index = probs.max(dim=-1)

        rows = []
        for q, conf, idx in zip(questions, confidence.tolist(), index.tolist(), strict=True):
            predicted = vocab.decode(idx)
            rows.append((q, predicted, conf, predicted == q.answer))
        report.append((path, rows))

    def good(rows: list) -> list:
        return [r for r in rows if r[3] and r[2] >= args.min_confidence]

    ranked = sorted(report, key=lambda item: len(good(item[1])), reverse=True)
    for path, rows in ranked[:6]:
        print(f"\n=== {path.name}  ({len(good(rows))} confident-correct of {len(rows)} sampled)")
        for q, predicted, conf, correct in sorted(rows, key=lambda r: -r[2]):
            if correct and conf >= args.min_confidence:
                mark = "OK  "
            elif not correct and conf >= args.min_confidence:
                mark = "WRONG"
            else:
                continue
            print(f"  {mark} [{q.question_type:11s}] {q.question}  -> {predicted} ({conf:.0%})"
                  f"  truth: {q.answer}")


if __name__ == "__main__":
    main()
