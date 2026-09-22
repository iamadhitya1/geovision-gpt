"""Evaluate a checkpoint from train_generalist.py on multiple held-out test sets at once.

Three evaluations matter for the "does this generalise" question, and this script runs all
three from one checkpoint so they're directly comparable:

* lr:test        - RSVQA-LR's own held-out test (compare directly against the published
                    baseline: 72.74% overall / 75.16% average)
* hr:test         - RSVQA-HR's held-out test, same cities/distribution as its training data
* hr:test_phili   - RSVQA-HR's Philadelphia test set: a city the model never saw in training.
                    This is the one real generalisation number this project has.

Uses the checkpoint's OWN answer vocabulary (<tag>_answer_vocab.json) — it was built jointly
from LR+HR training data, not the original LR-only vocabulary, so results from this model are
not directly comparable answer-for-answer with the original xcit_nano_elementwise_epoch9.pt,
only via the accuracy percentages.

    python scripts/evaluate_generalist.py --tag gen_xcit --which best
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dataset.rsvqa_lr_dataset import AnswerVocabulary
from dataset.vqa_examples_dataset import VqaExamplesDataset, in_vocabulary
from models.vqa_model import LitRSVQAModel, build_image_transform, build_tokenizer
from scripts.train import evaluate, make_collate_fn, resolve_device
from scripts.train_generalist import CHECKPOINT_DIR, load_examples

EVAL_SPECS = [("lr", "test"), ("hr", "test"), ("hr", "test_phili")]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tag", required=True)
    parser.add_argument("--which", choices=["best", "final", "latest"], default="best")
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()

    device = resolve_device()
    vocab = AnswerVocabulary.from_json(CHECKPOINT_DIR / f"{args.tag}_answer_vocab.json")
    checkpoint = torch.load(
        CHECKPOINT_DIR / f"{args.tag}_{args.which}.pt", map_location=device, weights_only=False
    )
    config = checkpoint["config"]
    model = LitRSVQAModel(config, fusion=checkpoint["fusion"]).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    print(
        f"Loaded {args.tag}_{args.which}.pt — {config.image_encoder} + {checkpoint['fusion']} "
        f"fusion, {model.num_parameters():,} params, from epoch {checkpoint['state']['epoch']}, "
        f"vocab size {len(vocab)}"
    )

    tokenizer = build_tokenizer()
    collate = make_collate_fn(tokenizer, build_image_transform(config.image_encoder))

    results = {}
    for dataset_name, split in EVAL_SPECS:
        examples = in_vocabulary(load_examples(dataset_name, split), vocab)
        loader = DataLoader(
            VqaExamplesDataset(examples, vocab),
            batch_size=args.batch_size,
            shuffle=False,
            collate_fn=collate,
            num_workers=3,
        )
        metrics = evaluate(model, loader, device)
        key = f"{dataset_name}:{split}"
        results[key] = metrics
        print(f"\n=== {key} ({metrics['num_examples']} questions) ===")
        print(f"  overall accuracy: {metrics['overall_accuracy']:.4f}")
        print(f"  average accuracy: {metrics['average_accuracy']:.4f}")
        print(f"  per-type: {json.dumps(metrics['per_type_accuracy'])}")

    out_path = CHECKPOINT_DIR / f"{args.tag}_{args.which}_eval.json"
    out_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nWritten: {out_path}")


if __name__ == "__main__":
    main()
