"""Evaluate a trained checkpoint on the RSVQA-LR test split, reporting results in the same
shape as Table 1 of the LiT-4-RSVQA paper — overall accuracy, average accuracy (macro over
question types), per-type accuracy, and parameter count — so the numbers in the project
report are directly comparable to a real published benchmark, not just internally
consistent.

Usage:
    python scripts/evaluate.py --checkpoint checkpoints/xcit_nano_elementwise_epoch9.pt
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dataset.rsvqa_lr_dataset import AnswerVocabulary, RSVQALowResDataset
from models.vqa_model import (
    CrossAttentionFusion,
    LitRSVQAModel,
    build_image_transform,
    build_tokenizer,
)
from scripts.train import evaluate, make_collate_fn, resolve_device

DATA_ROOT = Path(__file__).resolve().parent.parent / "dataset" / "rsvqa_lr"
VOCAB_PATH = DATA_ROOT / "answer_vocab.json"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--fusion", choices=["elementwise", "cross_attention"], default="elementwise"
    )
    parser.add_argument("--split", choices=["val", "test"], default="test")
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()

    device = resolve_device()
    vocab = AnswerVocabulary.from_json(VOCAB_PATH)

    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    config = checkpoint["config"]

    model = LitRSVQAModel(config)
    if args.fusion == "cross_attention":
        model.fusion = CrossAttentionFusion(
            model.image_encoder.output_dim, model.text_encoder.output_dim, config.fusion_dim
        )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)

    dataset = RSVQALowResDataset(DATA_ROOT, args.split, vocab)
    tokenizer = build_tokenizer()
    image_transform = build_image_transform(config.image_encoder)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=make_collate_fn(tokenizer, image_transform),
    )

    metrics = evaluate(model, loader, device)
    metrics["parameter_count"] = model.num_parameters()
    metrics["trainable_parameter_count"] = model.num_parameters(trainable_only=True)
    metrics["checkpoint"] = str(args.checkpoint)
    metrics["split"] = args.split

    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
