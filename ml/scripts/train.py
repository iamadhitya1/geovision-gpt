"""Fine-tune the LiT-4-RSVQA-style model on RSVQA-LR.

Runs identically on CPU or GPU — device selection is automatic (see ``resolve_device``
below). On this dev machine it runs on CPU, useful for validating the pipeline end to end
and for very short smoke runs; real training happens once institute HPC access lands, by
running this exact script with ``--epochs`` set to a real value and nothing else changed.

Usage:
    python scripts/train.py --smoke                 # ~30 examples, 1 epoch, proves the
                                                      # loop runs correctly, nothing more
    python scripts/train.py --epochs 10              # a real run, once on a real GPU
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dataset.rsvqa_lr_dataset import AnswerVocabulary, RSVQALowResDataset
from models.vqa_model import (
    ImageEncoderName,
    LitRSVQAModel,
    VqaModelConfig,
    build_image_transform,
    build_tokenizer,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)

DATA_ROOT = Path(__file__).resolve().parent.parent / "dataset" / "rsvqa_lr"
VOCAB_PATH = DATA_ROOT / "answer_vocab.json"
CHECKPOINT_DIR = Path(__file__).resolve().parent.parent / "checkpoints"


def resolve_device() -> torch.device:
    """The one line that changes behaviour between this dev machine and the institute GPU
    — everything else in this file is device-agnostic."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    logger.warning(
        "No CUDA device found — running on CPU. Fine for smoke-testing the "
        "pipeline; a real training run should happen on the institute HPC."
    )
    return torch.device("cpu")


def make_collate_fn(tokenizer, image_transform):
    """Applies tokenization and image preprocessing per-batch, not per-example — this
    lets padding be computed from the actual longest question in each batch rather than a
    fixed worst-case length, and keeps ``RSVQALowResDataset`` itself free of any dependency
    on which pretrained checkpoint is in use."""

    def collate(batch: list[dict]) -> dict:
        images = torch.stack([image_transform(item["image"]) for item in batch])
        encoded = tokenizer(
            [item["question"] for item in batch], padding=True, truncation=True, return_tensors="pt"
        )
        labels = torch.tensor([item["label"] for item in batch], dtype=torch.long)
        return {
            "pixel_values": images,
            "input_ids": encoded["input_ids"],
            "attention_mask": encoded["attention_mask"],
            "labels": labels,
            "question_types": [item["question_type"] for item in batch],
        }

    return collate


def evaluate(model: LitRSVQAModel, loader: DataLoader, device: torch.device) -> dict:
    model.eval()
    correct = 0
    total = 0
    per_type_correct: dict[str, int] = {}
    per_type_total: dict[str, int] = {}

    with torch.no_grad():
        for batch in loader:
            pixel_values = batch["pixel_values"].to(device)
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            logits = model(pixel_values, input_ids, attention_mask)
            predictions = logits.argmax(dim=-1)
            is_correct = predictions == labels

            correct += int(is_correct.sum())
            total += len(labels)

            for qtype, ok in zip(batch["question_types"], is_correct.tolist(), strict=True):
                per_type_total[qtype] = per_type_total.get(qtype, 0) + 1
                per_type_correct[qtype] = per_type_correct.get(qtype, 0) + int(ok)

    per_type_accuracy = {
        qtype: per_type_correct.get(qtype, 0) / count for qtype, count in per_type_total.items()
    }
    # Average accuracy (AA): the macro average across question types — matches how the
    # LiT-4-RSVQA paper reports it, so results are directly comparable to Table 1 there.
    average_accuracy = (
        sum(per_type_accuracy.values()) / len(per_type_accuracy) if per_type_accuracy else 0.0
    )

    return {
        "overall_accuracy": correct / total if total else 0.0,
        "average_accuracy": average_accuracy,
        "per_type_accuracy": per_type_accuracy,
        "num_examples": total,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--image-encoder", choices=["xcit_nano", "deit_tiny", "mobilevit_s"], default="xcit_nano"
    )
    parser.add_argument(
        "--fusion", choices=["elementwise", "cross_attention"], default="elementwise"
    )
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=5e-4)  # matches the paper's reported LR
    parser.add_argument(
        "--smoke",
        action="store_true",
        help=(
            "Tiny CPU run: ~30 train / 20 val examples, proves the pipeline "
            "works. Not a real training run."
        ),
    )
    args = parser.parse_args()

    device = resolve_device()
    CHECKPOINT_DIR.mkdir(exist_ok=True)

    if not VOCAB_PATH.exists():
        raise FileNotFoundError(f"{VOCAB_PATH} not found — run scripts/build_vocab.py first.")
    vocab = AnswerVocabulary.from_json(VOCAB_PATH)

    train_dataset = RSVQALowResDataset(DATA_ROOT, "train", vocab)
    val_dataset = RSVQALowResDataset(DATA_ROOT, "val", vocab)

    if args.smoke:
        from torch.utils.data import Subset

        train_dataset = Subset(train_dataset, range(min(32, len(train_dataset))))
        val_dataset = Subset(val_dataset, range(min(20, len(val_dataset))))
        args.epochs = 1
        logger.info(
            "Smoke mode: %d train / %d val examples, 1 epoch.", len(train_dataset), len(val_dataset)
        )

    image_encoder: ImageEncoderName = args.image_encoder
    tokenizer = build_tokenizer()
    image_transform = build_image_transform(image_encoder)
    collate_fn = make_collate_fn(tokenizer, image_transform)

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn
    )

    config = VqaModelConfig(image_encoder=image_encoder, num_answers=len(vocab))
    model = LitRSVQAModel(config)
    if args.fusion == "cross_attention":
        from models.vqa_model import CrossAttentionFusion

        model.fusion = CrossAttentionFusion(
            model.image_encoder.output_dim, model.text_encoder.output_dim, config.fusion_dim
        )
    model.to(device)

    logger.info(
        "Model: %s image encoder + BERT-Tiny + %s fusion — %s total params (%s trainable)",
        image_encoder,
        args.fusion,
        f"{model.num_parameters():,}",
        f"{model.num_parameters(trainable_only=True):,}",
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(args.epochs):
        model.train()
        epoch_start = time.time()
        running_loss = 0.0

        for step, batch in enumerate(train_loader):
            pixel_values = batch["pixel_values"].to(device)
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            optimizer.zero_grad()
            logits = model(pixel_values, input_ids, attention_mask)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if step % 20 == 0:
                logger.info("epoch %d step %d: loss=%.4f", epoch, step, loss.item())

        avg_loss = running_loss / max(1, len(train_loader))
        val_metrics = evaluate(model, val_loader, device)
        logger.info(
            "epoch %d done in %.1fs — train_loss=%.4f val_overall_acc=%.4f val_avg_acc=%.4f",
            epoch,
            time.time() - epoch_start,
            avg_loss,
            val_metrics["overall_accuracy"],
            val_metrics["average_accuracy"],
        )
        logger.info(
            "  per-type accuracy: %s", json.dumps(val_metrics["per_type_accuracy"], indent=2)
        )

        checkpoint_path = CHECKPOINT_DIR / f"{image_encoder}_{args.fusion}_epoch{epoch}.pt"
        torch.save(
            {"model_state_dict": model.state_dict(), "config": config, "epoch": epoch},
            checkpoint_path,
        )
        logger.info("  saved checkpoint: %s", checkpoint_path)

    logger.info("Training complete.")


if __name__ == "__main__":
    main()
