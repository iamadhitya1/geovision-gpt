"""Train the small VQA model on RSVQA-LR plus RSVQA-HR, with augmentation. Resumable.

This is the "make it generalise" trainer. ``scripts/train.py`` is left untouched so the paper's
baseline stays exactly reproducible; this script reuses its collate/evaluate code and adds:

* more, more varied data: RSVQA-LR (10 m Sentinel-2) mixed with a fixed number of RSVQA-HR
  (15 cm aerial) questions per epoch, so one epoch has a predictable length
* label-preserving augmentation (``dataset/augment.py``) against the brittleness that
  ``scripts/robustness_eval.py`` measured
* a time budget and resume: the cluster kills jobs after 4 hours, so the script stops itself
  cleanly before that, saves its full state, and picks up from the same step when run again
* optional bfloat16 mixed precision on the GPU

    python scripts/train_generalist.py --tag gen_xcit --datasets lr+hr --epochs 8 --amp
    (run it again with the same arguments to resume; exit code 3 means "not finished yet")
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
import sys
import time
from pathlib import Path

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import torch
from torch import nn
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dataset.augment import TrainAugmentation
from dataset.rsvqa_lr_dataset import AnswerVocabulary, VqaExample, load_split_examples
from dataset.vqa_examples_dataset import VqaExamplesDataset, in_vocabulary
from models.count_loss import build_range_table, range_mass_loss
from models.vqa_model import (
    ImageEncoderName,
    LitRSVQAModel,
    VqaModelConfig,
    build_image_transform,
    build_tokenizer,
)
from scripts.train import evaluate, make_collate_fn, resolve_device

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)

ML_ROOT = Path(__file__).resolve().parent.parent
LR_ROOT = ML_ROOT / "dataset" / "rsvqa_lr"
HR_ROOT = ML_ROOT / "dataset" / "rsvqa_hr"
CHECKPOINT_DIR = ML_ROOT / "checkpoints"
EXIT_NOT_FINISHED = 3


def load_examples(name: str, split: str) -> list[VqaExample]:
    if name == "lr":
        return load_split_examples(LR_ROOT, split)  # type: ignore[arg-type]
    from dataset.rsvqa_hr_dataset import load_hr_split_examples

    return load_hr_split_examples(HR_ROOT, split)


def reseed_worker(_worker_id: int) -> None:
    """Give each data-loader worker (and each epoch) its own augmentation randomness."""
    info = torch.utils.data.get_worker_info()
    augmentation = getattr(info.dataset, "augment", None) if info else None
    if augmentation is not None:
        augmentation.reseed(torch.initial_seed())


def save_atomically(payload: dict, path: Path) -> None:
    """A job killed mid-write must not leave a corrupt checkpoint that breaks the resume."""
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def epoch_plan(
    lr_train: list[VqaExample],
    hr_train: list[VqaExample],
    hr_per_epoch: int,
    epoch: int,
    seed: int,
) -> list[VqaExample]:
    """The examples of one epoch, in a deterministic order for that (seed, epoch) — which is
    what lets a resumed run skip exactly the steps it already did."""
    examples = list(lr_train)
    if hr_train:
        chooser = random.Random(seed * 1000 + epoch)
        examples += chooser.sample(hr_train, min(hr_per_epoch, len(hr_train)))
    random.Random(seed * 1000 + epoch + 500).shuffle(examples)
    return examples


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tag", required=True, help="prefix for this run's checkpoint files")
    parser.add_argument("--datasets", choices=["lr", "lr+hr"], default="lr+hr")
    parser.add_argument(
        "--image-encoder", choices=["xcit_nano", "deit_tiny", "mobilevit_s"], default="xcit_nano"
    )
    parser.add_argument(
        "--fusion", choices=["elementwise", "cross_attention"], default="elementwise"
    )
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--hr-per-epoch", type=int, default=60000)
    parser.add_argument("--hr-val-size", type=int, default=4000)
    parser.add_argument("--vocab-size", type=int, default=1000)
    parser.add_argument("--no-augment", action="store_true")
    parser.add_argument(
        "--count-loss-weight",
        type=float,
        default=0.0,
        help="weight of the counting-aware loss (models/count_loss.py); 0 = off, the default",
    )
    parser.add_argument("--amp", action="store_true", help="bfloat16 mixed precision (CUDA only)")
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--max-minutes", type=float, default=215.0, help="stop cleanly after this")
    parser.add_argument("--save-every-minutes", type=float, default=10.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--smoke", action="store_true", help="tiny CPU run to test the loop")
    parser.add_argument("--stop-after-steps", type=int, default=0, help="test hook for resuming")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = resolve_device()
    use_amp = args.amp and device.type == "cuda"
    CHECKPOINT_DIR.mkdir(exist_ok=True)
    latest_path = CHECKPOINT_DIR / f"{args.tag}_latest.pt"
    best_path = CHECKPOINT_DIR / f"{args.tag}_best.pt"
    vocab_path = CHECKPOINT_DIR / f"{args.tag}_answer_vocab.json"
    history_path = CHECKPOINT_DIR / f"{args.tag}_history.json"

    names = args.datasets.split("+")
    train_pools = {n: load_examples(n, "train") for n in names}
    val_pools = {n: load_examples(n, "val") for n in names}

    if vocab_path.exists():
        vocab = AnswerVocabulary.from_json(vocab_path)
    else:
        vocab = AnswerVocabulary.build(
            [e for pool in train_pools.values() for e in pool], max_size=args.vocab_size
        )
        vocab.to_json(vocab_path)
    logger.info("Answer vocabulary: %d classes (%s)", len(vocab), vocab_path.name)

    train_pools = {n: in_vocabulary(p, vocab) for n, p in train_pools.items()}
    val_pools = {n: in_vocabulary(p, vocab) for n, p in val_pools.items()}
    if "hr" in val_pools:
        # A fixed subset: the full HR validation set is too slow to run every epoch.
        val_pools["hr"] = random.Random(1).sample(
            val_pools["hr"], min(args.hr_val_size, len(val_pools["hr"]))
        )
    if args.smoke:
        train_pools = {n: p[:64] for n, p in train_pools.items()}
        val_pools = {n: p[:32] for n, p in val_pools.items()}
        args.hr_per_epoch = 32

    lr_train = train_pools["lr"]
    hr_train = train_pools.get("hr", [])
    logger.info(
        "Training pools: LR %d questions, HR %d (using %d per epoch)",
        len(lr_train),
        len(hr_train),
        min(args.hr_per_epoch, len(hr_train)),
    )

    encoder: ImageEncoderName = args.image_encoder
    tokenizer = build_tokenizer()
    collate = make_collate_fn(tokenizer, build_image_transform(encoder))
    augment = None if args.no_augment else TrainAugmentation(rng=random.Random(args.seed))

    config = VqaModelConfig(image_encoder=encoder, num_answers=len(vocab))
    model = LitRSVQAModel(config, fusion=args.fusion).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()
    range_table = build_range_table(len(vocab), vocab.decode)
    if args.count_loss_weight > 0:
        logger.info("Counting-aware loss ON, weight %.3f", args.count_loss_weight)
    logger.info(
        "Model: %s + BERT-Tiny + %s fusion, %s parameters, amp=%s, augment=%s",
        encoder,
        args.fusion,
        f"{model.num_parameters():,}",
        use_amp,
        augment is not None,
    )

    state = {"epoch": 0, "step": 0, "history": [], "best_score": -1.0, "minutes": 0.0}
    if latest_path.exists():
        saved = torch.load(latest_path, map_location=device, weights_only=False)
        model.load_state_dict(saved["model_state_dict"])
        optimizer.load_state_dict(saved["optimizer_state_dict"])
        state = saved["state"]
        logger.info(
            "RESUMING at epoch %d step %d (%.1f minutes already trained)",
            state["epoch"],
            state["step"],
            state["minutes"],
        )

    def snapshot() -> dict:
        return {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "state": state,
            "config": config,
            "fusion": args.fusion,
        }

    def validate() -> dict[str, dict]:
        results = {}
        for name, pool in val_pools.items():
            loader = DataLoader(
                VqaExamplesDataset(pool, vocab),
                batch_size=args.batch_size,
                shuffle=False,
                collate_fn=collate,
                num_workers=args.num_workers,
            )
            with torch.autocast(device.type, dtype=torch.bfloat16, enabled=use_amp):
                results[name] = evaluate(model, loader, device)
        return results

    minutes_before = state["minutes"]
    run_started = time.time()
    last_saved = run_started
    steps_this_run = 0

    def minutes_this_run() -> float:
        return (time.time() - run_started) / 60

    while state["epoch"] < args.epochs:
        epoch = state["epoch"]
        examples = epoch_plan(lr_train, hr_train, args.hr_per_epoch, epoch, args.seed)
        first_index = state["step"] * args.batch_size
        loader = DataLoader(
            VqaExamplesDataset(examples, vocab, augment),
            batch_size=args.batch_size,
            sampler=list(range(first_index, len(examples))),
            collate_fn=collate,
            num_workers=args.num_workers,
            worker_init_fn=reseed_worker,
            pin_memory=device.type == "cuda",
        )
        model.train()
        steps_total = (len(examples) + args.batch_size - 1) // args.batch_size
        running, counted = 0.0, 0

        for batch in loader:
            optimizer.zero_grad()
            with torch.autocast(device.type, dtype=torch.bfloat16, enabled=use_amp):
                logits = model(
                    batch["pixel_values"].to(device),
                    batch["input_ids"].to(device),
                    batch["attention_mask"].to(device),
                )
                labels = batch["labels"].to(device)
                loss = criterion(logits.float(), labels)
                if args.count_loss_weight > 0:
                    is_count = torch.tensor([t == "count" for t in batch["question_types"]])
                    loss = loss + args.count_loss_weight * range_mass_loss(
                        logits.float(), labels, is_count, range_table
                    )
            loss.backward()
            optimizer.step()

            state["step"] += 1
            steps_this_run += 1
            running += loss.item()
            counted += 1
            if state["step"] % 50 == 0:
                logger.info(
                    "epoch %d step %d/%d loss=%.4f (%.1f min this run)",
                    epoch,
                    state["step"],
                    steps_total,
                    running / counted,
                    minutes_this_run(),
                )
                running, counted = 0.0, 0

            out_of_time = minutes_this_run() >= args.max_minutes
            test_stop = args.stop_after_steps and steps_this_run >= args.stop_after_steps
            if (
                out_of_time
                or test_stop
                or (time.time() - last_saved) / 60 >= args.save_every_minutes
            ):
                state["minutes"] = minutes_before + minutes_this_run()
                save_atomically(snapshot(), latest_path)
                last_saved = time.time()
            if out_of_time or test_stop:
                logger.info(
                    "STOPPING at epoch %d step %d — state saved to %s. Run again to resume.",
                    epoch,
                    state["step"],
                    latest_path.name,
                )
                sys.exit(EXIT_NOT_FINISHED)

        # ---- end of epoch: validate, record, checkpoint --------------------------------
        results = validate()
        score = sum(r["average_accuracy"] for r in results.values()) / len(results)
        state["history"].append(
            {"epoch": epoch, "score": score, **{n: r for n, r in results.items()}}
        )
        logger.info(
            "epoch %d done — %s",
            epoch,
            "  ".join(
                f"{n}: overall {r['overall_accuracy']:.4f} avg {r['average_accuracy']:.4f}"
                for n, r in results.items()
            ),
        )
        for name, r in results.items():
            logger.info("  %s per-type: %s", name, json.dumps(r["per_type_accuracy"]))

        state["minutes"] = minutes_before + minutes_this_run()
        state["epoch"] += 1
        state["step"] = 0
        if score > state["best_score"]:
            state["best_score"] = score
            save_atomically(snapshot() | {"epoch": epoch}, best_path)
            logger.info("  new best (validation score %.4f) -> %s", score, best_path.name)
        save_atomically(snapshot(), latest_path)
        history_path.write_text(json.dumps(state["history"], indent=2), encoding="utf-8")
        last_saved = time.time()

    final_path = CHECKPOINT_DIR / f"{args.tag}_final.pt"
    save_atomically(snapshot(), final_path)
    logger.info(
        "Training complete. Final: %s   Best-by-validation: %s", final_path.name, best_path.name
    )


if __name__ == "__main__":
    main()
