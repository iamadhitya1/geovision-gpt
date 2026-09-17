"""Build the answer vocabulary from the training split and save it to disk.

Run once, before training: ``python scripts/build_vocab.py``. The saved vocabulary is then
loaded by both the training script and the evaluation script, so train/val/test all agree
on which integer means which answer string.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dataset.rsvqa_lr_dataset import AnswerVocabulary, load_split_examples

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

DATA_ROOT = Path(__file__).resolve().parent.parent / "dataset" / "rsvqa_lr"
VOCAB_PATH = DATA_ROOT / "answer_vocab.json"


def main() -> None:
    train_examples = load_split_examples(DATA_ROOT, "train")
    vocab = AnswerVocabulary.build(train_examples)
    vocab.to_json(VOCAB_PATH)

    logger.info(
        "Built vocabulary: %d answers, from %d training examples.", len(vocab), len(train_examples)
    )
    logger.info("Saved to %s", VOCAB_PATH)

    # Sanity-report coverage on val/test — how many of their answers actually appear in a
    # vocabulary built only from train. This number belongs in the report as a stated
    # limitation if it's not close to 100%.
    for split in ("val", "test"):
        examples = load_split_examples(DATA_ROOT, split)
        covered = sum(1 for ex in examples if vocab.encode(ex.answer) is not None)
        logger.info(
            "%s coverage: %d / %d examples (%.1f%%) have an answer in the train vocabulary.",
            split,
            covered,
            len(examples),
            100 * covered / len(examples),
        )


if __name__ == "__main__":
    main()
