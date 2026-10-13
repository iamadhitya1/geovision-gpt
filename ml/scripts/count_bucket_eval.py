"""How well do the models count, scored the way the RSVQA paper scores it?

RSVQA-LR's released labels are exact integers ("0", "7", "158", ...), and this project's
evaluation has always scored them by exact match — about 28% on count questions for both
widened models. But the RSVQA paper (Lobry et al., 2020, Section II-B) quantises
low-resolution counts into five ranges before scoring: 0 / 1-10 / 11-100 / 101-1000 / >1000,
and reports 67.01% count accuracy that way. The two numbers are therefore not comparable.
This module scores the same models on the same RSVQA-LR test count questions both ways, next
to constant-guess baselines, so the count number can be stated honestly.

For each model, two decodings are scored on the bucketed metric:

* unrestricted - the model's normal argmax over the whole 1,000-answer vocabulary
* numeric-only - argmax restricted to numeric answers (a count question should never be
                 answered "yes"; this is what a count-aware decoder would do)

``evaluate_counts`` is also called by ``evaluate_generalist.py``, so every evaluated
checkpoint records these numbers. Run this file directly to score the two deployed models:

    python scripts/count_bucket_eval.py
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dataset.count_ranges import RANGE_NAMES, range_index
from dataset.rsvqa_lr_dataset import AnswerVocabulary
from models.vqa_model import LitRSVQAModel, build_image_transform, build_tokenizer
from scripts.train import make_collate_fn, resolve_device
from scripts.train_generalist import load_examples

BACKEND_CHECKPOINTS = Path(__file__).resolve().parent.parent.parent / "backend" / "checkpoints"
OUT_PATH = Path(__file__).resolve().parent.parent / "checkpoints" / "count_bucket_eval.json"

MODELS = {
    "elementwise": BACKEND_CHECKPOINTS / "xcit_nano_elementwise_generalist.pt",
    "cross_attention": BACKEND_CHECKPOINTS / "xcit_nano_cross_attention_generalist.pt",
}
VOCAB_PATH = BACKEND_CHECKPOINTS / "answer_vocab_generalist.json"

BUCKET_NAMES = RANGE_NAMES
bucket = range_index  # the paper's five count ranges, shared with the web app


def count_examples(split: str = "test"):
    """All RSVQA-LR count questions of a split, including those whose exact answer falls
    outside the model's vocabulary (the usual evaluation drops them; the bucketed metric can
    keep them, because a large rare number still has a well-defined range)."""
    return [e for e in load_examples("lr", split) if e.question_type == "count"]


class CountQuestions(Dataset):
    """Label is -1 where the answer is outside the vocabulary (the collate function needs one)."""

    def __init__(self, examples, vocab: AnswerVocabulary) -> None:
        self._examples = examples
        self._vocab = vocab

    def __len__(self) -> int:
        return len(self._examples)

    def __getitem__(self, index: int) -> dict:
        ex = self._examples[index]
        with Image.open(ex.image_path) as opened:
            image = opened.convert("RGB")
        label = self._vocab.encode(ex.answer)
        return {
            "image": image,
            "question": ex.question,
            "question_type": ex.question_type,
            "answer": ex.answer,
            "label": -1 if label is None else label,
        }


def evaluate_counts(
    model: LitRSVQAModel,
    vocab: AnswerVocabulary,
    tokenizer,
    device: torch.device,
    examples,
    num_workers: int = 0,
) -> dict:
    """Score one model on RSVQA-LR test count questions, exact-match and paper-bucketed.

    num_workers defaults to 0 because Windows spawn can't pickle the collate closure; the
    cluster's Linux fork can use more.
    """
    truth_bucket = [bucket(int(e.answer)) for e in examples]
    n = len(examples)
    in_vocab = [vocab.encode(e.answer) is not None for e in examples]

    numeric_ids = [i for i in range(len(vocab)) if vocab.decode(i).isdigit()]
    numeric_mask = torch.full((len(vocab),), float("-inf"))
    numeric_mask[numeric_ids] = 0.0

    loader = DataLoader(
        CountQuestions(examples, vocab),
        batch_size=64,
        shuffle=False,
        collate_fn=make_collate_fn(tokenizer, build_image_transform(model.config.image_encoder)),
        num_workers=num_workers,
    )

    # Ids of the numeric answers falling in each range, for the probability-mass decoding below.
    ids_in_bucket = [
        torch.tensor(
            [i for i in numeric_ids if bucket(int(vocab.decode(i))) == b], dtype=torch.long
        )
        for b in range(5)
    ]

    unrestricted_ids: list[int] = []
    numeric_ids_pred: list[int] = []
    mass_bucket_pred: list[int] = []
    model.eval()
    with torch.no_grad():
        for batch in loader:
            logits = model(
                batch["pixel_values"].to(device),
                batch["input_ids"].to(device),
                batch["attention_mask"].to(device),
            ).cpu()
            unrestricted_ids.extend(logits.argmax(dim=-1).tolist())
            numeric_ids_pred.extend((logits + numeric_mask).argmax(dim=-1).tolist())
            # Range with the most total probability among numeric answers. The exact-number
            # argmax can pick "0" even when "somewhere in 11-100" holds far more probability,
            # because that probability is spread over ~100 different numbers.
            probs = torch.softmax(logits + numeric_mask, dim=-1)
            mass = torch.stack([probs[:, ids].sum(dim=-1) for ids in ids_in_bucket], dim=-1)
            mass_bucket_pred.extend(mass.argmax(dim=-1).tolist())

    def exact_in_vocab(pred_ids: list[int]) -> float:
        hits = [
            vocab.decode(p) == e.answer
            for p, e, ok in zip(pred_ids, examples, in_vocab, strict=True)
            if ok
        ]
        return sum(hits) / len(hits)

    def bucketed(pred_ids: list[int]) -> tuple[float, list[int]]:
        """Bucketed accuracy over ALL count questions; a non-numeric answer is wrong."""
        pred_bucket = []
        for p in pred_ids:
            text = vocab.decode(p)
            pred_bucket.append(bucket(int(text)) if text.isdigit() else -1)
        return sum(
            pb == tb for pb, tb in zip(pred_bucket, truth_bucket, strict=True)
        ) / n, pred_bucket

    acc_unrestricted, pred_bucket_unrestricted = bucketed(unrestricted_ids)
    acc_numeric, pred_bucket_numeric = bucketed(numeric_ids_pred)
    acc_mass = sum(pb == tb for pb, tb in zip(mass_bucket_pred, truth_bucket, strict=True)) / n

    per_bucket = {}
    for b in range(5):
        idx = [i for i, t in enumerate(truth_bucket) if t == b]
        per_bucket[BUCKET_NAMES[b]] = {
            "n": len(idx),
            "recall_numeric_only": sum(pred_bucket_numeric[i] == b for i in idx) / len(idx),
            "recall_range_mass": sum(mass_bucket_pred[i] == b for i in idx) / len(idx),
        }

    return {
        "n_count_questions": n,
        "bucketed_range_mass": acc_mass,
        "predicted_bucket_distribution_range_mass": dict(
            Counter(BUCKET_NAMES[p] for p in mass_bucket_pred)
        ),
        "exact_match_in_vocab_unrestricted": exact_in_vocab(unrestricted_ids),
        "exact_match_in_vocab_numeric_only": exact_in_vocab(numeric_ids_pred),
        "bucketed_unrestricted": acc_unrestricted,
        "bucketed_numeric_only": acc_numeric,
        "non_numeric_answers_given": sum(1 for p in pred_bucket_unrestricted if p == -1),
        "predicted_bucket_distribution_numeric_only": dict(
            Counter(BUCKET_NAMES[p] for p in pred_bucket_numeric)
        ),
        "per_true_bucket": per_bucket,
    }


def baselines(examples, vocab: AnswerVocabulary) -> dict:
    """Constant guesses: the floor any model has to beat."""
    n = len(examples)
    truth_bucket = [bucket(int(e.answer)) for e in examples]
    in_vocab = [e for e in examples if vocab.encode(e.answer) is not None]
    out = {"always_0_exact_in_vocab": sum(1 for e in in_vocab if e.answer == "0") / len(in_vocab)}
    for b in range(5):
        out[f"always_bucket_{BUCKET_NAMES[b]}"] = truth_bucket.count(b) / n

    # The fair "never looks at the image" floor: for each question text, the range most
    # common in training. RSVQA's questions are templated, so this is far above a single
    # constant guess — and the gap between a model and it is what the image really adds.
    by_question: dict[str, Counter] = {}
    for ex in count_examples("train"):
        by_question.setdefault(ex.question, Counter())[bucket(int(ex.answer))] += 1
    overall = Counter(bucket(int(ex.answer)) for ex in count_examples("train")).most_common(1)[0][0]
    hits = sum(
        (by_question[e.question].most_common(1)[0][0] if e.question in by_question else overall)
        == bucket(int(e.answer))
        for e in examples
    )
    out["question_only_bucketed"] = hits / n
    return out


def main() -> None:
    device = resolve_device()
    vocab = AnswerVocabulary.from_json(VOCAB_PATH)
    examples = count_examples("test")
    truth_bucket = [bucket(int(e.answer)) for e in examples]
    n = len(examples)
    print(f"RSVQA-LR test count questions: {n}")

    summary: dict = {
        "n_count_questions": n,
        "bucket_names": BUCKET_NAMES,
        "truth_bucket_distribution": {BUCKET_NAMES[b]: truth_bucket.count(b) / n for b in range(5)},
        "baselines": baselines(examples, vocab),
        "models": {},
    }

    tokenizer = build_tokenizer()
    for name, path in MODELS.items():
        checkpoint = torch.load(path, map_location=device, weights_only=False)
        model = LitRSVQAModel(checkpoint["config"], fusion=checkpoint["fusion"]).to(device)
        model.load_state_dict(checkpoint["model_state_dict"])
        summary["models"][name] = evaluate_counts(model, vocab, tokenizer, device, examples)
        print(f"\n{name}: {json.dumps(summary['models'][name], indent=2)}")
        del model

    print("\nbaselines:", json.dumps(summary["baselines"], indent=2))
    OUT_PATH.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\nWritten: {OUT_PATH}")


if __name__ == "__main__":
    main()
