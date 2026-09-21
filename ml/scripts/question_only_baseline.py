"""How much of the accuracy comes from the question alone, with no image at all?

Two trivial predictors, both built from the TRAINING split only and scored on the test split:

* per-type    - always answer the most common training answer for the question's type
* per-question - answer the most common training answer for that exact question text
                 (falling back to the per-type answer for an unseen question)

RSVQA-LR's questions are templated, so many answers are strongly predictable from the words
alone. Any trained model has to be judged against these numbers, not against zero: the
difference is what the image actually contributes.

    python scripts/question_only_baseline.py
"""

from __future__ import annotations

import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dataset.rsvqa_lr_dataset import AnswerVocabulary, load_split_examples

DATA_ROOT = Path(__file__).resolve().parent.parent / "dataset" / "rsvqa_lr"
VOCAB_PATH = DATA_ROOT / "answer_vocab.json"


def main() -> None:
    vocab = AnswerVocabulary.from_json(VOCAB_PATH)
    train = load_split_examples(DATA_ROOT, "train")
    # Same filter the models are evaluated with: answers outside the 1,000-answer vocabulary
    # are dropped, so these numbers are directly comparable to evaluate.py's.
    test = [e for e in load_split_examples(DATA_ROOT, "test") if vocab.encode(e.answer) is not None]

    by_type: dict[str, Counter[str]] = defaultdict(Counter)
    by_question: dict[str, Counter[str]] = defaultdict(Counter)
    for e in train:
        by_type[e.question_type][e.answer] += 1
        by_question[e.question][e.answer] += 1

    majority_by_type = {t: c.most_common(1)[0][0] for t, c in by_type.items()}

    def per_question(e) -> str:
        counts = by_question.get(e.question)
        return counts.most_common(1)[0][0] if counts else majority_by_type[e.question_type]

    predictors = {
        "per-type": lambda e: majority_by_type[e.question_type],
        "per-question": per_question,
    }

    print(f"test questions: {len(test):,}\n")
    for name, predict in predictors.items():
        correct: Counter[str] = Counter()
        total: Counter[str] = Counter()
        for e in test:
            total[e.question_type] += 1
            correct[e.question_type] += int(predict(e) == e.answer)
        per_type = {t: correct[t] / total[t] for t in total}
        overall = sum(correct.values()) / sum(total.values())
        average = sum(per_type.values()) / len(per_type)
        print(f"{name:13s} overall {overall:.2%}  average {average:.2%}")
        cells = (f"{t} {a:.1%} (n={total[t]})" for t, a in sorted(per_type.items()))
        print("              " + "  ".join(cells))
    print("\nmost common training answer per type:", majority_by_type)


if __name__ == "__main__":
    main()
