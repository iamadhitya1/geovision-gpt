"""Answering counting questions as ranges instead of exact numbers.

The classifier head spreads a count question's probability over many exact numbers ("14",
"15", "16", ...), so a single exact value rarely wins by much — and "0", a single answer
holding a large share of the training labels, can win the argmax even when "somewhere in
11-100" holds far more probability in total. Summing probability per range first and then
taking the best range fixes that. Measured on the RSVQA-LR test set (``ml/scripts/
count_bucket_eval.py``): elementwise 64.5% -> 68.1%, cross-attention 40.4% -> 61.7% of count
questions placed in the right range, with no retraining.
"""

from __future__ import annotations

import torch
from dataset.count_ranges import RANGE_NAMES, range_index
from dataset.rsvqa_lr_dataset import AnswerVocabulary


class CountDecoder:
    """Precomputed from the answer vocabulary once; ``range_probabilities`` is then cheap."""

    def __init__(self, vocab: AnswerVocabulary) -> None:
        numeric_ids = [i for i in range(len(vocab)) if vocab.decode(i).isdigit()]
        # Added to the logits so a non-numeric answer ("yes") can never take count probability.
        self._numeric_mask = torch.full((len(vocab),), float("-inf"))
        self._numeric_mask[numeric_ids] = 0.0
        self._ids_by_range = [
            torch.tensor(
                [i for i in numeric_ids if range_index(int(vocab.decode(i))) == r],
                dtype=torch.long,
            )
            for r in range(len(RANGE_NAMES))
        ]

    def range_probabilities(self, logits: torch.Tensor) -> list[float]:
        """``logits`` is one question's raw output over the whole vocabulary; returns the
        probability of each of the five ranges (summing to 1.0)."""
        mask = self._numeric_mask.to(logits.device)
        probabilities = torch.softmax(logits + mask, dim=-1)
        return [
            float(probabilities[ids.to(logits.device)].sum().item()) for ids in self._ids_by_range
        ]
