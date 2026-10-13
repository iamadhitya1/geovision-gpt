"""Counting-aware training: an extra loss that teaches the model the five count ranges.

The classifier picks one of 1,000 answers, and for a count question the right answer is an exact
integer ("1067"). At 10 m per pixel the model cannot resolve that exactly, and the app does not
ask it to: it answers in the five ranges the RSVQA paper scores (0 / 1-10 / 11-100 / 101-1000 /
>1000) by adding up the model's probability inside each range
(``backend/app/infrastructure/vqa/count_decoding.py``). The ordinary cross-entropy never sees
that decoding — it only rewards the exact integer — so this module adds the missing objective:

    for every count question, the probability mass the model puts in the *true range*,
    among the numeric answers, should be high (loss = -log of that mass).

It is the training-time twin of the decoding the app already uses: it renormalises over numeric
answers and sums per range exactly as ``CountDecoder`` does, so the quantity being trained is
the quantity being served. It is an auxiliary term added to the usual loss with a small weight;
at weight 0 (the default) training is unchanged.
"""

from __future__ import annotations

from collections.abc import Callable

import torch
from torch import Tensor

from dataset.count_ranges import RANGE_NAMES, range_index

NUM_RANGES = len(RANGE_NAMES)


def build_range_table(num_answers: int, answer_of_id: Callable[[int], str]) -> Tensor:
    """For each answer id: the range index (0-4) if the answer is a whole number, else -1."""
    table = torch.full((num_answers,), -1, dtype=torch.long)
    for answer_id in range(num_answers):
        text = answer_of_id(answer_id)
        if text.isdigit():
            table[answer_id] = range_index(int(text))
    return table


def range_mass_loss(
    logits: Tensor,
    labels: Tensor,
    is_count: Tensor,
    range_table: Tensor,
    *,
    eps: float = 1e-6,
) -> Tensor:
    """Mean negative log of the probability mass in the true range, over the count questions.

    ``logits`` is (batch, answers), ``labels`` the true answer ids, ``is_count`` a boolean
    (batch,) mask of the count questions. Rows that aren't count questions, or whose answer
    isn't a number, are ignored; with none left the loss is a zero that carries no gradient.
    """
    range_table = range_table.to(logits.device)
    true_range = range_table[labels]
    rows = is_count.to(logits.device) & (true_range >= 0)
    if not bool(rows.any()):
        return logits.new_zeros(())

    probs = torch.softmax(logits[rows].float(), dim=-1)
    numeric = range_table >= 0
    mass = torch.zeros(probs.shape[0], NUM_RANGES, device=probs.device)
    # Sum the probability of every numeric answer into its range (no Python loop over answers).
    index = range_table.clamp(min=0).unsqueeze(0).expand_as(probs)
    mass = mass.scatter_add(1, index, probs * numeric.unsqueeze(0))
    mass = mass / mass.sum(dim=1, keepdim=True).clamp(min=eps)
    chosen = mass.gather(1, true_range[rows].unsqueeze(1)).squeeze(1)
    return -torch.log(chosen.clamp(min=eps)).mean()
