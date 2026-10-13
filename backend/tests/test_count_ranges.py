"""Counting questions are answered as ranges — see ``app/infrastructure/vqa/count_decoding.py``."""

from __future__ import annotations

import pytest
import torch
from dataset.count_ranges import RANGE_LABELS, RANGE_NAMES, is_count_question, range_index
from dataset.rsvqa_lr_dataset import AnswerVocabulary

from app.infrastructure.vqa.count_decoding import CountDecoder


@pytest.mark.parametrize(
    ("count", "expected"),
    [(0, 0), (1, 1), (10, 1), (11, 2), (100, 2), (101, 3), (1000, 3), (1001, 4), (17139, 4)],
)
def test_range_boundaries_match_the_papers_five_ranges(count: int, expected: int) -> None:
    assert range_index(count) == expected


def test_every_range_has_a_name_and_a_label() -> None:
    assert len(RANGE_NAMES) == len(RANGE_LABELS) == 5


@pytest.mark.parametrize(
    "question",
    [
        "How many residential buildings are there?",
        "What is the number of roads?",
        "  how many water areas are there?",
        "What's the number of grass areas?",
    ],
)
def test_count_questions_are_recognised(question: str) -> None:
    assert is_count_question(question)


@pytest.mark.parametrize(
    "question",
    [
        "Is it a rural or an urban area?",
        "Is a cemetery present?",
        "Are there more roads than buildings?",
        "Describe how many people live here.",  # contains the words, but isn't a count template
    ],
)
def test_other_questions_are_not_count_questions(question: str) -> None:
    assert not is_count_question(question)


def _decoder(answers: list[str]) -> CountDecoder:
    return CountDecoder(AnswerVocabulary({answer: i for i, answer in enumerate(answers)}))


def test_range_with_most_total_probability_beats_the_single_most_likely_number() -> None:
    # "0" is the single likeliest answer, but 15 + 20 + 30 together (all in 11-100) hold more.
    decoder = _decoder(["0", "yes", "15", "20", "30"])
    logits = torch.tensor([2.0, 5.0, 1.5, 1.5, 1.5])

    probs = decoder.range_probabilities(logits)

    assert probs.index(max(probs)) == range_index(20) == 2  # the 11-100 range
    assert probs[0] < probs[2]


def test_range_probabilities_sum_to_one_and_ignore_non_numeric_answers() -> None:
    decoder = _decoder(["yes", "no", "0", "5", "50", "500", "5000"])
    logits = torch.tensor([9.0, 9.0, 1.0, 2.0, 1.0, 0.5, 0.1])  # "yes"/"no" would dominate

    probs = decoder.range_probabilities(logits)

    assert sum(probs) == pytest.approx(1.0)
    assert len(probs) == 5
    assert probs.index(max(probs)) == range_index(5)
