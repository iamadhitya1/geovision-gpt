"""The counting-aware loss: it must reward probability in the true range, and only for counts."""

from __future__ import annotations

import math

import pytest
import torch

from models.count_loss import NUM_RANGES, build_range_table, range_mass_loss

# A tiny vocabulary: ids 0-1 are yes/no, then counts that fall in each of the five ranges.
ANSWERS = ["yes", "no", "0", "3", "10", "11", "50", "150", "900", "1500", "rural"]
RANGES_OF = {"0": 0, "3": 1, "10": 1, "11": 2, "50": 2, "150": 3, "900": 3, "1500": 4}


def _table() -> torch.Tensor:
    return build_range_table(len(ANSWERS), lambda i: ANSWERS[i])


def _logits_favouring(answer: str, strength: float = 8.0) -> torch.Tensor:
    logits = torch.zeros(1, len(ANSWERS))
    logits[0, ANSWERS.index(answer)] = strength
    return logits


def test_the_table_maps_numbers_to_their_range_and_everything_else_to_minus_one() -> None:
    table = _table().tolist()

    for index, answer in enumerate(ANSWERS):
        assert table[index] == RANGES_OF.get(answer, -1), answer
    assert NUM_RANGES == 5


def test_loss_is_small_when_the_mass_is_in_the_true_range() -> None:
    labels = torch.tensor([ANSWERS.index("50")])  # true range 11-100
    in_range = range_mass_loss(_logits_favouring("11"), labels, torch.tensor([True]), _table())
    wrong_range = range_mass_loss(_logits_favouring("1500"), labels, torch.tensor([True]), _table())

    assert in_range.item() < 0.05
    assert wrong_range.item() > 5


def test_an_off_by_a_few_answer_in_the_right_range_costs_nothing_extra() -> None:
    """The point of the loss: guessing 11 when the truth is 50 is right as far as ranges go."""
    labels = torch.tensor([ANSWERS.index("50")])
    loss = range_mass_loss(_logits_favouring("11"), labels, torch.tensor([True]), _table())

    assert loss.item() < 0.05


def test_loss_sums_probability_across_a_whole_range() -> None:
    # Half the mass spread over the two answers of range 1-10, none elsewhere among numbers.
    logits = torch.full((1, len(ANSWERS)), -20.0)
    logits[0, ANSWERS.index("3")] = 0.0
    logits[0, ANSWERS.index("10")] = 0.0
    labels = torch.tensor([ANSWERS.index("3")])

    loss = range_mass_loss(logits, labels, torch.tensor([True]), _table())

    assert loss.item() == pytest.approx(0.0, abs=1e-3)  # all numeric mass is in the true range


def test_non_numeric_probability_is_ignored_by_renormalising_over_numbers() -> None:
    logits = _logits_favouring("yes", strength=12.0)  # almost all mass on a non-number
    logits[0, ANSWERS.index("150")] = 10.0
    labels = torch.tensor([ANSWERS.index("900")])  # same range as 150

    loss = range_mass_loss(logits, labels, torch.tensor([True]), _table())

    assert loss.item() < 0.05


def test_rows_that_are_not_count_questions_are_ignored() -> None:
    logits = torch.cat([_logits_favouring("1500"), _logits_favouring("11")])
    labels = torch.tensor([ANSWERS.index("yes"), ANSWERS.index("50")])
    is_count = torch.tensor([False, True])

    loss = range_mass_loss(logits, labels, is_count, _table())

    assert loss.item() < 0.05  # only the second row counts, and it is right


def test_a_batch_without_count_questions_gives_a_zero_loss() -> None:
    logits = _logits_favouring("1500")
    loss = range_mass_loss(
        logits, torch.tensor([ANSWERS.index("yes")]), torch.tensor([False]), _table()
    )

    assert loss.item() == 0.0


def test_the_gradient_pushes_mass_toward_the_true_range() -> None:
    logits = torch.zeros(1, len(ANSWERS), requires_grad=True)
    labels = torch.tensor([ANSWERS.index("900")])  # range 101-1000

    range_mass_loss(logits, labels, torch.tensor([True]), _table()).backward()

    grad = logits.grad[0]
    in_range = [ANSWERS.index("150"), ANSWERS.index("900")]
    out_of_range = [ANSWERS.index(a) for a in ("0", "3", "10", "11", "50", "1500")]
    assert all(grad[i] < 0 for i in in_range)  # descending the loss raises these
    assert all(grad[i] > 0 for i in out_of_range)  # and lowers these


def test_uniform_logits_give_the_log_of_the_inverse_range_share() -> None:
    logits = torch.zeros(1, len(ANSWERS))
    labels = torch.tensor([ANSWERS.index("0")])  # range 0 holds 1 of the 8 numeric answers

    loss = range_mass_loss(logits, labels, torch.tensor([True]), _table())

    assert loss.item() == pytest.approx(math.log(8), rel=1e-4)
