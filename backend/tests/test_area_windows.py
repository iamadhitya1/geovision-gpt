"""Sliding-window bookkeeping: where the windows go and how their answers are combined."""

from __future__ import annotations

import pytest

from app.domain.area import (
    WindowAnswer,
    aggregate_cells,
    cell_pixel_box,
    plan_windows,
    summarise,
)


def test_one_window_when_the_area_is_exactly_the_model_size() -> None:
    plan = plan_windows(256, 256)

    assert plan.origins == ((0, 0),)
    assert (plan.cell_rows, plan.cell_cols) == (2, 2)


def test_windows_step_by_the_stride_and_the_last_one_sits_flush_with_the_edge() -> None:
    plan = plan_windows(1000, 256)

    xs = [x for x, _ in plan.origins]
    assert xs[:3] == [0, 128, 256]
    assert xs[-1] == 1000 - 256  # 744, not a multiple of 128, but the edge is covered
    assert all(y == 0 for _, y in plan.origins)


def test_every_pixel_is_seen_by_at_least_one_window() -> None:
    plan = plan_windows(777, 531)

    covered_x = {px for x, _ in plan.origins for px in range(x, x + 256)}
    covered_y = {px for _, y in plan.origins for px in range(y, y + 256)}
    assert covered_x == set(range(777))
    assert covered_y == set(range(531))


def test_an_area_smaller_than_one_window_is_rejected() -> None:
    with pytest.raises(ValueError, match="smaller than one"):
        plan_windows(255, 400)


@pytest.mark.parametrize("stride", [0, -5, 257])
def test_a_stride_that_leaves_gaps_or_never_moves_is_rejected(stride: int) -> None:
    with pytest.raises(ValueError, match="stride"):
        plan_windows(600, 600, stride_px=stride)


def test_cells_are_one_stride_wide_and_the_last_ones_are_clipped() -> None:
    plan = plan_windows(300, 300)

    assert cell_pixel_box(plan, 0, 0) == (0, 0, 128, 128)
    assert cell_pixel_box(plan, 2, 2) == (256, 256, 300, 300)  # 44 px left over
    assert (plan.cell_rows, plan.cell_cols) == (3, 3)


def _all(plan, answer: str, confidence: float = 0.9) -> list[WindowAnswer]:
    return [WindowAnswer(x, y, answer, confidence) for x, y in plan.origins]


def test_when_every_window_agrees_every_cell_gets_that_answer() -> None:
    plan = plan_windows(512, 512)

    cells = aggregate_cells(plan, _all(plan, "yes"))

    assert len(cells) == plan.cell_rows * plan.cell_cols
    assert {c.answer for c in cells} == {"yes"}
    assert all(c.agreement == 1.0 for c in cells)


def test_a_cell_takes_the_majority_of_the_windows_over_it() -> None:
    plan = plan_windows(512, 256)  # windows at x = 0, 128, 256
    answers = [
        WindowAnswer(0, 0, "no", 0.8),
        WindowAnswer(128, 0, "yes", 0.9),
        WindowAnswer(256, 0, "yes", 0.7),
    ]

    cells = {(c.row, c.col): c for c in aggregate_cells(plan, answers)}

    # Cell (0, 2) is centred at x = 320: only the windows starting at 128 and 256 contain it.
    assert cells[(0, 2)].answer == "yes"
    assert cells[(0, 2)].windows == 2
    # Cell (0, 0) is centred at x = 64: only the window at 0 contains it.
    assert cells[(0, 0)].answer == "no"
    assert cells[(0, 0)].agreement == 1.0


def test_a_tie_goes_to_the_answer_the_model_was_more_sure_of() -> None:
    plan = plan_windows(384, 256)  # windows at x = 0, 128
    answers = [WindowAnswer(0, 0, "no", 0.55), WindowAnswer(128, 0, "yes", 0.95)]

    cells = {(c.row, c.col): c for c in aggregate_cells(plan, answers)}

    # Cell (0, 1) is centred at x = 192, inside both windows: one vote each.
    assert cells[(0, 1)].windows == 2
    assert cells[(0, 1)].answer == "yes"
    assert cells[(0, 1)].agreement == pytest.approx(0.5)


def test_a_window_that_failed_leaves_its_cells_to_the_others_or_out_entirely() -> None:
    plan = plan_windows(512, 256)
    survivors = [WindowAnswer(0, 0, "yes", 0.9)]  # the other two windows failed

    cells = aggregate_cells(plan, survivors)

    # Only cells whose centre lies inside the surviving window are reported.
    assert {c.col for c in cells} == {0, 1}


def test_summary_is_the_share_of_cells_per_answer_largest_first() -> None:
    plan = plan_windows(512, 512)
    answers = [WindowAnswer(x, y, "urban" if x < 256 else "rural", 0.9) for x, y in plan.origins]

    summary = summarise(aggregate_cells(plan, answers))

    assert next(name for name, _ in summary) in {"urban", "rural"}
    assert sum(share for _, share in summary) == pytest.approx(1.0)


def test_summary_of_nothing_is_empty() -> None:
    assert summarise([]) == []


def test_an_exact_tie_is_settled_the_same_way_whatever_order_the_windows_arrive_in() -> None:
    plan = plan_windows(384, 256)  # windows at x = 0, 128; both contain the cell centred at 192
    forward = [WindowAnswer(0, 0, "no", 0.8), WindowAnswer(128, 0, "yes", 0.8)]

    first = {(c.row, c.col): c.answer for c in aggregate_cells(plan, forward)}
    second = {(c.row, c.col): c.answer for c in aggregate_cells(plan, forward[::-1])}

    assert first == second
