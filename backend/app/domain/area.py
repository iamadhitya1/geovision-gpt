"""Sliding-window analysis: asking a fixed-size model about an area larger than its input.

The VQA models read one 256 x 256 px image at about 10 m per pixel (2.56 km across). A bigger
area cannot simply be shrunk to that size, because the models are fragile to a change of scale
(see the robustness study in ``ml/README.md``). Instead a 256 px window is slid across the area,
overlapping its neighbours by half, and the model is asked the same question about every
window. Each window is exactly the kind of input the model was trained on.

The windows' answers are then combined into a grid of cells (one cell per stride) by majority
vote among the windows that cover each cell, and summarised as the share of the area that
received each answer. This module is only that bookkeeping: it has no model, image or network
in it, so it is tested completely on its own.

It is a *coarse* map. A cell is half a window wide (1.28 km), so it says "windows around here
answered yes", not "there is water on this exact spot". How well the models answer on stitched
or tiled scenes has not been measured against labels.
"""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass

from app.domain.enums import FusionType, ImagerySource

# What the models were trained on: RSVQA-LR scenes are 256 x 256 px at about 10 m per pixel.
WINDOW_PX = 256
GROUND_RESOLUTION_M = 10.0
# Half a window: every spot is seen by about four windows, so one odd window cannot decide a cell.
DEFAULT_STRIDE_PX = WINDOW_PX // 2


@dataclass(frozen=True)
class WindowPlan:
    """Where the windows go on a ``width_px`` x ``height_px`` area, and the grid of cells the
    results are reported on."""

    width_px: int
    height_px: int
    window_px: int
    stride_px: int
    origins: tuple[tuple[int, int], ...]  # (x, y) of each window's top-left, row by row
    cell_rows: int
    cell_cols: int

    @property
    def window_count(self) -> int:
        return len(self.origins)


@dataclass(frozen=True)
class WindowAnswer:
    """The model's answer for the window whose top-left corner is at pixel (x, y)."""

    x: int
    y: int
    answer: str
    confidence: float


@dataclass(frozen=True)
class CellResult:
    """One cell of the result grid: the answer most of the windows over it gave.

    ``agreement`` is the share of those windows that gave that answer (1.0 means all of them
    agreed, 0.5 means a split decision), ``windows`` how many covered the cell.
    """

    row: int
    col: int
    answer: str
    agreement: float
    windows: int
    mean_confidence: float


def _starts(length: int, window: int, stride: int) -> list[int]:
    """Window start positions along one axis, stepping by ``stride``, with a final window
    flush against the far edge so the whole length is covered."""
    last = length - window
    starts = list(range(0, last + 1, stride))
    if starts[-1] != last:
        starts.append(last)
    return starts


def plan_windows(
    width_px: int,
    height_px: int,
    *,
    window_px: int = WINDOW_PX,
    stride_px: int = DEFAULT_STRIDE_PX,
) -> WindowPlan:
    """Lay out the windows. Raises ``ValueError`` for an area smaller than one window or a
    stride that would leave gaps (larger than the window) or never move (zero)."""
    if width_px < window_px or height_px < window_px:
        raise ValueError(
            f"area {width_px}x{height_px} px is smaller than one {window_px} px window"
        )
    if not 0 < stride_px <= window_px:
        raise ValueError(f"stride must be between 1 and the window size, got {stride_px}")

    xs = _starts(width_px, window_px, stride_px)
    ys = _starts(height_px, window_px, stride_px)
    return WindowPlan(
        width_px=width_px,
        height_px=height_px,
        window_px=window_px,
        stride_px=stride_px,
        origins=tuple((x, y) for y in ys for x in xs),
        cell_rows=math.ceil(height_px / stride_px),
        cell_cols=math.ceil(width_px / stride_px),
    )


def cell_pixel_box(plan: WindowPlan, row: int, col: int) -> tuple[int, int, int, int]:
    """(left, top, right, bottom) of a cell in area pixels; the last row and column are
    clipped to the area, so they can be narrower than ``stride_px``."""
    left = col * plan.stride_px
    top = row * plan.stride_px
    return (
        left,
        top,
        min(left + plan.stride_px, plan.width_px),
        min(top + plan.stride_px, plan.height_px),
    )


def _covering_windows(
    plan: WindowPlan, by_origin: dict[tuple[int, int], WindowAnswer], row: int, col: int
) -> list[WindowAnswer]:
    """The windows that contain the centre of the cell."""
    left, top, right, bottom = cell_pixel_box(plan, row, col)
    centre_x, centre_y = (left + right) / 2, (top + bottom) / 2
    return [
        by_origin[(x, y)]
        for (x, y) in plan.origins
        if (x, y) in by_origin
        and x <= centre_x < x + plan.window_px
        and y <= centre_y < y + plan.window_px
    ]


def aggregate_cells(plan: WindowPlan, answers: Sequence[WindowAnswer]) -> list[CellResult]:
    """Combine window answers into one result per cell, row by row.

    A cell takes the answer most of its covering windows gave; a tie goes to the answer with
    the larger summed confidence, and a tie of that too to the later answer alphabetically, so
    the result never depends on the order the windows happened to finish in. A cell none of the
    answered windows cover (only possible if some windows failed) is left out rather than
    invented.
    """
    by_origin = {(a.x, a.y): a for a in answers}
    cells: list[CellResult] = []
    for row in range(plan.cell_rows):
        for col in range(plan.cell_cols):
            covering = _covering_windows(plan, by_origin, row, col)
            if not covering:
                continue
            votes = Counter(a.answer for a in covering)
            weight: dict[str, float] = {}
            for a in covering:
                weight[a.answer] = weight.get(a.answer, 0.0) + a.confidence
            winner = max(votes, key=lambda answer: (votes[answer], weight[answer], answer))
            cells.append(
                CellResult(
                    row=row,
                    col=col,
                    answer=winner,
                    agreement=votes[winner] / len(covering),
                    windows=len(covering),
                    mean_confidence=sum(a.confidence for a in covering if a.answer == winner)
                    / votes[winner],
                )
            )
    return cells


def summarise(cells: Sequence[CellResult]) -> list[tuple[str, float]]:
    """The share of the area (by cells, so overlapping windows are not double counted) that
    received each answer, largest first."""
    if not cells:
        return []
    counts = Counter(c.answer for c in cells)
    return [(answer, n / len(cells)) for answer, n in counts.most_common()]


# --- What an area analysis returns --------------------------------------------------------


@dataclass(frozen=True)
class AreaCell:
    """A :class:`CellResult` placed on the map: ``bounds_wgs84`` is (west, south, east, north)."""

    row: int
    col: int
    bounds_wgs84: tuple[float, float, float, float]
    answer: str
    agreement: float
    windows: int
    mean_confidence: float


@dataclass(frozen=True)
class FusionAreaResult:
    """One model variant's view of the area. ``share_by_answer`` is the share of the area (0-1)
    that received each answer; ``failed_windows`` counts windows the model could not answer."""

    fusion: FusionType
    share_by_answer: tuple[tuple[str, float], ...]
    cells: tuple[AreaCell, ...]
    failed_windows: int


@dataclass(frozen=True)
class AreaAnalysis:
    """The answer to one question about one area, for every model variant asked."""

    question: str
    question_used: str
    note: str | None
    source: ImagerySource
    captured: str | None  # e.g. "newest scene 2026-10-24"; None if the source has no date
    bounds_wgs84: tuple[float, float, float, float]
    width_px: int
    height_px: int
    window_px: int
    stride_px: int
    window_count: int
    results: tuple[FusionAreaResult, ...]
