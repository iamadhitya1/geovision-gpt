"""Gujarat tiles as VQA examples: the questions asked about each tile and the label behind each.

The tiles are built by ``backend/scripts/gujarat_set.py`` (a picture from Earth Engine plus labels
from Earth Engine's own layers: ESA WorldCover, JRC surface water, GHSL settlement, Google Open
Buildings). This module is the single place that turns a tile's labels into the app's question
templates and answers, so training and scoring can never disagree about what "right" means.

The labels are model-made, not surveyed, and the thresholds below are judgement calls written down
here rather than hidden. Accuracy measured on these labels is accuracy against *them*.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path

from dataset.count_ranges import range_index
from dataset.rsvqa_lr_dataset import AnswerVocabulary, VqaExample

GUJARAT_ROOT = Path(__file__).resolve().parent / "gujarat" / "set"

# A thing counts as "present" in a 2.56 km tile when it covers at least this share of it.
PRESENT_FRACTION = 0.005
# GHSL degree of urbanisation: 30 urban centre, 23 dense urban cluster, 22 semi-dense urban cluster.
URBAN_SETTLEMENT_CLASSES = frozenset({22, 23, 30})

WATER = "Is there a water area?"
FOREST = "Is there a forest?"
BUILDING = "Is there a building?"
RURAL_URBAN = "Is it a rural or an urban area?"
COUNT = "How many buildings are there?"

QUESTION_TYPES = {
    WATER: "presence",
    FOREST: "presence",
    BUILDING: "presence",
    RURAL_URBAN: "rural_urban",
    COUNT: "count",
}

TRUTH: dict[str, Callable[[dict], str]] = {
    WATER: lambda t: (
        "yes"
        if max(t["cover"].get("water", 0), t.get("surface_water", 0)) >= PRESENT_FRACTION
        else "no"
    ),
    FOREST: lambda t: "yes" if t["cover"].get("tree", 0) >= PRESENT_FRACTION else "no",
    BUILDING: lambda t: "yes" if t["buildings"] >= 1 else "no",
    RURAL_URBAN: lambda t: "urban" if t.get("settlement") in URBAN_SETTLEMENT_CLASSES else "rural",
}


def count_answer(count: int, vocab: AnswerVocabulary) -> str | None:
    """The answer-list number nearest to ``count`` that lies in the same range, so the training
    target says the right range even where the exact count is not an answer the model can give.
    Counts above the largest listed number clamp to it. ``None`` if the range has no answer."""
    wanted = range_index(count)
    best: tuple[int, int] | None = None  # (distance, number)
    for index in range(len(vocab)):
        text = vocab.decode(index)
        if not text.isdigit():
            continue
        number = int(text)
        if range_index(number) != wanted:
            continue
        distance = abs(number - count)
        if best is None or distance < best[0]:
            best = (distance, number)
    return None if best is None else str(best[1])


def tile_answers(tile: dict, vocab: AnswerVocabulary | None = None) -> dict[str, str]:
    """question -> label for one tile. The count question is included only when ``vocab`` lets a
    count be expressed (see :func:`count_answer`)."""
    answers = {question: rule(tile) for question, rule in TRUTH.items()}
    if vocab is not None:
        counted = count_answer(int(tile["buildings"]), vocab)
        if counted is not None:
            answers[COUNT] = counted
    return answers


def read_tiles(root: Path = GUJARAT_ROOT, split: str | None = None) -> list[dict]:
    index = root / "labels.jsonl"
    tiles = [json.loads(line) for line in index.read_text(encoding="utf-8").splitlines() if line]
    return [t for t in tiles if split is None or t["split"] == split]


def load_gujarat_examples(
    split: str, vocab: AnswerVocabulary, root: Path = GUJARAT_ROOT
) -> list[VqaExample]:
    """Every question about every tile of ``split`` ('train' / 'val' / 'test'): the split is by
    whole district, so a tile's neighbours are never in another split."""
    examples = []
    for tile in read_tiles(root, split):
        path = root / "images" / f"{tile['id']}.png"
        for question, answer in tile_answers(tile, vocab).items():
            examples.append(VqaExample(path, question, QUESTION_TYPES[question], answer))
    return examples
