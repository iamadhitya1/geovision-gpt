"""Gujarat tiles as VQA examples: what each tile is asked and what the right answer is."""

from __future__ import annotations

import json
from pathlib import Path

from dataset.gujarat_dataset import (
    BUILDING,
    COUNT,
    FOREST,
    RURAL_URBAN,
    WATER,
    count_answer,
    load_gujarat_examples,
    tile_answers,
)
from dataset.rsvqa_lr_dataset import AnswerVocabulary, VqaExample
from scripts.train_generalist import epoch_plan

NUMBERS = ["0", "1", "5", "10", "50", "150", "900", "1200", "15000"]
VOCAB = AnswerVocabulary({a: i for i, a in enumerate(["yes", "no", "rural", "urban", *NUMBERS])})


def _tile(**overrides) -> dict:
    tile = {
        "id": "train_00000",
        "split": "train",
        "cover": {"tree": 0.0, "water": 0.0},
        "surface_water": 0.0,
        "settlement": 11,
        "buildings": 0,
    }
    return {**tile, **overrides}


# --- the building count, expressed as an answer the model can give ----------------------------


def test_a_count_becomes_the_nearest_answer_in_the_same_range() -> None:
    assert count_answer(7, VOCAB) == "5"  # 1-10: nearer 5 than 10
    assert count_answer(2240, VOCAB) == "1200"  # over 1,000: nearer 1200 than 15000
    assert count_answer(0, VOCAB) == "0"
    assert count_answer(120, VOCAB) == "150"  # 101-1000, never leaks into the 11-100 range


def test_a_count_above_every_listed_number_clamps_to_the_largest() -> None:
    assert count_answer(40_000, VOCAB) == "15000"


def test_a_range_with_no_answer_gives_none_so_the_example_is_dropped() -> None:
    sparse = AnswerVocabulary({"yes": 0, "no": 1, "0": 2, "5": 3})

    assert count_answer(500, sparse) is None


# --- the labels -----------------------------------------------------------------------------


def test_water_counts_from_land_cover_or_from_long_term_surface_water() -> None:
    assert tile_answers(_tile())[WATER] == "no"
    assert tile_answers(_tile(cover={"water": 0.006}))[WATER] == "yes"
    assert tile_answers(_tile(surface_water=0.02))[WATER] == "yes"  # a pond the snapshot missed
    assert tile_answers(_tile(cover={"water": 0.001}))[WATER] == "no"  # under the threshold


def test_forest_and_building_follow_tree_cover_and_the_building_count() -> None:
    assert tile_answers(_tile(cover={"tree": 0.2}))[FOREST] == "yes"
    assert tile_answers(_tile())[FOREST] == "no"
    assert tile_answers(_tile(buildings=1))[BUILDING] == "yes"
    assert tile_answers(_tile(buildings=0))[BUILDING] == "no"


def test_rural_or_urban_follows_the_settlement_class_not_the_building_count() -> None:
    assert tile_answers(_tile(settlement=30, buildings=3))[RURAL_URBAN] == "urban"
    assert tile_answers(_tile(settlement=22))[RURAL_URBAN] == "urban"
    assert tile_answers(_tile(settlement=13, buildings=5000))[RURAL_URBAN] == "rural"
    assert tile_answers(_tile(settlement=None))[RURAL_URBAN] == "rural"


def test_the_count_question_is_only_asked_when_a_count_can_be_expressed() -> None:
    assert COUNT not in tile_answers(_tile(buildings=30))  # no vocabulary given
    assert tile_answers(_tile(buildings=30), VOCAB)[COUNT] == "50"


# --- loading ---------------------------------------------------------------------------------


def _write_set(root: Path) -> None:
    (root / "images").mkdir(parents=True)
    tiles = [
        _tile(id="train_00000", split="train", buildings=120),
        _tile(id="val_00001", split="val", buildings=0),
        _tile(id="test_00002", split="test", buildings=9),
    ]
    (root / "labels.jsonl").write_text("\n".join(json.dumps(t) for t in tiles), encoding="utf-8")


def test_each_split_gets_only_its_own_tiles_with_five_questions_each(tmp_path: Path) -> None:
    _write_set(tmp_path)

    train = load_gujarat_examples("train", VOCAB, tmp_path)
    val = load_gujarat_examples("val", VOCAB, tmp_path)

    assert len(train) == len(val) == 5
    assert {e.image_path.name for e in train} == {"train_00000.png"}
    assert {e.image_path.name for e in val} == {"val_00001.png"}
    assert {e.question_type for e in train} == {"presence", "rural_urban", "count"}
    assert all(isinstance(e, VqaExample) for e in train)


# --- how often Gujarat appears in an epoch ---------------------------------------------------


def _examples(prefix: str, n: int) -> list[VqaExample]:
    return [VqaExample(Path(f"{prefix}{i}.png"), "q", "presence", "yes") for i in range(n)]


def test_gujarat_examples_are_repeated_in_every_epoch() -> None:
    plan = epoch_plan(
        _examples("lr", 10), [], 0, epoch=0, seed=0, guj_train=_examples("g", 3), guj_repeat=4
    )

    names = [e.image_path.name for e in plan]
    assert len(plan) == 10 + 3 * 4
    assert names.count("g0.png") == 4


def test_without_gujarat_the_epoch_is_what_it_always_was() -> None:
    lr, hr = _examples("lr", 10), _examples("hr", 50)

    before = epoch_plan(lr, hr, 20, epoch=2, seed=1)
    with_empty = epoch_plan(lr, hr, 20, epoch=2, seed=1, guj_train=[], guj_repeat=4)

    assert before == with_empty
    assert len(before) == 30
