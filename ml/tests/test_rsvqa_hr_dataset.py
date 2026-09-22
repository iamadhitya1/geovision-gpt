"""Loader tests against small, hand-written fixtures matching the REAL downloaded file format
(confirmed by inspecting the actual Zenodo files directly — see rsvqa_hr_dataset.py's docstring).
No dependency on the 14GB dataset itself.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from dataset.rsvqa_hr_dataset import load_hr_split_examples


def _write_split(root: Path, split: str, questions: list[dict], answers: list[dict]) -> None:
    (root / f"USGS_split_{split}_questions.json").write_text(
        json.dumps({"questions": questions}), encoding="utf-8"
    )
    (root / f"USGS_split_{split}_answers.json").write_text(
        json.dumps({"answers": answers}), encoding="utf-8"
    )


def _question(id_: int, img_id: int, qtype: str, active: bool = True) -> dict:
    return {
        "id": id_,
        "img_id": img_id,
        "people_id": 0,
        "type": qtype,
        "question": f"question {id_}",
        "answers_ids": [id_],
        "active": active,
    }


def _answer(id_: int, question_id: int, value: str, active: bool = True) -> dict:
    return {
        "id": id_,
        "question_id": question_id,
        "people_id": 0,
        "answer": value,
        "active": active,
    }


def test_loads_active_examples_with_correct_fields(tmp_path: Path) -> None:
    _write_split(
        tmp_path,
        "val",
        [_question(0, img_id=7, qtype="presence")],
        [_answer(0, question_id=0, value="yes")],
    )

    examples = load_hr_split_examples(tmp_path, "val")

    assert len(examples) == 1
    example = examples[0]
    assert example.image_path == tmp_path / "Data" / "7.png"
    assert example.question == "question 0"
    assert example.question_type == "presence"
    assert example.answer == "yes"


def test_inactive_questions_are_dropped(tmp_path: Path) -> None:
    _write_split(
        tmp_path,
        "val",
        [_question(0, img_id=1, qtype="presence", active=False)],
        [_answer(0, question_id=0, value="yes")],
    )
    assert load_hr_split_examples(tmp_path, "val") == []


def test_area_questions_are_excluded() -> None:
    # area answers are near-unique measurements ("521m2") — unusable for a fixed vocabulary,
    # and excluding it is the whole reason HR's answer space stays compatible with LR's.
    from dataset.rsvqa_hr_dataset import EXCLUDED_TYPES

    assert frozenset({"area"}) == EXCLUDED_TYPES


def test_area_questions_are_excluded_end_to_end(tmp_path: Path) -> None:
    _write_split(
        tmp_path,
        "val",
        [_question(0, img_id=1, qtype="area"), _question(1, img_id=1, qtype="count")],
        [_answer(0, question_id=0, value="521m2"), _answer(1, question_id=1, value="4")],
    )

    examples = load_hr_split_examples(tmp_path, "val")

    assert [e.question_type for e in examples] == ["count"]


def test_answer_is_matched_by_question_id_not_answer_id(tmp_path: Path) -> None:
    # answers.json links back via "question_id"; its own "id" is unrelated and must not be
    # confused with it (an easy off-by-one to introduce when copying the RSVQA-LR loader).
    _write_split(
        tmp_path,
        "val",
        [_question(5, img_id=1, qtype="count")],
        [_answer(id_=99, question_id=5, value="3")],
    )
    assert load_hr_split_examples(tmp_path, "val")[0].answer == "3"


def test_all_four_named_splits_load(tmp_path: Path) -> None:
    for split in ("train", "val", "test", "test_phili"):
        _write_split(tmp_path, split, [_question(0, 1, "presence")], [_answer(0, 0, "no")])
    for split in ("train", "val", "test", "test_phili"):
        assert len(load_hr_split_examples(tmp_path, split)) == 1  # type: ignore[arg-type]


def test_unknown_split_raises() -> None:
    with pytest.raises(ValueError, match="unknown RSVQA-HR split"):
        load_hr_split_examples(Path("."), "phili")  # type: ignore[arg-type]


def test_inactive_answer_raises_keyerror_like_the_lr_loader(tmp_path: Path) -> None:
    # Mirrors RSVQA-LR's own loader: an active question whose only answer is inactive means the
    # on-disk data no longer matches what this loader was written against — surface it loudly
    # rather than silently drop the question.
    _write_split(
        tmp_path,
        "val",
        [_question(0, img_id=1, qtype="presence")],
        [_answer(0, question_id=0, value="yes", active=False)],
    )
    with pytest.raises(KeyError):
        load_hr_split_examples(tmp_path, "val")
