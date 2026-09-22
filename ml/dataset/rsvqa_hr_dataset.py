"""Loader for RSVQA-HR — 15cm aerial imagery, a much finer scale than RSVQA-LR's 10m Sentinel-2.

Source: Lobry, Marcos, Murray & Tuia, "RSVQA: Visual Question Answering for Remote Sensing
Data," IEEE TGRS 2020 (the same paper as RSVQA-LR). Downloaded from Zenodo, record 6344367,
CC-BY-4.0. 10,659 images, 512x512, USGS High Resolution Orthoimagery.

Confirmed directly against the real downloaded files (not assumed from documentation):

* Same active-flag quirk as RSVQA-LR: each split's questions/answers JSON is a cumulative
  superset, and only entries with ``"active": true`` belong to that split.
* Four question types — ``count``, ``area``, ``comp``, ``presence`` — NOT five: unlike
  RSVQA-LR, RSVQA-HR has no ``rural_urban`` type. Training on both datasets together still
  gets rural/urban questions, from RSVQA-LR.
* ``area`` answers are exact measurements (``"521m2"``), essentially all unique, not the
  bucketed ranges some documentation describes — unusable for a fixed-vocabulary classifier,
  so this loader drops the type entirely (see ``EXCLUDED_TYPES``).
* Every image ID exists on disk as BOTH ``<id>.tif`` (georeferenced) and ``<id>.png`` (plain
  raster) — 21,318 files for 10,659 images. The PNG is used: identical pixels, no need for
  ``rasterio``/GDAL just to read RGB values, and it is what ``dataset.augment`` and the model's
  ``build_image_transform`` already expect (a plain PIL image).
* There are two test splits, both usable through this loader by name: ``"test"`` (same
  distribution as training) and ``"test_phili"`` (Philadelphia — a city the model never saw in
  training, the one real generalisation check this project has).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Literal, get_args

from dataset.rsvqa_lr_dataset import VqaExample

HrSplit = Literal["train", "val", "test", "test_phili"]

# "area" has near-unique string answers ("521m2") and cannot be a classification target with a
# fixed answer vocabulary; excluding it, not RSVQA-LR's four types, is what makes the two
# datasets' answer spaces compatible for joint training.
EXCLUDED_TYPES = frozenset({"area"})

_FILE_PREFIX = "USGS_split"


def _load_json(path: Path, key: str) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        return json.load(f)[key]


def load_hr_split_examples(data_root: Path, split: HrSplit) -> list[VqaExample]:
    """Load one split's active (image, question, answer) triples, images-folder image
    lookup by numeric id (``Data/<img_id>.png``), matching ``load_split_examples``'s
    return type exactly so the two datasets are interchangeable to the trainer."""
    if split not in get_args(HrSplit):
        raise ValueError(f"unknown RSVQA-HR split {split!r}; expected one of {get_args(HrSplit)}")

    questions = _load_json(data_root / f"{_FILE_PREFIX}_{split}_questions.json", "questions")
    answers = _load_json(data_root / f"{_FILE_PREFIX}_{split}_answers.json", "answers")

    answer_by_question_id = {a["question_id"]: a for a in answers if a.get("active")}
    images_dir = data_root / "Data"

    examples: list[VqaExample] = []
    for q in questions:
        if not q.get("active") or q["type"] in EXCLUDED_TYPES:
            continue
        answer = answer_by_question_id[q["id"]]
        examples.append(
            VqaExample(
                image_path=images_dir / f"{q['img_id']}.png",
                question=q["question"],
                question_type=q["type"],
                answer=answer["answer"],
            )
        )
    return examples
