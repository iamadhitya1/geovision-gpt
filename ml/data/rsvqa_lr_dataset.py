"""PyTorch ``Dataset`` for RSVQA-LR.

Source: Lobry, Marcos, Murray & Tuia, "RSVQA: Visual Question Answering for Remote
Sensing Data," IEEE TGRS 2020. Downloaded from Zenodo (10.5281/zenodo.6344334).

The release's file format has a quirk worth documenting, because it's easy to load wrong:
each split's ``LR_split_<split>_questions.json`` / ``_answers.json`` file is NOT
pre-filtered to that split — it contains a cumulative superset of entries, most of them
placeholder stubs (``{"id": ..., "active": False}``) with no other fields. Only entries
with ``"active": True`` belong to that split. Filtering on this flag was verified against
the file contents directly: it recovers exactly 57,223 / 10,005 / 10,004 active
question-answer pairs for train/val/test, summing to the documented 77,232 total — so this
is confirmed correct, not assumed.

Every image file (via ``LR_split_<split>_images.json``) is a 256×256, 3-band, uint8 GeoTIFF
under EPSG:3857, verified directly with rasterio.
"""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from PIL import Image
from torch.utils.data import Dataset

Split = Literal["train", "val", "test"]

# The paper restricts the classifier to the nA=1000 most frequent answers; RSVQA-LR's
# actual vocabulary (1,527 unique strings) is small enough that capping rarely matters,
# but the cap is kept for a fair, literature-consistent comparison.
DEFAULT_VOCAB_SIZE = 1000


@dataclass(frozen=True, slots=True)
class VqaExample:
    image_path: Path
    question: str
    question_type: str
    answer: str


def _load_json(path: Path, key: str) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        return json.load(f)[key]


def load_split_examples(data_root: Path, split: Split) -> list[VqaExample]:
    """Load one split's active (image, question, answer) triples.

    Cross-referencing questions -> answers is by ``question_id``, and questions -> images
    is by ``img_id``. Every active question in this dataset has a matching active answer
    (verified directly against the release: zero unmatched active questions in any split) —
    a raised KeyError here means the on-disk data no longer matches what was verified when
    this loader was written, not a bug in this function.
    """
    questions = _load_json(data_root / f"LR_split_{split}_questions.json", "questions")
    answers = _load_json(data_root / f"LR_split_{split}_answers.json", "answers")

    answer_by_question_id = {a["question_id"]: a for a in answers if a.get("active")}
    images_dir = data_root / "images" / "Images_LR"

    examples: list[VqaExample] = []
    for q in questions:
        if not q.get("active"):
            continue
        answer = answer_by_question_id[q["id"]]
        examples.append(
            VqaExample(
                image_path=images_dir / f"{q['img_id']}.tif",
                question=q["question"],
                question_type=q["type"],
                answer=answer["answer"],
            )
        )
    return examples


class AnswerVocabulary:
    """Maps answer strings to class indices — built from the *training* split only, so
    validation/test evaluation never leaks information about which answers exist there but
    not in training (the standard, literature-consistent way to build this)."""

    def __init__(self, answer_to_index: dict[str, int]) -> None:
        self._answer_to_index = answer_to_index
        self._index_to_answer = {i: a for a, i in answer_to_index.items()}

    @classmethod
    def build(
        cls, examples: list[VqaExample], max_size: int = DEFAULT_VOCAB_SIZE
    ) -> AnswerVocabulary:
        counts = Counter(ex.answer for ex in examples)
        most_common = [answer for answer, _ in counts.most_common(max_size)]
        return cls({answer: i for i, answer in enumerate(most_common)})

    def __len__(self) -> int:
        return len(self._answer_to_index)

    def encode(self, answer: str) -> int | None:
        """Returns ``None`` for an answer outside the vocabulary — the caller decides
        whether to skip the example or treat it as automatically wrong; see
        ``RSVQALowResDataset``'s ``drop_unknown_answers`` flag."""
        return self._answer_to_index.get(answer)

    def decode(self, index: int) -> str:
        return self._index_to_answer[index]

    def to_json(self, path: Path) -> None:
        path.write_text(json.dumps(self._answer_to_index, indent=2), encoding="utf-8")

    @classmethod
    def from_json(cls, path: Path) -> AnswerVocabulary:
        return cls(json.loads(path.read_text(encoding="utf-8")))


class RSVQALowResDataset(Dataset):
    """One (image, question, answer) example per item, ready for a DataLoader.

    Images are returned as PIL Images, not tensors — the image encoder's own
    preprocessing (``timm.data.resolve_data_config`` + its transform) must be applied
    consistently with whatever pretrained checkpoint is in use, so that step lives with the
    model, not the dataset. Question text is likewise returned raw; tokenization is BERT's
    job.
    """

    def __init__(
        self,
        data_root: Path,
        split: Split,
        vocab: AnswerVocabulary,
        *,
        drop_unknown_answers: bool = True,
    ) -> None:
        self._data_root = data_root
        self._vocab = vocab
        examples = load_split_examples(data_root, split)

        if drop_unknown_answers:
            before = len(examples)
            examples = [ex for ex in examples if vocab.encode(ex.answer) is not None]
            dropped = before - len(examples)
            if dropped:
                import logging

                logging.getLogger(__name__).warning(
                    "%s: dropped %d/%d examples whose answer falls outside the %d-answer "
                    "vocabulary (built from the training split).",
                    split,
                    dropped,
                    before,
                    len(vocab),
                )

        self._examples = examples

    def __len__(self) -> int:
        return len(self._examples)

    def __getitem__(self, idx: int) -> dict:
        ex = self._examples[idx]
        with Image.open(ex.image_path) as img:
            image = img.convert("RGB")
        return {
            "image": image,
            "question": ex.question,
            "question_type": ex.question_type,
            "answer": ex.answer,
            "label": self._vocab.encode(ex.answer),
        }
