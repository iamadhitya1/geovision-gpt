"""A dataset over plain ``VqaExample`` lists, so RSVQA-LR and RSVQA-HR (and any later source)
look identical to the trainer, with optional augmentation applied per item."""

from __future__ import annotations

from collections.abc import Callable

from PIL import Image
from torch.utils.data import Dataset

from dataset.rsvqa_lr_dataset import AnswerVocabulary, VqaExample

Augmentation = Callable[[Image.Image, str], Image.Image]


class VqaExamplesDataset(Dataset):
    """Every example's answer must be in ``vocab`` — filter first with ``in_vocabulary``.
    Images are returned as PIL images; the model's own transform runs later, per batch."""

    def __init__(
        self,
        examples: list[VqaExample],
        vocab: AnswerVocabulary,
        augment: Augmentation | None = None,
    ) -> None:
        self._examples = examples
        self._vocab = vocab
        self.augment = augment  # public: the trainer reseeds it inside each worker process

    def __len__(self) -> int:
        return len(self._examples)

    def __getitem__(self, index: int) -> dict:
        example = self._examples[index]
        with Image.open(example.image_path) as opened:
            image = opened.convert("RGB")
        if self.augment is not None:
            image = self.augment(image, example.question)
        label = self._vocab.encode(example.answer)
        assert label is not None, "filter examples with in_vocabulary() before building the dataset"
        return {
            "image": image,
            "question": example.question,
            "question_type": example.question_type,
            "answer": example.answer,
            "label": label,
        }


def in_vocabulary(examples: list[VqaExample], vocab: AnswerVocabulary) -> list[VqaExample]:
    return [e for e in examples if vocab.encode(e.answer) is not None]
