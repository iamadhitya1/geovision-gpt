"""The augmentation must never change the true answer.

The dangerous case is geometry: mirroring or rotating an image changes the correct answer to a
question like "is there a road on the left of the building". These tests pin that down, along
with the basics (size preserved, seeded runs repeat, brightness actually changes).
"""

from __future__ import annotations

import random

import numpy as np
from PIL import Image

from dataset.augment import AugmentConfig, TrainAugmentation, mentions_direction

GEOMETRY_ONLY = AugmentConfig(
    color_probability=0.0, degrade_probability=0.0, geometry_probability=1.0
)
COLOUR_ONLY = AugmentConfig(
    color_probability=1.0, degrade_probability=0.0, geometry_probability=0.0
)


def _image(seed: int = 0, size: int = 32) -> Image.Image:
    rng = np.random.default_rng(seed)
    return Image.fromarray(rng.integers(0, 255, (size, size, 3), dtype=np.uint8))


def test_directional_questions_are_never_mirrored_or_rotated() -> None:
    image = _image()
    augment = TrainAugmentation(GEOMETRY_ONLY, random.Random(0))
    for question in (
        "Is there a road on the left of the building?",
        "Is a water area at the top of a road present?",
        "Is there a forest next to a building?",
    ):
        for _ in range(50):
            assert np.array_equal(np.asarray(augment(image, question)), np.asarray(image))


def test_non_directional_questions_do_get_geometric_changes() -> None:
    image = _image()
    augment = TrainAugmentation(GEOMETRY_ONLY, random.Random(0))
    outputs = {np.asarray(augment(image, "Is there a road?")).tobytes() for _ in range(60)}
    assert len(outputs) > 1
    assert np.asarray(image).tobytes() not in outputs or len(outputs) > 1


def test_output_size_is_preserved() -> None:
    augment = TrainAugmentation(rng=random.Random(1))
    for size in (32, 64, 100):
        assert augment(_image(size=size), "Is there a road?").size == (size, size)


def test_seeded_augmentation_repeats() -> None:
    image = _image()
    first = TrainAugmentation(rng=random.Random(7))(image, "Is there a road?")
    second = TrainAugmentation(rng=random.Random(7))(image, "Is there a road?")
    assert np.array_equal(np.asarray(first), np.asarray(second))


def test_colour_augmentation_changes_pixels_but_not_geometry() -> None:
    image = _image()
    out = TrainAugmentation(COLOUR_ONLY, random.Random(3))(image, "Is there a road?")
    assert not np.array_equal(np.asarray(out), np.asarray(image))
    # Same pixel positions: correlation with the original stays strongly positive.
    a, b = np.asarray(image, dtype=float).ravel(), np.asarray(out, dtype=float).ravel()
    assert np.corrcoef(a, b)[0, 1] > 0.5


def test_direction_detection() -> None:
    assert mentions_direction("Is there a road on the left of a building?")
    assert mentions_direction("Is a commercial building at the top of a road present?")
    assert not mentions_direction("Is it a rural or an urban area")
    assert not mentions_direction("Are there more roads than buildings?")


def test_reseeding_gives_different_workers_different_augmentations() -> None:
    """Two copies of one augmentation (as two data-loader workers get) must diverge once each
    is reseeded — otherwise every worker replays the same 'random' sequence."""
    image = _image()
    copy_a = TrainAugmentation(rng=random.Random(0))
    copy_b = TrainAugmentation(rng=random.Random(0))
    copy_a.reseed(101)
    copy_b.reseed(202)
    out_a = [np.asarray(copy_a(image, "Is there a road?")).tobytes() for _ in range(20)]
    out_b = [np.asarray(copy_b(image, "Is there a road?")).tobytes() for _ in range(20)]
    assert out_a != out_b
