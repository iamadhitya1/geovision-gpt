"""Training-time image augmentation that keeps the true answer correct.

The robustness study (``scripts/robustness_eval.py``) showed the trained models are brittle to
changes that leave the true answer unchanged: dimming an image cut rural/urban accuracy from 92%
to 62%, and halving its resolution cut it to 58%. This module randomises exactly those things
during training so the model learns not to rely on them.

Only label-preserving changes are used:

* brightness, contrast and saturation
* resolution loss (down-sample then up-sample, as a coarser sensor would give)
* mirroring and 90-degree rotations, but ONLY for questions that do not mention a direction;
  "is there a road on the left of the building" has a different true answer once the image is
  mirrored

Zoom is deliberately absent: cropping can remove the object a question asks about, which would
teach the model wrong labels. Scale diversity has to come from data at other scales instead
(the aerial RSVQA-HR set), not from cropping.
"""

from __future__ import annotations

import random
import re
from dataclasses import dataclass

from PIL import Image, ImageEnhance, ImageOps

DIRECTIONAL = re.compile(r"\b(left|right|top|bottom|next to|above|below)\b", re.IGNORECASE)


@dataclass(frozen=True, slots=True)
class AugmentConfig:
    brightness: tuple[float, float] = (0.5, 1.5)
    contrast: tuple[float, float] = (0.7, 1.3)
    saturation: tuple[float, float] = (0.7, 1.3)
    color_probability: float = 0.8
    degrade_probability: float = 0.3
    degrade_factors: tuple[int, ...] = (2, 3)
    geometry_probability: float = 0.5


def mentions_direction(question: str) -> bool:
    return DIRECTIONAL.search(question) is not None


class TrainAugmentation:
    """Callable ``(image, question) -> image``. Pass a seeded ``random.Random`` for
    reproducible behaviour (the tests do)."""

    def __init__(self, config: AugmentConfig | None = None, rng: random.Random | None = None):
        self.config = config or AugmentConfig()
        self._rng = rng or random.Random()

    def reseed(self, seed: int) -> None:
        """Data-loader workers each get a COPY of this object, so without reseeding every worker
        (and every epoch) would replay the identical "random" sequence."""
        self._rng.seed(seed)

    def __call__(self, image: Image.Image, question: str) -> Image.Image:
        cfg, rng = self.config, self._rng

        if rng.random() < cfg.color_probability:
            image = ImageEnhance.Brightness(image).enhance(rng.uniform(*cfg.brightness))
            image = ImageEnhance.Contrast(image).enhance(rng.uniform(*cfg.contrast))
            image = ImageEnhance.Color(image).enhance(rng.uniform(*cfg.saturation))

        if rng.random() < cfg.degrade_probability:
            factor = rng.choice(cfg.degrade_factors)
            width, height = image.size
            small = image.resize(
                (max(1, width // factor), max(1, height // factor)), Image.Resampling.BICUBIC
            )
            image = small.resize((width, height), Image.Resampling.BICUBIC)

        if not mentions_direction(question) and rng.random() < cfg.geometry_probability:
            operation = rng.choice(("mirror", "flip", "rot90", "rot180", "rot270"))
            if operation == "mirror":
                image = ImageOps.mirror(image)
            elif operation == "flip":
                image = ImageOps.flip(image)
            else:
                image = image.rotate({"rot90": 90, "rot180": 180, "rot270": 270}[operation])

        return image
