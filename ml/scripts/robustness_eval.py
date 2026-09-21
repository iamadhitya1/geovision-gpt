"""How much do the trained models' answers change when the image changes in ways a user's
upload might differ from the training data?

Each held-out test question is asked about the ORIGINAL image and about perturbed copies
(darker, brighter, half resolution, mirrored, rotated, zoomed). Two numbers per perturbation:

* accuracy - how often the answer is still correct
* unchanged - how often the answer is the same as for the original image

Read the perturbations honestly. Brightness and half-resolution do not change the true answer.
Mirroring and rotating do not either, EXCEPT for questions that mention a direction ("on the
left of", "at the top of", "next to"), so those questions are skipped for those two tests. Zoom
CAN change the true answer (a crop may not contain the object any more), so a drop under zoom is
not purely model fragility; "unchanged" is the more meaningful number there.

Rural/urban questions are reported separately: there are only 100 in the test split, but they are
where the image matters most (a no-image guess scores 56% on them).

    python scripts/robustness_eval.py --checkpoints ../backend/checkpoints
"""

from __future__ import annotations

import argparse
import random
import re
import sys
from collections.abc import Callable
from pathlib import Path

import torch
from PIL import Image, ImageEnhance, ImageOps

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dataset.rsvqa_lr_dataset import AnswerVocabulary, VqaExample, load_split_examples
from models.vqa_model import LitRSVQAModel, build_image_transform, build_tokenizer

DATA_ROOT = Path(__file__).resolve().parent.parent / "dataset" / "rsvqa_lr"
VOCAB_PATH = DATA_ROOT / "answer_vocab.json"
DIRECTIONAL = re.compile(r"\b(left|right|top|bottom|next to|above|below)\b", re.IGNORECASE)
BATCH = 128


def zoom(image: Image.Image, factor: float) -> Image.Image:
    width, height = image.size
    crop_w, crop_h = int(width / factor), int(height / factor)
    left, top = (width - crop_w) // 2, (height - crop_h) // 2
    crop = image.crop((left, top, left + crop_w, top + crop_h))
    return crop.resize((width, height), Image.Resampling.BICUBIC)


def half_resolution(image: Image.Image) -> Image.Image:
    width, height = image.size
    small = image.resize((width // 2, height // 2), Image.Resampling.BICUBIC)
    return small.resize((width, height), Image.Resampling.BICUBIC)


# name -> (transform, skip questions that mention a direction, does the true answer stay the same)
PERTURBATIONS: dict[str, tuple[Callable[[Image.Image], Image.Image], bool, bool]] = {
    "original": (lambda im: im, False, True),
    "darker (x0.6)": (lambda im: ImageEnhance.Brightness(im).enhance(0.6), False, True),
    "brighter (x1.4)": (lambda im: ImageEnhance.Brightness(im).enhance(1.4), False, True),
    "half resolution": (half_resolution, False, True),
    "mirrored": (ImageOps.mirror, True, True),
    "rotated 90 deg": (lambda im: im.rotate(90), True, True),
    "zoom 1.5x": (lambda im: zoom(im, 1.5), False, False),
    "zoom 2x": (lambda im: zoom(im, 2.0), False, False),
}


def load_models(checkpoint_dir: Path) -> dict[str, LitRSVQAModel]:
    models = {}
    for fusion in ("elementwise", "cross_attention"):
        checkpoint = torch.load(
            checkpoint_dir / f"xcit_nano_{fusion}_epoch9.pt", map_location="cpu", weights_only=False
        )
        model = LitRSVQAModel(checkpoint["config"], fusion=fusion)
        model.load_state_dict(checkpoint["model_state_dict"])
        models[fusion] = model.eval()
    return models


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoints", type=Path, required=True)
    parser.add_argument("--n", type=int, default=1000, help="non-rural/urban questions to sample")
    parser.add_argument("--seed", type=int, default=5)
    args = parser.parse_args()

    vocab = AnswerVocabulary.from_json(VOCAB_PATH)
    test = [e for e in load_split_examples(DATA_ROOT, "test") if vocab.encode(e.answer) is not None]
    rural = [e for e in test if e.question_type == "rural_urban"]
    others = [e for e in test if e.question_type != "rural_urban"]
    rng = random.Random(args.seed)
    groups = {
        f"rural/urban (n={len(rural)})": rural,
        f"other types (n={args.n})": rng.sample(others, args.n),
    }

    models = load_models(args.checkpoints)
    tokenizer = build_tokenizer()
    transform = build_image_transform("xcit_nano")
    originals: dict[Path, Image.Image] = {}

    def tensor_for(path: Path, name: str, cache: dict[tuple[Path, str], torch.Tensor]):
        if (path, name) not in cache:
            if path not in originals:
                with Image.open(path) as opened:
                    originals[path] = opened.convert("RGB")
            cache[(path, name)] = transform(PERTURBATIONS[name][0](originals[path]))
        return cache[(path, name)]

    def predict(examples: list[VqaExample], name: str, cache: dict) -> dict[str, list[str]]:
        answers: dict[str, list[str]] = {fusion: [] for fusion in models}
        for start in range(0, len(examples), BATCH):
            batch = examples[start : start + BATCH]
            pixels = torch.stack([tensor_for(e.image_path, name, cache) for e in batch])
            encoded = tokenizer(
                [e.question for e in batch], padding=True, truncation=True, return_tensors="pt"
            )
            with torch.no_grad():
                for fusion, model in models.items():
                    logits = model(pixels, encoded["input_ids"], encoded["attention_mask"])
                    answers[fusion] += [vocab.decode(int(i)) for i in logits.argmax(-1)]
        return answers

    for group_name, group in groups.items():
        print(f"\n=== {group_name}", flush=True)
        print(
            f"{'perturbation':18s} {'n':>5s}  "
            + "  ".join(f"{fusion[:5]:>5s} acc  unch" for fusion in models)
            + "   true answer unchanged?",
            flush=True,
        )
        cache: dict[tuple[Path, str], torch.Tensor] = {}
        reference: dict[str, dict[int, str]] = {}
        for name, (_, skip_directional, label_preserving) in PERTURBATIONS.items():
            eligible = [
                (i, e)
                for i, e in enumerate(group)
                if not (skip_directional and DIRECTIONAL.search(e.question))
            ]
            examples = [e for _, e in eligible]
            answers = predict(examples, name, cache)
            cells = []
            for fusion in models:
                acc = sum(a == e.answer for a, e in zip(answers[fusion], examples, strict=True))
                if name == "original":
                    reference[fusion] = {
                        i: a for (i, _), a in zip(eligible, answers[fusion], strict=True)
                    }
                same = sum(
                    reference[fusion][i] == a
                    for (i, _), a in zip(eligible, answers[fusion], strict=True)
                )
                cells.append(f"{acc / len(examples):>8.1%} {same / len(examples):>5.0%}")
            print(
                f"{name:18s} {len(examples):>5d}  "
                + "  ".join(cells)
                + f"   {'yes' if label_preserving else 'NOT necessarily'}",
                flush=True,
            )


if __name__ == "__main__":
    main()
