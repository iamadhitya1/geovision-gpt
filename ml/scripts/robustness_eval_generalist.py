"""Same robustness check as robustness_eval.py, but for a train_generalist.py checkpoint (its
own answer vocabulary, not the two original baseline/cross-attention checkpoints' shared one).

Tests on RSVQA-LR's test split specifically — not because that's the model's only training
data any more, but so the numbers are directly comparable to the baseline/cross-attention
robustness table already recorded in ml/README.md ("How much of the training-time augmentation
(dataset/augment.py: brightness, resolution loss, flips/rotations) actually closed that gap?").

    python scripts/robustness_eval_generalist.py --tag gen_xcit --which best
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dataset.rsvqa_lr_dataset import AnswerVocabulary, VqaExample, load_split_examples
from models.vqa_model import LitRSVQAModel, build_image_transform, build_tokenizer
from scripts.robustness_eval import DIRECTIONAL, PERTURBATIONS
from scripts.train_generalist import CHECKPOINT_DIR

DATA_ROOT = Path(__file__).resolve().parent.parent / "dataset" / "rsvqa_lr"
BATCH = 128


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tag", required=True)
    parser.add_argument("--which", choices=["best", "final", "latest"], default="best")
    parser.add_argument("--n", type=int, default=1000, help="non-rural/urban questions to sample")
    parser.add_argument("--seed", type=int, default=5)
    args = parser.parse_args()

    vocab = AnswerVocabulary.from_json(CHECKPOINT_DIR / f"{args.tag}_answer_vocab.json")
    checkpoint = torch.load(
        CHECKPOINT_DIR / f"{args.tag}_{args.which}.pt", map_location="cpu", weights_only=False
    )
    model = LitRSVQAModel(checkpoint["config"], fusion=checkpoint["fusion"])
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    print(f"Loaded {args.tag}_{args.which}.pt — {checkpoint['fusion']} fusion, vocab {len(vocab)}")

    test = [e for e in load_split_examples(DATA_ROOT, "test") if vocab.encode(e.answer) is not None]
    rural = [e for e in test if e.question_type == "rural_urban"]
    others = [e for e in test if e.question_type != "rural_urban"]
    rng = random.Random(args.seed)
    groups = {
        f"rural/urban (n={len(rural)})": rural,
        f"other types (n={args.n})": rng.sample(others, min(args.n, len(others))),
    }

    tokenizer = build_tokenizer()
    transform = build_image_transform(checkpoint["config"].image_encoder)
    originals: dict[Path, Image.Image] = {}

    def tensor_for(
        path: Path, name: str, cache: dict[tuple[Path, str], torch.Tensor]
    ) -> torch.Tensor:
        if (path, name) not in cache:
            if path not in originals:
                with Image.open(path) as opened:
                    originals[path] = opened.convert("RGB")
            cache[(path, name)] = transform(PERTURBATIONS[name][0](originals[path]))
        return cache[(path, name)]

    def predict(examples: list[VqaExample], name: str, cache: dict) -> list[str]:
        answers: list[str] = []
        for start in range(0, len(examples), BATCH):
            batch = examples[start : start + BATCH]
            pixels = torch.stack([tensor_for(e.image_path, name, cache) for e in batch])
            encoded = tokenizer(
                [e.question for e in batch], padding=True, truncation=True, return_tensors="pt"
            )
            with torch.no_grad():
                logits = model(pixels, encoded["input_ids"], encoded["attention_mask"])
            answers += [vocab.decode(int(i)) for i in logits.argmax(-1)]
        return answers

    for group_name, group in groups.items():
        print(f"\n=== {group_name}")
        header = f"{'perturbation':18s} {'n':>5s}  {'accuracy':>10s}  {'unchanged':>9s}"
        print(f"{header}   true answer unchanged?")
        cache: dict[tuple[Path, str], torch.Tensor] = {}
        reference: dict[int, str] = {}
        for name, (_, skip_directional, label_preserving) in PERTURBATIONS.items():
            eligible = [
                (i, e)
                for i, e in enumerate(group)
                if not (skip_directional and DIRECTIONAL.search(e.question))
            ]
            examples = [e for _, e in eligible]
            answers = predict(examples, name, cache)
            acc = sum(a == e.answer for a, e in zip(answers, examples, strict=True))
            if name == "original":
                reference = {i: a for (i, _), a in zip(eligible, answers, strict=True)}
            same = sum(reference[i] == a for (i, _), a in zip(eligible, answers, strict=True))
            print(
                f"{name:18s} {len(examples):>5d}  {acc / len(examples):>9.1%}  "
                f"{same / len(examples):>8.0%}   {'yes' if label_preserving else 'NOT necessarily'}"
            )


if __name__ == "__main__":
    main()
