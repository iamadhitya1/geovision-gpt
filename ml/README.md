# GeoVisionGPT — VQA Model

A LiT-4-RSVQA-style model (Hackel, Clasen, Ravanbakhsh & Demir, TU Berlin, 2023 —
[arXiv:2306.00758](https://arxiv.org/abs/2306.00758)): BERT-Tiny + a lightweight pretrained
image transformer, fused by an elementwise product, classified by an MLP over the
1,000 most frequent answers in RSVQA-LR.

See [`docs/vqa-model-internals`](../docs/) (or the published study-notes artifact) for how
every piece works. This README is the practical runbook.

## Status

| Stage | Status |
|---|---|
| Dataset downloaded, verified, loader tested | **Done** |
| Model architecture implemented, CPU smoke-tested | **Done** |
| Training/evaluation scripts written | **Done** |
| Real training run | **Blocked on IITRAM HPC access** |

Everything above the blocked line was built and verified without a GPU. The moment HPC
access lands, only the commands in "On GPU day" below are new — no code should need to
change.

## Setup

```bash
cd ml
uv venv --python 3.11
uv pip install -e ".[dev]"
```

## Dataset

[RSVQA-LR](https://zenodo.org/records/6344334) (Lobry, Marcos, Murray & Tuia, IEEE TGRS
2020) — already downloaded to `data/rsvqa_lr/`. 772 images (256×256, 3-band, uint8
GeoTIFF, Sentinel-2 over the Netherlands), 77,232 question-answer pairs across four
question types (`presence`, `count`, `comp`, `rural_urban`).

**A format quirk worth knowing** (documented in detail in
`data/rsvqa_lr_dataset.py`): each split's JSON files contain a cumulative superset of
entries; only ones flagged `"active": true` belong to that split. This was verified
directly against the file contents, not assumed — filtering on it recovers exactly
57,223 / 10,005 / 10,004 examples for train/val/test, summing to the documented 77,232
total.

To re-download from scratch: see the `curl` commands used originally, or just re-run
against `https://zenodo.org/records/6344334/files/<filename>?download=1` for each of the
12 files listed on the Zenodo page.

## Everyday commands

```bash
# Build the answer vocabulary (run once; already done, output is in data/rsvqa_lr/answer_vocab.json)
python scripts/build_vocab.py

# Smoke-test the model — proves the pipeline runs correctly, on ~30 examples, CPU, 1 epoch.
# Not a real training run; just confirms nothing is broken.
python scripts/train.py --smoke

# The model architecture's own smoke test — one forward + backward pass on random data
python models/vqa_model.py
```

## On GPU day

1. `ssh` (or however IITRAM's HPC access works) in, clone/pull this repo.
2. `cd ml && uv venv --python 3.11 && uv pip install -e ".[dev]"` — same as above. If the
   cluster has CUDA available, `torch` will detect it automatically; nothing in this
   codebase needs to change for that (see `resolve_device()` in `scripts/train.py`).
3. Confirm: `python -c "import torch; print(torch.cuda.is_available())"` should print `True`.
4. Real training run:
   ```bash
   python scripts/train.py --image-encoder xcit_nano --fusion elementwise --epochs 10 --batch-size 512
   ```
   (`xcit_nano` + `elementwise` reproduces the paper's best-performing configuration —
   64.61% average accuracy on their benchmark. `--batch-size 512` matches the paper; lower
   it if the GPU's memory can't fit that.)
5. Evaluate on the held-out test set:
   ```bash
   python scripts/evaluate.py --checkpoint checkpoints/xcit_nano_elementwise_epoch9.pt --split test
   ```
6. **The actual experiment** — repeat step 4 with `--fusion cross_attention`, then compare
   the two evaluation results. That comparison is the project's own contribution on top of
   the paper's baseline.

## Architecture

```
image ──► ImageEncoder (timm, pretrained, fine-tuned)  ──┐
                                                          ├──► Fusion ──► Classifier ──► answer
question ──► TextEncoder (BERT-Tiny, pretrained, fine-tuned) ──┘
```

Both encoders start from pretrained weights and are fine-tuned, not trained from scratch —
the paper's own ablation shows end-to-end-from-scratch training loses roughly 13 accuracy
points against the identical architecture started from pretrained weights. Fusion starts as
a simple elementwise product (`ElementwiseProductFusion` in `models/vqa_model.py`) with a
`CrossAttentionFusion` alternative already implemented for the experiment in step 6 above.

## Files

```
ml/
├── data/
│   ├── rsvqa_lr_dataset.py   # Dataset class, active-flag filtering, answer vocabulary
│   └── rsvqa_lr/             # downloaded dataset (images/, JSON files, answer_vocab.json)
├── models/
│   └── vqa_model.py          # ImageEncoder, TextEncoder, both Fusion variants, Classifier
├── scripts/
│   ├── build_vocab.py
│   ├── train.py
│   └── evaluate.py
└── checkpoints/               # saved model weights (created on first training run)
```
