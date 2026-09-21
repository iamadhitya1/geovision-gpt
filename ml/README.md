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
| Training/evaluation scripts written, full pipeline smoke-tested end to end | **Done** |
| SLURM job scripts written (`ml/slurm/`) | **Done** |
| HPC access | **Granted** — credentials received |
| Baseline (elementwise fusion) trained + evaluated on HPC | **Done** — see result below |
| Cross-attention fusion, first attempt | **Failed by design flaw** — see "First cross-attention attempt" below |
| Cross-attention fusion, corrected version (the actual experiment) | Fixed and locally tested; retrain on HPC pending |

### Baseline result (2026-09-08, job 4297 trained / job 4323 evaluated)

`xcit_nano` + BERT-Tiny + elementwise fusion, 10 epochs, batch-size 64 (reduced from the
paper's 512 — see "On GPU day" for why), tested on RSVQA-LR's true held-out test split
(9,838 examples):

| Metric | Value |
|---|---|
| Overall accuracy | **72.74%** |
| Average accuracy (macro, matches paper's convention) | **75.16%** |
| rural_urban / presence / comp / count | 92.0% / 90.7% / 90.4% / 27.5% |
| Parameters | 8,217,896 |

Above the paper's own reported 64.61% — though not a strict apples-to-apples comparison,
since the paper trained on the much larger RSVQAxBEN (~15M examples derived from
BigEarthNet-S2) rather than RSVQA-LR directly. `count` is the weak point, consistent with
counting being the hardest question type in VQA generally (see the literature in the
project report). The cross-attention comparison against this 75.16% baseline — the
project's actual original contribution — is running now; see `ml/slurm/train_cross_attention.sbatch`.

### First cross-attention attempt (2026-09-17, job 4324 trained / job 4330 evaluated) — a flawed design, not a result

Test-split accuracy was **36.06% overall / 27.07% average** against the baseline's
72.74% / 75.16%, with rural/urban and counting both at exactly 0.0. Validation accuracy sat
at 36.09% or 34.57% from epoch 0 to epoch 9 and training loss never left ~2.4 — the pattern
of a model that outputs nearly the same answer whatever it is asked.

The cause was a design bug, not something cross-attention inherently does. That first
`CrossAttentionFusion` reduced the image and the question to *one token each* and let the
question attend over the image. With a single key, softmax over one item is always exactly 1,
so the attention weights never depended on the question: changing the question changed the
module's output by **0.0** (checked directly), while changing the image did change it. The
classifier therefore never saw the question at all.

**The fix** (`ml/models/vqa_model.py`): cross-attention now runs over *sequences* — every
question-word token attends over all of the image's patch tokens, with a residual connection
and a padding-aware average. `LitRSVQAModel(config, fusion=...)` now builds the fusion itself
(previously `train.py`, `evaluate.py` and the backend each swapped it in by hand, which is how
the bug went unnoticed). `ml/tests/test_fusion.py` now asserts, on random tensors, that both
the question *and* the image change a fusion module's output — the check that would have
caught this immediately.

The old checkpoints are kept as evidence, not deleted, and this failed run is worth a
paragraph in the report: it shows why the two fusion styles are not interchangeable.

## Setup

```bash
cd ml
uv venv --python 3.11
uv pip install -e ".[dev]"
```

## Dataset

[RSVQA-LR](https://zenodo.org/records/6344334) (Lobry, Marcos, Murray & Tuia, IEEE TGRS
2020) — already downloaded to `dataset/rsvqa_lr/`. 772 images (256×256, 3-band, uint8
GeoTIFF, Sentinel-2 over the Netherlands), 77,232 question-answer pairs across four
question types (`presence`, `count`, `comp`, `rural_urban`).

**A format quirk worth knowing** (documented in detail in
`dataset/rsvqa_lr_dataset.py`): each split's JSON files contain a cumulative superset of
entries; only ones flagged `"active": true` belong to that split. This was verified
directly against the file contents, not assumed — filtering on it recovers exactly
57,223 / 10,005 / 10,004 examples for train/val/test, summing to the documented 77,232
total.

To re-download from scratch: see the `curl` commands used originally, or just re-run
against `https://zenodo.org/records/6344334/files/<filename>?download=1` for each of the
12 files listed on the Zenodo page.

## Everyday commands

```bash
# Build the answer vocabulary (run once; already done, output is in dataset/rsvqa_lr/answer_vocab.json)
python scripts/build_vocab.py

# Smoke-test the model — proves the pipeline runs correctly, on ~30 examples, CPU, 1 epoch.
# Not a real training run; just confirms nothing is broken.
python scripts/train.py --smoke

# The model architecture's own smoke test — one forward + backward pass on random data
python models/vqa_model.py
```

## On GPU day

IITRAM's CoE-AIML HPC runs everything through **SLURM** — no direct `python train.py` on
the GPU node, ever (the admin kills it and the job's progress is lost). Access is via the
college intranet only, master node `192.168.16.5`. Full manual: "Cluster Job Manual for
HPC.pdf" from the coordinator's email.

**Confirmed 2026-09-08** via `sinfo` + `scontrol show node gpunode` on a real login: this
cluster has exactly **one** Slurm partition, named `gpu` (the manual's own example name
turned out to be the real one, not a placeholder). Its `Gres` string is
`gpu:ampere:3,gpu:1g.20gb:2,gpu:1g.10gb:4` — three full GPUs tagged generically `ampere`
(GPU 0/1/2 from the coordinator's email; Slurm can't tell them apart by name, so a bare
`--gres=gpu:1` risks landing on reserved GPU 0) plus GPU 3's six MIG slices, two
`1g.20gb` and four `1g.10gb`, matching the email's 20/20/10/10/10/10GB split exactly.
**The only way to guarantee GPU 3's small tier and never GPU 0 is to request a named
slice** — `--gres=gpu:1g.10gb:1` — which both `smoke.sbatch` and `train.sbatch` already
use.

**MIG addressing quirk, also confirmed 2026-09-08 (jobs 4288-4290):** Slurm hands the job
a bare MIG device index in `$CUDA_VISIBLE_DEVICES` (e.g. `5`), but CUDA needs that index's
actual UUID (e.g. `MIG-0790625e-...`, from `nvidia-smi -L`) to see the device at all — the
bare index alone left `torch.cuda.is_available()` silently `False`, no error. Both scripts
now resolve the correct UUID from `nvidia-smi -L` at runtime before launching Python.
Separately, the cluster's driver only supports up to CUDA 12.4 — the default `pip install
torch` pulled a newer-CUDA build that silently fell back to CPU; fixed by installing with
`pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124`.

1. **Log in**: `ssh <username>@192.168.16.5` (Windows: via MobaXterm; Linux: any terminal).
   Change the emailed password with `passwd` on first login.
2. **Upload `ml/`**: zip `ml/` (code + `dataset/rsvqa_lr/` — skip `.venv/` and `checkpoints/`)
   and upload via MobaXterm's file panel, or `scp`/`rsync` from a terminal that can reach
   the intranet. Don't try to write this through `nano` — only the `.sbatch` scripts
   themselves have to be typed/pasted directly on the cluster (see step 4).
3. **Pretrained weights: confirmed working, no pre-caching needed.** The training script
   pulls BERT-Tiny and the image encoder's weights from huggingface.co on first run.
   **Confirmed 2026-09-08 on a real job (4289/4290): the GPU node has outbound internet**
   — the download succeeded live, even with `HF_HUB_OFFLINE=1` set in the script (that env
   var is left in as a no-op safety net; it didn't block the live calls this
   `huggingface_hub`/`transformers` version made anyway). No action needed here.
4. **Set up the environment** (HPC gives conda, not uv — see the manual §3.2). Conda isn't
   on PATH by default; load the module first, then (one-time only) `conda init` + reopen
   the shell before `conda activate` works interactively:
   ```bash
   module load anaconda3
   conda create -n geovision python=3.11 -y
   conda init && source ~/.bashrc      # one-time; skip on later logins
   conda activate geovision
   cd ~/geovision-gpt-ml && pip install -e .
   ```
   Confirm CUDA: `python -c "import torch; print(torch.cuda.is_available())"` — prints
   `False` here, correctly: the master node has no GPU, only `gpunode` does, only reachable
   through a submitted job. Don't treat `False` here as an error.

   Note where the env actually lives — `conda create -n geovision` fails to write under
   the shared `/apps/codes/anaconda3/envs/` (no permission), so conda silently falls back
   to `~/.conda/envs/geovision` instead. Confirm with `conda env list`. This matters for
   the next step: `conda activate` inside a non-interactive `sbatch` script is fragile
   (confirmed the hard way — a job failed with `ModuleNotFoundError: No module named
   'torch'` despite installing correctly interactively) even with `module load` +
   `source conda.sh` added. Both `.sbatch` scripts now call
   `~/.conda/envs/geovision/bin/python` directly instead of activating anything — more
   robust than relying on shell-init ordering inside a batch job.
5. **Write and submit the smoke job first** — this is the cheap way to catch environment
   problems (conda env not activating, cache path wrong) before spending the
   one-job-per-user slot on a real run. `ml/slurm/smoke.sbatch` has the full script,
   `--partition=gpu` and `--gres=gpu:1g.10gb:1` already filled in (see above); copy its
   contents into `nano smoke.sbatch` on the cluster, save (`Ctrl+X`, `Y`, `Enter`), then:
   ```bash
   sbatch smoke.sbatch
   squeue -u $USER                    # check it's running, not stuck pending
   tail -f smoke_<job_id>.log         # watch it live
   ```
   Look for `Training complete.` with no errors before moving on.
6. **Submit the real training job**, from `ml/slurm/train.sbatch` — **done**, 2026-09-08
   (job 4297). Runs `--image-encoder xcit_nano --fusion elementwise --epochs 10
   --batch-size 64` (512, the paper's value, OOM'd on this cluster's ~9.5GB MIG slice —
   see the script's own comments). Monitor with `squeue -u $USER` /
   `tail -f train_<job_id>.log`; cancel with `scancel <job_id>` if something's wrong.
   **Don't resubmit while it's pending** — it starts on its own once resources free up.
7. **Evaluate on the test split**, from `ml/slurm/evaluate.sbatch` — **done**, job 4323:
   **72.74% overall / 75.16% average accuracy**, see the Status section above for the
   full breakdown.
8. **The actual experiment** — **in progress**, job 4324 (`ml/slurm/train_cross_attention.sbatch`,
   `ml/slurm/evaluate_cross_attention.sbatch`): identical setup, `--fusion cross_attention`
   instead of `elementwise`. Comparing this against the 75.16% baseline — not just
   reproducing the paper — is the project's own contribution.

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
