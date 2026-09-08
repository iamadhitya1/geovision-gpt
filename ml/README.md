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
| Real training run | **Next step** — see "On GPU day" |

Everything above was built and verified without a GPU — the code hasn't changed since,
only the SLURM scripts and this runbook were added once access actually arrived.

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

1. **Log in**: `ssh <username>@192.168.16.5` (Windows: via MobaXterm; Linux: any terminal).
   Change the emailed password with `passwd` on first login.
2. **Upload `ml/`**: zip `ml/` (code + `data/rsvqa_lr/` — skip `.venv/` and `checkpoints/`)
   and upload via MobaXterm's file panel, or `scp`/`rsync` from a terminal that can reach
   the intranet. Don't try to write this through `nano` — only the `.sbatch` scripts
   themselves have to be typed/pasted directly on the cluster (see step 4).
3. **Pre-cache the pretrained weights.** The training script pulls BERT-Tiny
   (`google/bert_uncased_L-2_H-128_A-2`) and the image encoder's weights from
   huggingface.co the first time it runs — confirmed by the dev-machine smoke test, which
   made real HTTP calls to `huggingface.co`. Whether the GPU node has outbound internet
   from inside the college network is unverified, so don't assume it and get a training
   job killed by a network timeout after using the one-job slot. Two options:
   - **Try it live first** — if a job fails with a connection error, fall back to below.
   - **Safer: pre-cache and go offline.** On this dev machine, the weights are already
     cached at `~/.cache/huggingface/hub/` (specifically the
     `models--google--bert_uncased_L-2_H-128_A-2` and
     `models--timm--xcit_nano_12_p16_224.fb_dist_in1k` folders — ~117MB combined). Zip
     just those two folders, upload them to the same relative path
     (`~/.cache/huggingface/hub/...`) on the HPC, and both `smoke.sbatch` and
     `train.sbatch` already set `HF_HUB_OFFLINE=1` / `TRANSFORMERS_OFFLINE=1`, which makes
     `transformers` and `timm` read the local cache instead of hitting the network.
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
   Confirm CUDA: `python -c "import torch; print(torch.cuda.is_available())"` → `True`.
   Confirmed real conda path on this cluster: `/apps/codes/anaconda3/etc/profile.d/conda.sh`
   — already baked into both `.sbatch` scripts (batch jobs need `module load anaconda3` +
   this `source` line explicitly, since they don't inherit an interactive shell's init).
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
6. **Submit the real training job** the same way, from `ml/slurm/train.sbatch`:
   ```bash
   sbatch train.sbatch
   ```
   This runs `--image-encoder xcit_nano --fusion elementwise --epochs 10 --batch-size 512`
   — the paper's own best configuration (64.61% average accuracy on their benchmark).
   Monitor with `squeue -u $USER` / `tail -f train_<job_id>.log`; cancel with
   `scancel <job_id>` if something's wrong. **Don't resubmit while it's pending** — it
   starts on its own once resources free up.
7. **Copy the checkpoint back** before it's cleaned up — `scp` or MobaXterm's file panel —
   then evaluate on this machine or on the cluster:
   ```bash
   python scripts/evaluate.py --checkpoint checkpoints/xcit_nano_elementwise_epoch9.pt --split test
   ```
8. **The actual experiment**: copy `train.sbatch`, change `--fusion` to
   `cross_attention`, submit, evaluate, and compare the two results. That comparison —
   not just reproducing the paper — is the project's own contribution.

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
