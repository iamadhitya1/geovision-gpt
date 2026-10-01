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
| Cross-attention fusion, corrected version (the actual experiment) | **Done** — see "Cross-attention result" below |
| Widening the model: RSVQA-LR + RSVQA-HR, augmentation (elementwise) | **Done** — generalises to an unseen city, fixes the robustness weakness — see below |
| Widening the model: cross-attention variant | **Done** — same generalisation and robustness gains as the elementwise version — see below |
| Significance test on the elementwise-vs-cross-attention tie | **Done** — McNemar's + bootstrap both confirm the overall tie is real, not unmeasured noise — see below |
| Attention heatmap (cross-attention interpretability) | **Done** — shows where the model looked, verified live to concentrate on the relevant scene content — see below |

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

### Cross-attention result (2026-09-21, job 4332 trained / job 4333 evaluated)

Same setup as the baseline (`xcit_nano` + BERT-Tiny, 10 epochs, batch 64), with the corrected
token-sequence cross-attention fusion (9,269,544 parameters vs. 8,217,896). Held-out test
split, 9,838 questions:

| | Baseline (elementwise) | Cross-attention |
|---|---|---|
| Overall accuracy | **72.74%** | 72.33% |
| Average accuracy | **75.16%** | 72.29% |
| rural_urban (100 questions) | **92.0%** | 81.0% |
| presence (2,955) | **90.7%** | 90.2% |
| comp (4,002) | **90.4%** | 89.2% |
| count (2,781) | 27.5% | **28.8%** |

**Reading it honestly:** the two models are effectively tied overall (0.4 points, about 40
questions). The 3-point gap in *average* accuracy comes almost entirely from rural/urban: only
100 test questions, but the macro average weights it as a quarter of the score, so 11 more
questions right for the baseline moves the average by almost 3 points. On the three large
categories the models are within 1.3 points either way, with cross-attention slightly ahead on
counting. Each model was trained once (one seed), so differences this size are not evidence
that either fusion is better. The result is: **richer cross-attention fusion did not measurably
improve on the published elementwise baseline in this setup**, at about 13% more parameters.
Validation accuracy at epoch 9 tells the same story (73.8% vs. 74.1%).

### How much of the accuracy comes from the question alone? (`scripts/question_only_baseline.py`)

RSVQA-LR's questions are templated, so many answers are predictable from the words alone. A
predictor that **never sees the image** and answers each question with its most common
*training* answer (for that exact question text) scores, on the same 9,838 test questions:

| | No image | Baseline | Cross-attention |
|---|---|---|---|
| Overall | **70.42%** | 72.74% | 72.33% |
| Average | 64.84% | 75.16% | 72.29% |
| rural_urban | 56.0% | 92.0% | 81.0% |
| presence | 88.3% | 90.7% | 90.2% |
| comp | 87.1% | 90.4% | 89.2% |
| count | 28.0% | 27.5% | 28.8% |

So the image adds about **2 points overall**. It matters clearly for rural/urban (56% → 92% /
81%), a little for presence and comparison, and **not at all for counting** — both models are
level with always answering the most common count. This is a known weakness of templated
benchmarks like RSVQA, and it is the honest floor to judge the models against, not 0%. It is why
the dashboard shows a "no image" bar next to each model, and why the live demo leads with
rural/urban questions, where the image decides the answer.

### Robustness study (`scripts/robustness_eval.py`, 2026-09-21) — the measured weakness

Each held-out test question was asked about the original image and about altered copies. Numbers
are accuracy, then in brackets the share of answers that stayed the same as for the original image
(baseline / cross-attention). Brightness, half-resolution, mirroring and rotation do **not** change
the true answer (directional questions like "on the left of" are skipped for mirroring/rotation);
zoom can, so read its "unchanged" figure rather than its accuracy.

**Rural/urban (100 questions — where the image matters most; a no-image guess scores 56%)**

| Change | Baseline | Cross-attention |
|---|---|---|
| Original | 92% | 81% |
| Darker (×0.6) | **62%** (68%) | 66% (51%) |
| Brighter (×1.4) | 88% (94%) | 81% (70%) |
| Half resolution | **58%** (64%) | 73% (62%) |
| Mirrored / rotated 90° | 92% / 91% | 79% / 84% |
| Zoom 1.5× / 2× | 73% / 56% | 86% / 58% |

**Other question types (1,000 questions; ~70% of their accuracy is guessable from the wording)**

| Change | Baseline acc (unchanged) | Cross-attention acc (unchanged) |
|---|---|---|
| Original | 71.5% | 72.0% |
| Darker / brighter | 65.8% (73%) / 69.6% (83%) | 69.8% (78%) / 70.7% (80%) |
| Half resolution | 70.6% (79%) | 71.8% (81%) |
| Mirrored / rotated 90° | 72.5% (87%) / 71.8% (87%) | 72.3% (92%) / 72.5% (92%) |
| Zoom 1.5× / 2× | 70.9% (81%) / 67.0% (74%) | 72.2% (83%) / 70.1% (77%) |

**Reading it:** simply dimming the image drops the baseline's rural/urban accuracy from 92% to 62%,
and halving its resolution to 58% — close to the 56% of ignoring the image. Answers to the other
types barely lose accuracy only because most of it never came from the image. But even there,
**13% of the baseline's answers flip under a mirror or a rotation that cannot change the true
answer** — a model that understood the scene would flip none. Mirroring and rotation do not hurt
rural/urban, so the weakness is appearance and scale, not geometry. This is what
`dataset/augment.py` (brightness/contrast/saturation, resolution loss, and mirror/rotate for
non-directional questions) and the second, aerial-scale dataset (RSVQA-HR) are meant to fix.

## Widening the model — training beyond a single dataset

Both models above are trained on a single dataset (RSVQA-LR: one country, one sensor, one
10m resolution) and the robustness study above shows exactly what that costs: dimming an
image alone dropped rural/urban accuracy from 92% to 62%. Starting 2026-09-21, the model was
retrained on RSVQA-LR *and* RSVQA-HR (Lobry et al., same paper family — 15cm aerial imagery,
~1.07M questions, CC-BY-4.0, [Zenodo record 6344367](https://zenodo.org/records/6344367)),
with label-preserving augmentation (`dataset/augment.py`), to test whether a small model can
generalise beyond the one benchmark it happened to be trained on.

New code, `train.py`/`evaluate.py` untouched so the original baseline stays exactly
reproducible: `dataset/rsvqa_hr_dataset.py` (RSVQA-HR loader — same active-flag file quirk as
RSVQA-LR, but only 4 question types, no `rural_urban`, and `area` answers are excluded as
near-unique strings like `"521m2"` that a fixed vocabulary can't represent),
`scripts/train_generalist.py` (resumable — stops itself before the cluster's 4-hour limit and
picks up from the same step; verified by deliberately interrupting and resuming a run),
`scripts/evaluate_generalist.py` and `scripts/robustness_eval_generalist.py`.

### Elementwise generalist result (2026-09-22/23, `TAG=gen_xcit`, job 4341 trained / jobs 4349, 4364 evaluated)

10 epochs' worth of RSVQA-LR mixed with 60,000 RSVQA-HR questions per epoch (8 epochs total,
best-by-validation checkpoint was epoch 5), batch 64, bf16 mixed precision — completed in one
~95-minute job, no resume needed. Evaluated on three held-out test sets:

| | RSVQA-LR test | RSVQA-HR test (same cities as training) | RSVQA-HR Philadelphia (**unseen city**) |
|---|---|---|---|
| Overall | 72.02% | 81.87% | 77.21% |
| Average | 73.65% | 81.36% | 76.54% |
| Presence | 89.0% | 90.2% | 84.6% |
| Comparison | 89.8% | 87.7% | 84.7% |
| Counting | 27.9% | 66.2% | 60.3% |
| Rural/urban | 88.0% | — (RSVQA-HR has no rural/urban questions) | — |

**Reading it honestly.** Against the original single-dataset baseline (72.74% / 75.16%), the
generalist model costs almost nothing on RSVQA-LR (−0.7 / −1.5 points) — training on a second,
very different dataset did not come at the expense of the first. Moving from RSVQA-HR's own
test cities to Philadelphia, a city the model never saw in any form during training, costs
about 5 points (82%→77% overall) — a real but modest generalisation gap, not a collapse to
guessing. That is the actual evidence for "this model generalises beyond a handful of selected
images": a genuinely unseen city, on a dataset built independently of RSVQA-LR, still gets
answered correctly more than three-quarters of the time.

**One number not to over-read**: RSVQA-HR's 66% counting accuracy is *not* evidence that
counting was fixed — RSVQA-LR's own counting accuracy is unchanged at 27.9%. RSVQA-HR's count
answers are simply a different, easier distribution; counting remains the hard case for this
size of model in general.

### Did the augmentation fix the robustness problem? (`scripts/robustness_eval_generalist.py`, 2026-09-23)

Same perturbations, same RSVQA-LR test images, so this compares directly against the original
baseline's robustness table above.

| Change | Original baseline (rural/urban) | Generalist model (rural/urban) |
|---|---|---|
| Original | 92.0% | 88.0% |
| Darker (×0.6) | **62.0%** (−30 pts) | **88.0%** (±0) |
| Half resolution | **58.0%** (−34 pts) | **85.0%** (−3 pts) |
| Zoom 2× | 56.0% (−36 pts) | 81.0% (−7 pts) |
| Mirrored / rotated 90° | 92% / 91% | 84% / 90% |

On the other 1,000 questions, accuracy held steady or improved slightly under every
perturbation, and — a cleaner signal than accuracy alone — the share of answers that stayed
**exactly the same** as on the clean image rose from the 73–87% range to 90–98% across the
board. **Yes, this is a real, measured fix**: the specific weakness the robustness study found
(appearance and scale, not geometry) is what the second dataset and the augmentation directly
target, and the model that used to lose a third of its accuracy from a dimmer image now barely
notices.

### Cross-attention generalist result (2026-09-23/24, `TAG=gen_xcit_xattn`, job 4365 trained / jobs 4366, 4367 evaluated)

Same recipe as the elementwise generalist above, cross-attention fusion instead of
elementwise, to keep the original baseline-vs-cross-attention comparison alive on the wider
training setup — 8 epochs, best-by-validation checkpoint at epoch 5, one ~95-minute job, no
resume needed. Evaluated on the same three held-out test sets:

| | RSVQA-LR test | RSVQA-HR test (same cities as training) | RSVQA-HR Philadelphia (**unseen city**) |
|---|---|---|---|
| Overall | 71.96% | 80.42% | 74.42% |
| Average | 73.91% | 79.87% | 73.83% |
| Presence | 89.3% | 88.2% | 79.5% |
| Comparison | 89.1% | 86.7% | 80.9% |
| Counting | 28.3% | 64.8% | 61.1% |
| Rural/urban | 89.0% | — | — |

**Reading it honestly.** Against the original single-dataset cross-attention result
(72.33% / 72.29%), the generalist version costs almost nothing on RSVQA-LR (−0.4 points
overall, actually *up* 1.6 points on average) — same pattern as the elementwise generalist:
training on a second dataset didn't cost the first. The unseen-city gap (Philadelphia vs.
RSVQA-HR's own test cities) is about 6 points overall (80.4%→74.4%), essentially the same
size as the elementwise generalist's ~5-point gap — so this isn't an elementwise-only effect,
both fusion styles generalise to a city neither ever saw.

**Robustness** (`scripts/robustness_eval_generalist.py`, 2026-09-24), same perturbations,
same RSVQA-LR test images, compared against the *original* single-dataset cross-attention
model's robustness table above:

| Change | Original cross-attention (rural/urban) | Generalist cross-attention (rural/urban) |
|---|---|---|
| Original | 81.0% | 89.0% |
| Darker (×0.6) | **66.0%** (−15 pts) | **86.0%** (−3 pts) |
| Half resolution | 73.0% (−8 pts) | 81.0% (−8 pts) |
| Zoom 2× | 58.0% | 70.0% |
| Mirrored / rotated 90° | 79% / 84% | 88% / 92% |

Same story as the elementwise fix: the darker-image drop shrinks from 15 points to 3, and
clean-image accuracy itself is also higher (89% vs. 81%) simply from more training data. Half
resolution still costs about the same *relative* amount, so that particular sensitivity isn't
fully solved by this recipe — worth naming as an honest limit rather than claiming a full fix.
The wider 1,000-question set shows the same pattern as the elementwise model: accuracy held
steady across every perturbation (70.4%–72.4%), with 97–99% of answers staying exactly the
same as on the clean image under brightness/resolution/geometry changes.

**Which model does the app serve?** Both — the live demo shows the elementwise and
cross-attention generalists side by side, same as the two original models did, so the
project's core comparison (does richer fusion help?) stays intact on the wider training
setup. Short answer on that comparison: still no measurable win for cross-attention, now
confirmed on top of the generalization and robustness work too.

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

## Is the "tie" real? A significance test (`scripts/significance_test.py`, 2026-10-01)

Every result above calls elementwise and cross-attention "effectively tied" — a judgement
based on the gap being small relative to one training run's noise, not a measurement. This
script makes it a measurement: it runs both currently-deployed widened checkpoints over
RSVQA-LR's test split, records a right/wrong outcome per question for each model (both share
one answer vocabulary, so the filtered question set is identical and in the same order for
both — what makes a *paired* test valid here), then runs two tests.

**McNemar's exact test** — of the 9,838 questions, 582 are "discordant" (the two models
disagree): elementwise alone got 294 of those right, cross-attention alone got 288. A
near-50/50 split like that is exactly what you'd expect if the "tie" is real: **p = 0.84**,
nowhere near significant.

**Paired bootstrap, 10,000 resamples** — the 95% confidence interval on the accuracy gap
(elementwise − cross-attention) is **[−0.43%, +0.55%]**, comfortably containing zero.

Both tests agree: the overall tie is now a measured result, not an assumption.

**One nuance worth reporting honestly.** Broken down by question type, **comparison**
questions (4,002 of them, the largest category) show elementwise ahead 89.78% vs. 89.06%,
**p = 0.048** — just under the conventional 0.05 threshold. Read in isolation that looks like
a real effect. But four question types were tested here (comp, count, presence,
rural/urban), and running four tests raises the chance one comes out under 0.05 by chance
alone; a Bonferroni correction for that requires p < 0.0125 to call a single test
significant, and 0.048 doesn't clear that bar. The honest framing: **a suggestive signal on
comparison questions, not a confirmed one** — worth a dedicated follow-up (a repeat run with
a different seed, specifically on comparison questions) rather than citing as a finding.

| | Overall | Comparison (4,002) | Counting (2,781) | Presence (2,955) | Rural/urban (100) |
|---|---|---|---|---|---|
| p-value | 0.836 | **0.048** | 0.390 | 0.512 | 1.000 |

Reproduce: `python scripts/significance_test.py` (~20-25 minutes on a laptop CPU — no
cluster needed, since it only runs inference on already-trained checkpoints). Results also
written to `ml/checkpoints/significance_test_widened.json`.

## Seeing where cross-attention looks (attention heatmap, 2026-10-01)

Cross-attention's own mechanism — every question word attending over the image's 196
patches — was, until now, never actually *shown*. The model computes these weights on every
forward pass; they just weren't exposed anywhere. `CrossAttentionFusion.forward` and
`LitRSVQAModel.forward` now accept `return_attention=True` (opt-in, every existing call site
is unaffected) and return the raw `(B, L, N)` attention weights alongside the logits.

The backend (`TorchVqaModelRunner.explain`, `POST /images/{id}/attention`) averages those
weights over the question's real (non-padding) words, drops the one summary token XCiT-Nano
prepends (confirmed against timm's own source, not assumed — see the model_runner.py
comment), and reshapes the remaining 196 into the image's 14×14 patch grid. The frontend
renders it as a semi-transparent amber overlay, toggled by a "show where it looked" button on
the cross-attention answer card — elementwise has no per-patch attention to show, so the
button only appears on that card.

**This is interpretability, not a detector.** It shows where the model's attention mass went,
not a verified location for any object — stated plainly in the API response's own docstring
and the domain entity's. Verified live: on `237.tif` (urban, correctly answered by both
models), the heatmap concentrated almost entirely on the single patch covering the image's
one built-up cluster of buildings — not the surrounding farmland — for the question "is it a
rural or an urban area?". That's exactly where a person would look to answer that question,
and it's the first real evidence (beyond the accuracy numbers) that cross-attention's own
mechanism is doing something sensible, not just computing an expensive no-op that happens to
match elementwise's accuracy.

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
