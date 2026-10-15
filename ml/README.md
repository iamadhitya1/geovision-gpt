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

> **Superseded (2026-10-08).** This test compares the **seed-0** pair only — one trained model per
> fusion — so it can show those two models are not distinguishable on these questions, but it cannot
> see how much a model changes from one training run to the next. Training each fusion three times
> ("Three training runs each", below) showed that variation is as large as the gap between the
> fusions, and that across runs the elementwise model is consistently a little ahead. The "tie"
> below was real for seed 0 and a misleading summary of the two designs.

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

## Counting: why "28%" was the wrong number, and what the models actually do (2026-10-06)

Counting looked hopeless: both widened models score ~28% on RSVQA-LR's count questions, which
is also what "always answer 0" scores (27%). Two things were wrong with reading that as "the
models can't count":

1. **The metric.** RSVQA-LR's labels are exact integers ("0", "7", "158"…), and exact match
   is a very harsh test at 10 m per pixel. The original RSVQA paper (Lobry et al., 2020,
   Section II-B) quantises low-resolution counts into five ranges — 0 / 1–10 / 11–100 /
   101–1,000 / >1,000 — before scoring, and reports 67.01% on counting that way. Our 28% and
   its 67% were never measuring the same thing. The ranges live in one shared module,
   [`dataset/count_ranges.py`](dataset/count_ranges.py), used by both the evaluation scripts
   and the web app so they can't drift apart.
2. **The decoding.** The classifier spreads a count question's probability over many exact
   numbers ("14", "15", "16"…), so "0" — one answer holding a large share of the training
   labels — can win the argmax even when "somewhere in 11–100" holds far more probability in
   total. Summing probability per range first, then taking the best range, fixes that. It
   needs no retraining.

[`scripts/count_bucket_eval.py`](scripts/count_bucket_eval.py), over RSVQA-LR's 2,947 test
count questions (mean over the training runs; per-run values in "Three training runs each" below):

| | exact number | in ranges, usual argmax | **in ranges, summed per range** |
|---|---|---|---|
| Always guess one answer | 27.1% | ≤ 28.6% | — |
| **No image**: each question's most common range in training | — | **57.9%** | — |
| Elementwise (3 runs scored with the counting metric: 68.1, 68.5, 67.6 in ranges) | 28.4% | 61.8% | **68.1%** |
| Cross-attention (3 runs: 61.7, 56.7, 57.4) | 27.9% | 40.2% | **58.6%** ± 2.7 |

How to read it:

- **The image does help on counting, once it's scored sensibly — for elementwise**: it is about 10
  points above the no-image floor (57.9%). **Cross-attention is only about 1 point above it
  (58.6%)**: on counting it is roughly no better than guessing from the question alone. The
  earlier statement that the image adds nothing on counting is true only for exact match.
- The cross-attention model's low "usual argmax" figure is largely a decoding artefact: it
  answered "0" for 42–65% of count questions (depending on the run), while only 26% of true answers are 0. Summed
  per range it recovers about 18 points — but it still trails elementwise by about 10, and
  that gap **held on every one of its three runs** (61.7, 56.7, 57.4 against 68.1, 68.5 and 67.6), so
  it is a property of this fusion here, not seed luck.
- 68.1% vs the paper's 67.01% is **not** a like-for-like comparison (different models,
  different training data, our answer vocabulary cuts off rare large counts). Don't claim to
  have beaten it.
- The web app answers count questions this way ("101 to 1,000 (56%)", with the next-most-likely
  ranges and a note), and the dashboard shows both the exact-match row and the range row.

Every `evaluate_generalist.py` run now also records these numbers (`lr:test_count_bucketed` in
`<tag>_best_eval.json`), and `scripts/aggregate_seeds.py` reports their mean ± spread over
seeds.

## Pointing at the answer: highlighting, and how it differs from the VQA model (2026-10-06/07)

"Is there vegetation? → yes, *here*." The trained VQA model cannot do the "here": it
classifies, it was never shown a segmentation mask, and none of the datasets used in this
project contain one. So the web app does this with a **separate step that needs no training
data of our own**, says on screen which method found each thing, and never presents it as the
VQA model's doing.

**What an image-only method can honestly find at ~10 m per pixel** — tried with CLIPSeg
(Lüddecke & Ecker, 2022), a pretrained text-prompted segmenter, on the demo scenes:

| Thing | Image-only result | What the app does |
|---|---|---|
| water | Good for open water; bright beach sand also looks like "water" to it, so it is restricted to dark pixels | CLIPSeg + darkness |
| vegetation (any green) | Good | colour index (excess green, `2G − R − B`) |
| forest / grass / farmland | **Cannot be told apart**: prompted "forest", CLIPSeg lit up the farmland — and the town | map data; falls back to the colour index (all green) with a caveat |
| buildings | Shows built-up areas, not buildings | map data; falls back to CLIPSeg with a caveat |
| roads | **Fails**: about one pixel wide | map data only — no honest image-only fallback |
| cemetery | **Garbage**: it picked out a beach | map data only |

So for the things the image cannot show, a georeferenced image gets **OpenStreetMap map data**
drawn at its coordinates (Overpass API; roads, buildings, forest, grass, farmland,
cemeteries). That is real, precise geometry — and RSVQA's own questions and answers were built
from OpenStreetMap — but it is **map data, not something the model found in the image**: it can
be newer or older than the imagery, anything nobody mapped is missing, and roads are drawn wider
than they are so they can be seen. The panel under the image says so, per layer.

Behaviour worth knowing:

- Whatever the question names is highlighted; a comparison ("more roads than buildings?")
  highlights **both**, in two colours, with a legend. Size and shape words ("large",
  "circular") can't be honoured and the response says they weren't used.
- For a presence or count question the highlight is offered only when the models don't both
  say "none". For a comparison it is always offered — there a "no" only means one count isn't
  larger.
- If map data can't be reached, buildings and the green things fall back to the image (flagged
  in the caveat); roads and cemeteries say plainly they can't be shown.
- The free Overpass servers are the least reliable part of the app: the identical query has
  answered in 1.5 s and then returned a 504 on the next try. The provider asks one request at a
  time, retries once when told the server is busy, asks for dense layers (buildings) in four
  smaller pieces, and **caches everything on disk**. `backend/scripts/warm_map_cache.py` fills
  that cache for the places you'll demo, so they work instantly and offline.

This is **not segmentation accuracy**: no labelled masks were available, so there is no
measured IoU. It is a demonstrated, honestly-labelled pointer. Code:
`backend/app/infrastructure/vqa/segmenter.py`, `backend/app/infrastructure/imaging/osm_provider.py`;
the question → things rule is `backend/app/domain/concepts.py`.

The amber **"show model attention"** overlay on the cross-attention card is a different thing
and is *not* a locator: it shows where the model's attention weights went while it answered
(a coarse 14 × 14 grid, averaged over the question's words). On a water question it can light up
the surrounding green fields — plausibly because the model uses them as context for "no water
here". It is interpretability only, and the button and a caption beside the image say so.

## Live imagery from the map (2026-10-06)

The Map page's **Analyse this place** fetches a 256 × 256 px, ~10 m per pixel scene from the
Sentinel-2 cloudless mosaic (EOX IT Services GmbH; free for non-commercial use with
attribution) — the scale the models were trained on, because the robustness study above shows
they are fragile to scale and a screenshot of a sharp basemap would mislead them. The scene
fetched for `237.tif`'s coordinates shows the same town and canal as the real dataset image.

A first look at how the two widened models answer on six such scenes (not a measurement — one
scene each, no ground truth):

- Elementwise gave the answer a person would on rural/urban for all six, including Ahmedabad's
  dense centre (urban, 100%) and Mumbai harbour (urban, 88%), and said "no water" for a
  desert scene (at only 50% confidence).
- **Cross-attention called Ahmedabad's centre and Mumbai harbour "rural" at 99%.** That is
  consistent with the robustness findings (it is the more fragile fusion under a change of
  imagery) and is worth testing properly across seeds rather than waving away.

## Three training runs each (2026-10-05 to 10-08) — and what changed

Every widened-model result earlier in this file came from **one** training run per fusion
(seed 0). Each fusion was trained twice more with different seeds (`TAG=… EXTRA_ARGS="--seed N"`
for `train_generalist.sbatch`, evaluated with `evaluate_generalist.sbatch`), and
`scripts/aggregate_seeds.py` combines them. Overall accuracy, per run:

| Run | Tag | RSVQA-LR | RSVQA-HR | Philadelphia (unseen city) | LR rural/urban (100 Qs) |
|---|---|---|---|---|---|
| Elementwise, seed 0 | `gen_xcit` | 72.02 | 81.87 | 77.21 | 88 |
| Elementwise, seed 1 | `gen_xcit_s1` | 73.30 | 83.35 | 77.01 | 91 |
| Elementwise, seed 2 | `gen_xcit_s2` | 72.22 | 82.99 | 77.83 | 87 |
| **Elementwise, mean** | | **72.51 ± 0.69** | **82.74 ± 0.77** | **77.35 ± 0.43** | 88.7 ± 2.1 |
| Cross-attention, seed 0 | `gen_xcit_xattn` | 71.96 | 80.42 | 74.42 | 89 |
| Cross-attention, seed 1 | `gen_xcit_xattn_s1` | 70.43 | 76.81 | 76.09 | 78 |
| Cross-attention, seed 2 | `gen_xcit_xattn_s2` | 71.07 | 78.76 | 77.77 | 54 |
| **Cross-attention, mean** | | **71.15 ± 0.77** | **78.66 ± 1.80** | **76.09 ± 1.68** | 73.7 ± 17.9 |

(± is the standard deviation across the three runs.) What this does and does not show:

- **The seed-0 "tie" was luck.** Two single runs, 72.02 vs 71.96, looked identical; across three
  runs the elementwise model is ahead by about **1.4 points on RSVQA-LR** and **4.1 on
  RSVQA-HR**. On both, **every elementwise run beat every cross-attention run** — on RSVQA-LR by
  as little as 0.06 points (72.02 vs 71.96), so that edge is small but consistent.
- **On the unseen city the runs overlap** (the best cross-attention run, 77.77, beat the worst
  elementwise run, 77.01), so Philadelphia alone does not separate them. The means still favour
  elementwise (77.4 vs 76.1).
- **Cross-attention is also less stable**: its standard deviation is 1.1× to 3.9× larger on every
  test set, and its rural/urban score swung from 54% to 89%. That type has only 100 test
  questions, so the swing is partly sampling noise — but it is also why cross-attention's
  *average* accuracy (69.6%) is lower and far noisier than elementwise's (74.2%).
- **With three runs per side the strongest p-value an exact rank test can give is 0.10** (all
  three elementwise above all three cross-attention is 1 of 20 possible orderings, one-sided
  0.05, two-sided 0.10). So this is consistent, replicated evidence of a modest advantage, not a
  statistically significant one in the textbook sense. More seeds would sharpen it.
- **Counting** is the clearest difference — see the counting section above: cross-attention
  stays near the no-image floor on all three runs.
- **No claim of superiority for cross-attention survives.** The original hypothesis — that
  letting the question attend over the image would help — was not supported on this data; at best
  it matched elementwise on one lucky seed.

Caveats: three runs is small. The app still serves the seed-0 pair (the deployed
checkpoints), whose overall accuracy matches the cluster evaluation to the hundredth. The
range-summed counting metric exists for 5 of the 6 runs in the main table (the elementwise
seed-1 evaluation predates it, and seed 0 was re-scored on the deployed checkpoints). The
ensemble check below scored all three elementwise runs the same way (on the in-vocabulary count
questions only, so the figures differ by a fraction of a point from the table above): 68.0, 67.9
and 68.4, mean 68.1 — consistent with the 68.1 mean of the three runs in the counting table
(68.1, 68.5, 67.6; the third is run 4, below).

### A fourth elementwise run (`gen_xcit_s3`, 2026-10-13)

A fourth elementwise run was trained and evaluated after the comparison above was written. It
scored 72.70% on RSVQA-LR, 83.23% on RSVQA-HR's own cities and 79.31% on Philadelphia, and 67.6%
on counting in ranges. With it, the elementwise means over four runs are **72.56 ± 0.57**
(RSVQA-LR), **82.86 ± 0.68** (RSVQA-HR) and **77.84 ± 1.03** (Philadelphia), against the
unchanged three cross-attention runs (71.15, 78.66, 76.09): gaps of about +1.4, +4.2 and +1.8
points. All four elementwise runs are above all three cross-attention runs on RSVQA-LR and
RSVQA-HR; on Philadelphia they still overlap. The balanced three-against-three tables above
remain the headline (they are the three runs of each design that the app serves); this run only
confirms the direction. With four runs against three the best an exact rank test can give is about
p = 0.03 one-sided (0.06 two-sided), so it is still not conventionally significant.

### Would averaging the three elementwise runs help? (`scripts/ensemble_eval.py`, 2026-10-08)

A *seed ensemble* averages the three models' answer probabilities and then picks the top answer.
Scored on all 9,838 RSVQA-LR test questions that are in the answer vocabulary:

| | overall | average over question types | counting (5 ranges) |
|---|---|---|---|
| seed 0 / seed 1 / seed 2 | 72.02 / 73.30 / 72.22 | 73.65 / 75.32 / 73.58 | 68.03 / 67.93 / 68.39 |
| mean of the three single runs | 72.51 | 74.18 | 68.12 |
| **ensemble of the three** | **73.02** | **75.15** | **69.08** |

- Against a *typical* single run the ensemble is better by about 0.5 points overall and 1 point on
  the type average and on counting. It smooths out an unlucky seed.
- It does **not** beat the best single run (seed 1: 73.30 overall, 75.32 average), and it costs
  three times the inference and three checkpoints. So it is not the default; it would be an
  optional "robust mode".
- One test set (RSVQA-LR) and three runs: a small, plausible gain, not a tested one.

## Counting-aware training (run once on the cluster; no clear gain) — 2026-10-13/15

Counting in ranges (above) was obtained **without retraining**, by decoding the model's output
differently. The model itself was still trained only to name the exact integer. This is the
training-side follow-up: an auxiliary loss, `models/count_loss.py`, that rewards the probability
the model puts in the *true range* of a count question, computed the same way the app decodes it
(restrict to numeric answers, renormalise, sum per range — see
`backend/app/infrastructure/vqa/count_decoding.py`):

    loss = cross_entropy + w × mean over count questions of  −log( mass in the true range )

It is **off by default** (`--count-loss-weight 0`), so every earlier run reproduces exactly. The
loss itself is unit-tested (`tests/test_count_loss.py`, nine tests: it is small when the mass sits
in the true range, an answer that is off by a few but in the right range costs nothing extra,
non-count rows are ignored, and the gradient pushes mass toward the true range), and a CPU smoke
run of the full trainer with the loss on completed (`train_generalist.py --smoke
--count-loss-weight 0.5`). It was then run once on the cluster; the result is at the end of this
section.

To run it, one job at a time, on the cluster (copy the vocabulary first, as for every new tag;
`w = 0.5` is an untuned first guess — one run per design, compared against the three-run means
above, is what would tell you whether it helps):

```bash
cd ~/geovision-gpt-ml
cp checkpoints/gen_xcit_answer_vocab.json checkpoints/gen_xcit_cnt_answer_vocab.json
TAG=gen_xcit_cnt FUSION=elementwise EXTRA_ARGS="--count-loss-weight 0.5" sbatch train_generalist.sbatch
# when it finishes ("Training complete" in the log):
TAG=gen_xcit_cnt sbatch --time=03:00:00 evaluate_generalist.sbatch
```

`evaluate_generalist.py` already writes the counting-in-ranges figure, so the result is directly
comparable with the elementwise runs (68.1% in ranges, 61.8% by the usual argmax, 28% exact;
no-image floor 57.9%). The success criterion to decide in advance: counting in ranges clearly
above the 68% band of the existing runs (they span 67.6 to 68.5) **and** overall, presence and
comparison accuracy not lower than the existing runs by more than their run-to-run spread.

### Result (tag `gen_xcit_cnt`, elementwise, `--count-loss-weight 0.5`, default seed, jobs 4579 and 4580)

The criterion above was written before the run and is applied unchanged. **It was not met.**

| | `gen_xcit_cnt` (loss on) | `gen_xcit` (same seed, loss off) | other elementwise runs (s1, s2, s3) |
|---|---|---|---|
| Counting in ranges, LR test | 68.65% | 68.14% | 67.6 to 68.5 (the three-run band) |
| LR overall | 71.89 | 72.02 | 72.22 to 73.30 |
| HR overall | 81.56 | 81.87 | 82.99 to 83.35 |
| Philadelphia overall | 75.63 | 77.21 | 77.01 to 79.30 |
| HR comparison | 86.6 | 87.7 | 89.6 to 90.0 |

Counting in ranges is +0.5 over its same-seed baseline and ends 0.15 above the top of the old band,
which is not "clearly above". Philadelphia (the unseen city) is 1.6 points below its same-seed
baseline and below all four earlier elementwise runs; LR and HR overall are slightly below too,
within ordinary run-to-run variation. This is one run at one untuned weight and one seed, so it does
**not** show that the idea cannot work, and the Philadelphia drop cannot be pinned on the loss from a
single run. What it does show is that this setting gave no counting gain worth its apparent cost.

Outcome: the loss stays in the code, off by default; no model, table or app behaviour changed; no
further weights were tried. (The raw evaluation file is git-ignored with the other checkpoints; the
numbers above are copied from it.)

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
