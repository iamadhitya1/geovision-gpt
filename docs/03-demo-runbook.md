# Live demo runbook

A 6–8 minute walkthrough of GeoVisionGPT, plus what to do when something goes wrong.

## Before you present (10 minutes ahead)

1. Plug the laptop in. Close other heavy programs — the models run on the CPU.
2. From the repo root in PowerShell:

   ```powershell
   .\scripts\dev.ps1 demo
   ```

   This starts the backend (`:8000`), builds and serves the
   frontend (`:3000`), and opens the browser. Wait until the browser shows the sign-in page.
3. Sign in with an account you created beforehand. Check the top-right pill says
   **"2 of 2 models online"**. If it says 1 or 0, see *If something breaks* below.
4. Have these test images ready in a folder (all from `ml/dataset/rsvqa_lr/images/Images_LR/`,
   all from the held-out **test** split — the models never saw them in training):
   `243.tif` (urban), `246.tif` (rural), `315.tif` (farmland). Spares: `237.tif` (urban),
   `235.tif` (rural).
5. Have internet on: the map needs live basemap tiles. If the Wi-Fi is unreliable, skip the
   map or show it as a screenshot.

## The walkthrough

**1. What it is (30 s).** "You upload a satellite image, ask a question in plain English,
and two small models answer side by side — one uses simple fusion, the other cross-attention.
Both are trained on two datasets together, not just one, so they generalise beyond a single
benchmark — more on that in the close."

**2. Upload (30 s).** Library → drag in `243.tif`. Point out the metadata it extracted:
size, resolution (about 15 m per pixel), coordinate system, location. "It reads the
GeoTIFF's own geo-information, so the image already knows where on Earth it is."

**3. Ask (2–3 min).** Start with the question where the image decides the answer, so the
audience sees the models actually look at it. Use the suggestion chip "Is it a rural or an urban
area?" (checked with both models):

| Image | Expected | Both models say |
|---|---|---|
| `243.tif` | urban | urban (confidence TBD — re-check before presenting) |
| `246.tif` | rural | rural (confidence TBD — re-check before presenting) |

**Before you present**: the confidence numbers above were measured against the *original*
single-dataset models. The app now serves the widened models (see `ml/README.md`), which
should still get these right (rural/urban is 88–89% overall accuracy now, up from the old
92%/81%) but the exact confidence percentages haven't been re-measured. Run through this
exact walkthrough once yourself and fill in the real numbers — takes two minutes.

"Same question, two images, two different answers — so it's reading the picture, not just the
words." Then, on `315.tif`, show a few presence/comparison questions (both models answer all of
these correctly): "Is a cemetery present?" → yes; "Is there a rectangular road?" → no; "Are there
less roads than residential buildings in the image?" → yes.

Be upfront that these last ones are easier: on this dataset a good guess from the question's
words alone gets them right most of the time (see *The image's real contribution* below).

Point at the answer cards: the answer, the confidence bar, and the runner-up answers. Then
the agree/disagree line under them: "When the two models disagree, that's the interesting
case — and it's why the comparison is the point of the project."

**4. Map (1 min).** Map → the scene sits at its real location; toggle Dark/Satellite; click
the footprint → "Ask questions about this scene".

**5. Dashboard (1 min).** Model accuracy: elementwise vs. cross-attention fusion on the
9,838-question held-out test set, broken down by question type; then the live usage figures.

**6. The honest close (1 min).** "Counting is the weak spot for models this small, and the
dashboard shows it. That's a known limit of the approach, not something I hid."

## The image's real contribution (know this before you're asked)

A predictor that never sees the image and just answers each question with its most common
training answer scores **70.4% overall** on the test set; the models score 72.0% and 72.0%. The
dashboard shows this as a grey "no image" bar. The image clearly matters for rural/urban (56% →
88% / 89%), slightly for presence and comparison, and not at all for counting. Say it before an
examiner does — it's a known weakness of templated benchmarks like RSVQA, and stating it shows
you evaluated the result rather than just reporting it.

## Trying your own image

Any `.tif`, `.png` or `.jpg` uploads. A GeoTIFF with a coordinate system also appears on the map;
a PNG/JPG works for questions but has no location. Tested here: the same scene re-saved as PNG or
JPEG gives the same answers, and a larger picture is resized automatically.

**It is only reliable on imagery that looks like its training data**: Sentinel-2-style RGB at
about 10 m per pixel, a scene roughly 2.5 km across (the dataset's 256 × 256 tiles). Scale still
matters — the robustness study shows a 2× zoom drops rural/urban accuracy from 89% to 70% (see
`ml/README.md`) — though the wider training also softened this: brightness and resolution
changes barely move accuracy anymore, and mirroring/rotating a non-directional question's image
almost never flips the answer. Google Earth or Maps screenshots (far finer than 10 m per pixel)
are still the most likely to mislead, since scale is the one thing the fix didn't fully solve.
Word questions like the dataset's ("Is there a road?", "Are there more buildings than roads?",
"Is it a rural or an urban area?"); free-form wording is answered as its nearest template. There
is no ground truth for your own images, so treat results as illustration, not measurement.

## Questions to avoid on stage

- **Counts above zero** ("How many buildings are there?"). Counting is the weakest category
  (about 28% for both models). If someone asks for one, say so and show the dashboard.
- Questions outside the training style (open-ended "describe this image"). The models pick
  from a fixed list of 1,000 answers and can't write free text. The app shows a
  "low confidence" warning when it's close to guessing.

## Likely questions, and honest answers

**Why not a big model like GPT-4V or Qwen-VL?** The project follows the LiT-4-RSVQA paper's
argument: small models are cheap to train and run, and still do well on this task. The current
models trained in about 95 minutes on one GPU slice — on two datasets combined, RSVQA-LR and
the much larger RSVQA-HR — and a question takes about 0.4 seconds to answer on a laptop CPU
(measured, one model).

**What did the cross-attention experiment show?** "It's still tied with elementwise fusion:
72.0% vs 72.0% overall on 9,838 held-out questions, after widening the training to two
datasets. The honest result is that the richer fusion still didn't measurably help — that
held before the widening and holds after it." Then, if there's time, the story that shows
real engineering: the first attempt scored only 36% because I had collapsed each input to a
single token, so the question had no effect on the output. I found it from the flat training
curve, proved it with a direct test, fixed it by attending over all image patches, and added
a test that guards against it.

**Why widen the training to a second dataset?** "The first version only ever saw one dataset,
one country, one sensor. A robustness check showed the cost of that: just dimming an image
dropped its rural/urban accuracy from 92% to 62%. Retraining on RSVQA-LR plus a much larger,
independent aerial dataset (RSVQA-HR), with image augmentation, fixed that — the same
darkening now only costs 3 points — and the model still correctly answers most questions about
a city it never saw in training, Philadelphia, which is the real test of generalising beyond a
handful of selected images." Full numbers in `ml/README.md`.

**Why is your accuracy higher than the paper's 64.6%?** The paper trained on a much larger
dataset (RSVQAxBEN) and we train directly on RSVQA-LR, so the numbers aren't like-for-like.
Don't claim to have beaten the paper.

**Is it trained on the test images?** No. Accuracy is measured on the official held-out test
split, and every demo image comes from it.

## If something breaks

| Symptom | Fix |
|---|---|
| Pill says "no models loaded" | The checkpoints/vocab are missing from `backend/checkpoints/`. Copy `xcit_nano_elementwise_generalist.pt`, `xcit_nano_cross_attention_generalist.pt`, and `answer_vocab_generalist.json` there and restart. |
| Pill says "1 of 2" | One checkpoint is missing. The demo still works with one; say "the second model isn't loaded on this machine". |
| Blank page / can't sign in | Backend window closed. Re-run `.\scripts\dev.ps1 demo`. |
| Map is grey | No internet for tiles. Skip the map. |
| Answers look random and confidence is tiny (<1%) | A stand-in/early checkpoint is loaded. Check the file names in `backend/checkpoints/`. |
