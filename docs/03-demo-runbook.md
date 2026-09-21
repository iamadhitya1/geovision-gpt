# Live demo runbook

A 6–8 minute walkthrough of GeoVisionGPT, plus what to do when something goes wrong.

## Before you present (10 minutes ahead)

1. Plug the laptop in. Close other heavy programs — the models run on the CPU.
2. From the repo root in PowerShell:

   ```powershell
   .\scripts\dev.ps1 demo
   ```

   This applies database migrations, starts the backend (`:8000`), builds and serves the
   frontend (`:3000`), and opens the browser. Wait until the browser shows the sign-in page.
3. Sign in with an account you created beforehand. Check the top-right pill says
   **"2 of 2 models online"**. If it says 1 or 0, see *If something breaks* below.
4. Have these test images ready in a folder (all from `ml/dataset/rsvqa_lr/images/Images_LR/`,
   all from the held-out **test** split — the models never saw them in training):
   `243.tif`, `315.tif`, `304.tif`, `237.tif`.
5. Have internet on: the map needs live basemap tiles. If the Wi-Fi is unreliable, skip the
   map or show it as a screenshot.

## The walkthrough

**1. What it is (30 s).** "You upload a satellite image, ask a question in plain English,
and two small models answer side by side. One is a published baseline, the other is the
architecture I built to compare against it."

**2. Upload (30 s).** Library → drag in `243.tif`. Point out the metadata it extracted:
size, resolution (about 15 m per pixel), coordinate system, location. "It reads the
GeoTIFF's own geo-information, so the image already knows where on Earth it is."

**3. Ask (2 min).** Open the image and use these, which the baseline answers correctly:

| Question | Expected |
|---|---|
| Is there a forest? | yes |
| Is a road present? | yes |
| Are there more buildings than roads? | yes |
| What is the amount of square roads? | 0 |

Point at the answer cards: the answer, the confidence bar, and the runner-up answers. Then
the agree/disagree line under them: "When the two models disagree, that's the interesting
case — and it's why the comparison is the point of the project."

**4. Map (1 min).** Map → the scene sits at its real location; toggle Dark/Satellite; click
the footprint → "Ask questions about this scene".

**5. Dashboard (1 min).** Model accuracy: baseline vs. cross-attention on the 9,838-question
held-out test set, broken down by question type; then the live usage figures.

**6. The honest close (1 min).** "Counting is the weak spot for models this small, and the
dashboard shows it. That's a known limit of the approach, not something I hid."

## Questions to avoid on stage

- **Counts above zero** ("How many buildings are there?"). Counting is the weakest category
  (27.5% for the baseline). If someone asks for one, say so and show the dashboard.
- Questions outside the training style (open-ended "describe this image"). The models pick
  from a fixed list of 1,000 answers and can't write free text. The app shows a
  "low confidence" warning when it's close to guessing.

## Likely questions, and honest answers

**Why not a big model like GPT-4V or Qwen-VL?** The project follows the LiT-4-RSVQA paper's
argument: small models are cheap to train and run, and still do well on this task. The
baseline trained in about 73 minutes on one GPU slice, and a question takes about 0.4
seconds to answer on a laptop CPU (measured, one model).

**What did the cross-attention experiment show?** "It tied the baseline: 72.3% vs 72.7%
overall on 9,838 held-out questions. Cross-attention was slightly better at counting and
slightly worse at rural/urban, but that category has only 100 test questions, so I don't read
it as a real difference. The honest result is that the richer fusion didn't measurably help
here." Then, if there's time, the story that shows real engineering: the first attempt scored
only 36% because I had collapsed each input to a single token, so the question had no effect
on the output. I found it from the flat training curve, proved it with a direct test,
fixed it by attending over all image patches, and added a test that guards against it.

**Why is your accuracy higher than the paper's 64.6%?** The paper trained on a much larger
dataset (RSVQAxBEN) and we train directly on RSVQA-LR, so the numbers aren't like-for-like.
Don't claim to have beaten the paper.

**Is it trained on the test images?** No. Accuracy is measured on the official held-out test
split, and every demo image comes from it.

## If something breaks

| Symptom | Fix |
|---|---|
| Pill says "no models loaded" | The checkpoints/vocab are missing from `backend/checkpoints/`. Copy `xcit_nano_elementwise_epoch9.pt`, `xcit_nano_cross_attention_epoch9.pt`, and `answer_vocab.json` there and restart. |
| Pill says "1 of 2" | One checkpoint is missing. The demo still works with one; say "the second model isn't loaded on this machine". |
| Blank page / can't sign in | Backend window closed. Re-run `.\scripts\dev.ps1 demo`. |
| Map is grey | No internet for tiles. Skip the map. |
| Answers look random and confidence is tiny (<1%) | A stand-in/early checkpoint is loaded. Check the file names in `backend/checkpoints/`. |
