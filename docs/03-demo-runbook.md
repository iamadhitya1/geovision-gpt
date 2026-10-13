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
4. The test images are on the **Desktop, in `GeoVisionGPT Demo Images`** — no need to dig into
   the repo. All from the held-out **test** split (the models never saw them in training):
   `237.tif` (urban), `246.tif` (rural), `315.tif` (farmland). Spare: `235.tif` (rural).
   **Don't use `243.tif`** — re-verified 2026-09-24 against the current widened models and the
   two now disagree on it (elementwise: urban 71%; cross-attention: rural 59%). Genuinely
   interesting to discuss if it comes up, but not what you want for the "both models agree"
   opening beat.
5. Have internet on: the map needs live basemap tiles, and **Analyse this place** and the place
   search both call the internet. If the Wi-Fi is unreliable, skip the live-map step (6) and
   show the rest from the demo images, which need no internet.
6. **Warm the map-data cache while you're home and online** (once, and again if you add new
   places): from `backend`, `python scripts/warm_map_cache.py` (the three runbook images) and
   e.g. `python scripts/warm_map_cache.py --place 23.0225,72.5714` for each live-map place you'll
   click. Roads, buildings, forest, grass, farmland and cemeteries are drawn from OpenStreetMap
   map data, and the free servers are flaky — a place stored in advance highlights instantly,
   even offline. Re-run it if it reports failures; it only fetches what's missing.
7. The water highlight needs the segmentation weights (~577 MB, one-time download; see
   "Everything the app needs that isn't in git" in the root `README.md`). `dev.ps1 demo` loads
   them in the background at startup. Vegetation needs nothing.

## The walkthrough

**1. What it is (30 s).** "You upload a satellite image, ask a question in plain English,
and two small models answer side by side — one uses simple fusion, the other cross-attention.
Both are trained on two datasets together, not just one, so they generalise beyond a single
benchmark — more on that in the close."

**2. Upload (30 s).** Library → drag in `237.tif`. Point out the metadata it extracted:
size, resolution (about 15 m per pixel), coordinate system, location. "It reads the
GeoTIFF's own geo-information, so the image already knows where on Earth it is."

**3. Ask (2–3 min).** Start with the question where the image decides the answer, so the
audience sees the models actually look at it. Use the suggestion chip "Is it a rural or an urban
area?" (verified live 2026-09-24 against the current widened models):

| Image | Expected | Elementwise | Cross-attention |
|---|---|---|---|
| `237.tif` | urban | urban, 99% | urban, 79% |
| `246.tif` | rural | rural, 100% | rural, 99% |

"Same question, two images, two different answers — so it's reading the picture, not just the
words."

**3b. Where did it look? (1 min, on `237.tif`).** Click **"show where it looked"** on the
cross-attention card. Verified live 2026-10-01: the heatmap lights up almost entirely on the
single patch covering the image's one built-up cluster of buildings — not the surrounding
farmland. "This isn't a detector — it's showing the model's own attention weights, which it
computes anyway on every answer. But it's the first real evidence that cross-attention is
actually looking at the relevant part of the scene, not just coincidentally matching the
other model's accuracy." Click again to hide it before moving on.

Then, on `315.tif`, show a few presence/comparison questions (both models answer all of
these correctly, verified live 2026-09-24):

| Question | Expected | Elementwise | Cross-attention |
|---|---|---|---|
| "Is a cemetery present?" | yes | yes, 86% | yes, 97% |
| "Is there a rectangular road?" | no | no, 90% | no, 100% |
| "Are there less roads than residential buildings in the image?" | yes | yes, 84% | yes, 96% |

Type these exactly — the app's suggestion chips don't cover these three, only the rural/urban
one.

Be upfront that these last ones are easier: on this dataset a good guess from the question's
words alone gets them right most of the time (see *The image's real contribution* below).

Point at the answer cards: the answer, the confidence bar, and the runner-up answers. Then
the agree/disagree line under them: "When the two models disagree, that's the interesting
case — and it's why the comparison is the point of the project."

**3c. Counting (1 min, on `237.tif`).** Click the suggestion chip **"How many buildings are
there?"**. Both models answer as a *range* ("101 to 1,000", "more than 1,000"), with the
next-most-likely ranges underneath and a note saying why. Say: "A count to the exact number
isn't readable from 10 m imagery, so — like the original RSVQA paper — it answers in ranges.
Scored that way the elementwise model is right about 68% of the time, against 58% for a guess
that never looks at the image. Scored by exact number it's about 28%, which is why that number
looked hopeless." Don't promise an exact figure, and expect the two models to differ here.

**3d. Highlight the answer (2 min).** The highlight button appears in a box *below* the
answer cards, and highlights whatever the question names. On `237.tif`:
**"Is there any vegetation?"** → the farmland is picked out and the town dimmed.
**"Is there a road?"** → the real road network is outlined. **"Are there more roads than
buildings?"** → *both* are outlined, in two colours, with a legend. On `246.tif`:
**"Is there a water area?"** → the sea. Say plainly: "The answering model only says yes or no —
it can't point. Vegetation and water are found from the image; roads, buildings, forest and
the like are drawn from OpenStreetMap map data at this scene's coordinates, because at 10
metres per pixel a road is one pixel wide and the image alone can't show it. The panel under
the image says which is which." That honesty is the point: it's map data, not the model.

Don't confuse it with the amber **"show model attention"** button on the cross-attention card:
that is where the model's attention went, **not** where the answer is (on a water question it
can light up the green fields). If asked, say so.

**4. Map and live map (2–3 min).** Map → your uploaded scene sits at its real location;
toggle Dark / Satellite / **Sentinel-2** (the last is the imagery the models actually see).
Type a city in the search box (e.g. **Ahmedabad**) and press Go → the map flies there and a
dashed square shows the 2.5 km area that will be analysed → **Analyse this place** → a
Sentinel-2 scene is fetched and opens in the same Q&A workspace. Ask "Is there a water area?"
and highlight it: the river is outlined. Say: "This is the same scale as the data the models
were trained on — I deliberately don't feed them a sharp Google-Earth screenshot, because the
robustness study shows they're fragile to scale." Needs internet and takes a few seconds.
**Click an already-rehearsed place** (Ahmedabad, rural Gujarat, the NL coast); don't try a
random new one live. On dense city scenes the cross-attention model may call the place
"rural" while elementwise says "urban" — that disagreement is real, and the honest thing is to
point it out.

**5. Dashboard (1 min).** Model accuracy: elementwise vs. cross-attention fusion on the
9,838-question held-out test set, broken down by question type; then the live usage figures.

**6. The honest close (1 min).** "Three limits I'd rather say than have found. Exact counts
aren't readable at this resolution, so it answers in ranges. The highlight is a separate
zero-shot step, not something the answering model learned — good for vegetation, water and
built-up areas, weak for roads. And my first comparison of the two fusions, one training run each, said they
were tied — but repeating each three times showed that was luck: the simple model is a little
ahead and steadier, and the fancier one did not help."

## The image's real contribution (know this before you're asked)

A predictor that never sees the image and just answers each question with its most common
training answer scores **70.4% overall** on the test set; the models score 72.5% (elementwise) and 71.2% (cross-attention), each the mean of three training runs. The
dashboard shows this as a grey "no image" bar. The image clearly matters for rural/urban (56% →
88% / 89%), slightly for presence and comparison, and not at all for *exact* counts. Scored
as the original paper does — in five ranges — the image does help the elementwise model on counting: 68% against 58% for a no-image guess
(the row at the bottom of the dashboard) — but cross-attention gets 59%, barely above that guess. Say it before an examiner does — it's a known weakness of templated benchmarks like RSVQA, and stating it shows
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

## What it can be asked — exactly

The models were trained only on RSVQA's question templates, in four kinds:

| Kind | Looks like | Answer |
|---|---|---|
| Presence | "Is there a road?" · "Is a water area present?" · "Is there a large commercial building next to a water area?" | yes / no |
| Count | "How many buildings are there?" · "What is the number of roads?" | a range (0, 1–10, 11–100, 101–1,000, more than 1,000) |
| Comparison | "Are there more buildings than roads?" · "Are there less roads than water areas?" · "Is the number of buildings equal to the number of roads?" | yes / no |
| Rural / urban | "Is it a rural or an urban area?" | rural / urban |

**Things it knows** (the nouns the training questions use): buildings (and *residential* /
*commercial* buildings), roads, water areas, grass areas, forests, farmland, wetlands, parking,
orchards, scrub, meadows, islands, heaths, farmyards, cemeteries, playgrounds, pitches, gardens,
parks, schools, sports centres, golf courses and a few more map features. Questions may add a
size (small / medium / large), a shape (circular / square / rectangular) or a position ("on the
left of", "next to a …") — those were in the training questions, but they are the least reliable.

**What the app does with other wording** (`backend/app/domain/question_guard.py`):

- A question that already looks like a template goes to the models exactly as written.
- Loose wording is read as the nearest template and the answer card says so, e.g. "Is there a
  river?" → "Is there a water area?", "How many houses?" → "How many residential buildings are
  there?", "Are there fewer buildings than roads?" → "… less buildings than roads?".
- "Vegetation" isn't a trained noun (it knows grass areas, forests and farmland): it is let
  through with a caution, and the highlight is computed from the image colours.
- Anything of a different kind — "Describe this image", "What colour is the water?", "Is the
  road wide?", "How many cars?", "Why …" — is not sent to the models. The app shows what it *can*
  answer instead of a confident guess.

If sir asks something outside this, that message is the answer: the model picks from 1,000 learned
answers, so it works for these kinds of question and says so for the rest.

**Ground truth for the three demo images** (from the dataset's own labels, so you know when an
answer is right): `237.tif` urban, ~1,067 buildings (so "more than 1,000" is right), a road, farmland,
forest and water present, *not* more roads than buildings; `246.tif` rural, ~5 buildings and ~31
road segments, so "Is there a road?" → yes is right; `315.tif` urban, a cemetery, farmland, forest,
grass and roads present.

## Questions to avoid on stage

- **Asking for an exact count.** The app answers as a range ("101 to 1,000") on purpose. If
  someone wants "the exact number", say it can't be read reliably at 10 m per pixel and point
  at the dashboard's two counting rows (28% exact, ~68% in ranges). The two models often
  differ on counts — that's expected.
- **"Is there a road?" on `246.tif`** (mostly open sea with a beach). Both models answer "yes" at
  99–100%, and the dataset's own label agrees (OpenStreetMap records about 31 road segments in
  this scene, mostly small tracks), so it is *correct* — but it looks wrong to the eye at 10 m
  per pixel, and the highlight draws only thin strips. If you show it, say where the label comes
  from; otherwise stick to rural/urban and the water highlight on this image.
- **A highlight with no internet and no stored map data.** Roads and cemeteries then say they
  can't be shown; buildings and green things fall back to the image with a caveat. Warm the
  cache beforehand (see "Before you present").
- **"show model attention" as if it were a locator.** It isn't — see step 3d.
- **Random new places on the live map.** Cross-attention can call a dense city "rural" at 99%
  (it did for Ahmedabad's centre and Mumbai harbour); elementwise got those right. Rehearse the
  places you'll click, and treat any disagreement as something to point out, not hide.
- Questions outside the training style (open-ended "describe this image"). The models pick
  from a fixed list of 1,000 answers and can't write free text. The app shows a
  "low confidence" warning when it's close to guessing.

## Likely questions, and honest answers

**Why not a big model like GPT-4V or Qwen-VL?** The project follows the LiT-4-RSVQA paper's
argument: small models are cheap to train and run, and still do well on this task. The current
models trained in about 95 minutes on one GPU slice — on two datasets combined, RSVQA-LR and
the much larger RSVQA-HR — and a question takes about 0.4 seconds to answer on a laptop CPU
(measured, one model).

**What did the cross-attention experiment show?** "That the fancier fusion did not help. I
trained each design three times, because my first comparison — one run each, 72.0% vs 72.0% —
looked like a tie and I didn't trust it. Across three runs the simple elementwise model is
ahead: 72.5% against 71.2% on the standard test set, and 82.7% against 78.7% on the large aerial
set. All three elementwise runs beat all three cross-attention runs on both; on the unseen city
the runs overlap, so that test alone doesn't separate them. Cross-attention is also less steady
from run to run, and on counting it's barely better than guessing. So the honest result is a
negative one — letting the question attend over the image didn't pay off here." (Don't say 'proved':
with three runs each, the strongest exact rank test can give is p = 0.10; this is consistent,
replicated evidence of a modest edge, not a significance result.) Then, if there's time, the story that shows
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

**Does the model segment — can it show where the answer is?** "The answering model can't: it
classifies, and it was never shown a segmentation mask — none of the datasets I have contain
one. So the highlight is a separate step, and the method depends on what's asked. Vegetation
and water I find from the image — a colour test, and a pretrained segmenter called CLIPSeg. But
at 10 metres per pixel a road is a pixel wide, and a forest looks like any other green field, so
for roads, buildings, forest, grass, farmland and cemeteries I draw OpenStreetMap map data at
the scene's coordinates — and the app labels that as map data, not as something the model
found. The VQA model decides whether the thing is there; this only shows where. I haven't
measured segmentation accuracy because there are no labelled masks to measure against." To
train a real segmenter you would need a labelled land-cover dataset (e.g. LoveDA or
OpenEarthMap) and GPU time; that's future work.

**Can it count?** "It answers counts as a range, the way the original benchmark scores them:
about 68% right for the elementwise model against 58% for a guess that never looks at the image.
Exact numbers aren't readable at 10 m per pixel — scored by exact match it's ~28%, which is why
that number alone looked hopeless. The cross-attention model is about 10 points lower than
elementwise (59% against 68%), and that held on all three of its training runs, so it's a real
property of that design here — it's close to guessing from the question alone."

**Can it tell lengths or areas?** "No. RSVQA-LR has no length or distance questions, and
RSVQA-HR's area questions are exact measurements ('521m2') that a fixed-answer classifier can't
produce, so I excluded them. Measuring would need a detection or segmentation model plus the
image's metres-per-pixel, which the app already reads from the GeoTIFF — a natural next step,
not something it does now."

**Is the live map like Google Earth?** "It's a live map you can search and pan, and 'Analyse
this place' pulls a Sentinel-2 scene for that spot and asks the models about it. I deliberately
use ~10 m Sentinel-2 imagery rather than a sharp screenshot, because the models were trained on
that scale and my robustness study shows they mislead on very different scales. So it analyses
Sentinel-2 imagery of any place, but it isn't Google Earth's resolution, and I show it on places
I've rehearsed — on dense cities one of the two models can still be confidently wrong."

## If something breaks

| Symptom | Fix |
|---|---|
| Pill says "no models loaded" | The checkpoints/vocab are missing from `backend/checkpoints/`. Copy `xcit_nano_elementwise_generalist.pt`, `xcit_nano_cross_attention_generalist.pt`, and `answer_vocab_generalist.json` there and restart. |
| Pill says "1 of 2" | One checkpoint is missing. The demo still works with one; say "the second model isn't loaded on this machine". |
| Blank page / can't sign in | Backend window closed. Re-run `.\scripts\dev.ps1 demo`. |
| Map is grey | No internet for tiles. Skip the map. |
| "Analyse this place" says the imagery service couldn't be reached (502) | No internet, or the imagery service is down. Use an uploaded demo image instead. |
| Highlight says the segmentation model isn't available (503) | Vegetation and map-data layers still work. The CLIPSeg weights (needed for water) aren't downloaded: run the pre-download command in the root `README.md`, restart. |
| Highlight says it can't show roads/cemetery because map data couldn't be reached (502) | The free OpenStreetMap server is busy or you're offline. Try again in a minute, or use a place you warmed up (`scripts/warm_map_cache.py`) — those work offline. |
| First water highlight pauses for ~7 s | The segmentation weights load on first use. `dev.ps1 demo` warms them up at startup — if you started the backend another way, click one water highlight before the audience arrives. |
| "Place search isn't reachable" | OpenStreetMap's geocoder needs internet. Pan and zoom the map instead. |
| Answers look random and confidence is tiny (<1%) | A stand-in/early checkpoint is loaded. Check the file names in `backend/checkpoints/`. |
