# GeoVisionGPT

**A Multi-Modal Visual Question Answering System for Satellite and Earth Observation
Imagery**

B.Tech Project · Backend: FastAPI · Model: LiT-4-RSVQA-style (BERT-Tiny + lightweight ViT)

---

## What this is

Upload or select a satellite scene, ask a question in plain English, and get an answer.
Two tracks, both real:

- **The model** (`ml/`) — a from-first-principles-explained, actually-trained VQA model
  following [LiT-4-RSVQA](https://arxiv.org/abs/2306.00758) (Hackel, Clasen, Ravanbakhsh &
  Demir, TU Berlin, 2023): a lightweight pretrained image transformer + BERT-Tiny, fused,
  classified over the 1,000 most common answers in RSVQA-LR. Trained and evaluated on
  IITRAM's HPC GPU. See [`ml/README.md`](ml/README.md) for the full runbook — the
  [VQA Model Internals study notes](https://claude.ai/code/artifact/5bfca91a-3a98-41ba-87b5-e79441ab1a65)
  cover how every component works, from CNNs and attention up to this exact architecture.
- **The web app** (`backend/` + `frontend/`) — a FastAPI service (auth, image
  management, VQA inference) that serves the trained checkpoints, and a Next.js interface
  on top: image library, a workspace that shows both fusion variants' answers side by
  side, a map of georeferenced scenes, and a dashboard. Real engineering, real tests, but
  secondary to the model for grading purposes.

A training-free verification idea (**SGV** — checking a model's quantitative claims
against physically-measured spectral indices) was the original research angle before this
pivot; whether/how it returns as a layer on top of the trained classifier is still open.
See [`docs/02-research-contribution.md`](docs/02-research-contribution.md) for that
original framing.

## Run it on a new device

Everything needed to go from a fresh clone to the running app, in order. Windows PowerShell
shown; see [`frontend/README.md`](frontend/README.md) for the frontend piece on other
platforms — the rest is the same everywhere `uv` and `gh` run.

```powershell
git clone https://github.com/iamadhitya1/geovision-gpt.git
cd geovision-gpt
```

```powershell
cd backend; uv venv --python 3.11; uv pip install -e ".[dev]"; cd ..
```

```powershell
cd frontend; npm install; cd ..
```

```powershell
gh release download v1.0-trained-models --dir backend\checkpoints --pattern "*.pt" --pattern "*.json"
```

```powershell
.\scripts\dev.ps1 demo
```

The last command opens the app at <http://localhost:3000> after about 40 seconds. Requires:
Python 3.11 with [uv](https://docs.astral.sh/uv/) installed, Node.js 20+, and the
[GitHub CLI](https://cli.github.com/) (`gh`) signed in with access to this private repo.
No GPU is needed to run the app — only for retraining (see [`ml/README.md`](ml/README.md)).

The first launch also needs internet briefly: the model downloads two small pretrained
pieces (BERT-Tiny, XCiT-Nano) from Hugging Face before loading the trained weights on top,
then caches them for next time.

### Everything the app needs that isn't in git

Following the steps above is enough to run the app. These are the pieces that live outside the
repository (too large for git, or someone else's data), what each is for, and where it comes
from — nothing is hidden on one particular laptop.

| Asset | Size | Needed for | Where it comes from |
|---|---|---|---|
| Trained VQA checkpoints + answer vocabulary | ~35 MB each | Answering questions | `gh release download v1.0-trained-models …` (step above) |
| BERT-Tiny and XCiT-Nano pretrained weights | ~30 MB | Loading the trained models | Hugging Face, fetched automatically on first launch and cached |
| CLIPSeg segmentation weights (`CIDAS/clipseg-rd64-refined`) | ~577 MB | The **"highlight … in the image"** button for **water**, and for buildings when map data is unavailable | Hugging Face, fetched automatically the first time that button is used — or fetch it ahead of time with the command below |
| Map data (OpenStreetMap, via the free Overpass servers) | small, cached on disk | The highlight button for **roads, buildings, forest, grass, farmland and cemeteries** (needs the image's location) | Streamed from the internet when first needed, then kept in `backend/storage/map_cache/`. The free servers are unreliable — run `python scripts/warm_map_cache.py` once, online, to store the places you'll demo (see the runbook). Map data © OpenStreetMap contributors (ODbL) |
| Live basemap tiles (Esri) | — | The Map page | Streamed from the internet while the page is open; nothing to install |
| Demo images | ~100 scenes | Rehearsing the demo | The RSVQA-LR test split. Download `Images_LR.zip` from [Zenodo 6344334](https://zenodo.org/records/6344334) and use the images listed in `LR_split_test_images.json`; the runbook names the ones that work well |
| RSVQA-LR and RSVQA-HR datasets | ~GBs | **Retraining only**, not running the app | See [`ml/README.md`](ml/README.md) |

To fetch the segmentation weights ahead of time (so the first highlight click isn't slow, or
before going offline), run this once with the backend's Python. It skips the duplicate `.bin`
copy of the weights that the Hugging Face repo also ships:

```powershell
backend\.venv\Scripts\python.exe -c "from huggingface_hub import snapshot_download; snapshot_download('CIDAS/clipseg-rd64-refined', allow_patterns=['*.json','*.txt','model.safetensors'])"
```

Highlighting **vegetation** uses a plain colour index and needs no download at all. If the
CLIPSeg weights can't be fetched (offline, first use), water highlights return a clear
"segmentation model isn't available" message and everything else keeps working. To turn
highlighting off entirely, set `GVG_SEGMENTATION_ENABLED=false`; to stop it using the internet
for map data, set `GVG_MAP_DATA_ENABLED=false`.

## Status

| Track | Scope | Status |
|---|---|---|
| **Model** | Dataset (RSVQA-LR), architecture, training/eval scripts | **Complete** |
| **Model** | Widened training (RSVQA-LR + RSVQA-HR, augmentation), elementwise fusion — what the app serves | **Complete** — mean of 3 training runs: 72.5% ± 0.7 overall on RSVQA-LR (no regression vs. the single-dataset baseline), 82.7% ± 0.8 on RSVQA-HR's own cities, 77.4% ± 0.4 on an unseen city (Philadelphia); the dimmed-image robustness weakness below is fixed (62%→88%, seed 0). Details in [`ml/README.md`](ml/README.md) |
| **Model** | Widened training, cross-attention fusion — what the app serves | **Complete** — mean of 3 training runs: 71.2% ± 0.8 overall on RSVQA-LR, 78.7% ± 1.8 on RSVQA-HR, 76.1% ± 1.7 on the unseen city. **Slightly worse and less steady than elementwise** (every elementwise run beat every cross-attention run on RSVQA-LR and RSVQA-HR; by 0.06 points at the closest on RSVQA-LR). The richer fusion did not help. With one run each (seed 0) the two looked tied — that was luck of the draw |
| **Model** | Significance test on the elementwise-vs-cross-attention tie | **Complete, but superseded** — McNemar's test (p=0.84) and a 10,000-resample bootstrap (95% CI contains 0) show the two **seed-0** models are not distinguishable on those questions. It only compares one trained model per fusion, so it could not see run-to-run variation; the three-run comparison above is the stronger evidence. Details in [`ml/README.md`](ml/README.md) |
| **Model** | Attention heatmap — where cross-attention looked, shown in the app | **Complete** — toggle on the cross-attention answer card; verified live to concentrate on the relevant scene content, not scattered randomly |
| **Model** | Counting, answered as ranges (0 / 1–10 / 11–100 / 101–1,000 / >1,000) like the RSVQA paper scores it | **Complete** — summing probability per range instead of picking one exact number lifts counts placed in the right range to **68.1%** (elementwise, 3 runs) and **58.6%** (cross-attention, 3 runs), against **57.9%** for a guess that never looks at the image — so the image helps elementwise counting by ~10 points and cross-attention's by almost nothing. Exact-match stays ~28%, which is why it was misleading. No retraining. A counting-aware training loss (`ml/models/count_loss.py`, off by default) is written and tested but **not yet run on the cluster**; no result is claimed. Caveats in [`ml/README.md`](ml/README.md) |
| Web app | "Highlight … in the image": whatever the question names — vegetation, forest, grass, farmland, water, buildings, roads, cemeteries; **both things** for a comparison | **Complete, with stated limits** — never the trained VQA model, which cannot locate anything. Vegetation and water come from the image (colour index; CLIPSeg). Roads, buildings, forest, grass, farmland and cemeteries are **OpenStreetMap map data** drawn at the scene's coordinates, because at 10 m per pixel the image alone can't show them — labelled as map data on screen. Needs a georeferenced image and (once) internet; no accuracy measured, since there are no labelled masks |
| Web app | Question guard — what can be asked | **Complete** — the models only know RSVQA's four question kinds (presence, count, comparison, rural/urban). Loosely-worded questions are read as the nearest template and the answer card says so ("Is there a river?" → "Is there a water area?"); questions of another kind ("describe this", colours) are turned away with a list of what can be asked instead of a confident guess. Listed exactly in [`docs/03-demo-runbook.md`](docs/03-demo-runbook.md) |
| Web app | Live map: search any place, **Analyse this place** | **Complete** — fetches a 256 × 256 px, ~10 m Sentinel-2 scene (the scale the models were trained on) from the map, stores it like an upload, and opens it in the Q&A workspace |
| **Model** | Original single-dataset baseline (elementwise fusion) — superseded, kept for reference | 72.74% overall / 75.16% average on RSVQA-LR test; dimming an image dropped its rural/urban accuracy from 92%→62%, which is what motivated the widened training above |
| **Model** | Original single-dataset cross-attention (the project's first experiment) — superseded, kept for reference | 72.33% overall / 72.29% average: effectively tied with the original baseline; details and caveats in [`ml/README.md`](ml/README.md) |
| Web app — Module 1 | Authentication, RBAC, audit | **Complete** |
| Web app — Module 2 | Satellite image management | **Complete** |
| Web app — Module 3/4 | VQA inference service + model comparison API | **Complete** |
| Web app — Module 5 | Frontend: login, image library, side-by-side Q&A workspace | **Complete** |
| Web app — Module 6 | Map view of georeferenced scenes | **Complete** |
| Web app — Module 7 | Dashboard: model accuracy + live usage stats | **Complete** |

See "Run it on a new device" above to go from a fresh clone to the running app.

## Where the trained models actually live

Not in git — they're 33–37 MB each. Three copies exist, in order of how current they are:
the HPC cluster (`~/geovision-gpt-ml/checkpoints/`, the source of truth while a model is
still training), this repository's GitHub release `v1.0-trained-models` (what "Run it on a
new device" above pulls from — kept in step with the cluster after each result lands), and
whatever's currently in `backend/checkpoints/` on a given machine. To retrain from scratch,
see [`ml/README.md`](ml/README.md); the downloaded datasets (RSVQA-LR and RSVQA-HR) aren't
stored here either — that doc explains where to fetch them.

Note: this repository is private. It names the institute in a few places (this README, the ml
README, the report generator), so remove those first if you ever make it public.

## Backend only, no frontend (API via /docs)

For working on the API alone:

```bash
cd backend
uv venv --python 3.11
uv pip install -e ".[dev]"
GVG_SECRET_KEY=$(python -c "import secrets;print(secrets.token_urlsafe(64))") uvicorn app.main:app --reload
```

On Windows PowerShell:

```powershell
cd backend
uv venv --python 3.11
uv pip install -e ".[dev]"
$env:GVG_SECRET_KEY = "local-dev-secret-key-at-least-32-characters-long"; uvicorn app.main:app --reload
```

Then open <http://localhost:8000/docs>. Without trained checkpoints in `backend/checkpoints/`
the app still starts and reports "no models loaded" for `/vqa/fusions`.

On first start the bootstrap administrator is created from `GVG_FIRST_ADMIN_*`
(default `admin@geovisiongpt.io` / `ChangeMe_Please_2026`). **Change it.** In production the
app refuses to start with the default password.

## Full stack (PostgreSQL + Redis + MinIO + NGINX)

```bash
docker compose up -d
```

```bash
docker compose exec api alembic upgrade head
```

Enter through NGINX at <http://localhost/api/v1/> — the `api` container has no port
published to the host on purpose. NGINX is the trust boundary the per-IP rate limiter
relies on (it overwrites `X-Forwarded-For` with the real connecting IP); talking to `api`
directly would let a caller spoof that header and bypass rate limiting.

## Deployment profiles

Selected entirely by environment variables — no code changes.

| Profile | Database | Cache | Storage | Needs |
|---|---|---|---|---|
| `local-lite` | SQLite | in-memory | filesystem | nothing |
| `local-full` | PostgreSQL 16 + PostGIS | Redis 7 | MinIO | Docker |
| `production` | managed Postgres | managed Redis | S3-compatible | cloud |

The in-memory cache is **rejected at startup in production**: it is per-process, so rate
limits and session revocation would silently stop holding across replicas.

## Implemented API

Base path `/api/v1`.

| Method | Path | Access | Purpose |
|---|---|---|---|
| `POST` | `/auth/register` | public | Create a `researcher` account, returns tokens |
| `POST` | `/auth/login` | public | Exchange credentials for a token pair |
| `POST` | `/auth/refresh` | public | Rotate a refresh token (single-use) |
| `POST` | `/auth/logout` | public | End one session (idempotent) |
| `POST` | `/auth/logout-all` | authenticated | End every session |
| `POST` | `/auth/change-password` | authenticated | Change password; revokes all sessions |
| `GET` | `/auth/me` | authenticated | Current user |
| `GET` | `/users/me` | authenticated | Own profile |
| `PATCH` | `/users/me` | authenticated | Update own profile (name only) |
| `GET` | `/users` | **admin** | List users, filter and paginate |
| `GET` | `/users/{id}` | **admin** | Read a user |
| `PATCH` | `/users/{id}` | **admin** | Change role or activation |
| `GET` | `/users/{id}/audit` | **admin** | Audit trail |
| `POST` | `/images` | researcher+ | Upload a GeoTIFF/PNG/JPEG |
| `GET` | `/images` | researcher+ | List images (own only, unless admin) |
| `GET` | `/images/{id}` | researcher+ | Read an image's extracted metadata |
| `GET` | `/images/{id}/thumbnail` | researcher+ | Fetch the JPEG preview |
| `DELETE` | `/images/{id}` | researcher+ | Delete an image (owner or admin) |
| `GET` | `/health` | public | Liveness |
| `GET` | `/ready` | public | Readiness (checks DB and cache) |

### Module 2 — satellite image management

- Uploads are **content-sniffed, not trusted by extension** — a renamed file that doesn't
  match its claimed type is rejected before anything is stored.
- GeoTIFF metadata extraction via rasterio: CRS, native bounds, bounds reprojected to
  WGS84, ground sample distance (approximated in metres for geographic CRSs), band count
  and dtype. A TIFF with no embedded CRS is accepted, not rejected — the geospatial fields
  are simply `null`.
- Cloud coverage is a **brightness-ratio heuristic**, not a real cloud mask — Module 3 owns
  proper cloud handling. Documented as a heuristic everywhere it's surfaced.
- Thumbnails are generated from a downsampled read (never the full-resolution raster) and
  served as JPEG.
- Storage is swappable by configuration: filesystem locally, MinIO/S3 behind
  `GVG_STORAGE_BACKEND=minio`, same `ObjectStorage` port either way.
- `POST /images/capture` fetches a Sentinel-2 scene for a latitude/longitude (the Map page's
  **Analyse this place**) and stores it as the caller's own image, through the same pipeline
  as an upload. Imagery is Sentinel-2 cloudless by EOX IT Services GmbH, free for
  non-commercial use with attribution; needs internet. `GVG_IMAGERY_ENABLED=false` turns it
  off.

### Modules 3/4 — questions, counting and highlighting

- `POST /images/{id}/ask` answers with both fusion variants. **Counting questions** ("How
  many …", "What is the number of …") are answered as one of five ranges, with a note saying
  why; every other question is answered as before.
- `POST /images/{id}/attention` returns cross-attention's attention grid (interpretability).
- `POST /images/{id}/highlight` returns one overlay locating the thing(s) the question names —
  one thing, or two for a comparison ("more roads than buildings?"), each in its own colour.
  A question naming nothing supported is a 422. Each layer says **which method** found it (a
  colour index, zero-shot CLIPSeg, or OpenStreetMap map data) and **how far to trust it**; none
  of them is the trained VQA model. Size and shape words in the question ("large",
  "circular") are not used, and the response says so. `GVG_SEGMENTATION_ENABLED=false` and
  `GVG_MAP_DATA_ENABLED=false` turn it off.

### Security properties

- **Argon2id** password hashing with transparent cost upgrades on login.
- **Single-use refresh tokens with replay detection** — reusing a spent token revokes
  every session for that account, not just the request.
- **One-write global revocation** via `token_version`; password change, deactivation and
  logout-all all use it.
- **No user enumeration** — unknown email and wrong password return an identical response,
  and a dummy hash is computed for unknown emails to keep timing flat.
- **Hierarchical RBAC** — `guest` < `researcher` < `admin`.
- **Privilege escalation is unrepresentable**: `role` is absent from the self-service
  update model, not merely rejected.
- **Per-IP rate limiting** on credential endpoints, with `Retry-After`.
- **Audit trail on success and failure**, retained via `ON DELETE SET NULL`.
- An admin cannot demote or deactivate themselves.

## Development

```bash
cd backend && uv run pytest -q
```

```bash
cd backend && uv run ruff check . && uv run ruff format --check . && uv run mypy app
```

Current state: **103 tests passing**, ruff clean, mypy `--strict` clean.

The suite includes `tests/test_architecture.py`, which parses the AST of every module and
**fails the build** if `domain/` or `application/` import a framework, or if a router
constructs its own dependencies. The layering claim is enforced, not just documented.

## Documentation

| Document | Contents |
|---|---|
| [`docs/00-architecture.md`](docs/00-architecture.md) | Layering, module map, sequence/ER/deployment diagrams |
| [`docs/01-vlm-selection.md`](docs/01-vlm-selection.md) | Comparison of 7 candidate models and the decision |
| [`docs/02-research-contribution.md`](docs/02-research-contribution.md) | SGV framework, hypotheses, ablation plan |

## Tech stack

**Backend** FastAPI · SQLAlchemy 2.0 (async) · Alembic · PostgreSQL + PostGIS · Redis ·
MinIO · Argon2 · PyJWT · structlog
**ML** PyTorch · Transformers · Qwen2.5-VL · DOFA · RemoteCLIP · rasterio · GDAL · OpenCV ·
GeoPandas
**Frontend** Next.js (App Router) · Tailwind · Leaflet
**Ops** Docker · GitHub Actions · NGINX

## Licence

Academic project. Datasets retain their own licences (EarthVQA, BigEarthNet, EuroSAT,
RESISC45, LoveDA, Sentinel-2/Copernicus, OpenStreetMap).
