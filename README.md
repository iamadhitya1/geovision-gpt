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

## Status

| Track | Scope | Status |
|---|---|---|
| **Model** | Dataset (RSVQA-LR), architecture, training/eval scripts | **Complete** |
| **Model** | Widened training (RSVQA-LR + RSVQA-HR, augmentation), elementwise fusion — what the app serves | **Complete** — 72.0% overall on RSVQA-LR (no regression vs. the single-dataset baseline), 77.2% on an unseen city (Philadelphia), and the dimmed-image robustness weakness below is fixed (62%→88%). Details in [`ml/README.md`](ml/README.md) |
| **Model** | Widened training, cross-attention fusion — what the app serves | **Complete** — 72.0% overall on RSVQA-LR, 74.4% on the unseen city; still effectively tied with the elementwise model, confirming richer fusion doesn't measurably help here either |
| **Model** | Significance test on the elementwise-vs-cross-attention tie | **Complete** — McNemar's test (p=0.84) and a 10,000-resample bootstrap (95% CI contains 0) both confirm the tie is real, not just an unmeasured small gap. Details in [`ml/README.md`](ml/README.md) |
| **Model** | Attention heatmap — where cross-attention looked, shown in the app | **Complete** — toggle on the cross-attention answer card; verified live to concentrate on the relevant scene content, not scattered randomly |
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
