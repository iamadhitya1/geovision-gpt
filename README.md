# GeoVisionGPT

**A Multi-Modal Visual Question Answering System for Satellite and Earth Observation
Imagery using Vision-Language Models**

B.Tech Final Year Project · Backend: FastAPI · Frontend: Next.js · VLM: Qwen2.5-VL

---

## What this is

Upload or select a satellite scene, ask a question in plain English, and get an answer that
is **measured, not guessed** — grounded in spectral indices computed from the raster itself,
with the responsible region highlighted and a calibrated confidence attached.

```
"What percentage of this image is vegetation?"
→ 34.2% vegetated (NDVI > 0.3), covering 8.1 km² of the 23.7 km² scene.
  Confidence 0.91 · verified against computed NDVI · overlay attached
```

The research contribution is **SGV (Spectral-Grounded Verification)** — a training-free
framework that turns physics-based spectral indices into a verifier for VLM claims. See
[`docs/02-research-contribution.md`](docs/02-research-contribution.md).

## Status

| Module | Scope | Status |
|---|---|---|
| **1** | Authentication, RBAC, audit | **Complete** — 65 tests passing |
| 2 | Satellite image management | Not started |
| 3 | Preprocessing and spectral indices | Not started |
| 4 | Feature extraction | Not started |
| 5 | Question understanding | Not started |
| 6 | Vision-language fusion (SEI) | Not started |
| 7 | Answer generation (CEV) | Not started |
| 8 | Explainable AI | Not started |
| 9 | GIS visualisation | Not started |
| 10 | Analytics dashboard | Not started |

## Quick start (no Docker, no GPU required)

```bash
cd backend
uv venv --python 3.11
uv pip install -e ".[dev]"
```

Set a secret and run:

```bash
GVG_SECRET_KEY=$(python -c "import secrets;print(secrets.token_urlsafe(64))") uvicorn app.main:app --reload
```

On Windows PowerShell:

```powershell
$env:GVG_SECRET_KEY = "local-dev-secret-key-at-least-32-characters-long"; uvicorn app.main:app --reload
```

Then open <http://localhost:8000/docs>.

On first start the bootstrap administrator is created from `GVG_FIRST_ADMIN_*`
(default `admin@geovisiongpt.io` / `ChangeMe_Admin_123`). **Change it.** In production the
app refuses to start with the default password.

## Full stack (PostgreSQL + Redis + MinIO)

```bash
docker compose up -d
```

```bash
docker compose exec api alembic upgrade head
```

## Deployment profiles

Selected entirely by environment variables — no code changes.

| Profile | Database | Cache | Storage | Needs |
|---|---|---|---|---|
| `local-lite` | SQLite | in-memory | filesystem | nothing |
| `local-full` | PostgreSQL 16 + PostGIS | Redis 7 | MinIO | Docker |
| `production` | managed Postgres | managed Redis | S3-compatible | cloud |

The in-memory cache is **rejected at startup in production**: it is per-process, so rate
limits and session revocation would silently stop holding across replicas.

## Module 1 — implemented API

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
| `GET` | `/health` | public | Liveness |
| `GET` | `/ready` | public | Readiness (checks DB and cache) |

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

Current state: **65 tests passing**, ruff clean, mypy `--strict` clean.

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
**Frontend** Next.js · Leaflet/OpenLayers
**Ops** Docker · GitHub Actions · NGINX

## Licence

Academic project. Datasets retain their own licences (EarthVQA, BigEarthNet, EuroSAT,
RESISC45, LoveDA, Sentinel-2/Copernicus, OpenStreetMap).
