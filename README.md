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

## Status

| Track | Scope | Status |
|---|---|---|
| **Model** | Dataset (RSVQA-LR), architecture, training/eval scripts | **Complete** |
| **Model** | Baseline (elementwise fusion) trained + evaluated on HPC | **Complete** — 72.74% overall / 75.16% average accuracy on the held-out test split |
| **Model** | Cross-attention fusion (the project's own experiment) | Training run submitted; result pending — see [`ml/README.md`](ml/README.md) |
| Web app — Module 1 | Authentication, RBAC, audit | **Complete** |
| Web app — Module 2 | Satellite image management | **Complete** |
| Web app — Module 3/4 | VQA inference service + model comparison API | **Complete** |
| Web app — Module 5 | Frontend: login, image library, side-by-side Q&A workspace | **Complete** |
| Web app — Module 6 | Map view of georeferenced scenes | **Complete** |
| Web app — Module 7 | Dashboard: model accuracy + live usage stats | **Complete** |

Running it locally: [`backend/`](backend) (`uvicorn app.main:app`) and
[`frontend/`](frontend/README.md) (`npm run dev`). Without trained checkpoints in
`backend/checkpoints/` the app starts fine and reports "no models loaded"; copy the
`.pt` files and `answer_vocab.json` down from the HPC to enable questions.

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
