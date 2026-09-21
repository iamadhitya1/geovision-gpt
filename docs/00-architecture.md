# GeoVisionGPT — System Architecture

## 1. Architectural style

**Clean Architecture** with four concentric layers. The dependency rule is absolute: source-code
dependencies point *inward only*.

```
          ┌─────────────────────────────────────────────┐
          │  api/          FastAPI routers, DI wiring    │  ← frameworks
          │  ┌───────────────────────────────────────┐  │
          │  │ infrastructure/  SQLAlchemy, Redis,   │  │  ← adapters
          │  │                  MinIO, torch, GDAL   │  │
          │  │  ┌─────────────────────────────────┐  │  │
          │  │  │ application/  use-case services  │  │  │  ← use cases
          │  │  │  ┌───────────────────────────┐   │  │  │
          │  │  │  │ domain/  entities, enums, │   │  │  │  ← enterprise rules
          │  │  │  │          Protocols        │   │  │  │
          │  │  │  └───────────────────────────┘   │  │  │
          │  │  └─────────────────────────────────┘  │  │
          │  └───────────────────────────────────────┘  │
          └─────────────────────────────────────────────┘
```

**Enforced invariants** (checked by `tests/test_architecture.py`):

- `app/domain/**` imports nothing from `sqlalchemy`, `fastapi`, `redis`, or `torch`.
- `app/application/**` imports nothing from `sqlalchemy`, `fastapi`, or `redis`.
- Interfaces are declared as `typing.Protocol` in the domain; concrete classes live in
  `infrastructure` and are bound to routes in `api/deps.py`, which is the composition root.

This is what makes the "swap the VLM backend" and "swap Postgres for SQLite" claims real rather than
aspirational — nothing above `infrastructure` knows either exists.

## 2. High-level architecture

```mermaid
flowchart TB
    subgraph Client
        UI["Next.js 15 SPA<br/>dashboard · map · chat · analytics"]
    end

    subgraph Edge
        NGINX["NGINX<br/>TLS · gzip · static"]
    end

    subgraph API["FastAPI application"]
        MW["Middleware<br/>request-id · logging · rate limit"]
        AUTH["Auth & RBAC<br/>JWT · roles · audit"]
        IMG["Image service"]
        VQA["VQA orchestrator"]
        XAI["Explainability"]
        AN["Analytics"]
    end

    subgraph Workers["Async workers"]
        PRE["Preprocessing<br/>rasterio · GDAL · indices"]
        INF["Inference<br/>Qwen2.5-VL · DOFA"]
    end

    subgraph Data
        PG[("PostgreSQL<br/>+ PostGIS + pgvector")]
        RD[("Redis<br/>cache · rate limit · jti")]
        MO[("MinIO<br/>rasters · overlays · reports")]
    end

    subgraph External
        GEE["Google Earth Engine"]
        COP["Copernicus Open Access"]
        OSM["OpenStreetMap"]
    end

    UI --> NGINX --> MW --> AUTH
    AUTH --> IMG & VQA & XAI & AN
    IMG --> MO
    IMG --> PRE
    VQA --> INF
    XAI --> INF
    PRE --> MO
    INF --> PG
    AUTH --> PG & RD
    AN --> PG
    IMG -.-> GEE & COP
    UI -.-> OSM
```

## 3. Module → layer mapping

**Note:** this table (and the diagrams below it) described the original Phase-0 plan —
Qwen2.5-VL, DOFA, SGV, a fully generative pipeline. That plan was superseded after the
faculty guide's review (see `ml/README.md` and the project report); the model actually
built and trained is LiT-4-RSVQA-style (BERT-Tiny + a lightweight image transformer,
classification over 1,000 answers), and the web app's remaining modules were redesigned
around what that model actually is. Row 1-4 below reflect the real, current system;
rows 5+ and the diagrams in sections 2, 5, 8 still describe the superseded plan and
haven't been rewritten yet — treat them as historical, not current.

| # | Module | Domain | Application | Infrastructure | API |
|---|---|---|---|---|---|
| 1 | Authentication | `User`, `Role`, `AuditEvent` | `AuthService`, `AuditService` | SQLAlchemy repos, Argon2, JWT, Redis | `/api/v1/auth`, `/api/v1/users` |
| 2 | Image management | `SatelliteImage`, `ImageMetadata` | `ImageService` | MinIO, rasterio reader | `/api/v1/images` |
| 3 | VQA inference | `VqaQuery`, `VqaAnswer`, `FusionType` | `VqaService` | `TorchVqaModelRunner` (loads both trained checkpoints) | `/api/v1/images/{id}/ask` |
| 4 | Question/answer API + model comparison | (same as above) | (same as above) | (same as above) | `/api/v1/vqa/fusions`, `/api/v1/images/{id}/vqa` |
| 5 | Frontend | — | — | Next.js + Tailwind (`frontend/`) | separate app, not a backend router |
| 6 | Map view | — | — | Leaflet, reuses Module 2's `bounds_wgs84` | frontend-only |
| 7 | Dashboard | — (derived from `VqaQuery`) | `VqaService.stats` | `list_recent_by_user` | `/api/v1/vqa/stats` |

## 4. Repository layout

```
geovision-gpt/
├── docs/                       architecture, VLM study, research contribution, SRS, UML
├── backend/
│   ├── app/
│   │   ├── core/               config, logging, security primitives, errors
│   │   ├── domain/             entities, enums, Protocols   (zero framework imports)
│   │   ├── application/        use-case services, DTOs
│   │   ├── infrastructure/     db/, cache/, storage/, ml/
│   │   └── api/                deps.py (composition root), middleware, v1/ routers
│   ├── alembic/                migrations
│   └── tests/                  unit, integration, architecture
├── frontend/                   Next.js 15 (Module 2 onward)
├── ml/                         research notebooks, eval harness, SGV experiments
├── infra/                      docker-compose, nginx, k8s
└── scripts/                    dev tooling (PowerShell + shell)
```

## 5. Sequence — authenticated VQA request (target state)

```mermaid
sequenceDiagram
    actor R as Researcher
    participant UI as Next.js
    participant API as FastAPI
    participant RL as RateLimiter
    participant DB as PostgreSQL
    participant PRE as Preprocessor
    participant VLM as Qwen2.5-VL
    participant SGV as SGV Verifier

    R->>UI: ask "what % is vegetation?"
    UI->>API: POST /api/v1/vqa  (Bearer access token)
    API->>RL: check quota(user_id)
    RL-->>API: allowed
    API->>DB: load image + conversation
    API->>PRE: compute NDVI/NDWI/NDBI + area-per-pixel
    PRE-->>API: SpectralEvidence
    API->>VLM: prompt + EVIDENCE block (SEI)
    VLM-->>API: answer + grounding boxes + logprobs
    API->>SGV: extract claims, verify vs evidence
    alt contradiction beyond tau
        SGV->>VLM: regenerate with contradiction surfaced
        VLM-->>SGV: revised answer
    end
    SGV-->>API: answer + calibrated confidence
    API->>DB: persist answer, claims, inference log
    API-->>UI: answer + confidence + overlay URL
```

## 6. Authentication sequence (Module 1 — implemented)

```mermaid
sequenceDiagram
    actor U as User
    participant API as FastAPI
    participant RL as RateLimiter
    participant AS as AuthService
    participant UR as UserRepository
    participant TS as TokenService
    participant AU as AuditService

    U->>API: POST /auth/login {email, password}
    API->>RL: hit("login:<ip>")
    RL-->>API: allowed
    API->>AS: authenticate(credentials)
    AS->>UR: get_by_email(email)
    UR-->>AS: User | None
    AS->>AS: Argon2 verify (constant-time; dummy hash if user absent)
    alt invalid
        AS->>AU: record(LOGIN_FAILED)
        AS-->>API: InvalidCredentialsError
        API-->>U: 401
    else valid
        AS->>TS: issue access + refresh (jti stored)
        AS->>AU: record(LOGIN_SUCCESS)
        AS-->>API: TokenPair
        API-->>U: 200 + tokens
    end
```

## 7. ER diagram — current schema (Modules 1-4)

```mermaid
erDiagram
    USERS {
        uuid   id PK
        string email UK
        string full_name
        string password_hash
        string role
        bool   is_active
        bool   is_verified
        int    token_version
        ts     created_at
        ts     updated_at
        ts     last_login_at
    }
    AUDIT_EVENTS {
        uuid   id PK
        uuid   user_id FK
        string action
        string outcome
        string ip_address
        string user_agent
        json   detail
        ts     created_at
    }
    IMAGES {
        uuid   id PK
        uuid   owner_id FK
        string original_filename
        string image_format
        string status
        string storage_key UK
        string thumbnail_key
        bigint file_size_bytes
        string checksum_sha256
        json   metadata_json
        string error_message
        ts     created_at
        ts     updated_at
    }
    VQA_QUERIES {
        uuid   id PK
        uuid   image_id FK
        uuid   asked_by FK
        string question
        json   answers_json "one entry per fusion variant"
        ts     created_at
    }
    USERS ||--o{ AUDIT_EVENTS : "generates"
    USERS ||--o{ IMAGES : "owns"
    USERS ||--o{ VQA_QUERIES : "asks"
    IMAGES ||--o{ VQA_QUERIES : "is asked about"
```

`answers_json` holds every trained model variant's answer to that one question together
(elementwise baseline, cross-attention experiment), not as separate rows — see
`VqaQuery`'s docstring in `app/domain/entities.py` for why: the baseline-vs-cross-attention
comparison is one fact about a query, not something that could drift out of sync across
rows.

`token_version` is the revocation lever: bumping it invalidates every outstanding refresh token for
that user without needing a database-backed token table.

## 8. Deployment

```mermaid
flowchart LR
    subgraph GH["GitHub Actions"]
        L["lint + type check"] --> T["pytest"] --> B["build images"] --> P["push GHCR"]
    end
    P --> D["docker compose / k8s"]
    subgraph Runtime
        N["NGINX"] --> F["frontend"]
        N --> A["api replicas"]
        A --> W["gpu worker"]
        A --> PG[("PostgreSQL")]
        A --> RD[("Redis")]
        A --> MO[("MinIO")]
    end
```

## 9. Local development profiles

| Profile | Database | Cache | Storage | Requires |
|---|---|---|---|---|
| `local-lite` (**current**) | SQLite (aiosqlite) | in-memory | local filesystem | nothing extra |
| `local-full` | PostgreSQL 16 | Redis 7 | MinIO | Docker Desktop |
| `production` | managed Postgres + PostGIS | managed Redis | S3-compatible | cloud |

Profile is selected purely by environment variables; no code changes.
