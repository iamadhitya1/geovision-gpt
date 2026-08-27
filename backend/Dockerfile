# syntax=docker/dockerfile:1.7
# Multi-stage build: the wheel-building toolchain (compilers needed by argon2-cffi and
# asyncpg) never reaches the runtime image, which keeps it small and shrinks the attack
# surface.

# --- Stage 1: build dependencies ---------------------------------------------
FROM python:3.11-slim-bookworm AS builder

ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        libpq-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /build
COPY pyproject.toml ./
COPY app ./app

RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"
RUN pip install --upgrade pip && pip install .

# --- Stage 2: runtime ---------------------------------------------------------
FROM python:3.11-slim-bookworm AS runtime

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PATH="/opt/venv/bin:$PATH"

RUN apt-get update && apt-get install -y --no-install-recommends \
        libpq5 \
        curl \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --create-home --uid 10001 geovision

COPY --from=builder /opt/venv /opt/venv

WORKDIR /srv/app
COPY --chown=geovision:geovision app ./app
COPY --chown=geovision:geovision alembic ./alembic
COPY --chown=geovision:geovision alembic.ini ./

# Never run as root: a container escape from an application-level bug should not land
# the attacker with host-level privileges.
USER geovision

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD curl -fsS http://localhost:8000/api/v1/health || exit 1

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--proxy-headers"]
