"""Shared test fixtures.

The suite runs against an in-memory SQLite database and the in-process cache, so it needs
no Docker, no PostgreSQL and no Redis. Each test gets a fresh database and a fresh cache,
which keeps rate-limit counters and refresh-token allow-lists from leaking between tests.
"""

from __future__ import annotations

import io
from collections.abc import AsyncIterator
from pathlib import Path

import numpy as np
import pytest
import rasterio
from httpx import ASGITransport, AsyncClient
from PIL import Image
from rasterio.crs import CRS
from rasterio.transform import from_origin
from sqlalchemy.ext.asyncio import AsyncEngine, async_sessionmaker, create_async_engine
from sqlalchemy.pool import StaticPool

from app.application.password_policy import PasswordPolicy
from app.core.config import Environment, Settings
from app.domain.entities import User
from app.domain.enums import Role
from app.infrastructure.bootstrap import ensure_first_admin
from app.infrastructure.cache.memory import InMemoryCache
from app.infrastructure.db.repositories import SqlAlchemyUserRepository
from app.infrastructure.db.session import create_all, create_session_factory
from app.infrastructure.security.password import Argon2PasswordHasher
from app.infrastructure.security.tokens import JwtTokenService
from app.main import create_app

ADMIN_EMAIL = "admin@geovisiongpt.io"
ADMIN_PASSWORD = "Ge0Vision_Root_2026"  # must not contain the email local-part "admin"
USER_PASSWORD = "ResearchPass_2026x"


@pytest.fixture(scope="session")
def hasher() -> Argon2PasswordHasher:
    """Session-scoped: Argon2 is intentionally slow, and rebuilding it per test would
    dominate the suite's runtime. Cost parameters are lowered because these tests verify
    behaviour, not cracking resistance."""
    return Argon2PasswordHasher(time_cost=1, memory_cost_kib=8192, parallelism=1)


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    return Settings(
        environment=Environment.DEVELOPMENT,
        debug=True,
        secret_key="test-secret-key-that-is-definitely-long-enough-32+",
        database_url="sqlite+aiosqlite:///:memory:",
        cache_backend="memory",
        rate_limit_enabled=True,
        rate_limit_login_attempts=5,
        rate_limit_login_window_seconds=300,
        access_token_expire_minutes=30,
        refresh_token_expire_days=14,
        password_min_length=10,
        first_admin_email=ADMIN_EMAIL,
        first_admin_password=ADMIN_PASSWORD,
        first_admin_name="Test Administrator",
        # A per-test tmp_path, not the real ./storage default — otherwise every test run
        # would write real files into the repo's working directory.
        storage_backend="filesystem",
        storage_root=str(tmp_path / "storage"),
        max_upload_size_mb=5,
    )


@pytest.fixture
async def engine(settings: Settings) -> AsyncIterator[AsyncEngine]:
    """In-memory SQLite shared across connections.

    ``StaticPool`` is required: without it every connection would open its own private
    ``:memory:`` database and the schema created on one would be invisible to the next.
    """
    eng = create_async_engine(
        settings.database_url,
        poolclass=StaticPool,
        connect_args={"check_same_thread": False},
        future=True,
    )
    await create_all(eng)
    yield eng
    await eng.dispose()


@pytest.fixture
def session_factory(engine: AsyncEngine) -> async_sessionmaker:
    return create_session_factory(engine)


@pytest.fixture
def cache() -> InMemoryCache:
    return InMemoryCache()


@pytest.fixture
def token_service(settings: Settings) -> JwtTokenService:
    return JwtTokenService(
        secret_key=settings.secret_key.get_secret_value(),
        algorithm=settings.jwt_algorithm,
        access_token_expire_minutes=settings.access_token_expire_minutes,
        refresh_token_expire_days=settings.refresh_token_expire_days,
    )


@pytest.fixture
async def app(
    settings: Settings,
    engine: AsyncEngine,
    session_factory: async_sessionmaker,
    cache: InMemoryCache,
    hasher: Argon2PasswordHasher,
    token_service: JwtTokenService,
):
    """Application wired to the test doubles.

    Built via the real ``create_app`` so routing, middleware and error handlers are the
    production ones; only the process-scoped singletons are swapped. The lifespan hook is
    not run — its two jobs (schema creation, admin bootstrap) are performed explicitly
    here so each fixture's effect is visible rather than implicit.
    """
    application = create_app(settings)
    application.state.engine = engine
    application.state.session_factory = session_factory
    application.state.cache = cache
    application.state.password_hasher = hasher
    application.state.token_service = token_service
    application.state.password_policy = PasswordPolicy(min_length=settings.password_min_length)

    await ensure_first_admin(session_factory, settings, hasher)
    return application


@pytest.fixture
async def client(app) -> AsyncIterator[AsyncClient]:
    transport = ASGITransport(app=app)
    async with AsyncClient(
        transport=transport, base_url="http://testserver/api/v1"
    ) as async_client:
        yield async_client


# --- Convenience helpers ------------------------------------------------------


@pytest.fixture
async def researcher(session_factory: async_sessionmaker, hasher: Argon2PasswordHasher) -> User:
    """A pre-existing active researcher account."""
    session = session_factory()
    try:
        repo = SqlAlchemyUserRepository(session)
        user = User(
            email="researcher@geovisiongpt.io",
            full_name="Test Researcher",
            password_hash=hasher.hash(USER_PASSWORD),
            role=Role.RESEARCHER,
            is_verified=True,
        )
        created = await repo.add(user)
        await session.commit()
        return created
    finally:
        await session.close()


# --- Synthetic image fixtures --------------------------------------------------
#
# Generated in-process rather than committed as binary files: it keeps the repository
# text-only and makes every test's input data traceable to the code that produced it.


def make_geotiff_bytes(
    *, width: int = 64, height: int = 64, band_count: int = 3, with_crs: bool = True
) -> bytes:
    """A small synthetic GeoTIFF, georeferenced near Ahmedabad by default."""
    rng = np.random.default_rng(seed=42)
    data = rng.integers(0, 255, size=(band_count, height, width), dtype=np.uint8)
    profile = {
        "driver": "GTiff",
        "dtype": "uint8",
        "count": band_count,
        "height": height,
        "width": width,
    }
    if with_crs:
        profile["crs"] = CRS.from_epsg(4326)
        profile["transform"] = from_origin(72.5, 23.1, 0.0001, 0.0001)

    with rasterio.io.MemoryFile() as memfile:
        with memfile.open(**profile) as dst:
            dst.write(data)
        return memfile.read()


def make_bright_geotiff_bytes(*, width: int = 64, height: int = 64) -> bytes:
    """An all-white 3-band GeoTIFF — the cloud-coverage heuristic should score this near
    100%, giving the metadata-extraction tests a non-trivial value to assert on."""
    data = np.full((3, height, width), 255, dtype=np.uint8)
    profile = {
        "driver": "GTiff",
        "dtype": "uint8",
        "count": 3,
        "height": height,
        "width": width,
        "crs": CRS.from_epsg(4326),
        "transform": from_origin(72.5, 23.1, 0.0001, 0.0001),
    }
    with rasterio.io.MemoryFile() as memfile:
        with memfile.open(**profile) as dst:
            dst.write(data)
        return memfile.read()


def make_png_bytes(*, width: int = 32, height: int = 32) -> bytes:
    image = Image.new("RGB", (width, height), color=(10, 120, 60))
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def make_jpeg_bytes(*, width: int = 32, height: int = 32) -> bytes:
    image = Image.new("RGB", (width, height), color=(200, 30, 30))
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG")
    return buffer.getvalue()


async def login(client: AsyncClient, email: str, password: str) -> dict:
    """Log in and return the parsed response body."""
    response = await client.post("/auth/login", json={"email": email, "password": password})
    assert response.status_code == 200, response.text
    return response.json()


async def auth_headers(client: AsyncClient, email: str, password: str) -> dict[str, str]:
    body = await login(client, email, password)
    return {"Authorization": f"Bearer {body['tokens']['access_token']}"}
