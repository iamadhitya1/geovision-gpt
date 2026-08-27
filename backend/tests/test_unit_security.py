"""Unit tests for security primitives and domain rules."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import jwt
import pytest

from app.application.password_policy import PasswordPolicy
from app.domain.entities import User
from app.domain.enums import Role, TokenType
from app.domain.errors import InvalidTokenError, RateLimitExceededError, WeakPasswordError
from app.infrastructure.cache.memory import InMemoryCache
from app.infrastructure.cache.rate_limiter import FixedWindowRateLimiter
from app.infrastructure.security.password import Argon2PasswordHasher
from app.infrastructure.security.tokens import JwtTokenService

# --- Role hierarchy -----------------------------------------------------------


def test_admin_satisfies_researcher_requirement() -> None:
    assert Role.ADMIN.outranks_or_equals(Role.RESEARCHER)
    assert Role.RESEARCHER.outranks_or_equals(Role.GUEST)
    assert not Role.GUEST.outranks_or_equals(Role.RESEARCHER)


# --- User entity invariants ---------------------------------------------------


def test_password_change_revokes_outstanding_sessions() -> None:
    """The security property that makes a password change meaningful: an attacker holding
    a refresh token must not survive it."""
    user = User(email="a@example-domain.io", full_name="A", password_hash="old")
    original_version = user.token_version

    user.change_password("new")

    assert user.password_hash == "new"
    assert user.token_version == original_version + 1


def test_deactivation_revokes_sessions_and_blocks_login() -> None:
    user = User(email="a@example-domain.io", full_name="A", password_hash="h")
    user.deactivate()
    assert not user.can_login()
    assert user.token_version == 1


# --- Password policy ----------------------------------------------------------


@pytest.mark.parametrize(
    "password",
    ["short1A", "alllowercase123", "ALLUPPERCASE123", "NoDigitsHereAtAll", "password123"],
)
def test_policy_rejects_weak_passwords(password: str) -> None:
    with pytest.raises(WeakPasswordError):
        PasswordPolicy().validate(password)


def test_policy_reports_every_violation_at_once() -> None:
    """Users should not have to discover requirements one failed attempt at a time."""
    with pytest.raises(WeakPasswordError) as exc_info:
        PasswordPolicy().validate("short")
    violations = exc_info.value.detail["violations"]
    assert len(violations) >= 3


def test_policy_rejects_password_containing_email_local_part() -> None:
    with pytest.raises(WeakPasswordError):
        PasswordPolicy().validate("Adhitya_Secret9", email="adhitya@geovisiongpt.io")


def test_policy_accepts_strong_password() -> None:
    PasswordPolicy().validate("Ge0Vision_Strong_2026")


# --- Argon2 hasher ------------------------------------------------------------


def test_hash_verifies_and_is_salted(hasher: Argon2PasswordHasher) -> None:
    first = hasher.hash("Ge0Vision_Strong_2026")
    second = hasher.hash("Ge0Vision_Strong_2026")

    assert first != second, "identical passwords must not produce identical hashes"
    assert hasher.verify("Ge0Vision_Strong_2026", first)
    assert not hasher.verify("wrong-password", first)


def test_verify_rejects_malformed_hash_without_raising(hasher: Argon2PasswordHasher) -> None:
    assert hasher.verify("anything", "not-a-valid-argon2-hash") is False


def test_needs_rehash_detects_weaker_parameters() -> None:
    weak = Argon2PasswordHasher(time_cost=1, memory_cost_kib=8192, parallelism=1)
    strong = Argon2PasswordHasher(time_cost=3, memory_cost_kib=16384, parallelism=1)

    old_hash = weak.hash("Ge0Vision_Strong_2026")
    assert strong.needs_rehash(old_hash)
    assert not weak.needs_rehash(old_hash)


# --- JWT ----------------------------------------------------------------------


@pytest.fixture
def sample_user() -> User:
    return User(
        email="researcher@geovisiongpt.io",
        full_name="R",
        password_hash="h",
        role=Role.RESEARCHER,
    )


def test_access_token_round_trip(token_service: JwtTokenService, sample_user: User) -> None:
    token, expires_in = token_service.create_access_token(sample_user)
    claims = token_service.decode(token, expected_type=TokenType.ACCESS)

    assert claims.subject == sample_user.id
    assert claims.role is Role.RESEARCHER
    assert expires_in == 30 * 60


def test_refresh_token_cannot_be_used_as_an_access_token(
    token_service: JwtTokenService, sample_user: User
) -> None:
    """Without the ``typ`` claim check, a 14-day credential would be accepted wherever a
    30-minute one is expected."""
    refresh_token, _ = token_service.create_refresh_token(sample_user)
    with pytest.raises(InvalidTokenError):
        token_service.decode(refresh_token, expected_type=TokenType.ACCESS)


def test_each_refresh_token_has_a_unique_jti(
    token_service: JwtTokenService, sample_user: User
) -> None:
    _, first = token_service.create_refresh_token(sample_user)
    _, second = token_service.create_refresh_token(sample_user)
    assert first != second


def test_tampered_signature_is_rejected(token_service: JwtTokenService, sample_user: User) -> None:
    token, _ = token_service.create_access_token(sample_user)
    forged = token[:-4] + ("aaaa" if not token.endswith("aaaa") else "bbbb")
    with pytest.raises(InvalidTokenError):
        token_service.decode(forged, expected_type=TokenType.ACCESS)


def test_expired_token_is_rejected(sample_user: User) -> None:
    service = JwtTokenService(
        secret_key="test-secret-key-that-is-definitely-long-enough-32+",
        access_token_expire_minutes=1,
    )
    expired = jwt.encode(
        {
            "sub": str(sample_user.id),
            "role": sample_user.role.value,
            "typ": TokenType.ACCESS.value,
            "tv": 0,
            "jti": "expired",
            "iat": int((datetime.now(UTC) - timedelta(hours=2)).timestamp()),
            "exp": int((datetime.now(UTC) - timedelta(hours=1)).timestamp()),
            "iss": "geovisiongpt",
        },
        "test-secret-key-that-is-definitely-long-enough-32+",
        algorithm="HS256",
    )
    with pytest.raises(InvalidTokenError):
        service.decode(expired, expected_type=TokenType.ACCESS)


def test_token_signed_with_another_key_is_rejected(
    token_service: JwtTokenService, sample_user: User
) -> None:
    foreign = jwt.encode(
        {
            "sub": str(sample_user.id),
            "role": "admin",
            "typ": "access",
            "tv": 0,
            "jti": "x",
            "iat": int(datetime.now(UTC).timestamp()),
            "exp": int((datetime.now(UTC) + timedelta(hours=1)).timestamp()),
            "iss": "geovisiongpt",
        },
        "an-entirely-different-secret-key-32-characters",
        algorithm="HS256",
    )
    with pytest.raises(InvalidTokenError):
        token_service.decode(foreign, expected_type=TokenType.ACCESS)


# --- Cache and rate limiting --------------------------------------------------


async def test_increment_sets_ttl_only_on_first_hit() -> None:
    """A steady stream of requests must not keep pushing the window's expiry forward,
    which would make a tripped limit permanent."""
    cache = InMemoryCache()

    assert await cache.increment("k", ttl_seconds=60) == 1
    first_ttl = await cache.ttl("k")

    assert await cache.increment("k", ttl_seconds=60) == 2
    assert await cache.ttl("k") <= first_ttl


async def test_cache_expiry_removes_value() -> None:
    cache = InMemoryCache()
    await cache.set("k", "v", ttl_seconds=0)
    assert await cache.get("k") is None


async def test_rate_limiter_allows_up_to_the_limit_then_rejects() -> None:
    limiter = FixedWindowRateLimiter(InMemoryCache())

    for _ in range(5):
        await limiter.hit("login:1.2.3.4", limit=5, window_seconds=60)

    with pytest.raises(RateLimitExceededError) as exc_info:
        await limiter.hit("login:1.2.3.4", limit=5, window_seconds=60)

    assert exc_info.value.retry_after_seconds > 0


async def test_rate_limit_buckets_are_independent() -> None:
    limiter = FixedWindowRateLimiter(InMemoryCache())
    for _ in range(5):
        await limiter.hit("login:1.1.1.1", limit=5, window_seconds=60)

    # A different client must be unaffected by another's exhausted budget.
    await limiter.hit("login:2.2.2.2", limit=5, window_seconds=60)
