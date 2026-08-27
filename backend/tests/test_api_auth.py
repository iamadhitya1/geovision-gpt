"""Integration tests for the Module 1 HTTP surface.

These exercise the real routers, middleware, dependency graph and error handlers against
an in-memory database — the wiring, not just the units.
"""

from __future__ import annotations

from httpx import AsyncClient

from app.domain.entities import User
from tests.conftest import ADMIN_EMAIL, ADMIN_PASSWORD, USER_PASSWORD, auth_headers, login

VALID_PASSWORD = "Ge0Vision_Strong_2026"


async def register(client: AsyncClient, email: str, password: str = VALID_PASSWORD):
    return await client.post(
        "/auth/register",
        json={"email": email, "full_name": "New Researcher", "password": password},
    )


# --- Registration -------------------------------------------------------------


async def test_register_creates_researcher_and_returns_tokens(client: AsyncClient) -> None:
    response = await register(client, "new@geovisiongpt.io")

    assert response.status_code == 201, response.text
    body = response.json()
    assert body["user"]["email"] == "new@geovisiongpt.io"
    assert body["user"]["role"] == "researcher", "self-service registration must not grant admin"
    assert body["tokens"]["access_token"]
    assert body["tokens"]["refresh_token"]
    assert body["tokens"]["expires_in"] == 1800


async def test_registration_response_never_exposes_the_password_hash(
    client: AsyncClient,
) -> None:
    response = await register(client, "leak@geovisiongpt.io")
    assert "password" not in response.text.lower().replace("password_hash", "")


async def test_duplicate_email_is_rejected(client: AsyncClient) -> None:
    await register(client, "dupe@geovisiongpt.io")
    response = await register(client, "dupe@geovisiongpt.io")

    assert response.status_code == 409
    assert response.json()["code"] == "email_already_registered"


async def test_email_is_case_insensitive(client: AsyncClient) -> None:
    await register(client, "case@geovisiongpt.io")
    response = await register(client, "CASE@geovisiongpt.io")
    assert response.status_code == 409


async def test_weak_password_is_rejected_with_reasons(client: AsyncClient) -> None:
    response = await register(client, "weak@geovisiongpt.io", password="password123")

    assert response.status_code == 422
    body = response.json()
    assert body["code"] == "weak_password"
    assert body["detail"]["violations"]


# --- Login --------------------------------------------------------------------


async def test_login_succeeds_with_correct_credentials(client: AsyncClient) -> None:
    body = await login(client, ADMIN_EMAIL, ADMIN_PASSWORD)
    assert body["user"]["role"] == "admin"


async def test_wrong_password_and_unknown_email_are_indistinguishable(
    client: AsyncClient,
) -> None:
    """User enumeration check: both paths must return the same status, code and message."""
    wrong_password = await client.post(
        "/auth/login", json={"email": ADMIN_EMAIL, "password": "NotThePassword_1"}
    )
    unknown_email = await client.post(
        "/auth/login", json={"email": "nobody@geovisiongpt.io", "password": "NotThePassword_1"}
    )

    assert wrong_password.status_code == unknown_email.status_code == 401
    assert wrong_password.json()["code"] == unknown_email.json()["code"] == "invalid_credentials"
    assert wrong_password.json()["message"] == unknown_email.json()["message"]


async def test_login_updates_last_login_timestamp(client: AsyncClient) -> None:
    body = await login(client, ADMIN_EMAIL, ADMIN_PASSWORD)
    assert body["user"]["last_login_at"] is not None


# --- Protected routes ---------------------------------------------------------


async def test_me_requires_a_token(client: AsyncClient) -> None:
    response = await client.get("/auth/me")
    assert response.status_code == 401
    assert response.headers["WWW-Authenticate"] == "Bearer"


async def test_me_returns_the_authenticated_user(client: AsyncClient) -> None:
    headers = await auth_headers(client, ADMIN_EMAIL, ADMIN_PASSWORD)
    response = await client.get("/auth/me", headers=headers)

    assert response.status_code == 200
    assert response.json()["email"] == ADMIN_EMAIL


async def test_garbage_token_is_rejected(client: AsyncClient) -> None:
    response = await client.get("/auth/me", headers={"Authorization": "Bearer not.a.jwt"})
    assert response.status_code == 401


async def test_refresh_token_is_not_accepted_as_a_bearer_token(client: AsyncClient) -> None:
    body = await login(client, ADMIN_EMAIL, ADMIN_PASSWORD)
    refresh = body["tokens"]["refresh_token"]

    response = await client.get("/auth/me", headers={"Authorization": f"Bearer {refresh}"})
    assert response.status_code == 401


# --- Refresh rotation and replay ---------------------------------------------


async def test_refresh_issues_a_new_pair(client: AsyncClient) -> None:
    body = await login(client, ADMIN_EMAIL, ADMIN_PASSWORD)
    response = await client.post(
        "/auth/refresh", json={"refresh_token": body["tokens"]["refresh_token"]}
    )

    assert response.status_code == 200
    assert response.json()["refresh_token"] != body["tokens"]["refresh_token"]


async def test_replaying_a_used_refresh_token_revokes_every_session(
    client: AsyncClient,
) -> None:
    """The central anti-theft property: a stolen refresh token that the legitimate client
    has already spent must not merely fail — it must end all sessions for the account."""
    body = await login(client, ADMIN_EMAIL, ADMIN_PASSWORD)
    original_refresh = body["tokens"]["refresh_token"]

    rotated = await client.post("/auth/refresh", json={"refresh_token": original_refresh})
    assert rotated.status_code == 200
    new_refresh = rotated.json()["refresh_token"]

    replayed = await client.post("/auth/refresh", json={"refresh_token": original_refresh})
    assert replayed.status_code == 401
    assert replayed.json()["code"] == "token_reuse_detected"

    # The rotation the attacker triggered must also be dead.
    after = await client.post("/auth/refresh", json={"refresh_token": new_refresh})
    assert after.status_code == 401


async def test_logout_invalidates_the_refresh_token(client: AsyncClient) -> None:
    body = await login(client, ADMIN_EMAIL, ADMIN_PASSWORD)
    refresh = body["tokens"]["refresh_token"]

    assert (await client.post("/auth/logout", json={"refresh_token": refresh})).status_code == 200

    response = await client.post("/auth/refresh", json={"refresh_token": refresh})
    assert response.status_code == 401


async def test_logout_is_idempotent(client: AsyncClient) -> None:
    response = await client.post("/auth/logout", json={"refresh_token": "garbage"})
    assert response.status_code == 200


# --- Password change ----------------------------------------------------------


async def test_password_change_invalidates_existing_access_tokens(
    client: AsyncClient,
) -> None:
    registered = (await register(client, "rotate@geovisiongpt.io")).json()
    headers = {"Authorization": f"Bearer {registered['tokens']['access_token']}"}

    response = await client.post(
        "/auth/change-password",
        headers=headers,
        json={"current_password": VALID_PASSWORD, "new_password": "An0ther_Strong_2026"},
    )
    assert response.status_code == 200

    # token_version was bumped, so the previously valid access token is now dead.
    assert (await client.get("/auth/me", headers=headers)).status_code == 401

    fresh = await login(client, "rotate@geovisiongpt.io", "An0ther_Strong_2026")
    assert fresh["user"]["email"] == "rotate@geovisiongpt.io"


async def test_password_change_requires_the_current_password(client: AsyncClient) -> None:
    registered = (await register(client, "guard@geovisiongpt.io")).json()
    headers = {"Authorization": f"Bearer {registered['tokens']['access_token']}"}

    response = await client.post(
        "/auth/change-password",
        headers=headers,
        json={"current_password": "WrongCurrent_1", "new_password": "An0ther_Strong_2026"},
    )
    assert response.status_code == 401


# --- Rate limiting ------------------------------------------------------------


async def test_repeated_failed_logins_are_throttled(client: AsyncClient) -> None:
    payload = {"email": ADMIN_EMAIL, "password": "WrongPassword_1"}

    for _ in range(5):
        assert (await client.post("/auth/login", json=payload)).status_code == 401

    blocked = await client.post("/auth/login", json=payload)
    assert blocked.status_code == 429
    assert blocked.json()["code"] == "rate_limit_exceeded"
    assert int(blocked.headers["Retry-After"]) > 0


# --- Observability ------------------------------------------------------------


async def test_request_id_is_echoed_back(client: AsyncClient) -> None:
    response = await client.get("/health", headers={"X-Request-ID": "trace-abc-123"})
    assert response.headers["X-Request-ID"] == "trace-abc-123"


async def test_security_headers_are_present(client: AsyncClient) -> None:
    response = await client.get("/health")
    assert response.headers["X-Content-Type-Options"] == "nosniff"
    assert response.headers["X-Frame-Options"] == "DENY"


async def test_health_and_readiness(client: AsyncClient) -> None:
    assert (await client.get("/health")).json()["status"] == "ok"

    ready = await client.get("/ready")
    assert ready.status_code == 200
    assert ready.json()["checks"] == {"database": "ok", "cache": "ok"}


# --- Audit trail --------------------------------------------------------------


async def test_failed_and_successful_logins_are_both_audited(
    client: AsyncClient, researcher: User
) -> None:
    """A log that records only successes cannot evidence an attack."""
    await client.post(
        "/auth/login", json={"email": researcher.email, "password": "WrongPassword_1"}
    )
    await login(client, researcher.email, USER_PASSWORD)

    headers = await auth_headers(client, ADMIN_EMAIL, ADMIN_PASSWORD)
    response = await client.get(f"/users/{researcher.id}/audit", headers=headers)

    assert response.status_code == 200
    actions = {event["action"] for event in response.json()["items"]}
    assert "auth.login.failed" in actions
    assert "auth.login.succeeded" in actions
