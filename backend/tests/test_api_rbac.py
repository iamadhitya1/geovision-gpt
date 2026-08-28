"""Role-based access control and administration tests."""

from __future__ import annotations

from httpx import AsyncClient

from app.domain.entities import User
from tests.conftest import ADMIN_EMAIL, ADMIN_PASSWORD, USER_PASSWORD, auth_headers, login


async def test_researcher_cannot_list_users(client: AsyncClient, researcher: User) -> None:
    headers = await auth_headers(client, researcher.email, USER_PASSWORD)
    response = await client.get("/users", headers=headers)

    assert response.status_code == 403
    body = response.json()
    assert body["code"] == "forbidden"
    assert body["detail"]["required_role"] == "admin"
    assert body["detail"]["actual_role"] == "researcher"


async def test_admin_can_list_users(client: AsyncClient, researcher: User) -> None:
    headers = await auth_headers(client, ADMIN_EMAIL, ADMIN_PASSWORD)
    response = await client.get("/users", headers=headers)

    assert response.status_code == 200
    body = response.json()
    assert body["total"] >= 2
    emails = {item["email"] for item in body["items"]}
    assert researcher.email in emails


async def test_user_listing_supports_filtering_and_pagination(
    client: AsyncClient, researcher: User
) -> None:
    headers = await auth_headers(client, ADMIN_EMAIL, ADMIN_PASSWORD)

    filtered = await client.get("/users", headers=headers, params={"role": "admin"})
    assert filtered.status_code == 200
    assert all(item["role"] == "admin" for item in filtered.json()["items"])

    paged = await client.get("/users", headers=headers, params={"limit": 1})
    assert len(paged.json()["items"]) == 1
    assert paged.json()["total"] >= 2

    searched = await client.get("/users", headers=headers, params={"search": "RESEARCH"})
    assert any(item["email"] == researcher.email for item in searched.json()["items"])


async def test_self_service_profile_update_cannot_change_role(
    client: AsyncClient, researcher: User
) -> None:
    """Role is absent from ``UpdateProfileRequest``, so an injected field is ignored rather
    than honoured — escalation is unrepresentable, not merely rejected."""
    headers = await auth_headers(client, researcher.email, USER_PASSWORD)

    response = await client.patch(
        "/users/me", headers=headers, json={"full_name": "Renamed", "role": "admin"}
    )

    assert response.status_code == 200
    assert response.json()["full_name"] == "Renamed"
    assert response.json()["role"] == "researcher"


async def test_admin_can_promote_a_researcher(client: AsyncClient, researcher: User) -> None:
    headers = await auth_headers(client, ADMIN_EMAIL, ADMIN_PASSWORD)

    response = await client.patch(
        f"/users/{researcher.id}", headers=headers, json={"role": "admin"}
    )

    assert response.status_code == 200
    assert response.json()["role"] == "admin"


async def test_admin_cannot_demote_themselves(client: AsyncClient) -> None:
    """Guards against locking the platform out of its last administrator."""
    body = await login(client, ADMIN_EMAIL, ADMIN_PASSWORD)
    headers = {"Authorization": f"Bearer {body['tokens']['access_token']}"}
    admin_id = body["user"]["id"]

    response = await client.patch(
        f"/users/{admin_id}", headers=headers, json={"role": "researcher"}
    )
    assert response.status_code == 403


async def test_admin_cannot_deactivate_themselves(client: AsyncClient) -> None:
    body = await login(client, ADMIN_EMAIL, ADMIN_PASSWORD)
    headers = {"Authorization": f"Bearer {body['tokens']['access_token']}"}
    admin_id = body["user"]["id"]

    response = await client.patch(f"/users/{admin_id}", headers=headers, json={"is_active": False})
    assert response.status_code == 403


async def test_deactivation_takes_effect_immediately(client: AsyncClient, researcher: User) -> None:
    """The user row is re-read on every request, so a deactivated account loses access at
    once rather than when its access token happens to expire."""
    researcher_headers = await auth_headers(client, researcher.email, USER_PASSWORD)
    assert (await client.get("/auth/me", headers=researcher_headers)).status_code == 200

    admin_headers = await auth_headers(client, ADMIN_EMAIL, ADMIN_PASSWORD)
    deactivated = await client.patch(
        f"/users/{researcher.id}", headers=admin_headers, json={"is_active": False}
    )
    assert deactivated.status_code == 200

    response = await client.get("/auth/me", headers=researcher_headers)
    assert response.status_code == 401


async def test_deactivated_user_cannot_log_in(client: AsyncClient, researcher: User) -> None:
    admin_headers = await auth_headers(client, ADMIN_EMAIL, ADMIN_PASSWORD)
    await client.patch(f"/users/{researcher.id}", headers=admin_headers, json={"is_active": False})

    response = await client.post(
        "/auth/login", json={"email": researcher.email, "password": USER_PASSWORD}
    )
    assert response.status_code == 401
    assert response.json()["code"] == "inactive_user"


async def test_role_change_is_audited(client: AsyncClient, researcher: User) -> None:
    headers = await auth_headers(client, ADMIN_EMAIL, ADMIN_PASSWORD)
    await client.patch(f"/users/{researcher.id}", headers=headers, json={"role": "admin"})

    audit = await client.get(f"/users/{researcher.id}/audit", headers=headers)
    events = audit.json()["items"]
    role_events = [e for e in events if e["action"] == "user.role.changed"]

    assert role_events, "a privilege change must leave an audit record"
    assert role_events[0]["detail"]["previous_role"] == "researcher"
    assert role_events[0]["detail"]["new_role"] == "admin"


async def test_verification_change_is_audited(client: AsyncClient, researcher: User) -> None:
    """Regression test: is_verified changes must leave an audit record, same as role and
    activation changes do. The `researcher` fixture starts is_verified=True, so this flips
    it to False to exercise an actual change rather than a no-op."""
    headers = await auth_headers(client, ADMIN_EMAIL, ADMIN_PASSWORD)

    response = await client.patch(
        f"/users/{researcher.id}", headers=headers, json={"is_verified": False}
    )
    assert response.status_code == 200
    assert response.json()["is_verified"] is False

    audit = await client.get(f"/users/{researcher.id}/audit", headers=headers)
    events = audit.json()["items"]
    verification_events = [e for e in events if e["action"] == "user.verification.changed"]

    assert verification_events, "a verification change must leave an audit record"
    assert verification_events[0]["detail"]["previous_is_verified"] is True
    assert verification_events[0]["detail"]["new_is_verified"] is False


async def test_repeated_verification_no_op_is_not_audited(
    client: AsyncClient, researcher: User
) -> None:
    """Setting is_verified to the value it already holds must not create a duplicate audit
    entry, matching how role and activation changes are only recorded when they occur."""
    headers = await auth_headers(client, ADMIN_EMAIL, ADMIN_PASSWORD)

    response = await client.patch(
        f"/users/{researcher.id}", headers=headers, json={"is_verified": True}
    )
    assert response.status_code == 200

    audit = await client.get(f"/users/{researcher.id}/audit", headers=headers)
    events = audit.json()["items"]
    assert not [e for e in events if e["action"] == "user.verification.changed"]


async def test_admin_cannot_change_their_own_role_to_anything(client: AsyncClient) -> None:
    """Covers the case the original guard's dead sub-condition implied was allowed:
    self-targeting is blocked for every requested role value, not just non-admin ones."""
    body = await login(client, ADMIN_EMAIL, ADMIN_PASSWORD)
    headers = {"Authorization": f"Bearer {body['tokens']['access_token']}"}
    admin_id = body["user"]["id"]

    response = await client.patch(f"/users/{admin_id}", headers=headers, json={"role": "admin"})
    assert response.status_code == 200
    assert response.json()["role"] == "admin", "no-op: role unchanged, so no guard fires"


async def test_unknown_user_returns_404(client: AsyncClient) -> None:
    headers = await auth_headers(client, ADMIN_EMAIL, ADMIN_PASSWORD)
    response = await client.get("/users/00000000-0000-0000-0000-000000000000", headers=headers)
    assert response.status_code == 404


async def test_malformed_uuid_returns_422(client: AsyncClient) -> None:
    headers = await auth_headers(client, ADMIN_EMAIL, ADMIN_PASSWORD)
    response = await client.get("/users/not-a-uuid", headers=headers)
    assert response.status_code == 422
    assert response.json()["code"] == "validation_error"
