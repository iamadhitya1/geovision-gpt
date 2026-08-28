"""Integration tests for the Module 2 HTTP surface."""

from __future__ import annotations

from httpx import AsyncClient

from app.domain.entities import User
from tests.conftest import (
    ADMIN_EMAIL,
    ADMIN_PASSWORD,
    USER_PASSWORD,
    auth_headers,
    make_geotiff_bytes,
    make_jpeg_bytes,
    make_png_bytes,
)


async def upload(
    client: AsyncClient,
    headers: dict[str, str],
    *,
    filename: str,
    data: bytes,
    content_type: str,
):
    return await client.post(
        "/images",
        headers=headers,
        files={"file": (filename, data, content_type)},
    )


async def register_headers(client: AsyncClient, email: str, full_name: str) -> dict[str, str]:
    """Register a fresh researcher and return bearer headers for them — used wherever a
    test needs a second, independent account to exercise ownership scoping."""
    body = await client.post(
        "/auth/register",
        json={"email": email, "full_name": full_name, "password": "Str0ng_Password_2026"},
    )
    assert body.status_code == 201, body.text
    return {"Authorization": f"Bearer {body.json()['tokens']['access_token']}"}


# --- Upload: happy paths ----------------------------------------------------------------


async def test_upload_georeferenced_geotiff(client: AsyncClient, researcher: User) -> None:
    headers = await auth_headers(client, researcher.email, USER_PASSWORD)
    response = await upload(
        client,
        headers,
        filename="scene.tif",
        data=make_geotiff_bytes(width=64, height=64, band_count=3),
        content_type="image/tiff",
    )

    assert response.status_code == 201, response.text
    body = response.json()
    assert body["image_format"] == "geotiff"
    assert body["status"] == "ready"
    assert body["thumbnail_available"] is True
    assert body["metadata"]["crs"] == "EPSG:4326"
    assert body["metadata"]["bounds_wgs84"] is not None
    assert body["metadata"]["gsd_meters_approx"] > 0
    assert body["metadata"]["cloud_cover_percent"] is not None


async def test_upload_geotiff_without_crs_still_succeeds(
    client: AsyncClient, researcher: User
) -> None:
    """A plain, non-georeferenced TIFF is accepted — geospatial fields are just null."""
    headers = await auth_headers(client, researcher.email, USER_PASSWORD)
    response = await upload(
        client,
        headers,
        filename="plain.tif",
        data=make_geotiff_bytes(with_crs=False),
        content_type="image/tiff",
    )

    assert response.status_code == 201, response.text
    body = response.json()
    assert body["status"] == "ready"
    assert body["metadata"]["crs"] is None
    assert body["metadata"]["width_px"] == 64


async def test_upload_png(client: AsyncClient, researcher: User) -> None:
    headers = await auth_headers(client, researcher.email, USER_PASSWORD)
    response = await upload(
        client, headers, filename="quicklook.png", data=make_png_bytes(), content_type="image/png"
    )

    assert response.status_code == 201, response.text
    body = response.json()
    assert body["image_format"] == "png"
    assert body["status"] == "ready"
    assert body["metadata"]["crs"] is None


async def test_upload_jpeg(client: AsyncClient, researcher: User) -> None:
    headers = await auth_headers(client, researcher.email, USER_PASSWORD)
    response = await upload(
        client,
        headers,
        filename="quicklook.jpg",
        data=make_jpeg_bytes(),
        content_type="image/jpeg",
    )

    assert response.status_code == 201, response.text
    assert response.json()["image_format"] == "jpeg"


# --- Upload: rejections ------------------------------------------------------------------


async def test_upload_rejects_unsupported_extension(client: AsyncClient, researcher: User) -> None:
    headers = await auth_headers(client, researcher.email, USER_PASSWORD)
    response = await upload(
        client,
        headers,
        filename="malware.exe",
        data=b"whatever",
        content_type="application/octet-stream",
    )

    assert response.status_code == 422
    assert response.json()["code"] == "unsupported_file_type"


async def test_upload_rejects_content_mismatching_extension(
    client: AsyncClient, researcher: User
) -> None:
    """A JPEG renamed to .png — content sniffing catches what the filename alone can't."""
    headers = await auth_headers(client, researcher.email, USER_PASSWORD)
    response = await upload(
        client,
        headers,
        filename="fake.png",
        data=make_jpeg_bytes(),
        content_type="image/png",
    )

    assert response.status_code == 422
    assert response.json()["code"] == "corrupt_image"


async def test_upload_rejects_oversized_file(client: AsyncClient, researcher: User) -> None:
    """Test settings cap uploads at 5 MB (see conftest.settings)."""
    headers = await auth_headers(client, researcher.email, USER_PASSWORD)
    oversized = b"\x00" * (6 * 1024 * 1024)
    response = await upload(
        client, headers, filename="huge.png", data=oversized, content_type="image/png"
    )

    assert response.status_code == 413
    assert response.json()["code"] == "file_too_large"


async def test_upload_requires_authentication(client: AsyncClient) -> None:
    response = await client.post(
        "/images", files={"file": ("scene.png", make_png_bytes(), "image/png")}
    )
    assert response.status_code == 401


# --- Retrieval and ownership scoping -------------------------------------------------------


async def test_owner_can_read_their_own_image(client: AsyncClient, researcher: User) -> None:
    headers = await auth_headers(client, researcher.email, USER_PASSWORD)
    uploaded = await upload(
        client, headers, filename="a.png", data=make_png_bytes(), content_type="image/png"
    )
    image_id = uploaded.json()["id"]

    response = await client.get(f"/images/{image_id}", headers=headers)
    assert response.status_code == 200
    assert response.json()["id"] == image_id


async def test_other_researcher_cannot_read_someone_elses_image(client: AsyncClient) -> None:
    owner_headers = await register_headers(client, "owner@geovisiongpt.io", "Owner")
    uploaded = await upload(
        client, owner_headers, filename="a.png", data=make_png_bytes(), content_type="image/png"
    )
    image_id = uploaded.json()["id"]

    other_headers = await register_headers(client, "other@geovisiongpt.io", "Other")

    response = await client.get(f"/images/{image_id}", headers=other_headers)
    assert response.status_code == 404, "must not leak whether the id exists to a non-owner"


async def test_admin_can_read_any_image(client: AsyncClient, researcher: User) -> None:
    researcher_headers = await auth_headers(client, researcher.email, USER_PASSWORD)
    uploaded = await upload(
        client,
        researcher_headers,
        filename="a.png",
        data=make_png_bytes(),
        content_type="image/png",
    )
    image_id = uploaded.json()["id"]

    admin_headers = await auth_headers(client, ADMIN_EMAIL, ADMIN_PASSWORD)
    response = await client.get(f"/images/{image_id}", headers=admin_headers)
    assert response.status_code == 200


async def test_unknown_image_returns_404(client: AsyncClient, researcher: User) -> None:
    headers = await auth_headers(client, researcher.email, USER_PASSWORD)
    response = await client.get("/images/00000000-0000-0000-0000-000000000000", headers=headers)
    assert response.status_code == 404


# --- Listing -----------------------------------------------------------------------------


async def test_researcher_lists_only_their_own_images(client: AsyncClient) -> None:
    a_headers = await register_headers(client, "lister-a@geovisiongpt.io", "A")
    await upload(
        client, a_headers, filename="a.png", data=make_png_bytes(), content_type="image/png"
    )

    b_headers = await register_headers(client, "lister-b@geovisiongpt.io", "B")
    await upload(
        client, b_headers, filename="b.png", data=make_png_bytes(), content_type="image/png"
    )

    response = await client.get("/images", headers=a_headers)
    assert response.status_code == 200
    body = response.json()
    assert body["total"] == 1
    assert body["items"][0]["original_filename"] == "a.png"


async def test_admin_lists_every_image(client: AsyncClient, researcher: User) -> None:
    researcher_headers = await auth_headers(client, researcher.email, USER_PASSWORD)
    await upload(
        client,
        researcher_headers,
        filename="r.png",
        data=make_png_bytes(),
        content_type="image/png",
    )

    admin_headers = await auth_headers(client, ADMIN_EMAIL, ADMIN_PASSWORD)
    await upload(
        client, admin_headers, filename="ad.png", data=make_png_bytes(), content_type="image/png"
    )

    response = await client.get("/images", headers=admin_headers)
    assert response.status_code == 200
    assert response.json()["total"] >= 2


# --- Thumbnails --------------------------------------------------------------------------


async def test_thumbnail_is_retrievable(client: AsyncClient, researcher: User) -> None:
    headers = await auth_headers(client, researcher.email, USER_PASSWORD)
    uploaded = await upload(
        client, headers, filename="a.png", data=make_png_bytes(), content_type="image/png"
    )
    image_id = uploaded.json()["id"]

    response = await client.get(f"/images/{image_id}/thumbnail", headers=headers)
    assert response.status_code == 200
    assert response.headers["content-type"] == "image/jpeg"
    assert len(response.content) > 0


# --- Deletion ----------------------------------------------------------------------------


async def test_owner_can_delete_their_own_image(client: AsyncClient, researcher: User) -> None:
    headers = await auth_headers(client, researcher.email, USER_PASSWORD)
    uploaded = await upload(
        client, headers, filename="a.png", data=make_png_bytes(), content_type="image/png"
    )
    image_id = uploaded.json()["id"]

    response = await client.delete(f"/images/{image_id}", headers=headers)
    assert response.status_code == 200

    after = await client.get(f"/images/{image_id}", headers=headers)
    assert after.status_code == 404


async def test_deletion_removes_the_stored_thumbnail_too(
    client: AsyncClient, researcher: User
) -> None:
    headers = await auth_headers(client, researcher.email, USER_PASSWORD)
    uploaded = await upload(
        client, headers, filename="a.png", data=make_png_bytes(), content_type="image/png"
    )
    image_id = uploaded.json()["id"]

    await client.delete(f"/images/{image_id}", headers=headers)

    thumbnail_response = await client.get(f"/images/{image_id}/thumbnail", headers=headers)
    assert thumbnail_response.status_code == 404


async def test_non_owner_cannot_delete_someone_elses_image(client: AsyncClient) -> None:
    owner_headers = await register_headers(client, "delowner@geovisiongpt.io", "Owner")
    uploaded = await upload(
        client, owner_headers, filename="a.png", data=make_png_bytes(), content_type="image/png"
    )
    image_id = uploaded.json()["id"]

    intruder_headers = await register_headers(client, "intruder@geovisiongpt.io", "Intruder")

    response = await client.delete(f"/images/{image_id}", headers=intruder_headers)
    assert response.status_code == 403


async def test_admin_can_delete_any_image(client: AsyncClient, researcher: User) -> None:
    researcher_headers = await auth_headers(client, researcher.email, USER_PASSWORD)
    uploaded = await upload(
        client,
        researcher_headers,
        filename="a.png",
        data=make_png_bytes(),
        content_type="image/png",
    )
    image_id = uploaded.json()["id"]

    admin_headers = await auth_headers(client, ADMIN_EMAIL, ADMIN_PASSWORD)
    response = await client.delete(f"/images/{image_id}", headers=admin_headers)
    assert response.status_code == 200


# --- Audit trail -------------------------------------------------------------------------


async def test_upload_and_delete_are_both_audited(client: AsyncClient, researcher: User) -> None:
    headers = await auth_headers(client, researcher.email, USER_PASSWORD)
    uploaded = await upload(
        client, headers, filename="a.png", data=make_png_bytes(), content_type="image/png"
    )
    image_id = uploaded.json()["id"]
    await client.delete(f"/images/{image_id}", headers=headers)

    admin_headers = await auth_headers(client, ADMIN_EMAIL, ADMIN_PASSWORD)
    audit = await client.get(f"/users/{researcher.id}/audit", headers=admin_headers)
    actions = {e["action"] for e in audit.json()["items"]}

    assert "image.uploaded" in actions
    assert "image.deleted" in actions
