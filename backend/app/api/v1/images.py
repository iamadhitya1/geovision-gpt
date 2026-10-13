"""Satellite image management endpoints."""

from __future__ import annotations

from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, Depends, File, Query, UploadFile
from fastapi.responses import Response

from app.api.deps import ResearcherUser, get_image_service, get_request_context
from app.application.dto import (
    CaptureSceneRequest,
    ImageRead,
    MessageResponse,
    Page,
    RequestContext,
)
from app.application.image_service import ImageService
from app.domain.enums import ImageFormat, ImageStatus
from app.domain.errors import FileTooLargeError

router = APIRouter(prefix="/images", tags=["Images"])

ImageServiceDep = Annotated[ImageService, Depends(get_image_service)]
ContextDep = Annotated[RequestContext, Depends(get_request_context)]

_UPLOAD_CHUNK_SIZE = 1024 * 1024  # 1 MiB


async def _read_upload_bounded(file: UploadFile, max_bytes: int) -> bytes:
    """Read ``file`` in chunks, aborting as soon as ``max_bytes`` is exceeded.

    ``ImageService.upload`` also checks the final size, but that check runs *after* the
    bytes are already in memory — reading the whole thing first, then rejecting it,
    defeats the point of having a limit at all. Bounding the read itself means an
    oversized upload costs at most one chunk beyond the limit, not its full size.
    """
    total = 0
    chunks: list[bytes] = []
    while chunk := await file.read(_UPLOAD_CHUNK_SIZE):
        total += len(chunk)
        if total > max_bytes:
            raise FileTooLargeError(
                f"File exceeds the maximum of {max_bytes} bytes.",
                max_bytes=max_bytes,
            )
        chunks.append(chunk)
    return b"".join(chunks)


@router.post(
    "",
    response_model=ImageRead,
    status_code=201,
    summary="Upload a satellite image",
    responses={
        413: {"description": "File exceeds the maximum upload size"},
        422: {"description": "Unsupported file type, or the content doesn't match it"},
    },
)
async def upload_image(
    researcher: ResearcherUser,
    service: ImageServiceDep,
    context: ContextDep,
    file: Annotated[UploadFile, File(description="A .tif/.tiff, .png, or .jpg/.jpeg file")],
) -> ImageRead:
    """Accepts GeoTIFF, PNG, or JPEG. The file's actual content is verified against its
    extension — a mismatched or corrupt file is rejected before anything is stored.

    A 201 response does not by itself mean metadata extraction succeeded — check
    `status`. `ready` means metadata and (usually) a thumbnail are available; `failed`
    means the upload itself succeeded but the file couldn't be parsed as a raster, with
    `error_message` explaining why. The file bytes are never discarded either way.
    """
    data = await _read_upload_bounded(file, service.max_upload_size_bytes)
    image = await service.upload(
        owner=researcher,
        filename=file.filename or "upload",
        data=data,
        context=context,
    )
    return ImageRead.from_domain(image)


@router.post(
    "/capture",
    response_model=ImageRead,
    status_code=201,
    summary="Fetch a satellite scene for a point on the map",
    responses={
        502: {"description": "The imagery service couldn't be reached, or has nothing there"},
    },
)
async def capture_scene(
    request: CaptureSceneRequest,
    researcher: ResearcherUser,
    service: ImageServiceDep,
    context: ContextDep,
) -> ImageRead:
    """Cuts a 256 x 256 px, ~10 m per pixel scene (2.56 km across) from the Sentinel-2
    cloudless mosaic, centred on the point, and stores it as the caller's own image — the same
    shape of data the models were trained on, so a question about it is a fair one. From
    there it behaves exactly like an uploaded file."""
    image = await service.capture_scene(
        owner=researcher,
        latitude=request.latitude,
        longitude=request.longitude,
        context=context,
    )
    return ImageRead.from_domain(image)


@router.get(
    "",
    response_model=Page[ImageRead],
    summary="List images",
)
async def list_images(
    researcher: ResearcherUser,
    service: ImageServiceDep,
    offset: Annotated[int, Query(ge=0)] = 0,
    limit: Annotated[int, Query(ge=1, le=200)] = 50,
    image_format: ImageFormat | None = None,
    status: ImageStatus | None = None,
) -> Page[ImageRead]:
    """A researcher sees only their own uploads; an admin sees every image. See
    `ImageService.list_images` for where that scoping is enforced."""
    images, total = await service.list_images(
        requesting_user=researcher,
        offset=offset,
        limit=limit,
        image_format=image_format,
        status=status,
    )
    return Page[ImageRead](
        items=[ImageRead.from_domain(i) for i in images], total=total, offset=offset, limit=limit
    )


@router.get(
    "/{image_id}",
    response_model=ImageRead,
    summary="Read an image's metadata",
    responses={404: {"description": "Image not found, or not owned by the caller"}},
)
async def read_image(
    image_id: UUID, researcher: ResearcherUser, service: ImageServiceDep
) -> ImageRead:
    image = await service.get(image_id, researcher)
    return ImageRead.from_domain(image)


@router.get(
    "/{image_id}/thumbnail",
    summary="Fetch the preview thumbnail",
    responses={
        200: {"content": {"image/jpeg": {}}},
        404: {"description": "Image not found, not owned by the caller, or has no thumbnail"},
    },
)
async def read_thumbnail(
    image_id: UUID, researcher: ResearcherUser, service: ImageServiceDep
) -> Response:
    thumbnail_bytes = await service.get_thumbnail(image_id, researcher)
    return Response(content=thumbnail_bytes, media_type="image/jpeg")


@router.delete(
    "/{image_id}",
    response_model=MessageResponse,
    summary="Delete an image",
    responses={
        403: {"description": "Not the owner and not an admin"},
        404: {"description": "Image not found"},
    },
)
async def delete_image(
    image_id: UUID,
    researcher: ResearcherUser,
    service: ImageServiceDep,
    context: ContextDep,
) -> MessageResponse:
    """Removes the database record and both stored objects (original and thumbnail)."""
    await service.delete(image_id, researcher, context)
    return MessageResponse(message="Image deleted.")
