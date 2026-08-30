"""Satellite image management use cases — Module 2's core.

Like every service in this project, this imports no database driver, storage SDK, or
imaging library directly — only the domain ports. rasterio/Pillow/boto3 live behind
:class:`~app.domain.interfaces.ImageAnalyzer` and
:class:`~app.domain.interfaces.ObjectStorage`.
"""

from __future__ import annotations

import asyncio
import hashlib
from pathlib import PurePosixPath
from uuid import UUID

from app.application.audit_service import AuditService
from app.application.dto import RequestContext
from app.core.logging import get_logger
from app.domain.entities import SatelliteImage, User
from app.domain.enums import AuditAction, AuditOutcome, ImageFormat, ImageStatus
from app.domain.errors import AuthorizationError, FileTooLargeError, NotFoundError
from app.domain.interfaces import ImageAnalyzer, ImageRepository, ObjectStorage, UnitOfWork

logger = get_logger(__name__)


class ImageService:
    """Upload, inspection, listing, retrieval and deletion of satellite imagery."""

    def __init__(
        self,
        *,
        images: ImageRepository,
        storage: ObjectStorage,
        analyzer: ImageAnalyzer,
        audit: AuditService,
        uow: UnitOfWork,
        max_upload_size_bytes: int,
        thumbnail_max_size_px: int,
    ) -> None:
        self._images = images
        self._storage = storage
        self._analyzer = analyzer
        self._audit = audit
        self._uow = uow
        self._max_upload_size_bytes = max_upload_size_bytes
        self._thumbnail_max_size_px = thumbnail_max_size_px

    @property
    def max_upload_size_bytes(self) -> int:
        """Exposed so the route can bound how much of an oversized request body it reads
        into memory before rejecting it — see ``read_upload_bounded`` in
        ``api/v1/images.py``. The check inside :meth:`upload` below stays too, as a
        defensive backstop for any future caller that hands this service bytes directly."""
        return self._max_upload_size_bytes

    # --- Upload ----------------------------------------------------------------------

    async def upload(
        self,
        *,
        owner: User,
        filename: str,
        data: bytes,
        context: RequestContext | None = None,
    ) -> SatelliteImage:
        """Validate, store, and process an uploaded file.

        Content is sniffed for real, not trusted from the filename — a mismatch or
        unreadable file is rejected here, before anything is persisted or stored.
        Everything after that point (metadata extraction, thumbnailing) is recorded as
        having happened even if it subsequently fails: the upload itself succeeded, the
        file exists, and a processing failure is a fact about the record, not a reason to
        pretend the upload never happened.
        """
        if len(data) > self._max_upload_size_bytes:
            raise FileTooLargeError(
                f"File is {len(data)} bytes; the maximum is {self._max_upload_size_bytes}.",
                size_bytes=len(data),
                max_bytes=self._max_upload_size_bytes,
            )

        extension = PurePosixPath(filename).suffix.lstrip(".")
        image_format = await self._analyzer.sniff_format(data, extension)

        image = SatelliteImage(
            owner_id=owner.id,
            original_filename=filename,
            image_format=image_format,
            storage_key="",  # filled in once the id is known, just below
            file_size_bytes=len(data),
            checksum_sha256=hashlib.sha256(data).hexdigest(),
        )
        image.storage_key = self._original_key(owner.id, image.id, image_format)

        await self._storage.put(image.storage_key, data, content_type=_content_type(image_format))
        image = await self._images.add(image)
        await self._uow.commit()  # the file is stored and the record exists — durable now

        await self._process(image, data, context)

        await self._audit.record(
            action=AuditAction.IMAGE_UPLOADED,
            user_id=owner.id,
            context=context,
            image_id=str(image.id),
            image_format=image_format.value,
            file_size_bytes=image.file_size_bytes,
            status=image.status.value,
        )
        await self._uow.commit()

        logger.info(
            "image_uploaded",
            image_id=str(image.id),
            owner_id=str(owner.id),
            status=image.status.value,
        )
        return image

    async def _process(
        self, image: SatelliteImage, data: bytes, context: RequestContext | None
    ) -> None:
        """Extract metadata and generate a thumbnail. Metadata extraction failing marks
        the image FAILED; thumbnail generation failing does not — see
        :meth:`~app.domain.entities.SatelliteImage.mark_ready`.

        The two extraction calls run concurrently, not one after the other: they read the
        same immutable ``data``, neither depends on the other's result, and each already
        runs in its own worker thread (see ``RasterImageAnalyzer``) — running them
        sequentially would just make every upload wait twice as long as necessary,
        with nothing gained for the extra wait.
        """
        image.mark_processing()
        await self._images.update(image)

        metadata_result, thumbnail_result = await asyncio.gather(
            self._analyzer.inspect(data, image.image_format),
            self._analyzer.generate_thumbnail(
                data, image.image_format, max_size=self._thumbnail_max_size_px
            ),
            return_exceptions=True,
        )

        if isinstance(metadata_result, BaseException):
            # record with a human-readable reason, not a 500 for an upload that already
            # succeeded and whose bytes are already safely stored.
            logger.warning(
                "image_inspection_failed", image_id=str(image.id), error=str(metadata_result)
            )
            image.mark_failed(f"Metadata extraction failed: {metadata_result}")
            await self._images.update(image)
            await self._audit.record(
                action=AuditAction.IMAGE_PROCESSING_FAILED,
                outcome=AuditOutcome.FAILURE,
                user_id=image.owner_id,
                context=context,
                image_id=str(image.id),
                reason=str(metadata_result),
            )
            return

        metadata = metadata_result
        thumbnail_key: str | None = None
        if isinstance(thumbnail_result, BaseException):
            logger.warning(
                "thumbnail_generation_failed", image_id=str(image.id), error=str(thumbnail_result)
            )
        else:
            try:
                thumbnail_key = self._thumbnail_key(image.owner_id, image.id)
                await self._storage.put(thumbnail_key, thumbnail_result, content_type="image/jpeg")
            except Exception as exc:  # thumbnailing is best-effort; storing it can still fail
                logger.warning("thumbnail_storage_failed", image_id=str(image.id), error=str(exc))
                thumbnail_key = None

        image.mark_ready(metadata, thumbnail_key=thumbnail_key)
        await self._images.update(image)

    # --- Retrieval ---------------------------------------------------------------------

    async def get(self, image_id: UUID, requesting_user: User) -> SatelliteImage:
        """Raises :class:`NotFoundError` both when the image genuinely doesn't exist and
        when it exists but belongs to someone else — a 403 there would confirm to an
        unauthorised caller that a specific image id is in use, which a 404 does not."""
        image = await self._images.get_by_id(image_id)
        if image is None or not self._can_access(image, requesting_user):
            raise NotFoundError("Image not found.", image_id=str(image_id))
        return image

    async def get_thumbnail(self, image_id: UUID, requesting_user: User) -> bytes:
        image = await self.get(image_id, requesting_user)
        if image.thumbnail_key is None:
            raise NotFoundError("No thumbnail is available for this image.", image_id=str(image_id))
        return await self._storage.get(image.thumbnail_key)

    async def list_images(
        self,
        *,
        requesting_user: User,
        offset: int = 0,
        limit: int = 50,
        image_format: ImageFormat | None = None,
        status: ImageStatus | None = None,
    ) -> tuple[list[SatelliteImage], int]:
        """Researchers see only their own images; admins see every image. The scoping
        happens here, once, rather than trusting every call site to remember it."""
        owner_filter = None if requesting_user.is_admin else requesting_user.id
        return await self._images.list_images(
            offset=offset,
            limit=limit,
            owner_id=owner_filter,
            image_format=image_format,
            status=status,
        )

    # --- Deletion ------------------------------------------------------------------------

    async def delete(
        self, image_id: UUID, requesting_user: User, context: RequestContext | None = None
    ) -> None:
        image = await self._images.get_by_id(image_id)
        if image is None:
            raise NotFoundError("Image not found.", image_id=str(image_id))
        if not self._can_access(image, requesting_user):
            # Unlike get(), a delete attempt on an image the caller can prove exists (they
            # already fetched it, or are probing sequential ids) is worth recording as a
            # denied action rather than silently returning 404.
            await self._audit.record(
                action=AuditAction.ACCESS_DENIED,
                outcome=AuditOutcome.FAILURE,
                user_id=requesting_user.id,
                context=context,
                image_id=str(image_id),
                attempted_action="delete_image",
            )
            await self._uow.commit()
            raise AuthorizationError("You do not have permission to delete this image.")

        await self._storage.delete(image.storage_key)
        if image.thumbnail_key is not None:
            await self._storage.delete(image.thumbnail_key)
        await self._images.delete(image_id)

        await self._audit.record(
            action=AuditAction.IMAGE_DELETED,
            user_id=requesting_user.id,
            context=context,
            image_id=str(image_id),
            owner_id=str(image.owner_id),
        )
        await self._uow.commit()
        logger.info("image_deleted", image_id=str(image_id), actor_id=str(requesting_user.id))

    # --- Internals -----------------------------------------------------------------------

    @staticmethod
    def _can_access(image: SatelliteImage, user: User) -> bool:
        return user.is_admin or image.is_owned_by(user.id)

    @staticmethod
    def _original_key(owner_id: UUID, image_id: UUID, image_format: ImageFormat) -> str:
        ext = "tif" if image_format is ImageFormat.GEOTIFF else image_format.value
        return f"{owner_id}/{image_id}/original.{ext}"

    @staticmethod
    def _thumbnail_key(owner_id: UUID, image_id: UUID) -> str:
        return f"{owner_id}/{image_id}/thumbnail.jpg"


def _content_type(image_format: ImageFormat) -> str:
    return {
        ImageFormat.GEOTIFF: "image/tiff",
        ImageFormat.PNG: "image/png",
        ImageFormat.JPEG: "image/jpeg",
    }[image_format]
