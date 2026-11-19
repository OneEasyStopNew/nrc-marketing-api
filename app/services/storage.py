# The only place that talks to S3. Callers store the key, checksum and version id, never a URL.

import asyncio
import base64
import hashlib
import logging
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING, Any
from uuid import uuid4

import boto3
from botocore.config import Config
from botocore.exceptions import BotoCoreError, ClientError

from app.core.config import Settings

# Type stubs are a dev dependency only; never import them at runtime.
if TYPE_CHECKING:
    from mypy_boto3_s3 import S3Client

logger = logging.getLogger(__name__)

# Fixed parts of the object layout and tags, not per environment: bucket lifecycle rules and
# validation (NRC-7) match on them.
_ALLOWED_EXTENSIONS = frozenset({"png", "jpg"})
GENERATED_IMAGE_FOLDER = "generated-images"
GENERATED_IMAGE_DOC_TYPE = "generated-image"
# TODO(NRC-3): confirm retention class with Prabhu
GENERATED_IMAGE_RETENTION = "operational"
UPLOAD_FOLDER = "uploads"
UPLOAD_DOC_TYPE = "upload"
# TODO(NRC-3): confirm retention class with Prabhu
UPLOAD_RETENTION = "operational"
# Accepted upload MIME type to the extension its key carries.
_UPLOAD_EXTENSIONS: Mapping[str, str] = {"image/png": "png", "image/jpeg": "jpg"}
# HEAD responses have no body, so botocore reports the HTTP status as the error code.
_MISSING_OBJECT_CODES = frozenset({"404", "NoSuchKey", "NoSuchVersion", "NotFound"})


class StorageNotConfiguredError(RuntimeError):
    pass


class StorageError(RuntimeError):
    # Fixed message on purpose: boto3 errors can carry the bucket, key or a URL.
    def __init__(self) -> None:
        super().__init__("Object storage request failed")


@dataclass(frozen=True, slots=True)
class StoredObject:
    bucket: str
    key: str
    version_id: str | None
    size_bytes: int
    checksum_sha256: str
    content_type: str


@dataclass(frozen=True, slots=True)
class PresignedUpload:
    key: str
    url: str
    method: str
    # Exactly what the browser must send with the PUT for the signature to match.
    headers: Mapping[str, str]
    expires_at: datetime


@dataclass(frozen=True, slots=True)
class ObjectHead:
    size_bytes: int
    version_id: str | None
    # Lower-case hex, or None when the store reports no checksum.
    checksum_sha256: str | None
    content_type: str | None


def new_upload_key(content_type: str) -> str:
    # Callers validate the type first; this guards a programming error, not a job path.
    ext = _UPLOAD_EXTENSIONS.get(content_type)
    if ext is None:
        raise ValueError(f"No upload extension for content type {content_type!r}")
    now = datetime.now(UTC)
    return f"{UPLOAD_FOLDER}/{now:%Y}/{now:%m}/{uuid4()}.{ext}"


class ObjectStorage:
    def __init__(self, settings: Settings) -> None:
        if not settings.aws_s3_bucket:
            raise StorageNotConfiguredError("AWS_S3_BUCKET is not configured")
        self._bucket = settings.aws_s3_bucket
        self._max_url_ttl = settings.asset_url_ttl_seconds
        self._max_upload_url_ttl = settings.upload_url_ttl_seconds
        # Credentials come from the profile or environment locally, and the IAM role on servers.
        session = boto3.Session(profile_name=settings.aws_profile)
        client_config = Config(
            signature_version="s3v4",
            connect_timeout=settings.s3_connect_timeout_seconds,
            read_timeout=settings.s3_read_timeout_seconds,
            # botocore's max_attempts counts retries, not the first attempt.
            retries={"max_attempts": settings.s3_max_retries, "mode": "standard"},
            s3={"addressing_style": "path"} if settings.s3_force_path_style else {},
        )
        self._client: S3Client = session.client(
            "s3",
            region_name=settings.aws_region,
            endpoint_url=settings.aws_endpoint_url_s3,
            config=client_config,
        )

    async def save_generated_image(
        self, content: bytes, *, ext: str, content_type: str
    ) -> StoredObject:
        if ext not in _ALLOWED_EXTENSIONS:
            raise ValueError("ext must be png or jpg")
        digest = hashlib.sha256(content).digest()
        now = datetime.now(UTC)
        key = f"{GENERATED_IMAGE_FOLDER}/{now:%Y}/{now:%m}/{uuid4()}.{ext}"
        tagging = f"doc-type={GENERATED_IMAGE_DOC_TYPE}&retention={GENERATED_IMAGE_RETENTION}"
        try:
            response = await asyncio.to_thread(
                self._client.put_object,
                Bucket=self._bucket,
                Key=key,
                Body=content,
                ContentType=content_type,
                ChecksumSHA256=base64.b64encode(digest).decode("ascii"),
                Tagging=tagging,
            )
        except BotoCoreError, ClientError:
            raise StorageError from None
        version_id = response.get("VersionId")
        if version_id is None:
            logger.warning("Object storage returned no version id; is the bucket versioned?")
        return StoredObject(
            bucket=self._bucket,
            key=key,
            version_id=version_id,
            size_bytes=len(content),
            checksum_sha256=digest.hex(),
            content_type=content_type,
        )

    async def presigned_get(
        self, key: str, *, version_id: str | None = None, expires_in: int
    ) -> str:
        ttl = min(expires_in, self._max_url_ttl)
        return await asyncio.to_thread(self._presign, key, version_id, ttl)

    def _presign(self, key: str, version_id: str | None, expires_in: int) -> str:
        params: dict[str, Any] = {"Bucket": self._bucket, "Key": key}
        if version_id:
            params["VersionId"] = version_id
        try:
            return self._client.generate_presigned_url(
                "get_object", Params=params, ExpiresIn=expires_in
            )
        except BotoCoreError, ClientError:
            raise StorageError from None

    async def read(self, key: str, *, version_id: str | None = None) -> bytes:
        return await asyncio.to_thread(self._read, key, version_id)

    def _read(self, key: str, version_id: str | None) -> bytes:
        try:
            if version_id:
                response = self._client.get_object(
                    Bucket=self._bucket, Key=key, VersionId=version_id
                )
            else:
                response = self._client.get_object(Bucket=self._bucket, Key=key)
            body = response["Body"]
            try:
                return body.read()
            finally:
                body.close()
        except BotoCoreError, ClientError:
            raise StorageError from None

    async def presigned_put(
        self,
        key: str,
        *,
        content_type: str,
        size_bytes: int,
        checksum_sha256: str,
        expires_in: int,
    ) -> PresignedUpload:
        ttl = min(expires_in, self._max_upload_url_ttl)
        expires_at = datetime.now(UTC) + timedelta(seconds=ttl)
        params: dict[str, Any] = {
            "Bucket": self._bucket,
            "Key": key,
            "ContentType": content_type,
            "ContentLength": size_bytes,
            "ChecksumSHA256": base64.b64encode(bytes.fromhex(checksum_sha256)).decode("ascii"),
            "Tagging": f"doc-type={UPLOAD_DOC_TYPE}&retention={UPLOAD_RETENTION}",
        }
        try:
            url = await asyncio.to_thread(
                self._client.generate_presigned_url,
                "put_object",
                Params=params,
                ExpiresIn=ttl,
            )
        except BotoCoreError, ClientError:
            raise StorageError from None
        # The signed headers are content-length, content-type, host, x-amz-checksum-sha256 and
        # x-amz-tagging (checked on a generated URL). The browser sets host itself and sends the
        # real Content-Length, a forbidden header it can't be told to set: the signature only
        # matches when that length equals the declared size, which is the point.
        headers = {
            "Content-Type": content_type,
            "x-amz-checksum-sha256": params["ChecksumSHA256"],
            "x-amz-tagging": params["Tagging"],
        }
        return PresignedUpload(
            key=key, url=url, method="PUT", headers=headers, expires_at=expires_at
        )

    async def head(self, key: str, *, version_id: str | None = None) -> ObjectHead | None:
        return await asyncio.to_thread(self._head, key, version_id)

    def _head(self, key: str, version_id: str | None) -> ObjectHead | None:
        params: dict[str, Any] = {"Bucket": self._bucket, "Key": key, "ChecksumMode": "ENABLED"}
        if version_id:
            params["VersionId"] = version_id
        try:
            response = self._client.head_object(**params)
        except ClientError as error:
            if error.response.get("Error", {}).get("Code") in _MISSING_OBJECT_CODES:
                return None
            raise StorageError from None
        except BotoCoreError:
            raise StorageError from None
        checksum = response.get("ChecksumSHA256")
        return ObjectHead(
            size_bytes=response["ContentLength"],
            version_id=response.get("VersionId"),
            checksum_sha256=base64.b64decode(checksum).hex() if checksum else None,
            content_type=response.get("ContentType"),
        )
