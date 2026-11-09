# The only place that talks to S3. Callers store the key, checksum and version id, never a URL.

import asyncio
import base64
import hashlib
import logging
from dataclasses import dataclass
from datetime import UTC, datetime
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

_ALLOWED_EXTENSIONS = frozenset({"png", "jpg"})
GENERATED_IMAGE_FOLDER = "generated-images"
GENERATED_IMAGE_DOC_TYPE = "generated-image"
# TODO(NRC-3): confirm retention class with Prabhu
GENERATED_IMAGE_RETENTION = "operational"


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


class ObjectStorage:
    def __init__(self, settings: Settings) -> None:
        if not settings.aws_s3_bucket:
            raise StorageNotConfiguredError("AWS_S3_BUCKET is not configured")
        self._bucket = settings.aws_s3_bucket
        self._max_url_ttl = settings.asset_url_ttl_seconds
        # Credentials come from the profile or environment locally, and the IAM role on servers.
        session = boto3.Session(profile_name=settings.aws_profile)
        client_config = Config(
            signature_version="s3v4",
            connect_timeout=5,
            read_timeout=30,
            retries={"max_attempts": 3, "mode": "standard"},
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
