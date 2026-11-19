"""Object storage against the local SeaweedFS (bucket nrc-test-media)."""

import base64
import hashlib
import re
from datetime import UTC, datetime, timedelta
from typing import Any
from urllib.parse import parse_qs, urlsplit

import boto3
import httpx
import pytest

from app.core.config import Settings, get_settings
from app.services.storage import (
    GENERATED_IMAGE_RETENTION,
    UPLOAD_RETENTION,
    ObjectStorage,
    PresignedUpload,
    StorageError,
    StorageNotConfiguredError,
    new_upload_key,
)
from tests.conftest import BUCKET, ENDPOINT, needs_seaweedfs


@pytest.fixture(autouse=True)
def local_credentials(monkeypatch: pytest.MonkeyPatch) -> None:
    # Local-only values that open the SeaweedFS container, not AWS.
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "localdev")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "localdev-password")
    monkeypatch.delenv("AWS_PROFILE", raising=False)


@pytest.fixture
def settings() -> Settings:
    return get_settings().model_copy(
        update={
            "aws_s3_bucket": BUCKET,
            "aws_endpoint_url_s3": ENDPOINT,
            "s3_force_path_style": True,
            "aws_profile": None,
        }
    )


@pytest.fixture
def storage(settings: Settings) -> ObjectStorage:
    return ObjectStorage(settings)


def test_missing_bucket_is_not_configured(settings: Settings) -> None:
    with pytest.raises(StorageNotConfiguredError):
        ObjectStorage(settings.model_copy(update={"aws_s3_bucket": None}))


@needs_seaweedfs
async def test_save_then_read_round_trip(storage: ObjectStorage) -> None:
    content = b"\x89PNG-test-bytes"
    stored = await storage.save_generated_image(content, ext="png", content_type="image/png")

    assert stored.bucket == BUCKET
    assert re.fullmatch(r"generated-images/\d{4}/\d{2}/[0-9a-f-]{36}\.png", stored.key)
    assert stored.version_id
    assert stored.size_bytes == len(content)
    assert stored.checksum_sha256 == hashlib.sha256(content).hexdigest()
    assert stored.content_type == "image/png"
    assert await storage.read(stored.key, version_id=stored.version_id) == content
    assert await storage.read(stored.key) == content


@needs_seaweedfs
async def test_object_is_tagged(storage: ObjectStorage, settings: Settings) -> None:
    stored = await storage.save_generated_image(b"tagged", ext="jpg", content_type="image/jpeg")

    client = boto3.Session().client("s3", region_name=settings.aws_region, endpoint_url=ENDPOINT)
    tags = client.get_object_tagging(Bucket=BUCKET, Key=stored.key)["TagSet"]
    assert {tag["Key"]: tag["Value"] for tag in tags} == {
        "doc-type": "generated-image",
        "retention": GENERATED_IMAGE_RETENTION,
    }


@needs_seaweedfs
async def test_presigned_get_downloads_same_bytes(
    storage: ObjectStorage, settings: Settings
) -> None:
    content = b"presigned-bytes"
    stored = await storage.save_generated_image(content, ext="png", content_type="image/png")

    url = await storage.presigned_get(
        stored.key, version_id=stored.version_id, expires_in=settings.asset_url_ttl_seconds
    )

    query = parse_qs(urlsplit(url).query)
    assert int(query["X-Amz-Expires"][0]) <= settings.asset_url_ttl_seconds
    async with httpx.AsyncClient() as client:
        response = await client.get(url)
    assert response.status_code == 200
    assert response.content == content


@needs_seaweedfs
async def test_each_save_gets_a_new_key(storage: ObjectStorage) -> None:
    first = await storage.save_generated_image(b"same", ext="png", content_type="image/png")
    second = await storage.save_generated_image(b"same", ext="png", content_type="image/png")

    assert first.key != second.key
    assert first.checksum_sha256 == second.checksum_sha256


async def test_presigned_ttl_is_clamped_to_the_configured_maximum(storage: ObjectStorage) -> None:
    url = await storage.presigned_get("generated-images/x.png", expires_in=86_400)
    expires = int(parse_qs(urlsplit(url).query)["X-Amz-Expires"][0])
    assert expires == get_settings().asset_url_ttl_seconds


async def test_unsupported_extension_is_rejected(storage: ObjectStorage) -> None:
    with pytest.raises(ValueError, match="ext"):
        await storage.save_generated_image(b"x", ext="svg", content_type="image/png")


async def test_unreachable_endpoint_raises_safe_storage_error(settings: Settings) -> None:
    broken = ObjectStorage(
        settings.model_copy(update={"aws_endpoint_url_s3": "http://127.0.0.1:1"})
    )
    with pytest.raises(StorageError) as raised:
        await broken.save_generated_image(b"x", ext="png", content_type="image/png")
    assert raised.value.__cause__ is None
    assert raised.value.__suppress_context__
    assert BUCKET not in str(raised.value)
    assert "127.0.0.1" not in str(raised.value)
    with pytest.raises(StorageError):
        await broken.read("generated-images/x.png")


async def test_missing_version_id_is_logged_and_kept_none(
    storage: ObjectStorage, caplog: pytest.LogCaptureFixture
) -> None:
    class NoVersion:
        def put_object(self, **kwargs: object) -> dict[str, str]:
            return {}

    storage._client = NoVersion()  # type: ignore[assignment]
    with caplog.at_level("WARNING"):
        stored = await storage.save_generated_image(b"x", ext="png", content_type="image/png")
    assert stored.version_id is None
    assert "no version id" in caplog.text


def test_s3_client_timeouts_and_retries_come_from_settings() -> None:
    settings = get_settings().model_copy(
        update={
            "s3_connect_timeout_seconds": 2.5,
            "s3_read_timeout_seconds": 12.0,
            "s3_max_retries": 4,
        }
    )

    config: Any = ObjectStorage(settings)._client.meta.config

    assert (config.connect_timeout, config.read_timeout) == (2.5, 12.0)
    # botocore counts the first attempt too: 4 retries are 5 attempts.
    assert config.retries == {"total_max_attempts": 5, "mode": "standard"}


# Client uploads (NRC-3): presigned PUT and HEAD.
UPLOAD_BYTES = b"\x89PNG-declared-upload-bytes"
UPLOAD_ERROR_MESSAGE = "Object storage request failed"


async def _presign_upload(
    storage: ObjectStorage,
    content: bytes = UPLOAD_BYTES,
    *,
    content_type: str = "image/png",
    expires_in: int = 600,
) -> PresignedUpload:
    return await storage.presigned_put(
        new_upload_key(content_type),
        content_type=content_type,
        size_bytes=len(content),
        checksum_sha256=hashlib.sha256(content).hexdigest(),
        expires_in=expires_in,
    )


async def _put(upload: PresignedUpload, content: bytes) -> httpx.Response:
    # Exactly the returned headers; httpx adds the real Content-Length and Host itself.
    async with httpx.AsyncClient() as client:
        return await client.request(
            upload.method, upload.url, content=content, headers=dict(upload.headers)
        )


def test_new_upload_key_layout_and_extension() -> None:
    now = datetime.now(UTC)
    png = new_upload_key("image/png")
    jpeg = new_upload_key("image/jpeg")

    assert re.fullmatch(rf"uploads/{now:%Y}/{now:%m}/[0-9a-f-]{{36}}\.png", png)
    assert re.fullmatch(rf"uploads/{now:%Y}/{now:%m}/[0-9a-f-]{{36}}\.jpg", jpeg)
    assert new_upload_key("image/png") != png


def test_new_upload_key_rejects_an_unmapped_type() -> None:
    with pytest.raises(ValueError, match="No upload extension for content type"):
        new_upload_key("image/gif")


async def test_presigned_put_signs_type_length_checksum_and_tags(storage: ObjectStorage) -> None:
    upload = await _presign_upload(storage)

    query = parse_qs(urlsplit(upload.url).query)
    assert set(query["X-Amz-SignedHeaders"][0].split(";")) == {
        "content-length",
        "content-type",
        "host",
        "x-amz-checksum-sha256",
        "x-amz-tagging",
    }
    checksum = base64.b64encode(hashlib.sha256(UPLOAD_BYTES).digest()).decode("ascii")
    assert upload.method == "PUT"
    assert dict(upload.headers) == {
        "Content-Type": "image/png",
        "x-amz-checksum-sha256": checksum,
        "x-amz-tagging": f"doc-type=upload&retention={UPLOAD_RETENTION}",
    }


async def test_presigned_put_ttl_is_clamped_to_the_upload_setting(storage: ObjectStorage) -> None:
    before = datetime.now(UTC)
    upload = await _presign_upload(storage, expires_in=86_400)

    setting = get_settings().upload_url_ttl_seconds
    assert int(parse_qs(urlsplit(upload.url).query)["X-Amz-Expires"][0]) == setting
    assert before + timedelta(seconds=setting) <= upload.expires_at
    assert upload.expires_at <= datetime.now(UTC) + timedelta(seconds=setting)


async def test_presigned_put_error_never_carries_the_url(settings: Settings) -> None:
    # A bucket name boto3 refuses to build a URL for.
    broken = ObjectStorage(settings.model_copy(update={"aws_s3_bucket": "Not A Bucket!"}))

    with pytest.raises(StorageError, match=UPLOAD_ERROR_MESSAGE) as raised:
        await _presign_upload(broken)

    assert raised.value.__cause__ is None
    assert raised.value.__suppress_context__
    for fragment in ("Not A Bucket", "127.0.0.1", "uploads/", "X-Amz"):
        assert fragment not in str(raised.value)


@needs_seaweedfs
async def test_presigned_put_upload_then_head_and_read(
    storage: ObjectStorage, settings: Settings
) -> None:
    upload = await _presign_upload(storage)

    response = await _put(upload, UPLOAD_BYTES)
    assert response.status_code == 200

    head = await storage.head(upload.key)
    assert head is not None
    assert head.size_bytes == len(UPLOAD_BYTES)
    assert head.checksum_sha256 == hashlib.sha256(UPLOAD_BYTES).hexdigest()
    assert head.content_type == "image/png"
    # SeaweedFS versions the test bucket, as the generated-image round trip also relies on.
    assert head.version_id
    assert await storage.read(upload.key, version_id=head.version_id) == UPLOAD_BYTES
    assert await storage.head(upload.key, version_id=head.version_id) == head

    client = boto3.Session().client("s3", region_name=settings.aws_region, endpoint_url=ENDPOINT)
    tags = client.get_object_tagging(Bucket=BUCKET, Key=upload.key)["TagSet"]
    assert {tag["Key"]: tag["Value"] for tag in tags} == {
        "doc-type": "upload",
        "retention": UPLOAD_RETENTION,
    }


@needs_seaweedfs
async def test_presigned_put_refuses_wrong_bytes_of_the_same_length(
    storage: ObjectStorage,
) -> None:
    upload = await _presign_upload(storage)

    response = await _put(upload, b"X" * len(UPLOAD_BYTES))

    # SeaweedFS verifies the signed x-amz-checksum-sha256 and answers 400 BadDigest.
    assert response.status_code == 400
    assert "BadDigest" in response.text
    assert await storage.head(upload.key) is None


@needs_seaweedfs
async def test_presigned_put_refuses_a_different_length(storage: ObjectStorage) -> None:
    upload = await _presign_upload(storage)

    response = await _put(upload, UPLOAD_BYTES + b"extra")

    # Content-Length is signed, so a different length fails the signature check.
    assert response.status_code == 403
    assert "SignatureDoesNotMatch" in response.text
    assert await storage.head(upload.key) is None


@needs_seaweedfs
async def test_head_of_a_missing_key_or_version_is_none(storage: ObjectStorage) -> None:
    upload = await _presign_upload(storage)
    assert (await _put(upload, UPLOAD_BYTES)).status_code == 200

    assert await storage.head(new_upload_key("image/png")) is None
    assert await storage.head(upload.key, version_id="0" * 32) is None


async def test_head_unreachable_endpoint_raises_safe_storage_error(settings: Settings) -> None:
    broken = ObjectStorage(
        settings.model_copy(update={"aws_endpoint_url_s3": "http://127.0.0.1:1"})
    )
    with pytest.raises(StorageError, match=UPLOAD_ERROR_MESSAGE) as raised:
        await broken.head("uploads/2026/11/x.png")
    assert raised.value.__cause__ is None
    assert "127.0.0.1" not in str(raised.value)


@pytest.mark.parametrize(
    ("reported", "expected"),
    [
        pytest.param(
            base64.b64encode(hashlib.sha256(b"x").digest()).decode(),
            hashlib.sha256(b"x").hexdigest(),
            id="full-object",
        ),
        # A multipart object's checksum covers its parts, not the file (code review, NRC-3).
        pytest.param(
            base64.b64encode(hashlib.sha256(b"x").digest()).decode() + "-3",
            None,
            id="multipart-composite",
        ),
        pytest.param("not base64!", None, id="malformed"),
        pytest.param(base64.b64encode(b"short").decode(), None, id="wrong-length"),
        pytest.param(None, None, id="absent"),
    ],
)
async def test_head_reports_only_a_full_object_sha256(
    storage: ObjectStorage,
    monkeypatch: pytest.MonkeyPatch,
    reported: str | None,
    expected: str | None,
) -> None:
    response: dict[str, object] = {
        "ContentLength": 1,
        "VersionId": "v1",
        "ContentType": "image/png",
    }
    if reported is not None:
        response["ChecksumSHA256"] = reported
    monkeypatch.setattr(storage._client, "head_object", lambda **_: response)

    head = await storage.head("uploads/2026/11/x.png")

    assert head is not None
    assert head.checksum_sha256 == expected
