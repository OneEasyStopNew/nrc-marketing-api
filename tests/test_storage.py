"""Object storage against the local SeaweedFS (bucket nrc-test-media)."""

import hashlib
import os
import re
from urllib.parse import parse_qs, urlsplit

import boto3
import httpx
import pytest

from app.core.config import Settings, get_settings
from app.services.storage import (
    GENERATED_IMAGE_RETENTION,
    ObjectStorage,
    StorageError,
    StorageNotConfiguredError,
)

ENDPOINT = "http://127.0.0.1:8333"
BUCKET = "nrc-test-media"


def _seaweedfs_reachable() -> bool:
    try:
        httpx.get(ENDPOINT, timeout=2)
    except httpx.HTTPError:
        return False
    return True


# Locally the tests skip without SeaweedFS; in CI (CI env var set) they run and so fail loudly.
needs_seaweedfs = pytest.mark.skipif(
    "CI" not in os.environ and not _seaweedfs_reachable(),
    reason=f"SeaweedFS is not reachable at {ENDPOINT}",
)


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
