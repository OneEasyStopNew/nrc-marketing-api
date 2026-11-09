# Create the configured bucket with versioning on, in local SeaweedFS only.
# Usage: uv run python -m scripts.create_local_bucket

import sys

import boto3
from botocore.config import Config
from botocore.exceptions import ClientError
from pydantic_settings import BaseSettings, SettingsConfigDict


class StorageEnv(BaseSettings):
    # Only the storage values, so the script runs without the database or Cognito settings.

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    aws_region: str = "us-east-1"
    aws_s3_bucket: str | None = None
    aws_endpoint_url_s3: str | None = None
    s3_force_path_style: bool = False
    aws_profile: str | None = None


def main() -> int:
    settings = StorageEnv()
    if not settings.aws_endpoint_url_s3:
        print("AWS_ENDPOINT_URL_S3 is not set. This script only works against local storage.")
        return 1
    if not settings.aws_s3_bucket:
        print("AWS_S3_BUCKET is not set.")
        return 1
    bucket = settings.aws_s3_bucket
    session = boto3.Session(profile_name=settings.aws_profile)
    client = session.client(
        "s3",
        region_name=settings.aws_region,
        endpoint_url=settings.aws_endpoint_url_s3,
        config=Config(
            signature_version="s3v4",
            s3={"addressing_style": "path"} if settings.s3_force_path_style else {},
        ),
    )
    try:
        client.create_bucket(Bucket=bucket)
    except ClientError as exc:
        if exc.response.get("Error", {}).get("Code") not in {
            "BucketAlreadyOwnedByYou",
            "BucketAlreadyExists",
        }:
            raise
    client.put_bucket_versioning(Bucket=bucket, VersioningConfiguration={"Status": "Enabled"})
    print(f"Bucket {bucket} is ready with versioning enabled.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
