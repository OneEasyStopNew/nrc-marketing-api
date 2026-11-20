from decimal import Decimal
from typing import Any

import pytest
from pydantic import ValidationError

from app.core.config import Settings

REQUIRED: dict[str, Any] = {
    "database_url": "postgresql+asyncpg://u:p@localhost/db",
    "cognito_region": "ap-south-1",
    "cognito_user_pool_id": "ap-south-1_TESTPOOL",
    "cognito_client_ids": ("c",),
    "required_scope": "marketing/api",
}


def _settings(**overrides: Any) -> Settings:
    return Settings(_env_file=None, **{**REQUIRED, **overrides})


def test_asset_url_ttl_is_at_most_fifteen_minutes() -> None:
    assert _settings(asset_url_ttl_seconds=900).asset_url_ttl_seconds == 900

    with pytest.raises(ValidationError, match="asset_url_ttl_seconds"):
        _settings(asset_url_ttl_seconds=901)


@pytest.mark.parametrize(
    ("field", "too_high"),
    [
        ("generation_daily_credit_ceiling_per_tenant", 1001),
        ("generation_credits_per_image", 101),
        ("generation_max_cost_usd_per_image", Decimal("1.01")),
        ("generation_global_budget_usd", Decimal("10000.01")),
        ("max_generated_image_bytes", 52_428_801),
        ("max_image_pixels", 100_000_001),
    ],
)
def test_operator_limits_have_an_upper_bound(field: str, too_high: object) -> None:
    with pytest.raises(ValidationError, match=field):
        _settings(**{field: too_high})


def test_client_upload_defaults() -> None:
    settings = _settings()

    assert settings.upload_url_ttl_seconds == 600
    assert settings.upload_image_max_bytes == 20 * 1024 * 1024
    assert settings.upload_image_types == ("image/png", "image/jpeg")
    assert settings.malware_scan_mode == "live"
    assert settings.media_inspection_max_attempts == 3


@pytest.mark.parametrize(
    ("field", "bad_value"),
    [
        ("upload_url_ttl_seconds", 59),
        ("upload_url_ttl_seconds", 901),
        ("upload_image_max_bytes", 52_428_801),
        ("media_inspection_max_attempts", 0),
        ("media_inspection_max_attempts", 11),
    ],
)
def test_client_upload_limits_are_bounded(field: str, bad_value: int) -> None:
    with pytest.raises(ValidationError, match=field):
        _settings(**{field: bad_value})


def test_upload_image_types_are_normalised() -> None:
    settings = _settings(upload_image_types="image/png, IMAGE/JPEG")

    assert settings.upload_image_types == ("image/png", "image/jpeg")


def test_upload_image_types_reject_a_type_without_a_decoder() -> None:
    with pytest.raises(ValidationError, match=r"unsupported type\(s\): image/webp"):
        _settings(upload_image_types="image/png,image/webp")


@pytest.mark.parametrize("empty", ["", " , ", ()])
def test_upload_image_types_cannot_be_empty(empty: object) -> None:
    with pytest.raises(ValidationError, match="at least one MIME type"):
        _settings(upload_image_types=empty)


@pytest.mark.parametrize("app_env", ["local", "demo"])
def test_malware_scan_simulator_is_allowed_in_development(app_env: str) -> None:
    settings = _settings(app_env=app_env, malware_scan_mode="simulator")

    assert settings.malware_scan_mode == "simulator"


@pytest.mark.parametrize("app_env", ["staging", "production"])
def test_malware_scan_simulator_is_refused_outside_development(app_env: str) -> None:
    with pytest.raises(ValidationError, match="MALWARE_SCAN_MODE=simulator"):
        _settings(app_env=app_env, malware_scan_mode="simulator")


def test_policy_source_defaults() -> None:
    settings = _settings()

    assert settings.policy_source_bucket is None
    assert settings.policy_source_max_bytes == 50 * 1024 * 1024


@pytest.mark.parametrize("blank", ["", "  "])
def test_a_blank_policy_source_bucket_means_unset(blank: str) -> None:
    assert _settings(policy_source_bucket=blank).policy_source_bucket is None


def test_policy_source_bucket_is_read_as_given() -> None:
    assert _settings(policy_source_bucket="nrc-dev-policy").policy_source_bucket == "nrc-dev-policy"


@pytest.mark.parametrize("bad_value", [1023, 104_857_601])
def test_policy_source_max_bytes_is_bounded(bad_value: int) -> None:
    with pytest.raises(ValidationError, match="policy_source_max_bytes"):
        _settings(policy_source_max_bytes=bad_value)


def test_image_types_do_not_accept_policy_types() -> None:
    with pytest.raises(ValidationError, match=r"unsupported type\(s\): application/pdf"):
        _settings(upload_image_types="application/pdf")
