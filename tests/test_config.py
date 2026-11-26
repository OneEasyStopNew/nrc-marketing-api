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


def test_policy_context_timeout_must_be_below_the_db_timeout() -> None:
    # Otherwise asyncpg's own timeout fires first and fails the request instead of NEEDS_REVIEW.
    assert _settings(policy_context_timeout_seconds=9, db_timeout_seconds=10)
    with pytest.raises(ValidationError, match="must be below DB_TIMEOUT_SECONDS"):
        _settings(policy_context_timeout_seconds=10, db_timeout_seconds=10)


def test_policy_context_timeout_is_at_least_a_millisecond() -> None:
    # statement_timeout is whole milliseconds, and 0 would mean no limit at all.
    with pytest.raises(ValidationError, match="policy_context_timeout_seconds"):
        _settings(policy_context_timeout_seconds=0.0004)


def test_cancel_lock_timeout_never_rounds_to_postgres_no_timeout() -> None:
    # The value becomes SET LOCAL lock_timeout in whole milliseconds, and 0 ms there means wait
    # forever: anything that rounds to 0 is refused.
    assert _settings(job_cancel_lock_timeout_seconds=0.001).job_cancel_lock_timeout_seconds == 0.001
    assert _settings(job_cancel_lock_timeout_seconds=30).job_cancel_lock_timeout_seconds == 30

    for bad in (0, 0.0004, 0.0009, 30.001):
        with pytest.raises(ValidationError, match="job_cancel_lock_timeout_seconds"):
            _settings(job_cancel_lock_timeout_seconds=bad)


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
