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
