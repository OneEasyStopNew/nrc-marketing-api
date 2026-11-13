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
        ("max_generated_image_bytes", 52_428_801),
        ("max_image_pixels", 100_000_001),
    ],
)
def test_operator_limits_have_an_upper_bound(field: str, too_high: object) -> None:
    with pytest.raises(ValidationError, match=field):
        _settings(**{field: too_high})
