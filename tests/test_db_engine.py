# The app's engine never puts bound parameter values (post text, names) into database errors,
# so logging an error with its traceback can't leak customer content.

import pytest
from sqlalchemy import text
from sqlalchemy.exc import DBAPIError

from app.core.config import get_settings
from app.db.session import create_engine
from tests.conftest import IntegrationEnv

MARKER = "synthetic-post-text-7f3a"


async def test_database_errors_hide_parameter_values(test_env: IntegrationEnv) -> None:
    engine = create_engine(get_settings(), test_env.test_database_url)
    try:
        async with engine.connect() as conn:
            with pytest.raises(DBAPIError, match="division by zero") as caught:
                await conn.execute(
                    text("SELECT CAST(:marker AS text), 1 / :zero"),
                    {"marker": MARKER, "zero": 0},
                )
    finally:
        await engine.dispose()

    assert MARKER not in str(caught.value)
    assert "parameters hidden" in str(caught.value)
