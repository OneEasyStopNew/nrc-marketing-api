# The production image installs runtime dependencies only: app code must import without dev ones.

import re
import subprocess
import sys

from tests.conftest import ROOT

# Blocks every dev-only package, then imports each entry point in a fresh interpreter.
_SCRIPT = """
import sys
from importlib.abc import MetaPathFinder

DEV_ONLY = ("mypy_boto3", "boto3_stubs", "botocore_stubs", "mypy", "pytest", "asgi_lifespan")


class BlockDevOnly(MetaPathFinder):
    def find_spec(self, name, path=None, target=None):
        if name.split(".")[0].startswith(DEV_ONLY):
            raise ModuleNotFoundError(f"dev-only module imported at runtime: {name}")
        return None


sys.meta_path.insert(0, BlockDevOnly())
import app.main
import app.workers.connector
import app.services.providers.higgsfield
import app.services.storage
import scripts.list_unknown_jobs
import scripts.higgsfield_smoke
"""


def test_app_imports_without_dev_dependencies() -> None:
    result = subprocess.run(  # noqa: S603  fixed interpreter and script, no user input
        [sys.executable, "-c", _SCRIPT], cwd=ROOT, capture_output=True, text=True, check=False
    )

    assert result.returncode == 0, result.stderr


def test_no_test_module_imports_another() -> None:
    # Shared fixtures and helpers live in tests/conftest.py.
    offenders = [
        path.name
        for path in (ROOT / "tests").glob("test_*.py")
        if re.search(r"^\s*(from|import) tests\.test_", path.read_text(encoding="utf-8"), re.M)
    ]

    assert offenders == []


def test_api_process_does_not_load_the_provider_adapter() -> None:
    # Provider calls run only in the worker; the API never needs the adapter, boto3 or its client.
    script = "import sys, app.main; print('app.services.providers.higgsfield' in sys.modules)"
    result = subprocess.run(  # noqa: S603  fixed interpreter and script, no user input
        [sys.executable, "-c", script], cwd=ROOT, capture_output=True, text=True, check=False
    )

    assert (result.returncode, result.stdout.strip()) == (0, "False"), result.stderr
