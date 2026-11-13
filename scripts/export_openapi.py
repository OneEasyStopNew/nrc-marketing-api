"""Write the API contract to contracts/openapi.json for the frontend.

Usage: uv run python -m scripts.export_openapi
"""

import json
import os
import tempfile
from pathlib import Path

from app.main import create_app

CONTRACT_PATH = Path(__file__).resolve().parents[1] / "contracts" / "openapi.json"


def render() -> str:
    return json.dumps(create_app().openapi(), indent=2, ensure_ascii=False) + "\n"


def main() -> None:
    # Atomic replace, so a reader never sees a half-written contract.
    fd, tmp = tempfile.mkstemp(dir=CONTRACT_PATH.parent, suffix=".tmp")
    try:
        # newline="\n": on Windows text mode would write CRLF and the contract test would fail.
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(render())
        os.replace(tmp, CONTRACT_PATH)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise


if __name__ == "__main__":
    main()
