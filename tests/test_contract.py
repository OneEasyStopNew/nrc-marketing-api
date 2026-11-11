# The published contract files must match the code.

import json
from pathlib import Path

import pytest
from pydantic import BaseModel

from app.schemas.common import ErrorBody
from app.schemas.creatives import CreativeRead, CreativeVersionPage
from app.schemas.tenancy import ContextRead, MeRead
from app.schemas.validations import ValidationRunRead
from scripts.export_openapi import CONTRACT_PATH, render

EXAMPLES_DIR = Path(__file__).resolve().parents[1] / "contracts" / "examples"
SCHEMAS: dict[str, type[BaseModel]] = {
    model.__name__: model
    for model in (
        ContextRead,
        CreativeRead,
        CreativeVersionPage,
        ErrorBody,
        MeRead,
        ValidationRunRead,
    )
}


def test_published_openapi_is_current() -> None:
    assert CONTRACT_PATH.read_text(encoding="utf-8") == render(), (
        "contracts/openapi.json is stale: run `uv run python -m scripts.export_openapi`"
    )


CASES = [
    (path.name, name)
    for path in sorted(EXAMPLES_DIR.glob("*.json"))
    for name in json.loads(path.read_text(encoding="utf-8"))
    if not name.startswith("_")
]


@pytest.mark.parametrize(("file", "name"), CASES)
def test_example_matches_schema(file: str, name: str) -> None:
    example = json.loads((EXAMPLES_DIR / file).read_text(encoding="utf-8"))[name]

    model = SCHEMAS[example["schema"]].model_validate(example["body"])

    assert model.model_dump(mode="json") == example["body"]
