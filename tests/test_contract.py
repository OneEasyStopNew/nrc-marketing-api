# The published contract files must match the code.

import json
from pathlib import Path

import pytest
from pydantic import BaseModel

from app.models.enums import JobStage
from app.schemas.common import ErrorBody
from app.schemas.creatives import CreativeRead, CreativeVersionPage
from app.schemas.tenancy import ContextRead, MeRead
from scripts.export_openapi import CONTRACT_PATH, render

EXAMPLES = Path(__file__).resolve().parents[1] / "contracts" / "examples" / "d1.json"
SCHEMAS: dict[str, type[BaseModel]] = {
    model.__name__: model
    for model in (ContextRead, CreativeRead, CreativeVersionPage, ErrorBody, MeRead)
}


def test_published_openapi_is_current() -> None:
    assert CONTRACT_PATH.read_text(encoding="utf-8") == render(), (
        "contracts/openapi.json is stale: run `uv run python -m scripts.export_openapi`"
    )


def test_job_stage_is_a_published_enum() -> None:
    # The UI switches on stage, so the contract lists every value instead of any string.
    schemas = json.loads(CONTRACT_PATH.read_text(encoding="utf-8"))["components"]["schemas"]

    assert schemas["JobRead"]["properties"]["stage"]["anyOf"] == [
        {"$ref": "#/components/schemas/JobStage"},
        {"type": "null"},
    ]
    assert schemas["JobStage"]["enum"] == [stage.value for stage in JobStage]


@pytest.mark.parametrize(
    "name",
    [name for name in json.loads(EXAMPLES.read_text(encoding="utf-8")) if not name.startswith("_")],
)
def test_example_matches_schema(name: str) -> None:
    example = json.loads(EXAMPLES.read_text(encoding="utf-8"))[name]

    model = SCHEMAS[example["schema"]].model_validate(example["body"])

    assert model.model_dump(mode="json") == example["body"]
