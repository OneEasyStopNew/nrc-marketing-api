# The published contract files must match the code.

import json
from pathlib import Path

import pytest
from pydantic import BaseModel

from app.models.enums import JobStage
from app.schemas.assets import AssetRead
from app.schemas.common import ErrorBody
from app.schemas.creatives import CreativeRead, CreativeVersionPage
from app.schemas.jobs import JobAccepted, JobRead
from app.schemas.policies import GenerationEligibility, PolicyBundleRead
from app.schemas.tenancy import ContextRead, MeRead
from scripts.export_openapi import CONTRACT_PATH, render

EXAMPLE_FILES = sorted(
    (Path(__file__).resolve().parents[1] / "contracts" / "examples").glob("*.json")
)
SCHEMAS: dict[str, type[BaseModel]] = {
    model.__name__: model
    for model in (
        AssetRead,
        ContextRead,
        CreativeRead,
        CreativeVersionPage,
        ErrorBody,
        GenerationEligibility,
        JobAccepted,
        JobRead,
        MeRead,
        PolicyBundleRead,
    )
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
    assert [stage.value for stage in JobStage] == [
        "submitting",
        "retry_wait",
        "reconcile",
        "waiting_provider",
        "output_ready",
        "fetching_output",
        "stored",
    ]


def _examples() -> list[tuple[str, str]]:
    return [
        (path.name, name)
        for path in EXAMPLE_FILES
        for name in json.loads(path.read_text(encoding="utf-8"))
        if not name.startswith("_")
    ]


@pytest.mark.parametrize(("file", "name"), _examples())
def test_example_matches_schema(file: str, name: str) -> None:
    path = next(path for path in EXAMPLE_FILES if path.name == file)
    example = json.loads(path.read_text(encoding="utf-8"))[name]

    model = SCHEMAS[example["schema"]].model_validate(example["body"])

    assert model.model_dump(mode="json") == example["body"]
