# The published contract files must match the code.

import json
from collections.abc import Iterator, Mapping
from pathlib import Path
from types import MappingProxyType
from typing import Any, Final

import pytest
from pydantic import BaseModel

from app.api.errors import status_for
from app.core.errors import AppError
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
# NRC-25: the fixtures must show the frontend each of these responses.
REQUIRED_STATUSES: Final = frozenset({401, 403, 409, 412, 422, 429, 503})
# Codes built by the request-validation handler rather than raised as an AppError.
HANDLER_CODES: Final[Mapping[str, int]] = MappingProxyType({"VALIDATION_FAILED": 422})


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


def _example(file: str, name: str) -> dict[str, Any]:
    path = next(path for path in EXAMPLE_FILES if path.name == file)
    example: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))[name]
    return example


def _error_examples() -> list[tuple[str, str]]:
    return [(f, n) for f, n in _examples() if _example(f, n)["schema"] == "ErrorBody"]


def _subclasses(cls: type[AppError]) -> Iterator[type[AppError]]:
    for sub in cls.__subclasses__():
        yield sub
        yield from _subclasses(sub)


def _error_types_by_code() -> dict[str, type[AppError]]:
    by_code: dict[str, type[AppError]] = {}
    for error_type in _subclasses(AppError):
        # Subclasses that only add a default message share their parent's code and status.
        known = by_code.setdefault(error_type.code, error_type)
        assert status_for(known) == status_for(error_type), error_type.code
    return by_code


@pytest.mark.parametrize(("file", "name"), _examples())
def test_example_matches_schema(file: str, name: str) -> None:
    example = _example(file, name)

    model = SCHEMAS[example["schema"]].model_validate(example["body"])

    assert model.model_dump(mode="json") == example["body"]


def test_examples_cover_every_required_status() -> None:
    statuses = {_example(file, name)["status"] for file, name in _examples()}

    assert REQUIRED_STATUSES - statuses == set()
    assert any(200 <= status < 300 for status in statuses)


@pytest.mark.parametrize(("file", "name"), _error_examples())
def test_error_example_uses_the_status_the_api_sends(file: str, name: str) -> None:
    example = _example(file, name)
    code = example["body"]["code"]
    error_type = _error_types_by_code().get(code)

    if error_type is None:
        assert example["status"] == HANDLER_CODES[code]
        assert example["body"]["retryable"] is False
    else:
        assert example["status"] == status_for(error_type)
        assert example["body"]["retryable"] is error_type.retryable
