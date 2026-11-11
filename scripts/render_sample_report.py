"""Write the synthetic sample validation report to docs/samples/.

Usage: uv run python -m scripts.render_sample_report
"""

import json
from datetime import UTC, datetime
from pathlib import Path

from app.schemas.validations import ValidationRunRead
from app.services.reports import render_validation_report

ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / "contracts" / "examples" / "validation_runs.json"
SAMPLE_PATH = ROOT / "docs" / "samples" / "validation-report-sample.html"
# Fixed so the committed sample is reproducible.
SAMPLE_GENERATED_AT = datetime(2026, 11, 10, 9, 0, tzinfo=UTC)


def render() -> str:
    fixtures = json.loads(FIXTURES.read_text(encoding="utf-8"))
    run = ValidationRunRead.model_validate(fixtures["validation_run_needs_review"]["body"])
    return render_validation_report(run, generated_at=SAMPLE_GENERATED_AT)


def main() -> None:
    SAMPLE_PATH.parent.mkdir(parents=True, exist_ok=True)
    SAMPLE_PATH.write_text(render(), encoding="utf-8", newline="\n")


if __name__ == "__main__":
    main()
