# Proposed (NRC-30): download a version-bound validation evidence report.

from datetime import UTC, datetime
from typing import Annotated, Any
from uuid import UUID

from fastapi import APIRouter, Query
from fastapi.responses import HTMLResponse

from app.api.deps import CreativeReader, DbSession
from app.core.errors import VersionConflictError
from app.schemas.common import ErrorBody
from app.services import reports

router = APIRouter()

_RESPONSES: dict[int | str, dict[str, Any]] = {
    200: {
        "content": {"text/html": {"schema": {"type": "string"}}},
        "description": "Self-contained HTML report sent as an attachment.",
    },
    404: {"model": ErrorBody, "description": "No such validation run in this tenant."},
    412: {
        "model": ErrorBody,
        "description": "The run belongs to another creative version; see details.current_version.",
    },
    501: {"model": ErrorBody, "description": "Persisted validation runs are not available yet."},
}


@router.get(
    "/validations/{validation_run_id}/report",
    tags=["validations"],
    summary="Download the validation evidence report as HTML (proposed, NRC-30)",
    response_class=HTMLResponse,
    responses=_RESPONSES,
)
async def download_validation_report(
    validation_run_id: UUID,
    ctx: CreativeReader,
    db: DbSession,
    creative_version: Annotated[
        int, Query(ge=1, description="Creative version the preview currently shows.")
    ],
) -> HTMLResponse:
    run = await reports.load_validation_run(db, ctx, validation_run_id)
    if run.creative_version != creative_version:
        raise VersionConflictError(
            "The validation run belongs to a different creative version. Refresh and retry.",
            details={"current_version": run.creative_version},
        )
    html = reports.render_validation_report(run, generated_at=datetime.now(UTC))
    filename = f"validation-report-{run.validation_run_id}-v{run.creative_version}.html"
    return HTMLResponse(
        html,
        headers={
            "Content-Disposition": f'attachment; filename="{filename}"',
            "Content-Security-Policy": reports.REPORT_CSP,
            "Cache-Control": "no-store",
        },
    )
