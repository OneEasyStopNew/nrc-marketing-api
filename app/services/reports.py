# Version-bound validation evidence report (NRC-30, proposed). Renders a persisted run only.

import re
from datetime import UTC, datetime
from pathlib import Path
from uuid import UUID

from jinja2 import Environment, FileSystemLoader, StrictUndefined
from sqlalchemy.ext.asyncio import AsyncSession

from app.auth.access import TenantContext
from app.core.errors import FeatureNotAvailableError
from app.schemas.validations import ValidationRunRead

TEMPLATE_DIR = Path(__file__).resolve().parents[1] / "templates"
REPORT_CSP = "default-src 'none'; style-src 'unsafe-inline'"
_LINK_REMOVED = "[link removed]"
# Scheme-prefixed links, bare www hosts and scheme-relative links in free text.
_URL = re.compile(r"(?i)(?:\b[a-z][a-z0-9+.-]{1,15}://|\bwww\.|(?<![\w:])//)\S*")

# autoescape covers every {{ }}; StrictUndefined fails loudly when a field goes missing.
_env = Environment(
    loader=FileSystemLoader(TEMPLATE_DIR),
    autoescape=True,
    undefined=StrictUndefined,
)


def _plain(value: str | None) -> str:
    # Evidence text is shown as text only: links found in it are removed, never rendered.
    return _URL.sub(_LINK_REMOVED, value or "")


_env.filters["plain"] = _plain


def render_validation_report(run: ValidationRunRead, *, generated_at: datetime) -> str:
    template = _env.get_template("validation_report.html")
    return template.render(
        run=run, generated_at=generated_at.astimezone(UTC).strftime("%Y-%m-%d %H:%M UTC")
    )


async def load_validation_run(
    db: AsyncSession, ctx: TenantContext, validation_run_id: UUID
) -> ValidationRunRead:
    # TODO(NRC-7): call the persisted-read service for a tenant-scoped run (404 when not found).
    raise FeatureNotAvailableError(
        "The validation report needs persisted validation runs, which arrive with NRC-7."
    )
