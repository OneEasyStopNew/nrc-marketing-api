# NRC-30 (proposed): validation evidence report renderer and download route.

import json
from collections.abc import Awaitable, Callable
from datetime import UTC, datetime
from pathlib import Path
from uuid import UUID

import httpx
import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from app.auth.access import TenantContext
from app.schemas.validations import CoverageGap, Finding, ValidationRunRead
from app.services import reports
from tests.conftest import AuthHeaders, World

ROOT = Path(__file__).resolve().parents[1]
EXAMPLES = json.loads((ROOT / "contracts/examples/validation_runs.json").read_text("utf-8"))
NOW = datetime(2026, 11, 10, 9, 0, tzinfo=UTC)
SAMPLE = ROOT / "docs" / "samples" / "validation-report-sample.html"

Loader = Callable[[AsyncSession, TenantContext, UUID], Awaitable[ValidationRunRead]]


def _run(name: str) -> ValidationRunRead:
    return ValidationRunRead.model_validate(EXAMPLES[name]["body"])


def _hostile(run: ValidationRunRead) -> ValidationRunRead:
    finding = run.findings[0].model_copy(
        update={
            "explanation": "<script>alert(1)</script> see https://evil.example/x?token=abc",
            "offending_excerpt": '<img src=x onerror="steal()"> www.evil.example/a',
            "suggested_correction": "ftp://evil.example/file //evil.example/p",
        }
    )
    return run.model_copy(update={"findings": [finding]})


def _fake_loader(run: ValidationRunRead) -> Loader:
    async def load(
        db: AsyncSession, ctx: TenantContext, validation_run_id: UUID
    ) -> ValidationRunRead:
        return run

    return load


def _url(run: ValidationRunRead, version: int | None = None) -> str:
    v = run.creative_version if version is None else version
    return f"/v1/marketing/validations/{run.validation_run_id}/report?creative_version={v}"


def test_sample_report_is_written_from_the_synthetic_fixture() -> None:
    SAMPLE.parent.mkdir(parents=True, exist_ok=True)
    SAMPLE.write_text(
        reports.render_validation_report(_run("validation_run_needs_review"), generated_at=NOW),
        encoding="utf-8",
    )
    assert "Draft for review" in SAMPLE.read_text(encoding="utf-8")


def test_render_is_deterministic() -> None:
    run = _run("validation_run_fail")
    assert reports.render_validation_report(run, generated_at=NOW) == (
        reports.render_validation_report(run, generated_at=NOW)
    )


def test_render_has_label_versions_and_safe_head() -> None:
    run = _run("validation_run_pass")
    html = reports.render_validation_report(run, generated_at=NOW)

    assert "Draft for review" in html
    assert str(run.validation_run_id) in html
    assert "synthetic-2026.11" in html
    assert "2026-11-09 10:30 UTC" in html
    assert "default-src 'none'; style-src 'unsafe-inline'" in html
    assert "<script" not in html
    assert "overflow-wrap: anywhere" in html
    assert "none recorded" in html  # coverage gaps really empty
    for word in ("Approved", "Compliant"):
        assert word not in html


@pytest.mark.parametrize("name", ["validation_run_fail", "validation_run_needs_review"])
def test_fail_and_needs_review_stay_visible(name: str) -> None:
    run = _run(name)
    html = reports.render_validation_report(run, generated_at=NOW)

    assert run.outcome is not None
    assert f">{run.outcome.value}<" in html
    assert run.findings[0].rule_id in html
    assert run.findings[0].source.source_id in html


def test_coverage_gaps_are_listed_not_hidden() -> None:
    html = reports.render_validation_report(_run("validation_run_needs_review"), generated_at=NOW)

    assert "SYN-ELIG-003" in html
    assert "could not be assessed" in html
    assert "UNABLE_TO_ASSESS" in html
    assert "none recorded" not in html


def test_unfinished_run_has_no_fake_outcome() -> None:
    run = _run("validation_run_pass").model_copy(update={"outcome": None, "evaluated_at": None})
    html = reports.render_validation_report(run, generated_at=NOW)

    assert "NO OUTCOME YET" in html
    assert "not evaluated yet" in html


def test_dynamic_text_is_escaped_and_urls_removed() -> None:
    run = _hostile(_run("validation_run_fail"))
    run = run.model_copy(update={"coverage_gaps": [CoverageGap(reason="<b>gap</b> http://x.test")]})
    html = reports.render_validation_report(run, generated_at=NOW)

    assert "<script>" not in html
    assert "&lt;script&gt;" in html
    assert "<img" not in html
    assert "<b>gap" not in html
    assert "://" not in html
    assert "http:" not in html.lower()
    assert "https" not in html.lower()
    assert "evil.example" not in html
    assert "[link removed]" in html


def test_image_region_and_span_render_as_text() -> None:
    run = _run("validation_run_fail")
    finding = Finding.model_validate(
        {
            **run.findings[0].model_dump(mode="json"),
            "image_region": {"x": 0.1, "y": 0.2, "width": 0.5, "height": 0.25},
        }
    )
    html = reports.render_validation_report(
        run.model_copy(update={"findings": [finding]}), generated_at=NOW
    )

    assert "characters 10 to 24" in html
    assert "left 10.0%, top 20.0%, width 50.0%, height 25.0%" in html


async def test_route_is_501_until_persisted_runs_exist(
    client: httpx.AsyncClient, auth: AuthHeaders, world: World
) -> None:
    run = _run("validation_run_pass")

    response = await client.get(_url(run), headers=auth("creator-a", world.a.tenant_id))

    assert response.status_code == 501
    body = response.json()
    assert body["code"] == "NOT_IMPLEMENTED"
    assert set(body) == {"code", "message", "request_id", "details", "retryable"}


async def test_route_downloads_an_attachment(
    client: httpx.AsyncClient,
    auth: AuthHeaders,
    world: World,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run = _run("validation_run_fail")
    monkeypatch.setattr(reports, "load_validation_run", _fake_loader(run))

    response = await client.get(_url(run), headers=auth("reviewer-a", world.a.tenant_id))

    assert response.status_code == 200
    assert response.headers["content-type"] == "text/html; charset=utf-8"
    assert response.headers["content-disposition"] == (
        f'attachment; filename="validation-report-{run.validation_run_id}-v3.html"'
    )
    assert response.headers["content-security-policy"] == reports.REPORT_CSP
    assert "Draft for review" in response.text
    assert "SYN-CLAIM-001" in response.text


async def test_stale_creative_version_is_412_with_current_version(
    client: httpx.AsyncClient,
    auth: AuthHeaders,
    world: World,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run = _run("validation_run_fail")
    monkeypatch.setattr(reports, "load_validation_run", _fake_loader(run))

    response = await client.get(_url(run, version=2), headers=auth("creator-a", world.a.tenant_id))

    assert response.status_code == 412
    assert response.json()["code"] == "VERSION_CONFLICT"
    assert response.json()["details"] == {"current_version": 3}


async def test_creative_version_is_required_and_positive(
    client: httpx.AsyncClient, auth: AuthHeaders, world: World
) -> None:
    base = f"/v1/marketing/validations/{_run('validation_run_pass').validation_run_id}/report"
    headers = auth("creator-a", world.a.tenant_id)

    assert (await client.get(base, headers=headers)).status_code == 422
    assert (await client.get(f"{base}?creative_version=0", headers=headers)).status_code == 422


async def test_non_member_and_anonymous_are_refused(
    client: httpx.AsyncClient,
    auth: AuthHeaders,
    world: World,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run = _run("validation_run_pass")
    monkeypatch.setattr(reports, "load_validation_run", _fake_loader(run))

    other_tenant = await client.get(_url(run), headers=auth("creator-b", world.a.tenant_id))
    anonymous = await client.get(_url(run), headers={"X-Tenant-Id": str(world.a.tenant_id)})

    assert other_tenant.status_code == 403
    assert anonymous.status_code == 401
