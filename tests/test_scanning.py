from pathlib import Path
from typing import Any

import pytest

from app.core.config import Settings
from app.core.errors import AppError
from app.services.scanning import (
    MalwareScanner,
    ScannerNotConfiguredError,
    ScanOutcome,
    ScanUnavailableError,
    ScanVerdict,
    SimulatedScanner,
    build_scanner,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
REQUIRED: dict[str, Any] = {
    "database_url": "postgresql+asyncpg://u:p@localhost/db",
    "cognito_region": "ap-south-1",
    "cognito_user_pool_id": "ap-south-1_TESTPOOL",
    "cognito_client_ids": ("c",),
    "required_scope": "marketing/api",
}
# The standard EICAR test signature, assembled from pieces on purpose: Windows Defender
# quarantines any file that contains it whole. Kept independent of app.services.scanning so a
# wrong signature there fails these tests.
EICAR_PIECES = ("X5O!P%@AP[4\\PZX54(P^)", "7CC)7}$EICAR-STANDARD", "-ANTIVIRUS-TEST-FILE!$H+H*")
EICAR = "".join(EICAR_PIECES).encode("ascii")
PNG_ISH = b"\x89PNG\r\n\x1a\n" + bytes(range(256)) * 4
LOCATION: dict[str, Any] = {
    "bucket": "test-bucket",
    "key": "uploads/2026/11/x.png",
    "version_id": "v1",
}


def _settings(**overrides: Any) -> Settings:
    return Settings(_env_file=None, **{**REQUIRED, **overrides})


def _unvalidated(**overrides: Any) -> Settings:
    # Skips the settings validator, to prove the scanner refuses on its own.
    base = _settings(app_env="local", malware_scan_mode="simulator")
    return Settings.model_construct(**{**dict(base), **overrides})


async def test_simulator_passes_clean_content_and_labels_the_verdict() -> None:
    scanner = SimulatedScanner(_settings(app_env="local", malware_scan_mode="simulator"))

    verdict = await scanner.scan(PNG_ISH, **LOCATION)

    assert verdict == ScanVerdict(ScanOutcome.CLEAN, engine="SIMULATED", simulated=True)
    assert scanner.engine == "SIMULATED"
    assert scanner.simulated is True


async def test_simulator_passes_empty_content() -> None:
    scanner = SimulatedScanner(_settings(app_env="demo", malware_scan_mode="simulator"))

    verdict = await scanner.scan(b"", **LOCATION)

    assert verdict.outcome is ScanOutcome.CLEAN


@pytest.mark.parametrize(
    "content",
    [EICAR, PNG_ISH + EICAR, PNG_ISH + EICAR + PNG_ISH, EICAR + PNG_ISH],
    ids=["only", "at-end", "in-middle", "at-start"],
)
async def test_simulator_flags_the_eicar_signature(content: bytes) -> None:
    scanner = SimulatedScanner(_settings(app_env="local", malware_scan_mode="simulator"))

    verdict = await scanner.scan(content, **LOCATION)

    assert verdict == ScanVerdict(ScanOutcome.INFECTED, engine="SIMULATED", simulated=True)


async def test_simulator_does_not_flag_a_partial_signature() -> None:
    scanner = SimulatedScanner(_settings(app_env="local", malware_scan_mode="simulator"))

    verdict = await scanner.scan(EICAR[:-1], **LOCATION)

    assert verdict.outcome is ScanOutcome.CLEAN


async def test_build_scanner_returns_the_labelled_simulator_in_development() -> None:
    scanner: MalwareScanner = build_scanner(
        _settings(app_env="local", malware_scan_mode="simulator")
    )

    verdict = await scanner.scan(PNG_ISH, **LOCATION)

    assert isinstance(scanner, SimulatedScanner)
    assert verdict.simulated is True


@pytest.mark.parametrize("app_env", ["local", "demo", "staging", "production"])
def test_build_scanner_refuses_live_mode_until_a_scanner_exists(app_env: str) -> None:
    with pytest.raises(ScannerNotConfiguredError, match="MALWARE_SCAN_MODE=live needs a real"):
        build_scanner(_settings(app_env=app_env, malware_scan_mode="live"))


def test_simulator_is_refused_when_the_settings_say_live() -> None:
    with pytest.raises(ScannerNotConfiguredError, match="MALWARE_SCAN_MODE=simulator"):
        SimulatedScanner(_unvalidated(malware_scan_mode="live"))


@pytest.mark.parametrize("app_env", ["staging", "production"])
def test_simulator_is_refused_outside_development_even_without_the_validator(
    app_env: str,
) -> None:
    with pytest.raises(ScannerNotConfiguredError, match="APP_ENV is local or demo"):
        SimulatedScanner(_unvalidated(app_env=app_env, malware_scan_mode="simulator"))


@pytest.mark.parametrize("app_env", ["staging", "production"])
def test_build_scanner_refuses_the_simulator_outside_development(app_env: str) -> None:
    with pytest.raises(ScannerNotConfiguredError, match="APP_ENV is local or demo"):
        build_scanner(_unvalidated(app_env=app_env, malware_scan_mode="simulator"))


def test_scan_unavailable_is_retryable_with_a_stable_code() -> None:
    error = ScanUnavailableError()

    assert isinstance(error, AppError)
    assert error.code == "SCAN_UNAVAILABLE"
    assert error.retryable is True
    assert error.message == "The malware scan is temporarily unavailable."

    with pytest.raises(ScanUnavailableError, match="temporarily unavailable"):
        raise error


def test_no_source_file_contains_the_eicar_signature_whole() -> None:
    # Guards the developer machines: the antivirus quarantines any file holding the signature.
    needle = EICAR.decode("ascii")
    offenders = [
        str(path.relative_to(REPO_ROOT))
        for folder in ("app", "tests")
        for path in (REPO_ROOT / folder).rglob("*.py")
        if needle in path.read_text(encoding="utf-8")
    ]

    assert offenders == []
