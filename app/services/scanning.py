"""Malware scan before model access (NRC-3), behind an interface.

The real scanner (likely a managed S3 scanner) is not chosen yet; the infra owner decides. Until
then only a clearly labelled simulator exists, and only for local and demo. Nothing passes as
clean silently: live mode refuses to start without a real scanner, and an implementation that
cannot decide raises ScanUnavailableError instead of returning CLEAN.
"""

from dataclasses import dataclass
from enum import StrEnum
from typing import Final, Protocol

from app.core.config import NON_PRODUCTION_ENVS, Settings
from app.core.errors import AppError

SIMULATED_ENGINE: Final = "SIMULATED"

# The standard EICAR anti-malware test signature, assembled from pieces so that no source file
# holds it whole: endpoint protection on developer machines quarantines files that contain it.
_EICAR_PIECES: Final = (
    "X5O!P%@AP[4\\PZX54(P^)7CC)7}$",
    "EICAR-STANDARD-ANTIVIRUS-TEST-FILE!",
    "$H+H*",
)
EICAR_TEST_SIGNATURE: Final = "".join(_EICAR_PIECES).encode("ascii")


class ScanOutcome(StrEnum):
    CLEAN = "CLEAN"
    INFECTED = "INFECTED"


@dataclass(frozen=True, slots=True)
class ScanVerdict:
    outcome: ScanOutcome
    engine: str
    # True when the verdict comes from the simulator; it never counts as real scan evidence.
    simulated: bool


class ScanUnavailableError(AppError):
    # The media worker keeps the upload quarantined and retries; never treat this as clean.
    code = "SCAN_UNAVAILABLE"
    default_message = "The malware scan is temporarily unavailable."
    retryable = True


class ScannerNotConfiguredError(Exception):
    # The service cannot start with this configuration. Messages name settings, never values.
    pass


class MalwareScanner(Protocol):
    """Decides whether an uploaded object is clean.

    Both the bytes and the exact object version are passed because a managed S3 scanner reports
    per object version, while a stream scanner reads the bytes. Implementations must raise
    ScanUnavailableError when they cannot decide, and never return CLEAN by default.
    """

    @property
    def engine(self) -> str: ...

    @property
    def simulated(self) -> bool: ...

    async def scan(
        self, content: bytes, *, bucket: str, key: str, version_id: str | None
    ) -> ScanVerdict: ...


class SimulatedScanner:
    # Local and demo only. Flags the EICAR test signature, passes everything else, labelled
    # SIMULATED.
    engine = SIMULATED_ENGINE
    simulated = True

    def __init__(self, settings: Settings) -> None:
        # Defence in depth: the settings validator already refuses this in staging and production.
        if settings.malware_scan_mode != "simulator":
            raise ScannerNotConfiguredError(
                "The simulated malware scanner needs MALWARE_SCAN_MODE=simulator."
            )
        if settings.app_env not in NON_PRODUCTION_ENVS:
            raise ScannerNotConfiguredError(
                "The simulated malware scanner is only allowed when APP_ENV is local or demo."
            )

    async def scan(
        self, content: bytes, *, bucket: str, key: str, version_id: str | None
    ) -> ScanVerdict:
        outcome = ScanOutcome.INFECTED if EICAR_TEST_SIGNATURE in content else ScanOutcome.CLEAN
        return ScanVerdict(outcome=outcome, engine=self.engine, simulated=self.simulated)


def build_scanner(settings: Settings) -> MalwareScanner:
    if settings.malware_scan_mode == "simulator":
        return SimulatedScanner(settings)
    # TODO(NRC-3, owner Firoz): add the live scanner here once Prabhu picks it (a managed S3
    # scanner or a stream scanner). Until then live mode must keep refusing to start; remove this
    # raise and its test (test_build_scanner_refuses_live_mode_until_a_scanner_exists) together.
    raise ScannerNotConfiguredError(
        "MALWARE_SCAN_MODE=live needs a real malware scanner, and none is configured yet "
        "(scanner choice pending). Refusing to start."
    )
