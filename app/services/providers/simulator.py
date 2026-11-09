"""Simulated image provider for local runs and tests. Makes no network calls.

Every image it returns is stamped SIMULATED and never counts as live provider evidence. It keeps
the provider-side memory a real provider has (request ids per idempotency key), so retries and
reconciliation behave like Higgsfield: resubmitting with the same key returns the same request.
"""

import asyncio
import io
from collections import Counter
from decimal import Decimal
from enum import StrEnum
from typing import Final
from uuid import uuid4

from PIL import Image, ImageDraw, ImageFont

from app.services.providers.base import (
    FetchedImage,
    ImageRequest,
    ProviderJobStatus,
    ProviderOutcomeUnknownError,
    ProviderRateLimitedError,
    ProviderRejectedError,
    StatusResult,
    SubmitResult,
)

SIMULATOR_NAME: Final = "simulator"
SIMULATOR_MODEL: Final = "simulated-image-v1"
IMAGE_SIZE_PX: Final = 1080


class SimulatorScenario(StrEnum):
    SUCCESS = "success"
    # Accepted, then the provider reports a failed generation.
    PROVIDER_FAILED = "provider_failed"
    # The first submit is accepted but its response is lost; a resubmit with the same key
    # returns the original request id.
    TIMEOUT_AFTER_ACCEPT = "timeout_after_accept"
    RATE_LIMITED = "rate_limited"


class SimulatedImageProvider:
    name = SIMULATOR_NAME
    is_simulated = True

    def __init__(
        self,
        scenario: SimulatorScenario = SimulatorScenario.SUCCESS,
        *,
        cost_usd: Decimal | None = Decimal(0),
    ) -> None:
        self.model = SIMULATOR_MODEL
        self.scenario = scenario
        # Simulated images cost nothing; tests set a price to exercise the cost cap.
        self.cost_usd = cost_usd
        # Submits per idempotency key, so tests can assert the provider was not called twice.
        self.submits: Counter[str] = Counter()
        self._request_ids: dict[str, str] = {}

    async def estimate_cost_usd(self, request: ImageRequest) -> Decimal | None:
        return self.cost_usd

    async def submit(self, request: ImageRequest) -> SubmitResult:
        self.submits[request.idempotency_key] += 1
        if self.scenario is SimulatorScenario.RATE_LIMITED:
            raise ProviderRateLimitedError("Simulated rate limit.", retry_after_seconds=1)
        request_id = self._request_ids.setdefault(request.idempotency_key, f"sim-{uuid4()}")
        if (
            self.scenario is SimulatorScenario.TIMEOUT_AFTER_ACCEPT
            and self.submits[request.idempotency_key] == 1
        ):
            raise ProviderOutcomeUnknownError("Simulated timeout after the request was accepted.")
        return SubmitResult(request_id, ProviderJobStatus.QUEUED)

    async def get_status(self, provider_request_id: str) -> StatusResult:
        if provider_request_id not in self._request_ids.values():
            raise ProviderRejectedError("Unknown simulated request.")
        if self.scenario is SimulatorScenario.PROVIDER_FAILED:
            return StatusResult(
                provider_request_id, ProviderJobStatus.FAILED, failure_code="SIMULATED_FAILURE"
            )
        return StatusResult(
            provider_request_id,
            ProviderJobStatus.COMPLETED,
            output_urls=(f"simulator://{provider_request_id}.png",),
        )

    async def fetch_output(self, result: StatusResult) -> FetchedImage:
        content = await asyncio.to_thread(render_simulated_png)
        return FetchedImage(content=content, content_type="image/png", source_host=None)


def render_simulated_png() -> bytes:
    # A 1080x1080 PNG with SIMULATED across the middle.
    image = Image.new("RGB", (IMAGE_SIZE_PX, IMAGE_SIZE_PX), (226, 232, 240))
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default(size=120)
    center = IMAGE_SIZE_PX / 2
    draw.text((center, center), "SIMULATED", fill=(30, 41, 59), font=font, anchor="mm")
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()
