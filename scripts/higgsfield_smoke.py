"""Higgsfield connectivity check, run by a person outside the API and the worker.

  uv run python -m scripts.higgsfield_smoke estimate --models a,b,c
  uv run python -m scripts.higgsfield_smoke generate --model X --confirm-spend

`estimate` is free: it calls only /estimate. `generate` spends real money on one image and
stores it in local SeaweedFS only. Neither prints the API key, the Authorization header or any
URL; output hosts are named by host only.
"""

import argparse
import asyncio
import logging
import sys
import time
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass
from typing import Final, Protocol
from urllib.parse import urlsplit
from uuid import uuid4

from app.core.config import Settings, get_settings
from app.models.enums import FORMAT_SPECS, PostFormat
from app.services import media
from app.services.media import MediaRejectedError
from app.services.providers.base import (
    ImageRequest,
    ProviderError,
    ProviderJobStatus,
    StatusResult,
)
from app.services.providers.higgsfield import HiggsfieldProvider, create_provider
from app.services.storage import ObjectStorage, StoredObject


@dataclass(frozen=True, slots=True)
class Candidate:
    # None leaves resolution out of the body, so the model default applies.
    resolution: str | None
    source: str


# From docs.higgsfield.ai, checked 9 Nov 2026. Nothing here is guessed: llms.txt lists no model
# catalog (it lives in the account console), so these are the only text-to-image ids the docs
# name. num_images is left out (documented default 1); aspect_ratio is the job's 1:1.
CANDIDATES: Final = {
    "higgsfield-ai/soul/standard": Candidate(
        "2K",
        "openapi.json path /higgsfield-ai/soul/standard: prompt, num_images 1-4 (default 1), "
        "resolution 2K|4K (default 2K), aspect_ratio enum incl. 1:1",
    ),
    "higgsfield-ai/soul/v2/standard": Candidate(
        None,
        "quickstart.md, how-to/sdk.md and concepts/billing-and-retention.md (estimate example): "
        "prompt only; no other parameters documented",
    ),
}
SMOKE_PROMPT: Final = "A bright, empty clinic waiting room with plants, editorial photography"
# A person watches this one run, so these are fixed here rather than per-environment settings.
POLL_LIMIT_SECONDS: Final = 300.0
POLL_FIRST_WAIT_SECONDS: Final = 2.0
POLL_MAX_WAIT_SECONDS: Final = 15.0
REFUSED: Final = 2
FAILED: Final = 1
# Hosts that can only be this machine (RFC 6761 reserves .localhost and .test).
LOCAL_HOSTS: Final = frozenset({"localhost", "127.0.0.1", "::1"})
LOCAL_SUFFIXES: Final = (".localhost", ".test")


class ImageStore(Protocol):
    async def save_generated_image(
        self, content: bytes, *, ext: str, content_type: str
    ) -> StoredObject: ...


ProviderFactory = Callable[[str], HiggsfieldProvider]
Sleep = Callable[[float], Awaitable[None]]


def _help_epilog() -> str:
    lines = [
        "candidate text-to-image models (smallest documented settings; sources under",
        "https://docs.higgsfield.ai/docs/, checked 9 Nov 2026):",
    ]
    for model, candidate in CANDIDATES.items():
        resolution = candidate.resolution or "model default"
        lines.append(f"  {model}  resolution={resolution}")
        lines.append(f"      {candidate.source}")
    lines.append("other model ids are sent with prompt and aspect_ratio only.")
    return "\n".join(lines)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m scripts.higgsfield_smoke",
        description="Prove Higgsfield connectivity. estimate is free; generate costs money.",
        epilog=_help_epilog(),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    commands = parser.add_subparsers(dest="command", required=True)
    estimate = commands.add_parser(
        "estimate", help="price one image per model via /estimate (free)"
    )
    estimate.add_argument("--models", required=True, help="comma-separated model ids")
    generate = commands.add_parser(
        "generate", help="generate ONE image (real money) into local SeaweedFS"
    )
    generate.add_argument("--model", required=True)
    generate.add_argument(
        "--confirm-spend", action="store_true", help="required: you accept the cost"
    )
    return parser


def _request(idempotency_key: str) -> ImageRequest:
    return ImageRequest(
        prompt=SMOKE_PROMPT,
        width_px=1080,
        height_px=1080,
        aspect_ratio="1:1",
        idempotency_key=idempotency_key,
    )


def _is_local(endpoint: str | None) -> bool:
    # The stored image must stay on this machine: never a real S3 endpoint.
    try:
        host = urlsplit(endpoint or "").hostname
    except ValueError:
        return False
    return bool(host) and (host in LOCAL_HOSTS or str(host).endswith(LOCAL_SUFFIXES))


def provider_factory(settings: Settings) -> ProviderFactory:
    def make(model: str) -> HiggsfieldProvider:
        candidate = CANDIDATES.get(model)
        # A candidate's own resolution wins; otherwise HIGGSFIELD_RESOLUTION applies.
        resolution = candidate.resolution if candidate else None
        return create_provider(
            settings, model=model, resolution=resolution or settings.higgsfield_resolution
        )

    return make


async def estimate(models: Sequence[str], make_provider: ProviderFactory) -> int:
    result = 0
    for model in models:
        provider = make_provider(model)
        try:
            usd = await provider.estimate_cost_usd(_request("nrc-smoke-estimate"))
            print(f"model={model} usd={usd}")
        except ProviderError as error:
            print(f"model={model} error={error.code}: {error.message}")
            result = FAILED
        finally:
            await provider.aclose()
    return result


async def generate(
    settings: Settings,
    model: str,
    *,
    confirm_spend: bool,
    make_provider: ProviderFactory,
    make_storage: Callable[[Settings], ImageStore],
    sleep: Sleep = asyncio.sleep,
) -> int:
    # Every refusal happens before anything billable is sent.
    if not confirm_spend:
        print("Refused: generate costs real money. Re-run with --confirm-spend.")
        return REFUSED
    if not _is_local(settings.aws_endpoint_url_s3) or not settings.aws_s3_bucket:
        print("Refused: AWS_ENDPOINT_URL_S3 and AWS_S3_BUCKET must point at local SeaweedFS.")
        return REFUSED
    if not settings.higgsfield_output_hosts:
        print("Refused: set HIGGSFIELD_OUTPUT_HOSTS, or the image cannot be downloaded.")
        return REFUSED
    provider = make_provider(model)
    try:
        return await _generate(settings, provider, make_storage, sleep)
    except ProviderError as error:
        print(f"error={error.code}: {error.message}")
        return FAILED
    finally:
        await provider.aclose()


async def _generate(
    settings: Settings,
    provider: HiggsfieldProvider,
    make_storage: Callable[[Settings], ImageStore],
    sleep: Sleep,
) -> int:
    request = _request(f"nrc-smoke-{uuid4()}")
    usd = await provider.estimate_cost_usd(request)
    cap = settings.generation_max_cost_usd_per_image
    if usd is None or usd > cap:
        print(f"Refused: estimate usd={usd} is above GENERATION_MAX_COST_USD_PER_IMAGE={cap}.")
        return REFUSED
    print(f"model={provider.model} estimate_usd={usd}")
    submitted = await provider.submit(request)
    print(f"request_id={submitted.provider_request_id}")
    result = await _poll(provider, submitted.provider_request_id, sleep)
    if result is None:
        print(
            f"Stopped polling after {POLL_LIMIT_SECONDS:.0f} seconds; the request may still "
            "finish (and be billed)."
        )
        return FAILED
    print(f"status={result.status.value} actual_cost_usd={result.actual_cost_usd}")
    if result.status is not ProviderJobStatus.COMPLETED:
        return FAILED
    host = urlsplit(result.output_urls[0]).hostname
    if host not in {h.lower() for h in settings.higgsfield_output_hosts}:
        print(f"Output host {host} is not in HIGGSFIELD_OUTPUT_HOSTS; nothing downloaded.")
        return FAILED
    image = await provider.fetch_output(result)
    spec = FORMAT_SPECS[PostFormat.FEED_IMAGE]
    try:
        fitted = await asyncio.to_thread(
            media.fit_to_format,
            image.content,
            width_px=spec.width_px,
            height_px=spec.height_px,
            max_pixels=settings.max_image_pixels,
        )
    except MediaRejectedError as error:
        print(f"error={error.code}: {error}")
        return FAILED
    info = fitted.stored
    stored = await make_storage(settings).save_generated_image(
        fitted.content, ext=info.ext, content_type=info.mime_type
    )
    print(
        f"stored checksum_sha256={stored.checksum_sha256} size_bytes={stored.size_bytes} "
        f"width_px={info.width_px} height_px={info.height_px} mime_type={info.mime_type} "
        f"bucket={stored.bucket} s3_key={stored.key} s3_version_id={stored.version_id}"
    )
    return 0


async def _poll(provider: HiggsfieldProvider, request_id: str, sleep: Sleep) -> StatusResult | None:
    # Polls with backoff until a terminal status or the 5-minute limit; retryable errors wait.
    deadline = time.monotonic() + POLL_LIMIT_SECONDS
    wait = POLL_FIRST_WAIT_SECONDS
    while True:
        try:
            result = await provider.get_status(request_id)
        except ProviderError as error:
            if not error.retryable:
                raise
            print(f"status error={error.code}, retrying")
        else:
            if result.status in {ProviderJobStatus.QUEUED, ProviderJobStatus.IN_PROGRESS}:
                print(f"status={result.status.value}")
            else:
                return result
        if time.monotonic() + wait > deadline:
            return None
        await sleep(wait)
        wait = min(wait * 2, POLL_MAX_WAIT_SECONDS)


def parse_args(argv: Sequence[str]) -> argparse.Namespace:
    return _parser().parse_args(argv)


async def run(
    args: argparse.Namespace,
    settings: Settings,
    make_provider: ProviderFactory,
    make_storage: Callable[[Settings], ImageStore] = ObjectStorage,
    sleep: Sleep = asyncio.sleep,
) -> int:
    if args.command == "estimate":
        models = [model.strip() for model in args.models.split(",") if model.strip()]
        return await estimate(models, make_provider)
    return await generate(
        settings,
        args.model,
        confirm_spend=args.confirm_spend,
        make_provider=make_provider,
        make_storage=make_storage,
        sleep=sleep,
    )


def main() -> int:
    # Parsed first, so --help works without any settings.
    args = parse_args(sys.argv[1:])
    # Warnings only: the results are printed, and httpx logs request URLs at INFO.
    logging.basicConfig(level=logging.WARNING)
    settings = get_settings()
    return asyncio.run(run(args, settings, provider_factory(settings)))


if __name__ == "__main__":
    sys.exit(main())
