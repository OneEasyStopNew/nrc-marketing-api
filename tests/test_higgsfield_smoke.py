# The Higgsfield smoke script: refusal paths and output, against httpx.MockTransport only.

import io
from decimal import Decimal
from typing import Any

import httpx
import pytest
from PIL import Image
from pydantic import SecretStr

from app.core.config import Settings, get_settings
from app.services.providers.base import FetchedImage
from app.services.providers.higgsfield import HiggsfieldProvider
from app.services.storage import StoredObject
from scripts.higgsfield_smoke import CANDIDATES, parse_args, provider_factory, run

MODEL = "higgsfield-ai/soul/v2/standard"
KEY_ID, KEY_SECRET = "kid-SMOKE", "ksecret-SMOKE-VALUE"
OUTPUT_HOST = "images.higgsfield.example"
OUTPUT_URL = f"https://{OUTPUT_HOST}/out.png?signature=SIGNED"


class FakeApi:
    def __init__(self, usd: str = "0.05") -> None:
        self.usd = usd
        self.paths: list[str] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        path = request.url.path
        self.paths.append(f"{request.method} {path}")
        if path.startswith("/estimate/"):
            return httpx.Response(200, json={"usd": self.usd})
        if request.method == "POST":
            return httpx.Response(200, json={"status": "queued", "request_id": "req-1"})
        return httpx.Response(
            200,
            json={"status": "completed", "request_id": "req-1", "images": [{"url": OUTPUT_URL}]},
        )


class RecordingStore:
    def __init__(self) -> None:
        self.saved: list[bytes] = []

    async def save_generated_image(
        self, content: bytes, *, ext: str, content_type: str
    ) -> StoredObject:
        self.saved.append(content)
        return StoredObject(
            "nrc-test-media", "generated-images/x.png", "v1", len(content), "ab", ""
        )


def _png() -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", (1536, 1536), (10, 20, 30)).save(buffer, format="PNG")
    return buffer.getvalue()


async def _fetch(url: str, *, allowed_hosts: frozenset[str], max_bytes: int) -> FetchedImage:
    assert url == OUTPUT_URL
    return FetchedImage(_png(), "image/png", OUTPUT_HOST)


def _settings(**overrides: Any) -> Settings:
    values: dict[str, Any] = {
        "higgsfield_api_key": SecretStr(f"{KEY_ID}:{KEY_SECRET}"),
        "higgsfield_output_hosts": (OUTPUT_HOST,),
        "generation_max_cost_usd_per_image": Decimal("0.10"),
    }
    values.update(overrides)
    return get_settings().model_copy(update=values)


def _factory(api: FakeApi, made: list[str]) -> Any:
    def make(model: str) -> HiggsfieldProvider:
        made.append(model)
        return HiggsfieldProvider(
            client=httpx.AsyncClient(transport=httpx.MockTransport(api)),
            base_url="https://api.higgsfield.ai",
            model=model,
            api_key_id=SecretStr(KEY_ID),
            api_key_secret=SecretStr(KEY_SECRET),
            fetcher=_fetch,
            allowed_output_hosts=frozenset({OUTPUT_HOST}),
            max_output_bytes=10_000_000,
        )

    return make


async def _no_sleep(_: float) -> None:
    return None


async def _run(argv: list[str], settings: Settings, api: FakeApi, store: RecordingStore) -> int:
    made: list[str] = []
    return await run(
        parse_args(argv), settings, _factory(api, made), lambda _: store, sleep=_no_sleep
    )


def _assert_nothing_secret(text: str) -> None:
    for item in (KEY_ID, KEY_SECRET, "Authorization", "Key ", "https://", "SIGNED"):
        assert item not in text


async def test_generate_refuses_without_confirm_spend(capsys: pytest.CaptureFixture[str]) -> None:
    api, store = FakeApi(), RecordingStore()

    code = await _run(["generate", "--model", MODEL], _settings(), api, store)

    assert code == 2
    assert api.paths == []
    assert "--confirm-spend" in capsys.readouterr().out


@pytest.mark.parametrize("unset", ["aws_endpoint_url_s3", "aws_s3_bucket"])
async def test_generate_refuses_unless_storage_is_local(
    unset: str, capsys: pytest.CaptureFixture[str]
) -> None:
    api, store = FakeApi(), RecordingStore()

    code = await _run(
        ["generate", "--model", MODEL, "--confirm-spend"], _settings(**{unset: None}), api, store
    )

    assert code == 2
    assert api.paths == []
    assert "AWS_ENDPOINT_URL_S3" in capsys.readouterr().out


@pytest.mark.parametrize(
    "endpoint",
    [
        "https://s3.us-east-1.amazonaws.com",
        "https://bucket.s3.amazonaws.com",
        "http://localhost.example.com:8333",
        "http://10.0.0.5:8333",
        "not a url",
    ],
)
async def test_generate_refuses_a_storage_endpoint_that_is_not_local(
    endpoint: str, capsys: pytest.CaptureFixture[str]
) -> None:
    api, store = FakeApi(), RecordingStore()
    settings = _settings(aws_endpoint_url_s3=endpoint)

    code = await _run(["generate", "--model", MODEL, "--confirm-spend"], settings, api, store)

    assert code == 2
    assert api.paths == []
    assert store.saved == []


@pytest.mark.parametrize(
    "endpoint",
    ["http://127.0.0.1:8333", "http://localhost:8333", "http://[::1]:8333", "http://s3.nrc.test"],
)
async def test_generate_accepts_a_local_storage_endpoint(endpoint: str) -> None:
    api, store = FakeApi(), RecordingStore()
    settings = _settings(aws_endpoint_url_s3=endpoint)

    code = await _run(["generate", "--model", MODEL, "--confirm-spend"], settings, api, store)

    assert code == 0
    assert len(store.saved) == 1


async def test_generate_refuses_without_output_hosts(capsys: pytest.CaptureFixture[str]) -> None:
    api, store = FakeApi(), RecordingStore()

    code = await _run(
        ["generate", "--model", MODEL, "--confirm-spend"],
        _settings(higgsfield_output_hosts=()),
        api,
        store,
    )

    assert code == 2
    assert api.paths == []
    assert "HIGGSFIELD_OUTPUT_HOSTS" in capsys.readouterr().out


async def test_generate_refuses_an_estimate_above_the_per_image_cap(
    capsys: pytest.CaptureFixture[str],
) -> None:
    api, store = FakeApi(usd="0.11"), RecordingStore()

    code = await _run(["generate", "--model", MODEL, "--confirm-spend"], _settings(), api, store)

    out = capsys.readouterr().out
    assert code == 2
    assert api.paths == [f"POST /estimate/{MODEL}"]
    assert "GENERATION_MAX_COST_USD_PER_IMAGE" in out
    assert not store.saved
    _assert_nothing_secret(out)


async def test_generate_stores_one_image_and_prints_ids_only(
    capsys: pytest.CaptureFixture[str],
) -> None:
    api, store = FakeApi(), RecordingStore()

    code = await _run(["generate", "--model", MODEL, "--confirm-spend"], _settings(), api, store)

    out = capsys.readouterr().out
    assert code == 0
    assert api.paths == [
        f"POST /estimate/{MODEL}",
        f"POST /{MODEL}",
        "GET /requests/req-1/status",
    ]
    assert len(store.saved) == 1
    for expected in (
        "request_id=req-1",
        f"model={MODEL}",
        "estimate_usd=0.05",
        "actual_cost_usd=None",
        "width_px=1080 height_px=1080",
        "s3_key=generated-images/x.png",
    ):
        assert expected in out
    _assert_nothing_secret(out)


async def test_estimate_prices_each_model_and_never_submits(
    capsys: pytest.CaptureFixture[str],
) -> None:
    api, store = FakeApi(usd="0.094"), RecordingStore()
    models = list(CANDIDATES)

    code = await _run(["estimate", "--models", ",".join(models)], _settings(), api, store)

    out = capsys.readouterr().out
    assert code == 0
    assert api.paths == [f"POST /estimate/{model}" for model in models]
    for model in models:
        assert f"model={model} usd=0.094" in out
    _assert_nothing_secret(out)


async def test_candidates_use_their_smallest_documented_settings() -> None:
    make = provider_factory(_settings())

    soul, soul_v2 = make("higgsfield-ai/soul/standard"), make("higgsfield-ai/soul/v2/standard")

    assert soul._resolution == "2K"
    assert soul_v2._resolution == "1080p"
    for provider in (soul, soul_v2):
        await provider.aclose()
