# Pinned generation models (F24, Vishnu's 17 Nov review): each enabled model sends only its
# approved settings, the cost cap defaults to $0.09 and a job can only use an enabled model.

import hashlib
import json
from collections.abc import Iterator
from decimal import Decimal
from typing import Any

import httpx
import pytest
from pydantic import SecretStr

from app.core.config import Settings, get_settings
from app.core.errors import GenerationRequestError
from app.services.jobs import _choose_model
from app.services.providers.base import (
    HIGGSFIELD_NAME,
    FetchedImage,
    ImageRequest,
    ProviderRejectedError,
    ProviderUnavailableError,
)
from app.services.providers.higgsfield import HiggsfieldProvider, missing_live_settings
from app.services.providers.pinned_models import PINNED_MODELS, max_brief_chars
from app.workers.connector import WorkerConfigError, build_providers, provider_key
from tests.conftest import JOBS, AuthHeaders, World, _creative

SOUL = "higgsfield-ai/soul/v2/standard"
STUDIO = "marketing-studio/image"
IDEOGRAM = "ideogram/v4.0"
# Fake credentials: every provider call here goes to an httpx.MockTransport.
KEY_ID = "kid-test-123"
KEY_SECRET = "ksecret-test-456"  # noqa: S105


class RecordingFetcher:
    # Output fetching is covered by the adapter tests; these tests never reach it.
    async def __call__(
        self, url: str, *, allowed_hosts: frozenset[str], max_bytes: int
    ) -> FetchedImage:
        return FetchedImage(content=b"img", content_type="image/png", source_host="cdn.example.com")


def settings(**overrides: Any) -> Settings:
    values: dict[str, Any] = {
        "database_url": "postgresql+asyncpg://u:p@localhost/db",
        "cognito_region": "ap-south-1",
        "cognito_user_pool_id": "ap-south-1_TESTPOOL",
        "cognito_client_ids": ("c",),
        "required_scope": "marketing/api",
        "higgsfield_model": SOUL,
        "higgsfield_api_key": f"{KEY_ID}:{KEY_SECRET}",
        "higgsfield_output_hosts": ("cdn.example.com",),
    }
    values.update(overrides)
    return Settings(_env_file=None, **values)


def provider(model: str, seen: list[httpx.Request]) -> HiggsfieldProvider:
    def handler(req: httpx.Request) -> httpx.Response:
        seen.append(req)
        if "/estimate/" in str(req.url):
            return httpx.Response(200, json={"usd": "0.020"})
        return httpx.Response(200, json={"request_id": "r-1", "status": "queued"})

    return HiggsfieldProvider(
        client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
        base_url="https://api.higgsfield.ai",
        model=model,
        api_key_id=SecretStr(KEY_ID),
        api_key_secret=SecretStr(KEY_SECRET),
        fetcher=RecordingFetcher(),
        allowed_output_hosts=frozenset({"cdn.example.com"}),
        max_output_bytes=1234,
        # A caller-supplied resolution must never override a pinned model's settings.
        resolution="720p",
    )


def image_request(prompt: str = "A calm clinic waiting room", ratio: str = "1:1") -> ImageRequest:
    return ImageRequest(
        prompt=prompt, width_px=1080, height_px=1080, aspect_ratio=ratio, idempotency_key="k-1"
    )


@pytest.mark.parametrize(
    ("model", "pinned"),
    [
        (SOUL, {"resolution": "1080p"}),
        (STUDIO, {"quality": "low", "resolution": "2k"}),
        (IDEOGRAM, {"rendering_speed": "DEFAULT"}),
    ],
)
async def test_each_model_sends_exactly_its_pinned_settings(
    model: str, pinned: dict[str, str]
) -> None:
    seen: list[httpx.Request] = []
    hf = provider(model, seen)

    await hf.estimate_cost_usd(image_request())
    await hf.submit(image_request())

    expected = {"prompt": "A calm clinic waiting room", "aspect_ratio": "1:1", **pinned}
    # The estimate prices the exact body that is then submitted.
    assert [json.loads(req.content) for req in seen] == [expected, expected]


async def test_marketing_studio_never_relies_on_its_high_quality_default() -> None:
    # Its default (high quality, 2k) quotes $0.444; only quality=low is ever sent.
    seen: list[httpx.Request] = []

    await provider(STUDIO, seen).estimate_cost_usd(image_request())

    assert json.loads(seen[0].content)["quality"] == "low"


@pytest.mark.parametrize("model", [SOUL, STUDIO, IDEOGRAM])
async def test_a_prompt_over_the_model_limit_is_refused_before_any_request(model: str) -> None:
    seen: list[httpx.Request] = []
    too_long = "x" * (PINNED_MODELS[model].max_prompt_chars + 1)

    with pytest.raises(ProviderRejectedError, match="prompt is longer than"):
        await provider(model, seen).estimate_cost_usd(image_request(too_long))

    assert seen == []


async def test_a_ratio_outside_the_pinned_set_is_refused_before_any_request() -> None:
    seen: list[httpx.Request] = []

    with pytest.raises(ProviderRejectedError, match="aspect ratio"):
        await provider(SOUL, seen).estimate_cost_usd(image_request(ratio="4:5"))

    assert seen == []


async def test_a_token_priced_quote_without_a_dollar_price_is_never_usable() -> None:
    # GPT Image Flare/Sunburst answer with a pricing description, not a price: the worker then
    # never submits (it needs a price first), so they cannot run under the cap.
    def handler(req: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200, json={"type": "description", "pricing_description": "Per 1M tokens: ..."}
        )

    hf = HiggsfieldProvider(
        client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
        base_url="https://api.higgsfield.ai",
        model="marketing-studio/image/flare",
        api_key_id=SecretStr(KEY_ID),
        api_key_secret=SecretStr(KEY_SECRET),
        fetcher=RecordingFetcher(),
        allowed_output_hosts=frozenset(),
        max_output_bytes=1234,
    )

    with pytest.raises(ProviderUnavailableError, match="unreadable cost estimate"):
        await hf.estimate_cost_usd(image_request())


def test_the_cost_cap_defaults_to_nine_cents() -> None:
    assert Settings.model_fields["generation_max_cost_usd_per_image"].default == Decimal("0.09")


def test_brief_room_leaves_space_for_the_template() -> None:
    for model, pinned in PINNED_MODELS.items():
        room = max_brief_chars(model)
        assert 0 < room < pinned.max_prompt_chars, model
    # Ideogram's provider limit is 2,048 characters (free /estimate probe, 17 Nov 2026).
    assert PINNED_MODELS[IDEOGRAM].max_prompt_chars == 2048
    assert PINNED_MODELS[STUDIO].max_prompt_chars == 5000


@pytest.mark.parametrize(
    ("overrides", "problem"),
    [
        ({"higgsfield_models": (SOUL, "test/unpinned-model")}, "test/unpinned-model"),
        ({"higgsfield_models": (STUDIO,), "higgsfield_model": SOUL}, "HIGGSFIELD_MODEL"),
    ],
)
def test_live_mode_names_unpinned_or_unlisted_models(
    overrides: dict[str, Any], problem: str
) -> None:
    problems = missing_live_settings(settings(**overrides))

    assert any(problem in text for text in problems), problems


def test_enabled_models_default_to_the_single_configured_model() -> None:
    assert settings().enabled_image_models == (SOUL,)
    assert settings(higgsfield_models=(SOUL, STUDIO)).enabled_image_models == (SOUL, STUDIO)


async def test_live_worker_builds_one_provider_per_enabled_model() -> None:
    live = get_settings().model_copy(
        update={
            "generation_mode": "live",
            "higgsfield_api_key": SecretStr("test-id:test-secret"),
            "higgsfield_model": SOUL,
            "higgsfield_models": (SOUL, STUDIO, IDEOGRAM),
            "higgsfield_output_hosts": ("images.example.test",),
        }
    )

    providers = build_providers(live)
    try:
        assert sorted(providers) == sorted(
            provider_key(HIGGSFIELD_NAME, model) for model in (SOUL, STUDIO, IDEOGRAM)
        )
        assert {p.model for p in providers.values()} == {SOUL, STUDIO, IDEOGRAM}
    finally:
        for p in providers.values():
            if isinstance(p, HiggsfieldProvider):
                await p.aclose()


def test_live_worker_refuses_an_unpinned_enabled_model() -> None:
    live = get_settings().model_copy(
        update={
            "generation_mode": "live",
            "higgsfield_api_key": SecretStr("test-id:test-secret"),
            "higgsfield_model": SOUL,
            "higgsfield_models": (SOUL, "test/unpinned-model"),
            "higgsfield_output_hosts": ("images.example.test",),
        }
    )

    with pytest.raises(WorkerConfigError, match="test/unpinned-model"):
        build_providers(live)


# --- API: choosing a model when starting a job (simulator mode, so nothing is sent) ---


@pytest.fixture
def three_models() -> Iterator[None]:
    # Cleared again on teardown, so later tests never see these cached settings.
    with pytest.MonkeyPatch.context() as patch:
        patch.setenv("HIGGSFIELD_MODEL", SOUL)
        patch.setenv("HIGGSFIELD_MODELS", f"{SOUL},{STUDIO},{IDEOGRAM}")
        get_settings.cache_clear()
        yield
    get_settings.cache_clear()


async def _start(
    client: httpx.AsyncClient, auth: AuthHeaders, world: World, creative_id: str, model: str
) -> httpx.Response:
    # Fixed length: the header needs at least 8 characters, whatever the model name.
    key = f"job-{hashlib.sha256(model.encode()).hexdigest()[:12]}"
    headers = auth("creator-a", world.a.tenant_id, **{"Idempotency-Key": key})
    body = {
        "creative_id": creative_id,
        "creative_version": 1,
        "kind": "image_generation",
        "model": model,
    }
    return await client.post(JOBS, headers=headers, json=body)


@pytest.mark.usefixtures("three_models")
async def test_an_enabled_no_text_model_can_be_chosen(
    client: httpx.AsyncClient, world: World, auth: AuthHeaders
) -> None:
    creative_id = await _creative(client, auth, world.a)

    response = await _start(client, auth, world, creative_id, STUDIO)

    assert response.status_code == 202


@pytest.mark.usefixtures("three_models")
async def test_a_model_that_is_not_enabled_is_refused(
    client: httpx.AsyncClient, world: World, auth: AuthHeaders
) -> None:
    creative_id = await _creative(client, auth, world.a)

    response = await _start(client, auth, world, creative_id, "higgsfield-ai/soul/standard")

    assert response.status_code == 422
    body = response.json()
    assert (body["code"], body["details"]["reason"]) == (
        "GENERATION_REQUEST_INVALID",
        "MODEL_NOT_ENABLED",
    )


@pytest.mark.usefixtures("three_models")
async def test_an_empty_model_is_refused_not_replaced_by_the_default(
    client: httpx.AsyncClient, world: World, auth: AuthHeaders
) -> None:
    creative_id = await _creative(client, auth, world.a)

    response = await _start(client, auth, world, creative_id, "")

    assert response.status_code == 422
    body = response.json()
    assert body["code"] == "VALIDATION_FAILED"
    assert [field["loc"] for field in body["details"]["fields"]] == [["body", "model"]]


@pytest.mark.usefixtures("three_models")
async def test_a_blank_model_is_not_an_enabled_model(
    client: httpx.AsyncClient, world: World, auth: AuthHeaders
) -> None:
    # Only exact enabled names pass, so whitespace needs no pattern of its own.
    creative_id = await _creative(client, auth, world.a)

    response = await _start(client, auth, world, creative_id, " ")

    assert response.status_code == 422
    assert response.json()["details"]["reason"] == "MODEL_NOT_ENABLED"


def test_an_empty_model_name_never_falls_back_to_the_default() -> None:
    # Only a missing model (None) means the default; the schema already refuses "".
    with pytest.raises(GenerationRequestError, match="not available"):
        _choose_model(settings(), "")


@pytest.mark.usefixtures("three_models")
async def test_the_text_model_needs_image_text(
    client: httpx.AsyncClient, world: World, auth: AuthHeaders
) -> None:
    # A draft has no image-text field yet (drafts are Dinesh's), so every draft is no-text mode
    # and the text-in-image model cannot be chosen for it.
    creative_id = await _creative(client, auth, world.a)

    response = await _start(client, auth, world, creative_id, IDEOGRAM)

    assert response.status_code == 422
    assert response.json()["details"]["reason"] == "IMAGE_TEXT_REQUIRED"


@pytest.mark.usefixtures("three_models")
async def test_the_model_list_reports_brief_room_and_mode(
    client: httpx.AsyncClient, world: World, auth: AuthHeaders
) -> None:
    response = await client.get(
        "/v1/marketing/generation-models", headers=auth("creator-a", world.a.tenant_id)
    )

    assert response.status_code == 200
    models = {item["model"]: item for item in response.json()["items"]}
    assert list(models) == [SOUL, STUDIO, IDEOGRAM]
    assert models[SOUL]["is_default"] is True
    assert models[IDEOGRAM]["text_in_image"] is True
    assert models[STUDIO]["text_in_image"] is False
    assert models[IDEOGRAM]["max_brief_chars"] == max_brief_chars(IDEOGRAM)
    assert models[SOUL]["aspect_ratios"] == sorted(PINNED_MODELS[SOUL].aspect_ratios)


@pytest.mark.usefixtures("three_models")
async def test_the_model_list_carries_labels_and_estimated_prices(
    client: httpx.AsyncClient, world: World, auth: AuthHeaders
) -> None:
    # The web app builds its model picker and cost note only from this list (no hard-coded names
    # or prices), so each entry says what to show and how current the estimate is.
    response = await client.get(
        "/v1/marketing/generation-models", headers=auth("creator-a", world.a.tenant_id)
    )

    models = {item["model"]: item for item in response.json()["items"]}
    for model, item in models.items():
        pinned = PINNED_MODELS[model]
        assert item["label"] == pinned.label
        # Money goes out as a decimal string, never a float.
        assert item["estimated_usd_per_image"] == str(pinned.estimated_usd_per_image)
        assert item["price_checked_at"] == pinned.price_checked_at.isoformat()
        assert item["price_is_fixed"] is pinned.price_is_fixed
    assert models[STUDIO]["price_is_fixed"] is False
    assert models[SOUL]["price_is_fixed"] is True


def test_every_pinned_model_has_a_label_and_a_positive_price_estimate() -> None:
    for model, pinned in PINNED_MODELS.items():
        assert pinned.label.strip(), model
        assert pinned.estimated_usd_per_image > 0, model
