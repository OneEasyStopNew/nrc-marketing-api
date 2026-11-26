"""Embedding providers (NRC-5): simulator, settings checks, the Bedrock adapter without a network.

The adapter is exercised through botocore's Stubber, which validates each call against the real
bedrock-runtime service model, or through a client that raises. The request and response bodies
are the documented ones; they have not been checked against the live service (follow-up F30).
"""

import io
import json
import math
from collections.abc import Mapping, Sequence
from typing import Any
from unittest.mock import patch

import boto3
import pytest
from botocore.exceptions import (
    ConnectTimeoutError,
    CredentialRetrievalError,
    EndpointConnectionError,
    ReadTimeoutError,
    SSOError,
    TokenRetrievalError,
)
from botocore.response import StreamingBody
from botocore.stub import ANY, Stubber
from pydantic import ValidationError

from app.core.config import Settings
from app.models.policy_graph import EMBEDDING_DIMENSIONS
from app.services.providers.base import (
    ProviderConfigError,
    ProviderRejectedError,
    ProviderUnavailableError,
)
from app.services.providers.bedrock_embeddings import PINNED_MODELS, BedrockEmbeddingProvider
from app.services.providers.embeddings import (
    COHERE_MULTILINGUAL_V3_MODEL_ID,
    COHERE_V4_MODEL_ID,
    SIMULATED_MODEL_ID,
    SUPPORTED_BEDROCK_EMBEDDING_MODEL_IDS,
    TITAN_V2_MODEL_ID,
    EmbeddingPurpose,
    EmbeddingResponseError,
    SimulatedEmbeddingProvider,
    embedding_provider_from_settings,
    missing_live_embedding_settings,
)

DOCUMENT, QUERY = EmbeddingPurpose.DOCUMENT, EmbeddingPurpose.QUERY
REQUIRED: dict[str, Any] = {
    "database_url": "postgresql+asyncpg://u:p@localhost/db",
    "cognito_region": "ap-south-1",
    "cognito_user_pool_id": "ap-south-1_TESTPOOL",
    "cognito_client_ids": ("c",),
    "required_scope": "marketing/api",
}


def _settings(**overrides: Any) -> Settings:
    return Settings(_env_file=None, **{**REQUIRED, **overrides})


def _cosine(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b, strict=True))


# Simulator


async def test_simulator_is_deterministic_normalised_and_1024_wide() -> None:
    texts = ["Do not promise free treatment.", "Payment may be mentioned"]
    first = await SimulatedEmbeddingProvider().embed(texts, DOCUMENT)
    second = await SimulatedEmbeddingProvider().embed(texts, QUERY)

    assert first == second
    assert [len(vector) for vector in first] == [EMBEDDING_DIMENSIONS] * 2
    assert all(math.isclose(math.sqrt(_cosine(v, v)), 1.0) for v in first)


async def test_simulator_gives_a_fixed_unit_vector_for_text_without_words() -> None:
    empty, punctuation = await SimulatedEmbeddingProvider().embed(["", " ... !!"], DOCUMENT)

    assert empty == punctuation
    assert math.isclose(_cosine(empty, empty), 1.0)
    assert not any(math.isnan(value) for value in empty)


async def test_simulator_puts_similar_texts_closer_than_unrelated_ones() -> None:
    base, similar, unrelated = await SimulatedEmbeddingProvider().embed(
        [
            "Do not promise free treatment to participants",
            "Never promise free treatment",
            "The weather is sunny in the garden today",
        ],
        DOCUMENT,
    )

    assert _cosine(base, similar) > _cosine(base, unrelated) + 0.3


def test_simulator_is_labelled_simulated() -> None:
    provider = SimulatedEmbeddingProvider()

    assert provider.model_id == SIMULATED_MODEL_ID == "simulator:token-hash-v1"
    assert "SIMULATED" in (SimulatedEmbeddingProvider.__doc__ or "")


# Settings and the factory


def test_embeddings_are_off_by_default_and_the_factory_gives_no_provider() -> None:
    settings = _settings()

    assert settings.embedding_mode == "off"
    assert embedding_provider_from_settings(settings) is None
    assert missing_live_embedding_settings(settings) == []


def test_factory_gives_the_simulator_locally() -> None:
    provider = embedding_provider_from_settings(
        _settings(app_env="local", embedding_mode="simulator")
    )

    assert isinstance(provider, SimulatedEmbeddingProvider)


@pytest.mark.parametrize("app_env", ["staging", "production"])
def test_simulator_is_refused_outside_local_and_demo(app_env: str) -> None:
    with pytest.raises(ValidationError, match="EMBEDDING_MODE=simulator cannot be used"):
        _settings(app_env=app_env, embedding_mode="simulator")


def test_simulator_is_allowed_in_demo_and_live_in_staging() -> None:
    _settings(app_env="demo", embedding_mode="simulator")
    _settings(
        app_env="staging", embedding_mode="live", bedrock_embedding_model_id=TITAN_V2_MODEL_ID
    )


@pytest.mark.parametrize(
    ("overrides", "expected"),
    [
        ({}, "missing BEDROCK_EMBEDDING_MODEL_ID"),
        ({"bedrock_embedding_model_id": ""}, "missing BEDROCK_EMBEDDING_MODEL_ID"),
        ({"bedrock_embedding_model_id": "amazon.nova-embed:1"}, "not a supported model"),
    ],
)
def test_live_mode_lists_what_it_lacks_and_the_factory_refuses(
    overrides: dict[str, str], expected: str
) -> None:
    settings = _settings(embedding_mode="live", **overrides)

    problems = missing_live_embedding_settings(settings)
    assert len(problems) == 1
    assert expected in problems[0]
    with pytest.raises(ProviderConfigError, match=expected):
        embedding_provider_from_settings(settings)


def test_live_mode_builds_the_bedrock_adapter_for_a_pinned_model() -> None:
    settings = _settings(
        embedding_mode="live", bedrock_embedding_model_id=COHERE_V4_MODEL_ID, aws_region="us-east-1"
    )

    provider = embedding_provider_from_settings(settings)

    assert isinstance(provider, BedrockEmbeddingProvider)
    assert provider.model_id == COHERE_V4_MODEL_ID


@pytest.mark.parametrize(
    ("field", "bad"),
    [("bedrock_timeout_seconds", 61), ("bedrock_timeout_seconds", 0), ("bedrock_max_retries", 11)],
)
def test_bedrock_client_limits_are_bounded(field: str, bad: float) -> None:
    with pytest.raises(ValidationError, match=field):
        _settings(**{field: bad})


def test_pinned_models_match_the_models_the_settings_check_accepts() -> None:
    assert set(PINNED_MODELS) == SUPPORTED_BEDROCK_EMBEDDING_MODEL_IDS


# Bedrock adapter


def _vector(seed: float = 0.5) -> list[float]:
    return [seed] * EMBEDDING_DIMENSIONS


def _body(payload: object) -> StreamingBody:
    raw = json.dumps(payload).encode()
    return StreamingBody(io.BytesIO(raw), len(raw))


def _response(body: StreamingBody) -> dict[str, Any]:
    return {"body": body, "contentType": "application/json"}


def _expected(model_id: str, body: Mapping[str, object]) -> dict[str, object]:
    return {"modelId": model_id, "body": json.dumps(body), "contentType": ANY, "accept": ANY}


def _stubbed(model_id: str) -> tuple[BedrockEmbeddingProvider, Stubber]:
    # A real bedrock-runtime client whose calls are answered by the stubber, never sent.
    client = boto3.Session().client("bedrock-runtime", region_name="us-east-1")
    stubber = Stubber(client)
    stubber.activate()
    return BedrockEmbeddingProvider(model_id=model_id, client=client), stubber


async def test_titan_is_called_once_per_text_with_the_documented_body() -> None:
    provider, stubber = _stubbed(TITAN_V2_MODEL_ID)
    for text, seed in (("one", 0.1), ("two", 0.2), ("three", 0.3)):
        request = {"inputText": text, "dimensions": 1024, "normalize": True}
        stubber.add_response(
            "invoke_model",
            _response(_body({"embedding": _vector(seed), "inputTextTokenCount": 1})),
            _expected(TITAN_V2_MODEL_ID, request),
        )

    vectors = await provider.embed(["one", "two", "three"], DOCUMENT)

    assert [v[0] for v in vectors] == [0.1, 0.2, 0.3]
    stubber.assert_no_pending_responses()


@pytest.mark.parametrize(
    ("purpose", "input_type"), [(DOCUMENT, "search_document"), (QUERY, "search_query")]
)
async def test_cohere_multilingual_sends_the_texts_in_one_call(
    purpose: EmbeddingPurpose, input_type: str
) -> None:
    provider, stubber = _stubbed(COHERE_MULTILINGUAL_V3_MODEL_ID)
    request = {"texts": ["a", "b"], "input_type": input_type, "truncate": "END"}
    stubber.add_response(
        "invoke_model",
        _response(_body({"embeddings": [_vector(0.1), _vector(0.2)]})),
        _expected(COHERE_MULTILINGUAL_V3_MODEL_ID, request),
    )

    vectors = await provider.embed(["a", "b"], purpose)

    assert [v[0] for v in vectors] == [0.1, 0.2]
    stubber.assert_no_pending_responses()


async def test_cohere_v4_asks_for_float_vectors_of_1024_and_reads_them_by_type() -> None:
    provider, stubber = _stubbed(COHERE_V4_MODEL_ID)
    request = {
        "texts": ["a"],
        "input_type": "search_document",
        "embedding_types": ["float"],
        "output_dimension": 1024,
    }
    stubber.add_response(
        "invoke_model",
        _response(_body({"embeddings": {"float": [_vector(0.4)]}})),
        _expected(COHERE_V4_MODEL_ID, request),
    )

    assert [v[0] for v in await provider.embed(["a"], DOCUMENT)] == [0.4]


async def test_cohere_splits_more_than_96_texts_into_batches_in_order() -> None:
    provider, stubber = _stubbed(COHERE_V4_MODEL_ID)
    texts = [f"text {n}" for n in range(100)]
    for batch, seed in ((texts[:96], 0.1), (texts[96:], 0.2)):
        stubber.add_response(
            "invoke_model",
            _response(_body({"embeddings": {"float": [_vector(seed)] * len(batch)}})),
            {"modelId": COHERE_V4_MODEL_ID, "body": ANY, "contentType": ANY, "accept": ANY},
        )

    vectors = await provider.embed(texts, DOCUMENT)

    assert [v[0] for v in vectors] == [0.1] * 96 + [0.2] * 4
    stubber.assert_no_pending_responses()


async def test_no_texts_make_no_call() -> None:
    provider, stubber = _stubbed(TITAN_V2_MODEL_ID)

    assert await provider.embed([], DOCUMENT) == []
    stubber.assert_no_pending_responses()


@pytest.mark.parametrize(
    ("model_id", "payload", "message"),
    [
        (TITAN_V2_MODEL_ID, {"embedding": [0.1] * 512}, "1024 dimensions"),
        (TITAN_V2_MODEL_ID, {"vector": _vector()}, "no 'embedding' field"),
        (TITAN_V2_MODEL_ID, {"embedding": ["x"] * 1024}, "not a finite number"),
        (TITAN_V2_MODEL_ID, {"embedding": [True] * 1024}, "not a finite number"),
        (TITAN_V2_MODEL_ID, {"embedding": [None] * 1024}, "not a finite number"),
        (COHERE_MULTILINGUAL_V3_MODEL_ID, {"embeddings": [_vector()]}, "one vector per text"),
        (COHERE_MULTILINGUAL_V3_MODEL_ID, {"texts": []}, "no 'embeddings' field"),
        (COHERE_MULTILINGUAL_V3_MODEL_ID, {"embeddings": "nope"}, "one vector per text"),
        (COHERE_V4_MODEL_ID, {"embeddings": [_vector()] * 2}, "no 'float' field"),
        (COHERE_V4_MODEL_ID, {"embeddings": {"float": [_vector()]}}, "one vector per text"),
        (COHERE_V4_MODEL_ID, ["not", "an", "object"], "no 'embeddings' field"),
    ],
)
async def test_a_malformed_response_is_an_embedding_response_error(
    model_id: str, payload: object, message: str
) -> None:
    provider, stubber = _stubbed(model_id)
    stubber.add_response(
        "invoke_model",
        _response(_body(payload)),
        {"modelId": model_id, "body": ANY, "contentType": ANY, "accept": ANY},
    )
    texts: Sequence[str] = ["a", "b"] if model_id != TITAN_V2_MODEL_ID else ["a"]

    with pytest.raises(EmbeddingResponseError, match=message):
        await provider.embed(texts, DOCUMENT)


async def test_a_non_json_response_is_an_embedding_response_error() -> None:
    provider, stubber = _stubbed(TITAN_V2_MODEL_ID)
    stubber.add_response(
        "invoke_model",
        _response(StreamingBody(io.BytesIO(b"<html>"), 6)),
        {"modelId": TITAN_V2_MODEL_ID, "body": ANY, "contentType": ANY, "accept": ANY},
    )

    with pytest.raises(EmbeddingResponseError, match="not valid JSON"):
        await provider.embed(["a"], DOCUMENT)


@pytest.mark.parametrize(
    ("code", "status", "error_type"),
    [
        ("ThrottlingException", 429, ProviderUnavailableError),
        ("ServiceUnavailableException", 503, ProviderUnavailableError),
        ("ModelNotReadyException", 429, ProviderUnavailableError),
        ("SomethingNewException", 502, ProviderUnavailableError),
        ("AccessDeniedException", 403, ProviderConfigError),
        ("ValidationException", 400, ProviderConfigError),
        ("ResourceNotFoundException", 404, ProviderConfigError),
        ("ModelErrorException", 424, ProviderRejectedError),
    ],
)
async def test_bedrock_error_codes_map_to_typed_provider_errors(
    code: str, status: int, error_type: type[Exception]
) -> None:
    provider, stubber = _stubbed(TITAN_V2_MODEL_ID)
    stubber.add_client_error(
        "invoke_model",
        service_error_code=code,
        service_message="echo of the request text: SECRET POLICY TEXT",
        http_status_code=status,
    )

    with pytest.raises(error_type, match=f"{TITAN_V2_MODEL_ID} failed \\({code}\\)$") as caught:
        await provider.embed(["SECRET POLICY TEXT"], DOCUMENT)

    assert "SECRET" not in str(caught.value)
    assert caught.value.__cause__ is None


@pytest.mark.parametrize(
    "error",
    [
        ReadTimeoutError(endpoint_url="https://bedrock-runtime.example/model/x/invoke"),
        ConnectTimeoutError(endpoint_url="https://bedrock-runtime.example/model/x/invoke"),
        EndpointConnectionError(endpoint_url="https://bedrock-runtime.example/model/x/invoke"),
    ],
    ids=["read-timeout", "connect-timeout", "endpoint-connection"],
)
async def test_timeouts_and_connection_failures_are_unavailable(error: Exception) -> None:
    # A real client whose call raises the botocore error (a Stubber can't raise these).
    client = boto3.Session().client("bedrock-runtime", region_name="us-east-1")
    provider = BedrockEmbeddingProvider(model_id=TITAN_V2_MODEL_ID, client=client)

    with (
        patch.object(client, "invoke_model", side_effect=error),
        pytest.raises(ProviderUnavailableError, match="could not be reached") as caught,
    ):
        await provider.embed(["a"], DOCUMENT)

    assert caught.value.retryable
    assert "bedrock-runtime.example" not in str(caught.value)


@pytest.mark.parametrize(
    "error",
    [
        SSOError(),
        TokenRetrievalError(provider="sso", error_msg="the SSO session has expired"),
        CredentialRetrievalError(provider="assume-role", error_msg="role can't be assumed"),
    ],
    ids=["sso", "expired-token", "credential-retrieval"],
)
async def test_expired_or_unloadable_credentials_are_a_config_error(error: Exception) -> None:
    # Retrying won't renew a sign-in, so these must not be the retryable "unavailable" error.
    client = boto3.Session().client("bedrock-runtime", region_name="us-east-1")
    provider = BedrockEmbeddingProvider(model_id=TITAN_V2_MODEL_ID, client=client)

    with (
        patch.object(client, "invoke_model", side_effect=error),
        pytest.raises(ProviderConfigError, match="credentials missing or expired") as caught,
    ):
        await provider.embed(["a"], DOCUMENT)

    assert not caught.value.retryable


async def test_a_container_credential_blip_is_retried_not_a_config_error() -> None:
    # EKS Pod Identity serves credentials from a local agent ("container-role"); a short outage
    # there is temporary, so it is the retryable "unavailable" error.
    client = boto3.Session().client("bedrock-runtime", region_name="us-east-1")
    provider = BedrockEmbeddingProvider(model_id=TITAN_V2_MODEL_ID, client=client)
    error = CredentialRetrievalError(provider="container-role", error_msg="connection refused")

    with (
        patch.object(client, "invoke_model", side_effect=error),
        pytest.raises(
            ProviderUnavailableError, match="credentials temporarily unavailable"
        ) as caught,
    ):
        await provider.embed(["a"], DOCUMENT)

    assert caught.value.retryable


def test_unsupported_model_id_is_a_config_error() -> None:
    client = boto3.Session().client("bedrock-runtime", region_name="us-east-1")

    with pytest.raises(ProviderConfigError, match="not a supported model"):
        BedrockEmbeddingProvider(model_id="anthropic.claude-3", client=client)


def test_repr_shows_only_the_model_id() -> None:
    provider = embedding_provider_from_settings(
        _settings(embedding_mode="live", bedrock_embedding_model_id=TITAN_V2_MODEL_ID)
    )

    assert repr(provider) == f"BedrockEmbeddingProvider(model_id='{TITAN_V2_MODEL_ID}')"
