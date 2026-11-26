"""Bedrock text judge adapter (NRC-7) without a network.

Calls are answered by botocore's Stubber, which validates each one against the real bedrock-runtime
service model, or by a client that raises or answers off-contract. The shapes are the documented
ones; they have not been checked against the live service.
"""

import time
from typing import Any

import boto3
import pytest
from botocore.exceptions import (
    ConnectTimeoutError,
    CredentialRetrievalError,
    NoCredentialsError,
    ReadTimeoutError,
    SSOTokenLoadError,
    TokenRetrievalError,
    UnauthorizedSSOTokenError,
)
from botocore.stub import Stubber

from app.services.providers.base import (
    ProviderConfigError,
    ProviderError,
    ProviderOutputError,
    ProviderRateLimitedError,
    ProviderRejectedError,
    ProviderUnavailableError,
)
from app.services.providers.bedrock_judge import BedrockTextJudge
from app.services.providers.text_judge import (
    NOVA_LITE_MODEL_ID,
    SUPPORTED_BEDROCK_JUDGE_MODEL_IDS,
    JudgeRequest,
)

SECRET_TEXT = "SECRET TEXT UNDER REVIEW"  # noqa: S105  marker proving it is never echoed
# Long enough that no ordinary test call comes near it.
CALL_BUDGET_SECONDS = 30.0


def _request(max_output_tokens: int = 1500) -> JudgeRequest:
    return JudgeRequest(
        system="You judge text.",
        user=f"Check this: {SECRET_TEXT}",
        rule_ids=("R-1",),
        text=SECRET_TEXT,
        max_output_tokens=max_output_tokens,
    )


def _stubbed(max_output_tokens: int = 2000) -> tuple[BedrockTextJudge, Stubber]:
    # A real bedrock-runtime client whose calls are answered by the stubber, never sent.
    client = boto3.Session().client("bedrock-runtime", region_name="us-east-1")
    stubber = Stubber(client)
    stubber.activate()
    judge = BedrockTextJudge(
        model_id=NOVA_LITE_MODEL_ID,
        client=client,
        max_output_tokens=max_output_tokens,
        call_budget_seconds=CALL_BUDGET_SECONDS,
    )
    return judge, stubber


def _response(content: list[dict[str, Any]], **overrides: Any) -> dict[str, Any]:
    return {
        "output": {"message": {"role": "assistant", "content": content}},
        "stopReason": "end_turn",
        "usage": {"inputTokens": 12, "outputTokens": 5, "totalTokens": 17},
        "metrics": {"latencyMs": 1},
        **overrides,
    }


def _expected(max_tokens: int) -> dict[str, Any]:
    request = _request()
    return {
        "modelId": NOVA_LITE_MODEL_ID,
        "system": [{"text": request.system}],
        "messages": [{"role": "user", "content": [{"text": request.user}]}],
        "inferenceConfig": {"maxTokens": max_tokens},
    }


class _Client:
    # A client that raises, or answers with a fixed body (a Stubber cannot do either off-contract).

    def __init__(
        self, *, error: Exception | None = None, body: object = None, delay: float = 0
    ) -> None:
        self._error = error
        self._body = body
        self._delay = delay

    def converse(self, **kwargs: object) -> object:
        # Runs in the adapter's worker thread, so a blocking sleep stands in for a slow call.
        time.sleep(self._delay)
        if self._error is not None:
            raise self._error
        return self._body


def _judge_with(
    client: _Client, call_budget_seconds: float = CALL_BUDGET_SECONDS
) -> BedrockTextJudge:
    return BedrockTextJudge(
        model_id=NOVA_LITE_MODEL_ID,
        client=client,  # type: ignore[arg-type]  # a fake with converse() only, not a boto3 client
        max_output_tokens=500,
        call_budget_seconds=call_budget_seconds,
    )


async def test_a_call_past_its_budget_is_unavailable_and_retryable() -> None:
    # botocore's timeouts are per read, so the adapter bounds the whole call itself.
    slow = _judge_with(_Client(body={}, delay=0.3), call_budget_seconds=0.05)

    with pytest.raises(ProviderUnavailableError, match="did not answer in time") as raised:
        await slow.judge(_request())

    assert raised.value.retryable is True


async def test_request_has_system_and_user_message_and_the_lower_token_cap() -> None:
    judge, stubber = _stubbed(max_output_tokens=1000)
    # The stubber rejects any other parameter, so toolConfig and temperature are not sent.
    stubber.add_response("converse", _response([{"text": "{}"}]), _expected(1000))

    await judge.judge(_request(max_output_tokens=1500))

    stubber.assert_no_pending_responses()


async def test_the_request_cap_wins_when_it_is_lower_than_the_settings_cap() -> None:
    judge, stubber = _stubbed(max_output_tokens=1000)
    stubber.add_response("converse", _response([{"text": "{}"}]), _expected(300))

    await judge.judge(_request(max_output_tokens=300))

    stubber.assert_no_pending_responses()


async def test_reply_joins_every_text_block_and_reports_usage() -> None:
    judge, stubber = _stubbed()
    content: list[dict[str, Any]] = [
        {"reasoningContent": {"reasoningText": {"text": "thinking"}}},
        {"text": '{"results": '},
        {"text": "[]}"},
    ]
    stubber.add_response("converse", _response(content, stopReason="max_tokens"))

    reply = await judge.judge(_request())

    assert reply.raw == '{"results": []}'
    assert reply.model_id == NOVA_LITE_MODEL_ID
    assert reply.stop_reason == "max_tokens"
    assert (reply.input_tokens, reply.output_tokens) == (12, 5)
    assert reply.latency_ms >= 0
    assert not reply.is_simulated


@pytest.mark.parametrize(
    ("code", "status", "error_type"),
    [
        ("ThrottlingException", 429, ProviderRateLimitedError),
        ("ThrottlingException", 400, ProviderRateLimitedError),
        ("SomethingElse", 429, ProviderRateLimitedError),
        ("ServiceUnavailableException", 503, ProviderUnavailableError),
        ("ModelTimeoutException", 408, ProviderUnavailableError),
        ("SomethingElse", 502, ProviderUnavailableError),
        ("AccessDeniedException", 403, ProviderConfigError),
        ("ValidationException", 400, ProviderConfigError),
        ("ResourceNotFoundException", 404, ProviderConfigError),
        ("ModelErrorException", 424, ProviderRejectedError),
    ],
)
async def test_errors_are_classified_with_a_fixed_message(
    code: str, status: int, error_type: type[ProviderError]
) -> None:
    judge, stubber = _stubbed()
    stubber.add_client_error(
        "converse",
        service_error_code=code,
        service_message=f"echo of the request: {SECRET_TEXT}",
        http_status_code=status,
    )

    with pytest.raises(error_type, match=f"{NOVA_LITE_MODEL_ID} failed \\({code}\\)$") as caught:
        await judge.judge(_request())

    assert type(caught.value) is error_type
    assert SECRET_TEXT not in str(caught.value)
    assert caught.value.__cause__ is None


@pytest.mark.parametrize(
    "error",
    [
        ReadTimeoutError(endpoint_url="https://bedrock-runtime.example/model/x/converse"),
        ConnectTimeoutError(endpoint_url="https://bedrock-runtime.example/model/x/converse"),
    ],
    ids=["read-timeout", "connect-timeout"],
)
async def test_timeouts_are_unavailable(error: Exception) -> None:
    with pytest.raises(ProviderUnavailableError, match="could not be reached") as caught:
        await _judge_with(_Client(error=error)).judge(_request())

    assert caught.value.retryable
    assert "bedrock-runtime.example" not in str(caught.value)


async def test_missing_credentials_are_a_config_error() -> None:
    with pytest.raises(ProviderConfigError, match="credentials or region missing"):
        await _judge_with(_Client(error=NoCredentialsError())).judge(_request())


@pytest.mark.parametrize(
    "error",
    [
        TokenRetrievalError(provider="sso", error_msg="Token has expired and refresh failed"),
        SSOTokenLoadError(error_msg="no cached token"),
        UnauthorizedSSOTokenError(),
        CredentialRetrievalError(provider="container-role", error_msg="refused"),
    ],
)
async def test_expired_or_unloadable_credentials_are_a_config_error_not_retried(
    error: Exception,
) -> None:
    # Retrying won't renew a sign-in: the job must fail at once, not wait out its attempts.
    with pytest.raises(ProviderConfigError, match="credentials missing or expired") as raised:
        await _judge_with(_Client(error=error)).judge(_request())

    assert raised.value.retryable is False


async def test_a_response_without_message_content_is_an_output_error() -> None:
    body = {"output": {"message": {"role": "assistant"}}, "stopReason": "end_turn"}

    with pytest.raises(ProviderOutputError, match="no message content"):
        await _judge_with(_Client(body=body)).judge(_request())


async def test_a_response_with_no_output_is_an_output_error() -> None:
    with pytest.raises(ProviderOutputError, match="no message content"):
        await _judge_with(_Client(body={"stopReason": "end_turn"})).judge(_request())


def test_unsupported_model_id_is_a_config_error() -> None:
    client = boto3.Session().client("bedrock-runtime", region_name="us-east-1")

    with pytest.raises(ProviderConfigError, match="not a supported model"):
        BedrockTextJudge(
            model_id="anthropic.claude-3",
            client=client,
            max_output_tokens=500,
            call_budget_seconds=CALL_BUDGET_SECONDS,
        )


def test_every_supported_model_builds_an_adapter_and_repr_shows_only_the_model_id() -> None:
    client = boto3.Session().client("bedrock-runtime", region_name="us-east-1")

    for model_id in SUPPORTED_BEDROCK_JUDGE_MODEL_IDS:
        judge = BedrockTextJudge(
            model_id=model_id,
            client=client,
            max_output_tokens=500,
            call_budget_seconds=CALL_BUDGET_SECONDS,
        )
        assert repr(judge) == f"BedrockTextJudge(model_id='{model_id}')"
