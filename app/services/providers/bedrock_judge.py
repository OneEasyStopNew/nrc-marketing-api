"""Amazon Bedrock text judge adapter (NRC-7), through the Converse API.

UNVERIFIED AGAINST THE LIVE SERVICE (no Bedrock access yet). The request and response shapes
below come from AWS documentation and the botocore service model, not from a real call. The
first live run must confirm what each candidate model returns (reasoning blocks, stop reasons,
whether it accepts the plain request below).

Runs only in workers and owner scripts. Access comes from the default credential chain (the IAM
role on servers), never from keys in settings, and not from AWS_PROFILE, which locally names the
SeaweedFS-only profile.
"""

import asyncio
import time
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, Final

import boto3
from botocore.config import Config
from botocore.exceptions import BotoCoreError, ClientError

from app.core.config import Settings
from app.services.providers.aws_errors import (
    EXPIRED_CREDENTIAL_ERRORS,
    MISSING_CREDENTIAL_ERRORS,
    is_transient_credential_error,
)
from app.services.providers.base import (
    ProviderConfigError,
    ProviderError,
    ProviderOutputError,
    ProviderRateLimitedError,
    ProviderRejectedError,
    ProviderUnavailableError,
)
from app.services.providers.text_judge import (
    SUPPORTED_BEDROCK_JUDGE_MODEL_IDS,
    JudgeReply,
    JudgeRequest,
    judge_call_budget_seconds,
)

# Type stubs are a dev dependency only; never import them at runtime.
if TYPE_CHECKING:
    from mypy_boto3_bedrock_runtime import BedrockRuntimeClient

# Error codes a later attempt can succeed after (the client's own retries are already spent), and
# the ones that mean the deployment is set up wrongly. ThrottlingException is classified first.
_THROTTLED_CODE: Final = "ThrottlingException"
_UNAVAILABLE_CODES: Final = frozenset(
    {
        "ServiceUnavailableException",
        "ModelNotReadyException",
        "ModelTimeoutException",
        "InternalServerException",
    }
)
_CONFIG_CODES: Final = frozenset(
    {"AccessDeniedException", "ValidationException", "ResourceNotFoundException"}
)
_TOO_MANY_REQUESTS_STATUS: Final = 429
_SERVER_ERROR_STATUS: Final = 500


class BedrockTextJudge:
    is_simulated = False

    def __init__(
        self,
        *,
        model_id: str,
        client: BedrockRuntimeClient,
        max_output_tokens: int,
        call_budget_seconds: float,
    ) -> None:
        if model_id not in SUPPORTED_BEDROCK_JUDGE_MODEL_IDS:
            raise ProviderConfigError(f"Judge model {model_id} is not a supported model")
        self.model_id = model_id
        self._client = client
        self._max_output_tokens = max_output_tokens
        self._call_budget_seconds = call_budget_seconds

    @classmethod
    def from_settings(cls, settings: Settings) -> BedrockTextJudge:
        model_id = settings.bedrock_validation_model_id
        if not model_id:
            raise ProviderConfigError("The Bedrock judge is not configured; missing model id")
        client_config = Config(
            connect_timeout=settings.validation_model_timeout_seconds,
            read_timeout=settings.validation_model_timeout_seconds,
            # botocore's max_attempts counts retries after the first attempt here, as for S3.
            retries={"max_attempts": settings.bedrock_max_retries, "mode": "standard"},
        )
        client = boto3.Session().client(
            "bedrock-runtime", region_name=settings.aws_region, config=client_config
        )
        return cls(
            model_id=model_id,
            client=client,
            max_output_tokens=settings.validation_model_max_output_tokens,
            call_budget_seconds=judge_call_budget_seconds(settings),
        )

    def __repr__(self) -> str:
        return f"BedrockTextJudge(model_id={self.model_id!r})"

    async def judge(self, request: JudgeRequest) -> JudgeReply:
        started = time.monotonic()
        try:
            # The worker checks this budget against its lease. The thread can't be stopped: a
            # call past the budget finishes unseen and the job retries later.
            async with asyncio.timeout(self._call_budget_seconds):
                reply = await asyncio.to_thread(self._converse, request)
        except TimeoutError:
            raise ProviderUnavailableError(
                f"Bedrock judge model {self.model_id} did not answer in time"
            ) from None
        latency_ms = round((time.monotonic() - started) * 1000)
        return JudgeReply(
            raw=reply.raw,
            model_id=self.model_id,
            stop_reason=reply.stop_reason,
            input_tokens=reply.input_tokens,
            output_tokens=reply.output_tokens,
            latency_ms=latency_ms,
            is_simulated=False,
        )

    def _converse(self, request: JudgeRequest) -> JudgeReply:
        # Runs in a worker thread. Errors carry a fixed message with the model id and the error
        # code only: botocore's own text can echo the request, which holds the text under review.
        try:
            # No toolConfig (the answer is plain JSON text) and no temperature: it is unverified
            # whether the thinking-enabled models accept one.
            response = self._client.converse(
                modelId=self.model_id,
                system=[{"text": request.system}],
                messages=[{"role": "user", "content": [{"text": request.user}]}],
                inferenceConfig={
                    "maxTokens": min(request.max_output_tokens, self._max_output_tokens)
                },
            )
        except ClientError as error:
            raise self._classify(error) from None
        except MISSING_CREDENTIAL_ERRORS:
            raise ProviderConfigError(
                f"Bedrock judge model {self.model_id}: AWS credentials or region missing"
            ) from None
        except EXPIRED_CREDENTIAL_ERRORS as error:
            if is_transient_credential_error(error):
                raise ProviderUnavailableError(
                    f"Bedrock judge model {self.model_id}: AWS credentials temporarily unavailable"
                ) from None
            raise ProviderConfigError(
                f"Bedrock judge model {self.model_id}: AWS credentials missing or expired"
            ) from None
        except BotoCoreError:
            # Timeouts and connection failures; the client's retries are already spent.
            raise ProviderUnavailableError(
                f"Bedrock judge model {self.model_id} could not be reached"
            ) from None
        return self._parse(response)

    def _parse(self, response: Mapping[str, Any]) -> JudgeReply:
        content = response.get("output", {}).get("message", {}).get("content")
        if not isinstance(content, list):
            raise ProviderOutputError("The judge response has no message content")
        # Reasoning blocks have no "text" and are skipped; the answer is every text block joined.
        raw = "".join(
            block["text"]
            for block in content
            if isinstance(block, dict) and isinstance(block.get("text"), str)
        )
        usage = response.get("usage", {})
        return JudgeReply(
            raw=raw,
            model_id=self.model_id,
            stop_reason=response.get("stopReason"),
            input_tokens=usage.get("inputTokens"),
            output_tokens=usage.get("outputTokens"),
            latency_ms=0,
            is_simulated=False,
        )

    def _classify(self, error: ClientError) -> ProviderError:
        code = str(error.response.get("Error", {}).get("Code", "Unknown"))
        status = int(error.response.get("ResponseMetadata", {}).get("HTTPStatusCode", 0))
        message = f"Bedrock judge model {self.model_id} failed ({code})"
        if code == _THROTTLED_CODE or status == _TOO_MANY_REQUESTS_STATUS:
            return ProviderRateLimitedError(message)
        if code in _UNAVAILABLE_CODES or status >= _SERVER_ERROR_STATUS:
            return ProviderUnavailableError(message)
        if code in _CONFIG_CODES:
            return ProviderConfigError(message)
        # Any other client error: the request itself was refused, so repeating it will not help.
        return ProviderRejectedError(message)
