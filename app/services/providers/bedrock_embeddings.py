"""Amazon Bedrock embedding adapter (NRC-5): Titan Text Embeddings V2 and two Cohere models.

UNVERIFIED AGAINST THE LIVE SERVICE (follow-up F30: no Bedrock access yet). Every request and
response shape below comes from AWS documentation, not from a real call. Each model's shape lives
in exactly one place (its _build_* and _parse_* functions and its PINNED_MODELS entry), so a
correction after the first live run touches one function. The first live run must also confirm
that each model returns 1024 dimensions; a mismatch fails as EmbeddingResponseError, never as a
silently stored short vector.

Runs only in workers and owner scripts. Access comes from the default credential chain (the IAM
role on servers), never from keys in settings, and not from AWS_PROFILE, which locally names the
SeaweedFS-only profile.
"""

import asyncio
import json
import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import TYPE_CHECKING, Final

import boto3
from botocore.config import Config
from botocore.exceptions import BotoCoreError, ClientError

from app.core.config import EmbeddingSettings
from app.models.policy_graph import EMBEDDING_DIMENSIONS
from app.services.providers.aws_errors import EXPIRED_CREDENTIAL_ERRORS, MISSING_CREDENTIAL_ERRORS
from app.services.providers.base import (
    ProviderConfigError,
    ProviderError,
    ProviderRejectedError,
    ProviderUnavailableError,
)
from app.services.providers.embeddings import (
    COHERE_MULTILINGUAL_V3_MODEL_ID,
    COHERE_V4_MODEL_ID,
    SUPPORTED_BEDROCK_EMBEDDING_MODEL_IDS,
    TITAN_V2_MODEL_ID,
    EmbeddingPurpose,
    EmbeddingResponseError,
)

# Type stubs are a dev dependency only; never import them at runtime.
if TYPE_CHECKING:
    from mypy_boto3_bedrock_runtime import BedrockRuntimeClient

_JSON: Final = "application/json"

# Error codes Bedrock returns that a later attempt can succeed after (the client's own retries
# have already been spent), and the ones that mean the deployment is set up wrongly.
_UNAVAILABLE_CODES: Final = frozenset(
    {
        "ThrottlingException",
        "ServiceUnavailableException",
        "ModelNotReadyException",
        "ModelTimeoutException",
        "InternalServerException",
    }
)
_CONFIG_CODES: Final = frozenset(
    {"AccessDeniedException", "ValidationException", "ResourceNotFoundException"}
)
_SERVER_ERROR_STATUS: Final = 500


def _document_or_query(purpose: EmbeddingPurpose) -> str:
    return "search_document" if purpose is EmbeddingPurpose.DOCUMENT else "search_query"


# Request builders. Titan embeds one text per call and has no document/query distinction.


def _build_titan(texts: Sequence[str], purpose: EmbeddingPurpose) -> dict[str, object]:
    (text,) = texts
    return {"inputText": text, "dimensions": EMBEDDING_DIMENSIONS, "normalize": True}


def _build_cohere_v3(texts: Sequence[str], purpose: EmbeddingPurpose) -> dict[str, object]:
    return {"texts": list(texts), "input_type": _document_or_query(purpose), "truncate": "END"}


def _build_cohere_v4(texts: Sequence[str], purpose: EmbeddingPurpose) -> dict[str, object]:
    return {
        "texts": list(texts),
        "input_type": _document_or_query(purpose),
        "embedding_types": ["float"],
        "output_dimension": EMBEDDING_DIMENSIONS,
    }


# Response parsers: the vectors as the model returned them, one per input. Shape and numbers are
# checked afterwards by _checked_vectors, the same for every model.


def _field(body: object, name: str) -> object:
    if not isinstance(body, dict) or name not in body:
        raise EmbeddingResponseError(f"The embedding response has no {name!r} field")
    return body[name]


def _parse_titan(body: object) -> object:
    return [_field(body, "embedding")]


def _parse_cohere_v3(body: object) -> object:
    return _field(body, "embeddings")


def _parse_cohere_v4(body: object) -> object:
    # With embedding_types the vectors come back keyed by type.
    return _field(_field(body, "embeddings"), "float")


@dataclass(frozen=True, slots=True)
class EmbeddingModelSpec:
    max_batch: int
    build_request: Callable[[Sequence[str], EmbeddingPurpose], dict[str, object]]
    parse_response: Callable[[object], object]


PINNED_MODELS: Final = MappingProxyType(
    {
        TITAN_V2_MODEL_ID: EmbeddingModelSpec(1, _build_titan, _parse_titan),
        COHERE_MULTILINGUAL_V3_MODEL_ID: EmbeddingModelSpec(96, _build_cohere_v3, _parse_cohere_v3),
        COHERE_V4_MODEL_ID: EmbeddingModelSpec(96, _build_cohere_v4, _parse_cohere_v4),
    }
)
assert PINNED_MODELS.keys() == SUPPORTED_BEDROCK_EMBEDDING_MODEL_IDS  # noqa: S101


def _is_number(value: object) -> bool:
    # bool is an int subclass but never a valid vector component.
    return isinstance(value, int | float) and not isinstance(value, bool) and math.isfinite(value)


def _checked_vectors(raw: object, expected_count: int) -> list[list[float]]:
    if not isinstance(raw, list) or len(raw) != expected_count:
        raise EmbeddingResponseError("The embedding response does not hold one vector per text")
    vectors: list[list[float]] = []
    for vector in raw:
        if not isinstance(vector, list) or len(vector) != EMBEDDING_DIMENSIONS:
            raise EmbeddingResponseError(
                f"An embedding does not have {EMBEDDING_DIMENSIONS} dimensions"
            )
        if not all(_is_number(component) for component in vector):
            raise EmbeddingResponseError("An embedding holds a value that is not a finite number")
        vectors.append([float(component) for component in vector])
    return vectors


class BedrockEmbeddingProvider:
    def __init__(self, *, model_id: str, client: BedrockRuntimeClient) -> None:
        spec = PINNED_MODELS.get(model_id)
        if spec is None:
            raise ProviderConfigError(f"Embedding model {model_id} is not a supported model")
        self.model_id = model_id
        self._spec = spec
        self._client = client

    @classmethod
    def from_settings(cls, settings: EmbeddingSettings) -> BedrockEmbeddingProvider:
        model_id = settings.bedrock_embedding_model_id
        if not model_id:
            raise ProviderConfigError("Bedrock embeddings are not configured; missing model id")
        client_config = Config(
            connect_timeout=settings.bedrock_timeout_seconds,
            read_timeout=settings.bedrock_timeout_seconds,
            # botocore's max_attempts counts retries after the first attempt here, as for S3.
            retries={"max_attempts": settings.bedrock_max_retries, "mode": "standard"},
        )
        client = boto3.Session().client(
            "bedrock-runtime", region_name=settings.aws_region, config=client_config
        )
        return cls(model_id=model_id, client=client)

    def __repr__(self) -> str:
        return f"BedrockEmbeddingProvider(model_id={self.model_id!r})"

    async def embed(self, texts: Sequence[str], purpose: EmbeddingPurpose) -> list[list[float]]:
        vectors: list[list[float]] = []
        for start in range(0, len(texts), self._spec.max_batch):
            batch = texts[start : start + self._spec.max_batch]
            body = self._spec.build_request(batch, purpose)
            response = await asyncio.to_thread(self._invoke, body)
            raw = self._spec.parse_response(response)
            vectors.extend(_checked_vectors(raw, len(batch)))
        return vectors

    def _invoke(self, body: dict[str, object]) -> object:
        # Runs in a worker thread. Errors carry a fixed message with the model id and the error
        # code only: botocore's own text can echo the request, which holds policy text.
        try:
            response = self._client.invoke_model(
                modelId=self.model_id, body=json.dumps(body), contentType=_JSON, accept=_JSON
            )
            payload = response["body"].read()
        except ClientError as error:
            raise self._classify(error) from None
        except MISSING_CREDENTIAL_ERRORS:
            raise ProviderConfigError(
                f"Bedrock embedding model {self.model_id}: AWS credentials or region missing"
            ) from None
        except EXPIRED_CREDENTIAL_ERRORS:
            raise ProviderConfigError(
                f"Bedrock embedding model {self.model_id}: AWS credentials missing or expired"
            ) from None
        except BotoCoreError:
            # Timeouts and connection failures; the client's retries are already spent.
            raise ProviderUnavailableError(
                f"Bedrock embedding model {self.model_id} could not be reached"
            ) from None
        try:
            return json.loads(payload)
        except ValueError:
            raise EmbeddingResponseError("The embedding response is not valid JSON") from None

    def _classify(self, error: ClientError) -> ProviderError:
        code = str(error.response.get("Error", {}).get("Code", "Unknown"))
        status = int(error.response.get("ResponseMetadata", {}).get("HTTPStatusCode", 0))
        message = f"Bedrock embedding model {self.model_id} failed ({code})"
        if code in _UNAVAILABLE_CODES or status >= _SERVER_ERROR_STATUS:
            return ProviderUnavailableError(message)
        if code in _CONFIG_CODES:
            return ProviderConfigError(message)
        # Any other client error: the request itself was refused, so repeating it will not help.
        return ProviderRejectedError(message)
