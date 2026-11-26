"""Contract between policy indexing and retrieval (NRC-5) and the embedding models.

No boto3 here: the API process imports this module (settings checks, the contract) but never the
Bedrock adapter, which the factory imports only for live mode.
"""

import hashlib
import math
import re
from collections.abc import Sequence
from enum import StrEnum
from typing import Final, Protocol

from app.core.config import EmbeddingSettings
from app.models.policy_graph import EMBEDDING_DIMENSIONS
from app.services.providers.base import ProviderConfigError, ProviderError

# The models the evaluation compares (prices in NRC_Validation_Flow_Reference.md 13.3). The
# adapter's PINNED_MODELS has one entry for each; a test keeps the two in step.
TITAN_V2_MODEL_ID: Final = "amazon.titan-embed-text-v2:0"
COHERE_MULTILINGUAL_V3_MODEL_ID: Final = "cohere.embed-multilingual-v3"
COHERE_V4_MODEL_ID: Final = "cohere.embed-v4:0"
SUPPORTED_BEDROCK_EMBEDDING_MODEL_IDS: Final = frozenset(
    {TITAN_V2_MODEL_ID, COHERE_MULTILINGUAL_V3_MODEL_ID, COHERE_V4_MODEL_ID}
)

SIMULATED_MODEL_ID: Final = "simulator:token-hash-v1"
_TOKEN = re.compile(r"\w+")


class EmbeddingPurpose(StrEnum):
    # Cohere embeds a stored text and a search text differently; Titan does not.
    DOCUMENT = "document"
    QUERY = "query"


class EmbeddingResponseError(ProviderError):
    # The model's answer is not usable: wrong count or dimensions, a missing field, not numbers.
    # Retrying the same request will not fix it.
    code = "EMBEDDING_RESPONSE_INVALID"


class EmbeddingProvider(Protocol):
    """One embedding model. Implementations hold no per-call state."""

    model_id: str

    async def embed(self, texts: Sequence[str], purpose: EmbeddingPurpose) -> list[list[float]]:
        """One vector of EMBEDDING_DIMENSIONS floats per text, in the order given.

        Raises a ProviderError subclass on failure.
        """
        ...


class SimulatedEmbeddingProvider:
    """SIMULATED embeddings for local runs and tests. Makes no network calls.

    Lower-cased word tokens are hashed into signed buckets, so texts that share words are close
    and unrelated texts are not. The vectors say nothing about meaning (no synonyms, no other
    language): they prove the plumbing, never retrieval quality. Never counts as live evidence.
    """

    model_id = SIMULATED_MODEL_ID

    async def embed(self, texts: Sequence[str], purpose: EmbeddingPurpose) -> list[list[float]]:
        # Query and document vectors are the same here, so purpose is accepted and ignored.
        return [_simulated_vector(text) for text in texts]

    def __repr__(self) -> str:
        return f"SimulatedEmbeddingProvider(model_id={self.model_id!r})"


def _simulated_vector(text: str) -> list[float]:
    vector = [0.0] * EMBEDDING_DIMENSIONS
    for token in _TOKEN.findall(text.lower()):
        # hashlib, not hash(): Python's string hash changes per process, which would make the
        # stored vectors differ from run to run.
        digest = hashlib.sha256(token.encode()).digest()
        bucket = int.from_bytes(digest[:4]) % EMBEDDING_DIMENSIONS
        vector[bucket] += 1.0 if digest[4] & 1 else -1.0
    norm = math.sqrt(sum(value * value for value in vector))
    if norm == 0:
        # No tokens (or buckets that cancelled out): a fixed unit vector, never NaN.
        vector[0] = 1.0
        return vector
    return [value / norm for value in vector]


def missing_live_embedding_settings(settings: EmbeddingSettings) -> list[str]:
    # What live mode lacks for embeddings; messages name settings, never values. Empty for the
    # other modes.
    if settings.embedding_mode != "live":
        return []
    model_id = settings.bedrock_embedding_model_id
    if not model_id:
        return ["missing BEDROCK_EMBEDDING_MODEL_ID"]
    if model_id not in SUPPORTED_BEDROCK_EMBEDDING_MODEL_IDS:
        # The adapter only knows the pinned request shapes, so start-up refuses anything else.
        supported = ", ".join(sorted(SUPPORTED_BEDROCK_EMBEDDING_MODEL_IDS))
        return [f"BEDROCK_EMBEDDING_MODEL_ID is not a supported model (use one of: {supported})"]
    return []


def embedding_provider_from_settings(settings: EmbeddingSettings) -> EmbeddingProvider | None:
    """The provider for EMBEDDING_MODE; None when it is off (no vectors, lexical search only).

    Raises:
        ProviderConfigError: live mode lacks a setting. It never falls back to the simulator.
    """
    if settings.embedding_mode == "off":
        return None
    if settings.embedding_mode == "simulator":
        return SimulatedEmbeddingProvider()
    problems = missing_live_embedding_settings(settings)
    if problems:
        raise ProviderConfigError(f"Live embeddings are not configured: {'; '.join(problems)}")
    # Imported here so the API process never loads boto3 or the adapter.
    from app.services.providers.bedrock_embeddings import BedrockEmbeddingProvider

    return BedrockEmbeddingProvider.from_settings(settings)
