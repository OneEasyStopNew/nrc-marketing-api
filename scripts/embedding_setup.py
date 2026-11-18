# The embedder for owner scripts (seed, index build), from EMBEDDING_MODE.

from pydantic import ValidationError

from app.core.config import EmbeddingSettings
from app.services.providers.base import ProviderConfigError
from app.services.providers.embeddings import EmbeddingProvider, embedding_provider_from_settings


def load_embedder() -> EmbeddingProvider | None:
    """The configured embedder, or None when EMBEDDING_MODE is off.

    Loads only EmbeddingSettings, so the scripts need none of the API's settings. Invalid
    settings exit with their messages only: pydantic's error would also print the values it
    read, which include database URLs.
    """
    try:
        return embedding_provider_from_settings(EmbeddingSettings())
    except ValidationError as error:
        raise SystemExit("; ".join(str(item["msg"]) for item in error.errors())) from None
    except ProviderConfigError as error:
        raise SystemExit(str(error)) from None
