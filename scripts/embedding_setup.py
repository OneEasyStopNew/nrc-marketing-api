# Settings and the embedder for owner scripts (seed, index build, retrieval evaluation).

from pydantic import ValidationError
from pydantic_settings import BaseSettings

from app.core.config import EmbeddingSettings
from app.services.providers.base import ProviderConfigError
from app.services.providers.embeddings import EmbeddingProvider, embedding_provider_from_settings


def load_settings[T: BaseSettings](settings_class: type[T]) -> T:
    """Load a settings class, exiting with its error messages only.

    pydantic's error would also print the values it read, which include database URLs.
    """
    try:
        return settings_class()
    except ValidationError as error:
        raise SystemExit("; ".join(str(item["msg"]) for item in error.errors())) from None


def load_embedder(settings: EmbeddingSettings | None = None) -> EmbeddingProvider | None:
    """The configured embedder, or None when EMBEDDING_MODE is off.

    Loads only EmbeddingSettings unless given settings, so the scripts need none of the API's
    settings (Cognito, DATABASE_URL).
    """
    try:
        return embedding_provider_from_settings(settings or load_settings(EmbeddingSettings))
    except ProviderConfigError as error:
        raise SystemExit(str(error)) from None
