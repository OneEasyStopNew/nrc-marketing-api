# Loads the Higgsfield key from AWS Secrets Manager at worker startup (staging and production).

import json
import logging
from typing import TYPE_CHECKING, Final

import boto3
from botocore.exceptions import BotoCoreError, ClientError
from pydantic import SecretStr

from app.core.config import Settings

# Type stubs are a dev dependency only; never import them at runtime.
if TYPE_CHECKING:
    from mypy_boto3_secretsmanager import SecretsManagerClient

logger = logging.getLogger(__name__)

# The JSON key to read, then the legacy name ops used first (until the secret is renamed).
SECRET_JSON_KEYS: Final = ("HIGGSFIELD_API_KEY", "HF_KEY")


class ProviderSecretError(RuntimeError):
    # Messages name the secret id and JSON keys, never the secret's content.
    pass


def with_higgsfield_key(settings: Settings, client: SecretsManagerClient | None = None) -> Settings:
    """Return settings that hold the Higgsfield key, reading it from Secrets Manager if needed.

    Only when HIGGSFIELD_SECRET_ID is set and HIGGSFIELD_API_KEY is not. The key stays in memory
    (a copy of the frozen settings); it is never written anywhere or logged. On servers the
    default credential chain (the pod's IAM role) authorizes the read; no access keys are used.

    Raises:
        ProviderSecretError: the secret cannot be read, is not JSON, or holds neither key.
    """
    secret_id = settings.higgsfield_secret_id
    # An empty HIGGSFIELD_API_KEY= line in .env counts as not set.
    key = settings.higgsfield_api_key
    if not secret_id or (key is not None and key.get_secret_value()):
        return settings
    if client is None:
        client = boto3.Session().client("secretsmanager", region_name=settings.aws_region)
    try:
        response = client.get_secret_value(SecretId=secret_id)
    except BotoCoreError, ClientError:
        # Not chained: botocore messages can include request details.
        raise ProviderSecretError(f"Could not read secret {secret_id}.") from None
    try:
        data = json.loads(response.get("SecretString") or "")
    except ValueError:
        raise ProviderSecretError(f"Secret {secret_id} is not a JSON object.") from None
    if not isinstance(data, dict):
        raise ProviderSecretError(f"Secret {secret_id} is not a JSON object.")
    for name in SECRET_JSON_KEYS:
        value = data.get(name)
        if isinstance(value, str) and value:
            logger.info("higgsfield key loaded from secret=%s json_key=%s", secret_id, name)
            return settings.model_copy(update={"higgsfield_api_key": SecretStr(value)})
    raise ProviderSecretError(f"Secret {secret_id} has no {' or '.join(SECRET_JSON_KEYS)} entry.")
