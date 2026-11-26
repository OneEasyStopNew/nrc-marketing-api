"""botocore credential errors the Bedrock adapters (judge, embeddings) map to ProviderConfigError.

Both are configuration problems that a retry won't fix, so a job fails at once instead of
waiting out its attempts. The exception is a container credential agent that can't be reached
(EKS Pod Identity, ECS task roles): that is a short outage, retried like an unreachable endpoint.
Kept in one place so the adapters can't drift apart.
"""

from typing import Final

from botocore.exceptions import (
    CredentialRetrievalError,
    NoCredentialsError,
    NoRegionError,
    PartialCredentialsError,
    SSOError,
    TokenRetrievalError,
)

# Credentials or region not set up.
MISSING_CREDENTIAL_ERRORS: Final = (NoCredentialsError, PartialCredentialsError, NoRegionError)
# An expired sign-in, or a role or container credential that can't be retrieved.
EXPIRED_CREDENTIAL_ERRORS: Final = (SSOError, TokenRetrievalError, CredentialRetrievalError)
# botocore's ContainerProvider.METHOD: credentials from a local agent (EKS Pod Identity, ECS).
CONTAINER_CREDENTIAL_PROVIDER: Final = "container-role"


def is_transient_credential_error(error: Exception) -> bool:
    """True for a credential agent that didn't answer, which a later attempt may reach."""
    return (
        isinstance(error, CredentialRetrievalError)
        and error.kwargs.get("provider") == CONTAINER_CREDENTIAL_PROVIDER
    )
