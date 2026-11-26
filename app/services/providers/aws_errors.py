"""botocore credential errors the Bedrock adapters (judge, embeddings) map to ProviderConfigError.

Both are configuration problems that a retry won't fix, so a job fails at once instead of
waiting out its attempts. Kept in one place so the adapters can't drift apart.
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
