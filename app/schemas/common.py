# Shared schema bases and field types.

import unicodedata
from typing import Annotated

from pydantic import AfterValidator, BaseModel, ConfigDict, Field

MAX_PROMPT_CHARS = 4000
MAX_CAPTION_CHARS = 2200  # Instagram's limit, the stricter of the two first-pass channels


class ApiModel(BaseModel):
    # Base for response bodies.

    model_config = ConfigDict(from_attributes=True)


class RequestModel(BaseModel):
    # Base for request bodies: unknown fields are rejected, not silently dropped.

    model_config = ConfigDict(extra="forbid")


class ErrorBody(ApiModel):
    """Every error response has this shape."""

    code: str = Field(examples=["VERSION_CONFLICT"])
    message: str
    request_id: str | None
    details: dict[str, object] | None = None
    retryable: bool = False


def _normalize_text(value: str) -> str:
    # NFC keeps stored text, and the code-point offsets reported against it, stable.
    value = unicodedata.normalize("NFC", value)
    if not value.strip():
        raise ValueError("must not be blank")
    return value


PromptText = Annotated[
    str, Field(min_length=1, max_length=MAX_PROMPT_CHARS), AfterValidator(_normalize_text)
]
CaptionText = Annotated[
    str, Field(min_length=1, max_length=MAX_CAPTION_CHARS), AfterValidator(_normalize_text)
]
