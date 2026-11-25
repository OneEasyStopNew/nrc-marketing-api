# Shared schema bases and field types.

import unicodedata
from collections.abc import Callable
from typing import Annotated

from pydantic import AfterValidator, BaseModel, ConfigDict, Field, HttpUrl, UrlConstraints

MAX_PROMPT_CHARS = 4000
MAX_CAPTION_CHARS = 2200  # Instagram's limit, the stricter of the two first-pass channels
# Text rendered inside the image. The brief and this text must fit the provider's 1,448 limit.
MAX_IMAGE_TEXT_CHARS = 300
MAX_IMAGE_TEXT_LINES = 6
# Ideogram's limit for the prompt plus the image text; the generation gate (NRC-7) enforces it.
MAX_GENERATION_TEXT_CHARS = 1448
# Brief fields (NRC-2). Lengths are checked after NFC, which can make text longer.
MAX_HEADLINE_CHARS = 150
MAX_AUDIENCE_CHARS = 500
MAX_VISUAL_INSTRUCTIONS_CHARS = 2000
MAX_DESTINATION_URL_CHARS = 2048
MAX_SITE_CHARS = 300


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


def _normalize_image_text(value: str) -> str:
    # One line per rendered line: CRLF becomes LF, trailing spaces and a final newline go,
    # and blank lines in between are refused.
    value = unicodedata.normalize("NFC", value).replace("\r\n", "\n").replace("\r", "\n")
    lines = [line.rstrip() for line in value.strip("\n").split("\n")]
    if any(not line.strip() for line in lines):
        raise ValueError("must not contain blank lines")
    if len(lines) > MAX_IMAGE_TEXT_LINES:
        raise ValueError(f"must have at most {MAX_IMAGE_TEXT_LINES} lines")
    return "\n".join(lines)


def _brief_text(max_chars: int, *, multiline: bool) -> Callable[[str], str]:
    # NFC first, then the length, so normalizing can't push a value past its limit.
    def check(value: str) -> str:
        value = unicodedata.normalize("NFC", value).replace("\r\n", "\n").replace("\r", "\n")
        value = value.strip()
        if not value:
            raise ValueError("must not be blank")
        if len(value) > max_chars:
            raise ValueError(f"must be at most {max_chars} characters")
        allowed = {"\n"} if multiline else set()
        if any(unicodedata.category(c) == "Cc" and c not in allowed for c in value):
            raise ValueError(
                "must not contain control characters"
                if multiline
                else "must be one line without control characters"
            )
        return value

    return check


def _no_credentials(value: HttpUrl) -> HttpUrl:
    if value.username or value.password:
        raise ValueError("must not contain a user name or password")
    return value


HeadlineText = Annotated[
    str,
    Field(max_length=MAX_HEADLINE_CHARS, description="Link headline under the image, one line."),
    AfterValidator(_brief_text(MAX_HEADLINE_CHARS, multiline=False)),
]
AudienceText = Annotated[
    str,
    Field(max_length=MAX_AUDIENCE_CHARS, description="Who the post is for, one line."),
    AfterValidator(_brief_text(MAX_AUDIENCE_CHARS, multiline=False)),
]
VisualInstructionsText = Annotated[
    str,
    Field(max_length=MAX_VISUAL_INSTRUCTIONS_CHARS),
    AfterValidator(_brief_text(MAX_VISUAL_INSTRUCTIONS_CHARS, multiline=True)),
]
SiteName = Annotated[
    str,
    Field(max_length=MAX_SITE_CHARS, description="One of the study snapshot's sites, exactly."),
    AfterValidator(_brief_text(MAX_SITE_CHARS, multiline=False)),
]
DestinationUrl = Annotated[
    HttpUrl,
    UrlConstraints(max_length=MAX_DESTINATION_URL_CHARS, allowed_schemes=["https"]),
    AfterValidator(_no_credentials),
]

PromptText = Annotated[
    str, Field(min_length=1, max_length=MAX_PROMPT_CHARS), AfterValidator(_normalize_text)
]
CaptionText = Annotated[
    str, Field(min_length=1, max_length=MAX_CAPTION_CHARS), AfterValidator(_normalize_text)
]
ImageText = Annotated[
    str,
    Field(
        min_length=1,
        max_length=MAX_IMAGE_TEXT_CHARS,
        description="Exact text to render in the image, one line per line.",
    ),
    AfterValidator(_normalize_image_text),
]
