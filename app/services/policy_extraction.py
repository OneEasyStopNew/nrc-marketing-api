# Clause extraction from a policy source file (NRC-4). Pure functions: no database, no storage.
# Run only by the policy worker: pypdf parses untrusted files, so the API process never imports
# this module (tests/test_runtime_imports.py).
#
# A clause is a paragraph (text between blank lines) kept exactly as extracted: the AC7 check
# compares exact wording against clause text, so no word is ever rewritten. Only line endings
# become "\n", the ends of the paragraph are stripped, and NUL (which Postgres cannot store) is
# dropped.

import hashlib
import io
import re
from dataclasses import dataclass
from enum import StrEnum
from typing import Final

from pypdf import PdfReader

from app.core.errors import AppError

PDF_TYPE: Final = "application/pdf"
TEXT_TYPES: Final = frozenset({"text/plain", "text/markdown"})
MARKDOWN_TYPE: Final = "text/markdown"
_SECTION_MAX_CHARS: Final = 200  # policy_source_clauses.section column size
_PARAGRAPH_BREAK: Final = re.compile(r"\n\s*\n")
# Section headings. A Markdown heading line (# .. ######) is one in Markdown files. A numbered
# heading ("3", "3.2", "3.2." then a title) is a paragraph of one short line that does not end like
# a sentence, so a one-line numbered clause ("3.2 Claims must be true.") stays a clause. Both set
# the section of the clauses after them and are not clauses themselves.
_MARKDOWN_HEADING: Final = re.compile(r"^#{1,6}\s+(\S.*)$")
_NUMBERED_HEADING: Final = re.compile(r"^\d+(\.\d+)*\.?\s+\S.*$")
_NUMBERED_HEADING_MAX_CHARS: Final = 120
_SENTENCE_END: Final = (".", ";", ":", ",", "!", "?")


class ExtractionCode(StrEnum):
    # Stored in policy_sources.extraction_error_code; the first four are permanent failures of the
    # file itself, the rest are the worker's.
    UNREADABLE = "SOURCE_UNREADABLE"
    EMPTY = "SOURCE_EMPTY"
    TOO_MANY_PAGES = "SOURCE_TOO_MANY_PAGES"
    TOO_MANY_CLAUSES = "SOURCE_TOO_MANY_CLAUSES"
    STORAGE_UNAVAILABLE = "SOURCE_STORAGE_UNAVAILABLE"
    TIMEOUT = "SOURCE_EXTRACTION_TIMEOUT"
    FAILED = "SOURCE_EXTRACTION_FAILED"


class ExtractionError(AppError):
    # The file cannot be turned into clauses; failure_code says why (an ExtractionCode value).
    code = "POLICY_EXTRACTION_FAILED"
    default_message = "Clauses could not be extracted from this source."

    def __init__(self, failure_code: ExtractionCode, message: str) -> None:
        super().__init__(message)
        self.failure_code = failure_code


@dataclass(frozen=True, slots=True)
class ExtractedClause:
    clause_ref: str
    section: str | None
    page: int | None
    text: str
    checksum_sha256: str


@dataclass(frozen=True, slots=True)
class Extraction:
    # None for a text file: it has no pages.
    page_count: int | None
    clauses: tuple[ExtractedClause, ...]


def extract_clauses(
    content: bytes, mime_type: str, *, max_pages: int, max_clauses: int
) -> Extraction:
    if mime_type == PDF_TYPE:
        page_count, pages = _pdf_pages(content, max_pages)
    elif mime_type in TEXT_TYPES:
        page_count, pages = None, [_decode(content)]
    else:
        raise ExtractionError(ExtractionCode.UNREADABLE, "This file type cannot be read.")
    markdown = mime_type == MARKDOWN_TYPE
    clauses: list[ExtractedClause] = []
    section: str | None = None
    for index, page_text in enumerate(pages, start=1):
        page = index if page_count is not None else None
        for kind, text in _blocks(page_text, markdown):
            if kind == "heading":
                section = text[:_SECTION_MAX_CHARS]
                continue
            if len(clauses) >= max_clauses:
                raise ExtractionError(
                    ExtractionCode.TOO_MANY_CLAUSES,
                    f"The source has more than {max_clauses} clauses.",
                )
            ref = f"c{len(clauses) + 1}" if page is None else f"p{page}-c{len(clauses) + 1}"
            clauses.append(
                ExtractedClause(
                    ref, section, page, text, hashlib.sha256(text.encode("utf-8")).hexdigest()
                )
            )
    if not clauses:
        raise ExtractionError(ExtractionCode.EMPTY, "The source has no extractable text.")
    return Extraction(page_count, tuple(clauses))


def _pdf_pages(content: bytes, max_pages: int) -> tuple[int, list[str]]:
    # pypdf raises assorted exceptions on damaged files (not only PdfReadError), and this is the
    # one place that parses untrusted input: every failure here means "unreadable".
    try:
        reader = PdfReader(io.BytesIO(content))
        if reader.is_encrypted:
            raise ExtractionError(ExtractionCode.UNREADABLE, "The PDF is encrypted.")
        count = len(reader.pages)
        if count > max_pages:
            raise ExtractionError(
                ExtractionCode.TOO_MANY_PAGES, f"The PDF has more than {max_pages} pages."
            )
        # Layout mode keeps the blank line where a paragraph ends; the default mode joins every
        # line of a page, which would make each page one clause.
        return count, [page.extract_text(extraction_mode="layout") for page in reader.pages]
    except ExtractionError:
        raise
    except Exception:
        raise ExtractionError(ExtractionCode.UNREADABLE, "The PDF cannot be read.") from None


def _decode(content: bytes) -> str:
    try:
        return content.decode("utf-8-sig")
    except UnicodeDecodeError:
        raise ExtractionError(ExtractionCode.UNREADABLE, "The file is not UTF-8 text.") from None


def _blocks(page_text: str, markdown: bool) -> list[tuple[str, str]]:
    # ("heading" | "clause", text) in reading order.
    normal = page_text.replace("\x00", "").replace("\r\n", "\n").replace("\r", "\n")
    found: list[tuple[str, str]] = []
    for paragraph in _PARAGRAPH_BREAK.split(normal):
        lines = paragraph.split("\n")
        pending: list[str] = []
        for line in lines:
            heading = _MARKDOWN_HEADING.match(line.strip()) if markdown else None
            if heading is None:
                pending.append(line)
                continue
            # A Markdown heading ends the paragraph before it, even without a blank line.
            if text := "\n".join(pending).strip():
                found.append(("clause", text))
            found.append(("heading", heading.group(1).strip()))
            pending = []
        text = "\n".join(pending).strip()
        if not text:
            continue
        numbered = len(pending) == len(lines) and _is_numbered_heading(text)
        found.append(("heading" if numbered else "clause", text))
    return found


def _is_numbered_heading(text: str) -> bool:
    return (
        "\n" not in text
        and len(text) <= _NUMBERED_HEADING_MAX_CHARS
        and _NUMBERED_HEADING.match(text) is not None
        and not text.endswith(_SENTENCE_END)
    )
