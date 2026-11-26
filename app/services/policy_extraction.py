# Clause extraction from a policy source file (NRC-4). Pure functions: no database, no storage.
# Run only by the policy worker: pypdf parses untrusted files, so the API process never imports
# this module (tests/test_runtime_imports.py).
#
# A clause is a paragraph (text between blank lines) kept exactly as extracted: the AC7 check
# compares exact wording against clause text, so no word is ever rewritten. Only line endings
# become "\n", the ends of the paragraph are stripped, and NUL (which Postgres cannot store) is
# dropped.

import asyncio
import hashlib
import io
import multiprocessing
import re
from collections.abc import Callable, Generator, Iterable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from multiprocessing.connection import Connection
from typing import Final

from pypdf import PdfReader

from app.core.errors import AppError
from app.models.enums import ExtractionCode

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
    content: bytes, mime_type: str, *, max_pages: int, max_clauses: int, max_chars: int
) -> Extraction:
    pages: Iterable[str]
    if mime_type == PDF_TYPE:
        page_count, pages = _pdf_pages(content, max_pages)
    elif mime_type in TEXT_TYPES:
        page_count, pages = None, [_decode(content)]
    else:
        raise ExtractionError(ExtractionCode.UNREADABLE, "This file type cannot be read.")
    markdown = mime_type == MARKDOWN_TYPE
    clauses: list[ExtractedClause] = []
    section: str | None = None
    chars = 0
    # Pages arrive one at a time, so the text cap stops a large source after the page that
    # crosses it instead of after every page's text is in memory.
    for index, page_text in enumerate(pages, start=1):
        chars += len(page_text)
        if chars > max_chars:
            raise ExtractionError(
                ExtractionCode.TOO_MUCH_TEXT,
                f"The source has more than {max_chars} characters of text.",
            )
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


def _pdf_pages(content: bytes, max_pages: int) -> tuple[int, Iterator[str]]:
    # pypdf raises assorted exceptions on damaged files (not only PdfReadError), and this is the
    # one place that parses untrusted input: every failure here means "unreadable".
    with _unreadable_pdf():
        reader = PdfReader(io.BytesIO(content))
        if reader.is_encrypted:
            raise ExtractionError(ExtractionCode.UNREADABLE, "The PDF is encrypted.")
        count = len(reader.pages)
    if count > max_pages:
        raise ExtractionError(
            ExtractionCode.TOO_MANY_PAGES, f"The PDF has more than {max_pages} pages."
        )
    return count, _page_texts(reader)


def _page_texts(reader: PdfReader) -> Iterator[str]:
    # One page's text at a time. Layout mode keeps the blank line where a paragraph ends; the
    # default mode joins every line of a page, which would make each page one clause.
    for page in reader.pages:
        with _unreadable_pdf():
            text = page.extract_text(extraction_mode="layout")
        yield text


@contextmanager
def _unreadable_pdf() -> Generator[None]:
    try:
        yield
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


# ---- Running extraction in a child process
#
# pypdf parses untrusted files and can run long. A thread can't be stopped, so a timed-out one
# would keep parsing and every retry would add another; a child process is killed instead.

type _ChildResult = Extraction | tuple[ExtractionCode, str]
type ChildTarget = Callable[[Connection, bytes, str, int, int, int], None]


def _extract_in_child(
    writer: Connection,
    content: bytes,
    mime_type: str,
    max_pages: int,
    max_clauses: int,
    max_chars: int,
) -> None:
    # ExtractionError takes two arguments, which pickling an exception doesn't keep; send the
    # code and message instead.
    result: _ChildResult
    try:
        result = extract_clauses(
            content, mime_type, max_pages=max_pages, max_clauses=max_clauses, max_chars=max_chars
        )
    except ExtractionError as error:
        result = (error.failure_code, error.message)
    writer.send(result)
    writer.close()


async def extract_in_child_process(
    content: bytes,
    mime_type: str,
    *,
    max_pages: int,
    max_clauses: int,
    max_chars: int,
    timeout_seconds: float,
    target: ChildTarget = _extract_in_child,
) -> Extraction:
    """extract_clauses in a spawned process, killed when timeout_seconds pass (TimeoutError).

    A child that dies without an answer (a crash, out of memory) makes the file UNREADABLE: it
    would die again on every retry. target is replaceable for tests only.
    """
    context = multiprocessing.get_context("spawn")
    reader, writer = context.Pipe(duplex=False)
    process = context.Process(
        target=target,
        args=(writer, content, mime_type, max_pages, max_clauses, max_chars),
        daemon=True,
    )
    process.start()
    writer.close()
    try:
        # poll returns early when the child answers or exits (end of file).
        if not await asyncio.to_thread(reader.poll, timeout_seconds):
            raise TimeoutError
        try:
            result: _ChildResult = await asyncio.to_thread(reader.recv)
        except EOFError:
            raise ExtractionError(
                ExtractionCode.UNREADABLE, "Extraction stopped without a result."
            ) from None
    finally:
        # Also on cancellation: the child never outlives the call.
        process.kill()
        await asyncio.to_thread(process.join)
        reader.close()
    if isinstance(result, Extraction):
        return result
    raise ExtractionError(*result)
