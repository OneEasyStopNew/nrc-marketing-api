"""Clause extraction from policy sources (NRC-4): pure functions, no database.

Every file here is built in the test and labelled [SYNTHETIC]; no binary fixtures are checked in.
"""

import hashlib
import io

import pytest
from pypdf import PdfWriter

from app.services.policy_extraction import (
    ExtractionCode,
    ExtractionError,
    extract_clauses,
)
from tests.conftest import make_pdf

PDF = "application/pdf"
MARKDOWN = "text/markdown"
PLAIN = "text/plain"
LIMITS = {"max_pages": 50, "max_clauses": 100}


def texts(content: bytes, mime: str, **limits: int) -> list[str]:
    return [c.text for c in extract_clauses(content, mime, **{**LIMITS, **limits}).clauses]


def test_a_pdf_gives_clauses_with_page_numbers_and_refs() -> None:
    pdf = make_pdf(
        [
            ["[SYNTHETIC] First clause.", None, "Second clause, line one", "line two."],
            ["Third clause on page two."],
        ]
    )

    result = extract_clauses(pdf, PDF, **LIMITS)

    assert result.page_count == 2
    assert [(c.clause_ref, c.page, c.text) for c in result.clauses] == [
        ("p1-c1", 1, "[SYNTHETIC] First clause."),
        ("p1-c2", 1, "Second clause, line one\nline two."),
        ("p2-c3", 2, "Third clause on page two."),
    ]


def test_a_pdf_heading_sets_the_section_across_pages() -> None:
    pdf = make_pdf(
        [
            ["1. Scope", None, "[SYNTHETIC] Clause under scope."],
            ["Still under scope.", None, "2.1 Claims", None, "Claim clause."],
        ]
    )

    result = extract_clauses(pdf, PDF, **LIMITS)

    assert [(c.section, c.text) for c in result.clauses] == [
        ("1. Scope", "[SYNTHETIC] Clause under scope."),
        ("1. Scope", "Still under scope."),
        ("2.1 Claims", "Claim clause."),
    ]


def test_pdf_text_keeps_internal_spacing_and_punctuation() -> None:
    pdf = make_pdf([["[SYNTHETIC] Never say  100% safe -- ever (really); ok?"]])

    assert texts(pdf, PDF) == ["[SYNTHETIC] Never say  100% safe -- ever (really); ok?"]


def test_markdown_headings_set_the_section_and_are_not_clauses() -> None:
    content = (
        b"# [SYNTHETIC] Policy\n"
        b"Intro text right after the heading.\n\n"
        b"## Claims\n\n"
        b"A claim clause.\n\n"
        b"### Disclosures\n"
        b"A disclosure clause.\n"
    )

    result = extract_clauses(content, MARKDOWN, **LIMITS)

    assert [(c.clause_ref, c.section, c.page, c.text) for c in result.clauses] == [
        ("c1", "[SYNTHETIC] Policy", None, "Intro text right after the heading."),
        ("c2", "Claims", None, "A claim clause."),
        ("c3", "Disclosures", None, "A disclosure clause."),
    ]
    assert result.page_count is None


def test_numbered_headings_set_the_section() -> None:
    content = (
        b"1. Scope\n\n[SYNTHETIC] First.\n\n3.2.1 Exact wording\n\nSecond.\n\n4. Short sentence.\n"
    )

    result = extract_clauses(content, PLAIN, **LIMITS)

    # "4. Short sentence." ends like a sentence, so it is a clause, never a dropped heading.
    assert [(c.section, c.text) for c in result.clauses] == [
        ("1. Scope", "[SYNTHETIC] First."),
        ("3.2.1 Exact wording", "Second."),
        ("3.2.1 Exact wording", "4. Short sentence."),
    ]


def test_a_long_numbered_line_is_a_clause() -> None:
    line = "1. " + "word " * 40

    assert texts(line.encode(), PLAIN) == [line.strip()]


def test_hash_lines_are_not_headings_in_plain_text() -> None:
    assert texts(b"# not a heading here\n", PLAIN) == ["# not a heading here"]


def test_text_clauses_have_no_page() -> None:
    result = extract_clauses(b"One.\n\nTwo.\n", PLAIN, **LIMITS)

    assert [(c.clause_ref, c.page) for c in result.clauses] == [("c1", None), ("c2", None)]


def test_text_is_verbatim_apart_from_line_endings_and_the_paragraph_ends() -> None:
    content = (
        "  Double  spaces,\t“curly” quotes; 100%!  \r\nsecond line\r\n\r\n\r\n   \r\nNext.".encode()
    )

    assert texts(content, MARKDOWN) == [
        "Double  spaces,\t“curly” quotes; 100%!  \nsecond line",
        "Next.",
    ]


def test_a_byte_order_mark_is_not_part_of_the_text() -> None:
    assert texts(b"\xef\xbb\xbfFirst.", PLAIN) == ["First."]


def test_the_checksum_is_the_sha256_of_the_stored_text() -> None:
    text = "Données [SYNTHETIC]"
    (clause,) = extract_clauses(text.encode(), PLAIN, **LIMITS).clauses

    assert clause.checksum_sha256 == hashlib.sha256(text.encode("utf-8")).hexdigest()


def test_more_clauses_than_the_cap_fail() -> None:
    content = b"One.\n\nTwo.\n\nThree.\n"

    assert len(texts(content, PLAIN, max_clauses=3)) == 3
    with pytest.raises(ExtractionError, match="more than 2 clauses") as error:
        extract_clauses(content, PLAIN, max_pages=1, max_clauses=2)
    assert error.value.failure_code is ExtractionCode.TOO_MANY_CLAUSES


def test_more_pages_than_the_cap_fail_before_any_text_is_read() -> None:
    pdf = make_pdf([["One."], ["Two."], ["Three."]])

    with pytest.raises(ExtractionError, match="more than 2 pages") as error:
        extract_clauses(pdf, PDF, max_pages=2, max_clauses=100)
    assert error.value.failure_code is ExtractionCode.TOO_MANY_PAGES
    assert extract_clauses(pdf, PDF, max_pages=3, max_clauses=100).page_count == 3


def test_an_encrypted_pdf_is_unreadable() -> None:
    writer = PdfWriter(clone_from=io.BytesIO(make_pdf([["[SYNTHETIC] Secret."]])))
    writer.encrypt("synthetic-test-password")
    buffer = io.BytesIO()
    writer.write(buffer)

    with pytest.raises(ExtractionError, match="encrypted") as error:
        extract_clauses(buffer.getvalue(), PDF, **LIMITS)
    assert error.value.failure_code is ExtractionCode.UNREADABLE


@pytest.mark.parametrize(
    "content",
    [
        pytest.param(b"not a pdf at all", id="garbage"),
        pytest.param(b"%PDF-1.7\n1 0 obj\n<<>>\n", id="truncated"),
        pytest.param(b"", id="empty"),
    ],
)
def test_a_damaged_pdf_is_unreadable(content: bytes) -> None:
    with pytest.raises(ExtractionError, match="cannot be read") as error:
        extract_clauses(content, PDF, **LIMITS)
    assert error.value.failure_code is ExtractionCode.UNREADABLE


def test_a_text_file_that_is_not_utf8_is_unreadable() -> None:
    with pytest.raises(ExtractionError, match="not UTF-8") as error:
        extract_clauses(b"caf\xe9", PLAIN, **LIMITS)
    assert error.value.failure_code is ExtractionCode.UNREADABLE


def test_an_unknown_file_type_is_unreadable() -> None:
    with pytest.raises(ExtractionError, match="type cannot be read") as error:
        extract_clauses(b"<html></html>", "text/html", **LIMITS)
    assert error.value.failure_code is ExtractionCode.UNREADABLE


def test_a_pdf_without_text_is_empty() -> None:
    # An image-only (scanned) PDF has pages but no text layer; a blank page stands in for one.
    writer = PdfWriter()
    writer.add_blank_page(width=200, height=200)
    buffer = io.BytesIO()
    writer.write(buffer)

    with pytest.raises(ExtractionError, match="no extractable text") as error:
        extract_clauses(buffer.getvalue(), PDF, **LIMITS)
    assert error.value.failure_code is ExtractionCode.EMPTY


@pytest.mark.parametrize("content", [b"", b"   \n\n  \n", b"# Only a heading\n"])
def test_a_text_file_without_clauses_is_empty(content: bytes) -> None:
    with pytest.raises(ExtractionError, match="no extractable text") as error:
        extract_clauses(content, MARKDOWN, **LIMITS)
    assert error.value.failure_code is ExtractionCode.EMPTY
