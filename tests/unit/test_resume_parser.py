"""Resume parsing and chunking.

Runs against real PDF and DOCX bytes rather than a mocked extractor. The whole
risk in this module is what pypdf and python-docx actually hand back -- a
hand-written string fixture would test the sectioning logic and nothing else.
"""

import pathlib

import pytest

from app.modules.resume import parser
from app.modules.resume.chunker import MAX_CHUNK_CHARS, chunk_sections

FIXTURES = pathlib.Path(__file__).parent.parent / "fixtures"
PDF = "application/pdf"
DOCX = "application/vnd.openxmlformats-officedocument.wordprocessingml.document"


@pytest.fixture(scope="module")
def pdf_bytes() -> bytes:
    return (FIXTURES / "resume.pdf").read_bytes()


@pytest.fixture(scope="module")
def docx_bytes() -> bytes:
    return (FIXTURES / "resume.docx").read_bytes()


# --- Extraction -------------------------------------------------------------


@pytest.mark.parametrize("fixture,content_type", [("pdf_bytes", PDF), ("docx_bytes", DOCX)])
def test_both_formats_yield_the_same_sections(request, fixture, content_type):
    """A candidate's format choice must not change what the interviewer sees."""
    parsed = parser.parse(request.getfixturevalue(fixture), content_type)
    assert {"summary", "experience", "education", "skills"} <= set(parsed.sections)
    assert "Northwind Systems" in parsed.sections["experience"]
    assert "Kafka" in parsed.sections["skills"]


def test_contact_details_are_extracted(pdf_bytes):
    parsed = parser.parse(pdf_bytes, PDF)
    assert parsed.emails == ["priya.raman@example.com"]
    assert parsed.links == ["https://github.com/priyaraman"]
    assert any("98765" in phone for phone in parsed.phones)


def test_a_year_range_is_not_mistaken_for_a_phone_number(pdf_bytes):
    """"2021 - 2024" is eight digits and would match a loose phone pattern."""
    parsed = parser.parse(pdf_bytes, PDF)
    assert not any(phone.strip().startswith("2021") for phone in parsed.phones)


def test_an_email_is_not_harvested_as_a_phone_number():
    emails, phones, _ = parser.extract_contacts("reach me at 12345678@example.com today")
    assert emails == ["12345678@example.com"]
    assert phones == [], "the digits inside an email address became a phone number"


def test_parsed_payload_excludes_the_full_text(pdf_bytes):
    """Storing the text twice would double the row for no reader."""
    payload = parser.parse(pdf_bytes, PDF).as_dict()
    assert "text" not in payload
    assert payload["char_count"] > 0


# --- Failure modes ----------------------------------------------------------


def test_an_unsupported_content_type_is_refused():
    with pytest.raises(parser.ResumeParseError, match="Unsupported"):
        parser.parse(b"whatever", "image/png")


def test_a_corrupt_pdf_raises_rather_than_returning_junk():
    with pytest.raises(parser.ResumeParseError):
        parser.parse(b"%PDF-1.4 this is not really a pdf", PDF)


def test_a_text_free_document_is_a_parse_failure():
    """A scanned CV has no text layer. OCR is out of scope, so this must fail
    loudly rather than storing an empty resume that poisons retrieval."""
    import io

    from reportlab.lib.pagesizes import LETTER  # noqa: PLC0415
    from reportlab.pdfgen import canvas  # noqa: PLC0415

    buf = io.BytesIO()
    canvas.Canvas(buf, pagesize=LETTER).save()
    with pytest.raises(parser.ResumeParseError, match="scan"):
        parser.parse(buf.getvalue(), PDF)


# --- Normalisation ----------------------------------------------------------


def test_normalize_folds_ligatures_and_bullets():
    # NFKC turns the "ffi" ligature back into three characters; without it the
    # word tokenises differently from the same word typed normally.
    assert "efficient" in parser.normalize("eﬃcient")
    assert parser.normalize("• first • second").count("\n") >= 1


def test_headings_must_be_short_lines():
    """A sentence containing "experience" must not start a section."""
    text = "Header line\nI have considerable experience across many domains and teams here\nBody"
    assert "experience" not in parser.split_sections(text)


# --- Chunking ---------------------------------------------------------------


def test_chunks_carry_their_section_as_a_prefix(pdf_bytes):
    parsed = parser.parse(pdf_bytes, PDF)
    chunks = chunk_sections(parsed.sections)

    assert chunks
    assert all(c.content.startswith(f"[{c.section}]") for c in chunks)
    assert [c.ordinal for c in chunks] == list(range(len(chunks)))


def test_each_role_becomes_its_own_chunk(pdf_bytes):
    """A blind sliding window splits one job across two chunks; this must not."""
    parsed = parser.parse(pdf_bytes, PDF)
    experience = [c for c in chunk_sections(parsed.sections) if c.section == "experience"]

    northwind = [c for c in experience if "Northwind" in c.content]
    assert len(northwind) == 1
    # The whole role stays together: title line and its bullets in one chunk.
    assert "payments ledger" in northwind[0].content
    assert "Contoso" not in northwind[0].content


def test_chunking_is_deterministic(pdf_bytes):
    """A retry after a failed embedding must line up with the rows already stored."""
    sections = parser.parse(pdf_bytes, PDF).sections
    assert chunk_sections(sections) == chunk_sections(sections)


def test_oversized_sections_are_split_on_line_boundaries():
    line = "Delivered a measurable improvement to the platform this quarter."
    body = "\n".join([line] * 120)
    chunks = chunk_sections({"experience": body})

    assert len(chunks) > 1
    for chunk in chunks:
        # The prefix adds a little; the payload itself must respect the limit.
        assert len(chunk.content) <= MAX_CHUNK_CHARS + len("[experience] ")
        # No line was cut in half.
        assert all(part == line for part in chunk.content.split("\n")[1:] if part)


def test_tiny_trailing_spans_are_merged_not_emitted_alone():
    chunks = chunk_sections({"skills": "Python, Go, Postgres\n\nGo"})
    assert len(chunks) == 1


def test_chunk_count_is_bounded():
    body = "\n\n".join(f"Entry number {i} describing a role in some detail." for i in range(500))
    assert len(chunk_sections({"experience": body})) <= 120


def test_empty_sections_produce_no_chunks():
    assert chunk_sections({"skills": "   ", "experience": ""}) == []
