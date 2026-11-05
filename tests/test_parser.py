import re
from pathlib import Path

import pytest

from docqa.ingestion.parser import Page, clean_text, parse_pdf

FIXTURE = Path(__file__).parent / "fixtures" / "sample.pdf"

# --- Edit these if you swap in a different PDF ---------------------------
EXPECTED_PAGE_COUNT = 3
EXPECTED_PHRASES = {
    1: ["Mars Mission Design Project Worksheet", "9.5 km/s", "Isp of 450s"],
    2: ["Mission Overview", "Delta-v Budget Calculation", "Crew Needs"],
    3: ["Reflect on the challenges", "Submission"],
}
# --------------------------------------------------------------------------


def normalize(s: str) -> str:
    """Lowercase and collapse whitespace so tests survive small layout changes."""
    return re.sub(r"\s+", " ", s).lower()


@pytest.fixture(scope="module")
def pages() -> list[Page]:
    #Parsed once per test run: the parser can run OCR, which is slow.
    return parse_pdf(FIXTURE)


# ---- Tests that use the real PDF ----------------------------------------

def test_returns_page_objects(pages):
    assert pages
    assert all(isinstance(p, Page) for p in pages)


def test_page_count(pages):
    assert len(pages) == EXPECTED_PAGE_COUNT


def test_page_numbers_are_one_based_and_sequential(pages):
    assert [p.page_number for p in pages] == list(range(1, EXPECTED_PAGE_COUNT + 1))


def test_no_page_is_empty(pages):
    assert all(p.text.strip() for p in pages)


@pytest.mark.parametrize("page_number,phrases", EXPECTED_PHRASES.items())
def test_expected_phrases_on_correct_page(pages, page_number, phrases):
    text = normalize(pages[page_number - 1].text)
    for phrase in phrases:
        assert normalize(phrase) in text, f"{phrase!r} missing from page {page_number}"


def test_headings_come_out_as_markdown(pages):
    assert re.search(r"^#{1,6}\s+\S", pages[0].text, re.MULTILINE)


def test_cleanup_applied_to_real_output(pages):
    for p in pages:
        assert "<u>" not in p.text
        last_line = p.text.splitlines()[-1].strip()
        assert not re.fullmatch(r"\d{1,3}", last_line), "bare page number left at end"


def test_missing_file_raises():
    with pytest.raises(Exception):  # tighten to the specific error once you see which one it is
        parse_pdf(Path(__file__).parent / "fixtures" / "does_not_exist.pdf")


# ---- Pure unit tests for clean_text (no PDF, instant) -------------------

def test_clean_text_removes_trailing_page_number():
    assert clean_text("Some text\n\n3\n") == "Some text"


def test_clean_text_removes_underline_tags():
    assert clean_text("Use the <u>NASA Trajectory Browser.</u>") == "Use the NASA Trajectory Browser."


def test_clean_text_keeps_numbers_inside_content():
    text = "Earth to LEO: 9.5 km/s\nMars Orbit Insertion: 1.0 km/s"
    assert clean_text(text) == text


def test_clean_text_only_strips_one_trailing_number():
    assert clean_text("Totals\n\n42\n") == "Totals"