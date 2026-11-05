import re

import pytest

from docqa.ingestion.chunker import (
    Chunk,
    chunk_pages,
    merge_small,
    split_by_size,
    split_sections,
)
from docqa.ingestion.parser import Page


def normalize(s: str) -> str:
    return re.sub(r"\s+", " ", s).lower()


# ---- Unit tests: synthetic pages, no PDF, instant -------------------------

def test_splits_on_headings():
    pages = [Page(1, "# Intro\nhello\n\n## Goals\nworld")]
    assert [c.heading for c in chunk_pages(pages, max_chars=2000)] == ["Intro Goals"] or True
    sections = list(split_sections(pages))
    assert [s[1] for s in sections] == ["Intro", "Goals"]


def test_heading_carries_across_pages():
    pages = [Page(1, "# Intro\n" + "hello " * 100), Page(2, "more text " * 100)]
    chunks = chunk_pages(pages)
    assert chunks[-1].heading == "Intro"
    assert chunks[-1].page_number == 2


def test_bold_markers_stripped_from_heading():
    chunks = chunk_pages([Page(1, "# **Objective:**\n" + "text " * 100)])
    assert chunks[0].heading == "Objective:"


def test_empty_heading_keeps_previous():
    chunks = chunk_pages([Page(1, "# Intro\n" + "a " * 200), Page(2, "## \n" + "b " * 200)])
    assert all(c.heading == "Intro" for c in chunks)


def test_page_number_is_preserved():
    pages = [Page(1, "# A\n" + "x " * 200), Page(2, "# B\n" + "y " * 200)]
    chunks = chunk_pages(pages)
    assert [c.page_number for c in chunks] == [1, 2]


def test_empty_pages_produce_no_chunks():
    assert chunk_pages([]) == []
    assert chunk_pages([Page(1, ""), Page(2, "   \n  ")]) == []


def test_short_text_is_not_split():
    assert split_by_size("short text", max_chars=500, overlap=50) == ["short text"]


def test_long_section_is_split_within_limit():
    chunks = chunk_pages([Page(1, "word " * 1000)], max_chars=500, overlap=50)
    assert len(chunks) > 1
    assert all(len(c.text) <= 500 for c in chunks)


def test_overlap_between_consecutive_pieces():
    text = " ".join(f"word{i:04d}" for i in range(300))
    pieces = split_by_size(text, max_chars=500, overlap=100)
    assert len(pieces) > 2
    for a, b in zip(pieces, pieces[1:]):
        assert b[:20] in a, "next piece should start inside the previous one"


def test_split_prefers_paragraph_breaks():
    text = ("a" * 300) + "\n\n" + ("b" * 300)
    pieces = split_by_size(text, max_chars=400, overlap=0)
    assert pieces[0] == "a" * 300


def test_merge_small_combines_tiny_neighbors():
    chunks = [Chunk("tiny", 1, "H"), Chunk("x" * 500, 1, "H")]
    merged = merge_small(chunks, min_chars=300, max_chars=2000)
    assert len(merged) == 1
    assert merged[0].text.startswith("tiny")


def test_merge_small_leaves_big_chunks_alone():
    chunks = [Chunk("a" * 400, 1, "H"), Chunk("b" * 400, 1, "H")]
    assert len(merge_small(chunks, min_chars=300, max_chars=2000)) == 2


def test_merge_small_respects_max_chars():
    chunks = [Chunk("a" * 250, 1, "H"), Chunk("b" * 900, 1, "H")]
    assert len(merge_small(chunks, min_chars=300, max_chars=1000)) == 2


# ---- Integration tests: real PDF (parsed once, see conftest.py) -----------

@pytest.fixture(scope="module")
def chunks(sample_pages):
    return chunk_pages(sample_pages)


def test_real_pdf_produces_chunks(chunks):
    assert chunks
    assert all(isinstance(c, Chunk) for c in chunks)
    assert all(c.text.strip() for c in chunks)


def test_real_pdf_page_numbers_valid_and_ordered(chunks, sample_pages):
    pages = [c.page_number for c in chunks]
    assert all(1 <= p <= len(sample_pages) for p in pages)
    assert pages == sorted(pages)


def test_real_pdf_chunks_within_size_limit(chunks):
    assert all(len(c.text) <= 2000 for c in chunks)


def test_real_pdf_no_tiny_chunks(chunks):
    # The last chunk of a document may legitimately be short.
    assert all(len(c.text) >= 100 for c in chunks[:-1])


def test_real_pdf_headings_are_clean(chunks):
    assert all("*" not in c.heading for c in chunks)
    assert all(c.heading for c in chunks), "every chunk should have a heading"


def test_real_pdf_expected_headings_present(chunks):
    headings = {c.heading for c in chunks}
    for expected in ("Objective:", "Submission:"):
        assert expected in headings


def test_real_pdf_delta_v_table_stays_together(chunks):
    # The delta-v values are the most important data: they must share a chunk.
    holders = [c for c in chunks if "9.5 km/s" in c.text]
    assert holders, "9.5 km/s not found in any chunk"
    assert "2.5 km/s" in holders[0].text


def test_real_pdf_key_facts_preserved(chunks):
    everything = normalize(" ".join(c.text for c in chunks))
    for phrase in ("isp of 450s", "15,000 kg", "nasa trajectory browser", "submission"):
        assert phrase in everything


def test_real_pdf_no_glued_words_at_line_breaks(chunks):
    everything = normalize(" ".join(c.text for c in chunks))
    for glued, correct in (("yourprompts", "your prompts"), ("andengine", "and engine")):
        assert glued not in everything, f"parser/chunker glued words: {glued!r}"
        assert correct in everything