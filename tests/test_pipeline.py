import math
import re
from pathlib import Path

import pytest

import docqa.ingestion.pipeline as pipeline_module
from docqa.ingestion.parser import Page, parse_pdf
from docqa.ingestion.pipeline import IngestionPipeline, compute_doc_id
from docqa.retrieval.retriever import Retriever
from docqa.retrieval.vector_store import VectorStore

VOCAB = ["mars", "delta", "canvas", "radiation", "trajectory"]
SAMPLE_PDF = Path(__file__).parent / "fixtures" / "sample.pdf"

#### Fixtures and helpers ---------------------------------------------------
class FakeEmbedder:
    """Bag-of-words vectors: instant and deterministic (same idea as in test_retriever.py)."""

    name = "fake"

    def __init__(self) -> None:
        self.doc_calls = 0

    def _vec(self, text: str) -> list[float]:
        words = re.findall(r"[a-z]+", text.lower())
        return [float(words.count(w)) for w in VOCAB] + [0.01]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        self.doc_calls += 1
        return [self._vec(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._vec(text)


class FakeParser:
    def __init__(self, pages: list[Page]) -> None:
        self.pages = pages
        self.calls = 0

    def __call__(self, path) -> list[Page]:
        self.calls += 1
        return self.pages


def mars_pages() -> list[Page]:
    #each section is over 300 characters so merge_small leaves them alone,and the chunk count is always exactly 3.
    return [
        Page(1, "# Budget\n" + "The delta-v budget for the Mars mission is the sum of each segment. " * 8),
        Page(2, "# Submission\n" + "Submit your assignment on Canvas by the deadline. " * 10),
        Page(3, "# Safety\n" + "Radiation shielding protects the crew. " * 12),
    ]


@pytest.fixture
def env(tmp_path):
    embedder = FakeEmbedder()
    store = VectorStore(embedder.name, path=tmp_path / "chroma")
    return embedder, store, tmp_path


def make_pipeline(env, pages=None, **kwargs):
    embedder, store, tmp_path = env
    parser = FakeParser(mars_pages() if pages is None else pages)
    pipe = IngestionPipeline(
        embedder, store, parse=parser, upload_dir=tmp_path / "uploads", **kwargs
    )
    return pipe, parser


def write_pdf(directory: Path, name: str, body: str = "a") -> Path:
    p = directory / name
    p.write_bytes(b"%PDF-1.4\n" + body.encode())
    return p


# ---- compute_doc_id ------------------------------------------------------

def test_doc_id_depends_on_content_not_name(tmp_path):
    a = write_pdf(tmp_path, "a.pdf", "same")
    b = write_pdf(tmp_path, "b.pdf", "same")
    c = write_pdf(tmp_path, "c.pdf", "different")
    assert compute_doc_id(a) == compute_doc_id(b)
    assert compute_doc_id(a) != compute_doc_id(c)
    assert len(compute_doc_id(a)) == 16


# ---- ingest: the normal path ---------------------------------------------

def test_ingest_stores_chunks_and_reports_result(env):
    _, store, tmp_path = env
    pipe, _ = make_pipeline(env)
    pdf = write_pdf(tmp_path, "worksheet.pdf")

    result = pipe.ingest(pdf)

    assert result.status == "ingested"
    assert (result.chunks, result.pages) == (3, 3)
    assert result.doc_id == compute_doc_id(pdf)
    assert store.count() == 3
    assert store.list_documents()[0]["source"] == "worksheet.pdf"


def test_ingested_document_can_be_retrieved(env):
    embedder, store, tmp_path = env
    pipe, _ = make_pipeline(env)
    pipe.ingest(write_pdf(tmp_path, "worksheet.pdf"))

    retriever = Retriever(embedder, store)
    top = retriever.retrieve("Where do I submit on Canvas?", top_k=1)[0]
    assert (top.heading, top.page_number) == ("Submission", 2)
    assert retriever.retrieve("radiation", top_k=1)[0].heading == "Safety"


def test_copy_of_pdf_saved_to_upload_dir(env):
    _, _, tmp_path = env
    pipe, _ = make_pipeline(env)
    pdf = write_pdf(tmp_path, "worksheet.pdf", "contents")
    result = pipe.ingest(pdf)
    saved = tmp_path / "uploads" / f"{result.doc_id}.pdf"
    assert saved.read_bytes() == pdf.read_bytes()


# ---- re-ingesting ---------------------------------------------------------

def test_same_file_twice_is_skipped(env):
    _, store, tmp_path = env
    pipe, parser = make_pipeline(env)
    pdf = write_pdf(tmp_path, "worksheet.pdf")
    pipe.ingest(pdf)

    again = pipe.ingest(pdf)

    assert again.status == "unchanged"
    assert again.chunks == 3
    assert again.pages is None
    assert parser.calls == 1       #no second (slow) parse
    assert store.count() == 3      #no duplicates


def test_same_content_under_new_name_is_skipped(env):
    _, store, tmp_path = env
    pipe, parser = make_pipeline(env)
    pipe.ingest(write_pdf(tmp_path, "a.pdf", "same"))
    assert pipe.ingest(write_pdf(tmp_path, "copy_of_a.pdf", "same")).status == "unchanged"
    assert parser.calls == 1


def test_force_reingests_without_duplicates(env):
    _, store, tmp_path = env
    pipe, parser = make_pipeline(env)
    pdf = write_pdf(tmp_path, "worksheet.pdf")
    pipe.ingest(pdf)

    result = pipe.ingest(pdf, force=True)

    assert result.status == "replaced"
    assert parser.calls == 2
    assert store.count() == 3


def test_different_files_are_stored_separately(env):
    _, store, tmp_path = env
    pipe, _ = make_pipeline(env)
    pipe.ingest(write_pdf(tmp_path, "a.pdf", "one"))
    pipe.ingest(write_pdf(tmp_path, "b.pdf", "two"))
    assert store.count() == 6
    assert len(store.list_documents()) == 2


# ---- bad input ------------------------------------------------------------

def test_missing_file_raises(env):
    _, _, tmp_path = env
    pipe, _ = make_pipeline(env)
    with pytest.raises(FileNotFoundError):
        pipe.ingest(tmp_path / "nope.pdf")


def test_non_pdf_raises_before_parsing(env):
    _, store, tmp_path = env
    pipe, parser = make_pipeline(env)
    fake = tmp_path / "notes.pdf"
    fake.write_bytes(b"just some text")
    with pytest.raises(ValueError, match="Not a PDF"):
        pipe.ingest(fake)
    assert parser.calls == 0
    assert store.count() == 0


def test_pdf_with_no_text_raises_and_stores_nothing(env):
    _, store, tmp_path = env
    pipe, _ = make_pipeline(env, pages=[Page(1, "   "), Page(2, "")])
    with pytest.raises(ValueError, match="No text"):
        pipe.ingest(write_pdf(tmp_path, "blank.pdf"))
    assert store.count() == 0


def test_failed_reingest_keeps_existing_data(env):
    _, store, tmp_path = env
    pdf = write_pdf(tmp_path, "worksheet.pdf")
    make_pipeline(env)[0].ingest(pdf)

    broken, _ = make_pipeline(env, pages=[])
    with pytest.raises(ValueError):
        broken.ingest(pdf, force=True)

    assert store.count() == 3  #the old copy is still there


# ---- batching and settings ------------------------------------------------

def test_embeddings_are_sent_in_batches(env, monkeypatch):
    embedder, store, tmp_path = env
    monkeypatch.setattr(pipeline_module, "EMBED_BATCH", 10)
    pipe, _ = make_pipeline(
        env, pages=[Page(1, "# Long\n" + "word " * 2000)], max_chars=100, overlap=0
    )

    result = pipe.ingest(write_pdf(tmp_path, "long.pdf"))

    assert result.chunks > 50
    assert store.count() == result.chunks
    assert embedder.doc_calls == math.ceil(result.chunks / 10)


# ---- Integration: the real parser (runs OCR, so it takes several seconds) --

def test_real_pdf_end_to_end(env):
    embedder, store, tmp_path = env
    pipe = IngestionPipeline(
        embedder, store, parse=parse_pdf, upload_dir=tmp_path / "uploads"
    )

    result = pipe.ingest(SAMPLE_PDF)

    assert result.status == "ingested"
    assert result.pages == 3
    assert result.chunks >= 5

    top = Retriever(embedder, store).retrieve("radiation", top_k=1)[0]
    assert "radiation" in top.text.lower()
    assert top.page_number == 2