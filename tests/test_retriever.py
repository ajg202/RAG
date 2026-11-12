import re

import pytest

from docqa.config import settings
from docqa.ingestion.chunker import Chunk
from docqa.retrieval.embeddings import get_embedder
from docqa.retrieval.retriever import Retriever
from docqa.retrieval.vector_store import VectorStore

VOCAB = ["mars", "delta", "canvas", "radiation", "trajectory"]

#fake embedder is a simple bag-of-words embedding model for testing. It counts the occurrences of a small set of words and adds a small constant to avoid zero vectors.
class FakeEmbedder:
    name = "fake"

    def __init__(self) -> None:
        self.query_calls = 0

    def _vec(self, text: str) -> list[float]:
        words = re.findall(r"[a-z]+", text.lower())
        return [float(words.count(w)) for w in VOCAB] + [0.01]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._vec(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        self.query_calls += 1
        return self._vec(text)


@pytest.fixture
def env(tmp_path):
    embedder = FakeEmbedder()
    store = VectorStore(embedder.name, path=tmp_path / "chroma")
    return embedder, store, Retriever(embedder, store)


def ingest(embedder, store, doc_id, chunks):
    store.add_document(doc_id, chunks, embedder.embed_documents([c.text for c in chunks]))


def mars_chunks():
    return [
        Chunk("Delta-v budget for the Mars mission", 1, "Budget"),
        Chunk("Submit the assignment on Canvas", 2, "Submission"),
        Chunk("Radiation protection for the crew", 3, "Safety"),
        Chunk("Select a trajectory in the browser", 4, "Trajectory"),
    ]


def test_most_relevant_chunk_ranks_first(env):
    embedder, store, retriever = env
    ingest(embedder, store, "doc1", mars_chunks())
    results = retriever.retrieve("How much delta-v does Mars need?", top_k=4)
    assert results[0].text.startswith("Delta-v budget")
    assert results[0].score > results[1].score


def test_different_question_finds_different_chunk(env):
    embedder, store, retriever = env
    ingest(embedder, store, "doc1", mars_chunks())
    assert retriever.retrieve("radiation", top_k=1)[0].heading == "Safety"


def test_results_carry_page_and_heading(env):
    embedder, store, retriever = env
    ingest(embedder, store, "doc1", mars_chunks())
    top = retriever.retrieve("canvas", top_k=1)[0]
    assert (top.page_number, top.heading, top.doc_id) == (2, "Submission", "doc1")


def test_top_k_limits_results(env):
    embedder, store, retriever = env
    ingest(embedder, store, "doc1", mars_chunks())
    assert len(retriever.retrieve("mars", top_k=2)) == 2


def test_default_top_k_comes_from_settings(env):
    embedder, store, retriever = env
    chunks = [Chunk(f"mars filler {i}", 1, "H") for i in range(settings.top_k + 3)]
    ingest(embedder, store, "doc1", chunks)
    assert len(retriever.retrieve("mars")) == settings.top_k


def test_empty_store_returns_empty_list(env):
    _, _, retriever = env
    assert retriever.retrieve("mars") == []


def test_blank_question_returns_empty_and_skips_embedding(env):
    embedder, store, retriever = env
    ingest(embedder, store, "doc1", mars_chunks())
    assert retriever.retrieve("") == []
    assert retriever.retrieve("   \n ") == []
    assert embedder.query_calls == 0


def test_doc_filter_restricts_results(env):
    embedder, store, retriever = env
    ingest(embedder, store, "doc1", [Chunk("mars mine", 1, "H")])
    ingest(embedder, store, "doc2", [Chunk("mars theirs", 1, "H")])
    results = retriever.retrieve("mars", doc_ids=["doc1"])
    assert [r.doc_id for r in results] == ["doc1"]


def test_empty_doc_filter_returns_nothing_and_skips_embedding(env):
    embedder, store, retriever = env
    ingest(embedder, store, "doc1", mars_chunks())
    assert retriever.retrieve("mars", doc_ids=[]) == []
    assert embedder.query_calls == 0


def test_min_score_drops_weak_matches(env):
    embedder, store, retriever = env
    ingest(embedder, store, "doc1", mars_chunks())
    everything = retriever.retrieve("delta mars", top_k=4)
    strong = retriever.retrieve("delta mars", top_k=4, min_score=0.5)
    assert len(everything) == 4
    assert [r.heading for r in strong] == ["Budget"]


def test_min_score_can_filter_everything(env):
    embedder, store, retriever = env
    ingest(embedder, store, "doc1", mars_chunks())
    assert retriever.retrieve("delta mars", min_score=1.5) == []


def test_question_whitespace_is_ignored(env):
    embedder, store, retriever = env
    ingest(embedder, store, "doc1", mars_chunks())
    assert retriever.retrieve("  radiation \n", top_k=1)[0].heading == "Safety"


# ---- Integration: the real embedding model (needs the model downloaded) ----

def test_real_embedder_finds_semantically_related_chunk(tmp_path):
    embedder = get_embedder()
    store = VectorStore(embedder.name, path=tmp_path / "chroma")
    chunks = [
        Chunk("The total delta-v budget for the Mars round trip is the sum of each segment.", 1, "Budget"),
        Chunk("Submit your completed assignment through the Canvas portal.", 2, "Submission"),
        Chunk("Shielding protects the crew from cosmic radiation during transit.", 3, "Safety"),
    ]
    store.add_document("doc1", chunks, embedder.embed_documents([c.text for c in chunks]))
    retriever = Retriever(embedder, store)

    assert retriever.retrieve("How much propulsion change do we need?", top_k=1)[0].heading == "Budget"
    assert retriever.retrieve("Where do I turn in my work?", top_k=1)[0].heading == "Submission"