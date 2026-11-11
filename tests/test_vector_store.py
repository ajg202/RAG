import pytest

from docqa.ingestion.chunker import Chunk
from docqa.retrieval.vector_store import VectorStore

#three orthoganal vectors
A = [1.0, 0.0, 0.0]
B = [0.0, 1.0, 0.0]
C = [0.0, 0.0, 1.0]


@pytest.fixture
def store(tmp_path):
    return VectorStore("test-embedder", path=tmp_path / "chroma")


def three_chunks():
    return [
        Chunk("alpha text", 1, "Intro"),
        Chunk("beta text", 2, "Methods"),
        Chunk("gamma text", 3, ""),
    ]


def test_add_and_count(store):
    assert store.count() == 0
    assert store.add_document("doc1", three_chunks(), [A, B, C]) == 3
    assert store.count() == 3


def test_query_returns_nearest_first(store):
    store.add_document("doc1", three_chunks(), [A, B, C])
    results = store.query([1.0, 0.1, 0.0], top_k=3)
    assert [r.text for r in results] == ["alpha text", "beta text", "gamma text"]


def test_metadata_round_trips(store):
    store.add_document("doc1", three_chunks(), [A, B, C], source="paper.pdf")
    top = store.query(B, top_k=1)[0]
    assert top.text == "beta text"
    assert top.page_number == 2
    assert top.heading == "Methods"
    assert top.doc_id == "doc1"
    assert top.chunk_index == 1


def test_empty_heading_is_preserved(store):
    store.add_document("doc1", three_chunks(), [A, B, C])
    assert store.query(C, top_k=1)[0].heading == ""


def test_score_is_cosine_similarity(store):
    store.add_document("doc1", three_chunks(), [A, B, C])
    results = store.query(A, top_k=2)
    assert results[0].score == pytest.approx(1.0, abs=1e-3)  # identical
    assert results[1].score == pytest.approx(0.0, abs=1e-3)  # orthogonal


def test_top_k_limits_results(store):
    store.add_document("doc1", three_chunks(), [A, B, C])
    assert len(store.query(A, top_k=2)) == 2


def test_top_k_larger_than_stored_returns_all(store):
    store.add_document("doc1", three_chunks(), [A, B, C])
    assert len(store.query(A, top_k=50)) == 3


def test_query_empty_store_returns_empty_list(store):
    assert store.query(A, top_k=5) == []


def test_doc_filter_isolates_documents(store):
    store.add_document("doc1", [Chunk("mine", 1, "H")], [A])
    store.add_document("doc2", [Chunk("theirs", 1, "H")], [A])
    results = store.query(A, top_k=10, doc_ids=["doc1"])
    assert [r.text for r in results] == ["mine"]


def test_doc_filter_with_multiple_ids(store):
    store.add_document("doc1", [Chunk("one", 1, "H")], [A])
    store.add_document("doc2", [Chunk("two", 1, "H")], [B])
    store.add_document("doc3", [Chunk("three", 1, "H")], [C])
    results = store.query(A, top_k=10, doc_ids=["doc1", "doc2"])
    assert {r.doc_id for r in results} == {"doc1", "doc2"}


def test_empty_doc_filter_searches_nothing(store):
    # Important for per-user isolation: [] must never mean "everything".
    store.add_document("doc1", three_chunks(), [A, B, C])
    assert store.query(A, top_k=5, doc_ids=[]) == []


def test_filter_matching_nothing_returns_empty(store):
    store.add_document("doc1", three_chunks(), [A, B, C])
    assert store.query(A, top_k=5, doc_ids=["nope"]) == []


def test_readding_a_document_replaces_it(store):
    store.add_document("doc1", three_chunks(), [A, B, C])
    store.add_document("doc1", [Chunk("new only", 1, "H")], [A])
    assert store.count() == 1  # no duplicates, no stale leftovers
    assert store.query(A, top_k=5)[0].text == "new only"


def test_delete_removes_only_that_document(store):
    store.add_document("doc1", three_chunks(), [A, B, C])
    store.add_document("doc2", [Chunk("keep me", 1, "H")], [A])
    assert store.delete_document("doc1") == 3
    assert store.count() == 1
    assert store.query(A, top_k=5)[0].doc_id == "doc2"


def test_delete_unknown_document_returns_zero(store):
    assert store.delete_document("nope") == 0


def test_list_documents(store):
    store.add_document("doc1", three_chunks(), [A, B, C], source="a.pdf")
    store.add_document("doc2", [Chunk("x", 1, "H")], [A], source="b.pdf")
    docs = {d["doc_id"]: d for d in store.list_documents()}
    assert docs["doc1"] == {"doc_id": "doc1", "source": "a.pdf", "chunks": 3}
    assert docs["doc2"]["chunks"] == 1


def test_mismatched_lengths_raise(store):
    with pytest.raises(ValueError):
        store.add_document("doc1", three_chunks(), [A, B])


def test_adding_empty_document_stores_nothing(store):
    assert store.add_document("doc1", [], []) == 0
    assert store.count() == 0


def test_large_document_is_batched(store):
    n = 1200  # more than one batch
    chunks = [Chunk(f"chunk {i}", 1, "H") for i in range(n)]
    store.add_document("big", chunks, [A] * n)
    assert store.count() == n


def test_data_persists_across_instances(tmp_path):
    path = tmp_path / "chroma"
    VectorStore("test-embedder", path=path).add_document("doc1", three_chunks(), [A, B, C])
    assert VectorStore("test-embedder", path=path).count() == 3


def test_different_embedders_use_separate_collections(tmp_path):
    path = tmp_path / "chroma"
    VectorStore("model-a", path=path).add_document("doc1", three_chunks(), [A, B, C])
    assert VectorStore("model-b", path=path).count() == 0