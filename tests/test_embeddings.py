import math

import pytest

from docqa.retrieval.embeddings import get_embedder


@pytest.fixture(scope="module")
def embedder():
    return get_embedder()


def cosine(a, b):
    dot = sum(x * y for x, y in zip(a, b))
    return dot / (math.sqrt(sum(x * x for x in a)) * math.sqrt(sum(y * y for y in b)))


def test_empty_input_returns_empty_list(embedder):
    assert embedder.embed_documents([]) == []


def test_vectors_have_consistent_dimension(embedder):
    vecs = embedder.embed_documents(["first text", "a much longer second text about Mars"])
    assert len(vecs) == 2
    assert len(vecs[0]) == len(vecs[1]) > 0


def test_query_matches_document_dimension(embedder):
    doc = embedder.embed_documents(["rocket fuel"])[0]
    assert len(embedder.embed_query("rocket fuel")) == len(doc)


def test_related_text_scores_higher_than_unrelated(embedder):
    query = embedder.embed_query("How much delta-v does a Mars mission need?")
    related, unrelated = embedder.embed_documents([
        "The total delta-v budget for the Mars round trip is the sum of each segment.",
        "Submit your completed assignment through the Canvas portal.",
    ])
    assert cosine(query, related) > cosine(query, unrelated)