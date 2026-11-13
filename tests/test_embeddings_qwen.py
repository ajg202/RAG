import math

import pytest

pytest.importorskip("sentence_transformers")

from docqa.retrieval.embeddings import Qwen3Embedder


@pytest.fixture(scope="module")
def embedder():
    return Qwen3Embedder()  # loads the model once for this file


def cosine(a, b):
    return sum(x * y for x, y in zip(a, b))  # vectors are normalized


def test_empty_input_returns_empty_list(embedder):
    assert embedder.embed_documents([]) == []


def test_dimension_is_1024(embedder):
    assert len(embedder.embed_query("hello")) == 1024


def test_vectors_are_normalized(embedder):
    vec = embedder.embed_documents(["some text about Mars"])[0]
    assert math.sqrt(sum(x * x for x in vec)) == pytest.approx(1.0, abs=1e-2)


def test_more_texts_than_batch_size(embedder):
    vecs = embedder.embed_documents([f"text number {i}" for i in range(40)])
    assert len(vecs) == 40


def test_query_and_document_prompts_differ(embedder):
    text = "specific impulse"
    assert embedder.embed_query(text) != embedder.embed_documents([text])[0]


def test_related_text_scores_higher_than_unrelated(embedder):
    q = embedder.embed_query("How much delta-v does a Mars mission need?")
    related, unrelated = embedder.embed_documents([
        "The total delta-v budget for the Mars round trip is the sum of each segment.",
        "Submit your completed assignment through the Canvas portal.",
    ])
    assert cosine(q, related) > cosine(q, unrelated)


def test_acronym_question_finds_the_right_passage(embedder):
    # MiniLM missed this one on your worksheet.
    q = embedder.embed_query("What is the Isp?")
    isp, other = embedder.embed_documents([
        "Propellant: base calculations on an average Isp of 450s for chemical engines.",
        "Submit your completed assignment through the Canvas portal.",
    ])
    assert cosine(q, isp) > cosine(q, other)