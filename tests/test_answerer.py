from dataclasses import replace
from types import SimpleNamespace

import pytest

import docqa.generation.answerer as answerer_module
from docqa.config import settings
from docqa.generation.answerer import NOT_FOUND, Answerer, extract_citations, get_answerer
from docqa.generation.prompts import SYSTEM_PROMPT
from docqa.ingestion.chunker import Chunk
from docqa.retrieval.embeddings import get_embedder
from docqa.retrieval.retriever import Retriever
from docqa.retrieval.vector_store import SearchResult, VectorStore


def result(text, page=1, heading="H"):
    return SearchResult(text=text, page_number=page, heading=heading,
                        doc_id="d", chunk_index=0, score=0.5)


class FakeRetriever:
    def __init__(self, results):
        self.results = results
        self.calls = []

    def retrieve(self, question, top_k=None, doc_ids=None, min_score=None):
        self.calls.append(dict(question=question, top_k=top_k, doc_ids=doc_ids, min_score=min_score))
        return self.results


class FakeClient:
    """Mimics client.messages.create(...) and records what it was sent."""

    def __init__(self, reply="Answer [1].", stop_reason="end_turn"):
        self.calls = []
        self._reply = reply
        self._stop = stop_reason
        self.messages = self  # so client.messages.create works

    def create(self, **kwargs):
        self.calls.append(kwargs)
        blocks = self._reply if isinstance(self._reply, list) else [self._reply]
        return SimpleNamespace(
            content=[SimpleNamespace(type="text", text=t) for t in blocks],
            stop_reason=self._stop,
        )


def make(results, **client_kwargs):
    retriever, client = FakeRetriever(results), FakeClient(**client_kwargs)
    return Answerer(retriever, client), retriever, client


# ---- extract_citations -----------------------------------------------------

def test_extract_single_and_adjacent_citations():
    assert extract_citations("A [1]. B [3][2].", 3) == [1, 3, 2]


def test_extract_comma_separated_citations():
    assert extract_citations("Both agree [1, 2].", 2) == [1, 2]


def test_extract_dedupes_and_keeps_first_use_order():
    assert extract_citations("[2] then [1] then [2] again", 2) == [2, 1]


def test_extract_drops_numbers_that_are_not_sources():
    assert extract_citations("Made up [9] and [0] and [2023]", 2) == []


def test_extract_with_no_citations():
    assert extract_citations("No brackets here", 3) == []


# ---- Answerer ----------------------------------------------------------------

def test_answer_returns_text_and_cited_sources():
    ans, _, _ = make([result("Isp is 450 s", 1, "Data"), result("Submit on Canvas", 3, "Submission")],
                     reply="The Isp is 450 s [1].")
    out = ans.answer("What is the Isp?")
    assert out.text == "The Isp is 450 s [1]."
    assert out.cited == [1]
    assert out.cited_sources[0].heading == "Data"
    assert len(out.sources) == 2  # everything sent, not just what was cited


def test_request_has_model_system_prompt_and_context():
    ans, _, client = make([result("Isp is 450 s", 1, "Data")])
    ans.answer("What is the Isp?")
    call = client.calls[0]
    assert call["model"] == settings.claude_model
    assert call["system"] == SYSTEM_PROMPT
    assert call["max_tokens"] > 0
    content = call["messages"][0]["content"]
    assert "Isp is 450 s" in content and "Question: What is the Isp?" in content


def test_retrieval_options_are_passed_through():
    ans, retriever, _ = make([result("x")])
    ans.answer("q", doc_ids=["doc1"], top_k=3, min_score=0.2)
    assert retriever.calls[0] == dict(question="q", top_k=3, doc_ids=["doc1"], min_score=0.2)


def test_no_sources_returns_not_found_without_calling_the_api():
    ans, _, client = make([])
    out = ans.answer("What is the Isp?")
    assert out.text == NOT_FOUND
    assert out.sources == [] and out.cited == []
    assert client.calls == []


def test_blank_question_raises_before_anything_runs():
    ans, retriever, client = make([result("x")])
    with pytest.raises(ValueError):
        ans.answer("   ")
    assert retriever.calls == [] and client.calls == []


def test_multiple_text_blocks_are_joined():
    ans, _, _ = make([result("x")], reply=["Part one ", "part two [1]."])
    assert ans.answer("q").text == "Part one part two [1]."


def test_hallucinated_citation_is_dropped():
    ans, _, _ = make([result("x")], reply="Claim [1] and another [4].")
    assert ans.answer("q").cited == [1]


def test_answer_with_no_citations():
    ans, _, _ = make([result("x")], reply="The documents don't say.")
    out = ans.answer("q")
    assert out.cited == [] and out.cited_sources == []


def test_truncation_is_reported():
    ans, _, _ = make([result("x")], stop_reason="max_tokens")
    assert ans.answer("q").truncated is True
    ans2, _, _ = make([result("x")], stop_reason="end_turn")
    assert ans2.answer("q").truncated is False


def test_get_answerer_without_key_raises_clear_error(monkeypatch):
    monkeypatch.setattr(answerer_module, "settings", replace(settings, anthropic_api_key=""))
    with pytest.raises(RuntimeError, match="ANTHROPIC_API_KEY"):
        get_answerer()


# ---- Live test: real API, tiny cost, skipped when no key is configured --------

@pytest.mark.skipif(not settings.anthropic_api_key, reason="ANTHROPIC_API_KEY not set")
def test_live_answer_is_grounded_and_declines_unknowns(tmp_path):
    import anthropic

    embedder = get_embedder()
    store = VectorStore(embedder.name, path=tmp_path / "chroma")
    chunks = [
        Chunk("Propellant: assume an average Isp of 450 s for chemical engines.", 1, "Vehicle data"),
        Chunk("Submit your assignment on Canvas.", 3, "Submission"),
    ]
    store.add_document("doc1", chunks, embedder.embed_documents([c.text for c in chunks]))
    ans = Answerer(Retriever(embedder, store), anthropic.Anthropic(api_key=settings.anthropic_api_key))

    found = ans.answer("What Isp should I assume?")
    assert "450" in found.text
    assert any(s.heading == "Vehicle data" for s in found.cited_sources)

    missing = ans.answer("Who won the 1998 World Cup?")
    assert missing.cited == []