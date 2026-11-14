from types import SimpleNamespace

import pytest
import requests

from docqa.generation import ollama_client as oc
from docqa.generation.ollama_client import OllamaClient
from docqa.retrieval.vector_store import SearchResult


class FakeResponse:
    def __init__(self, data=None, status=200):
        self._data = data if data is not None else {}
        self.status_code = status
        self.text = str(self._data)

    def json(self):
        return self._data


def reply(content="Answer [1].", done_reason="stop", prompt_tokens=500):
    return {"message": {"role": "assistant", "content": content},
            "done_reason": done_reason, "prompt_eval_count": prompt_tokens}


@pytest.fixture
def post(monkeypatch):
    state = SimpleNamespace(calls=[], response=FakeResponse(reply()))

    def fake_post(url, json=None, timeout=None):
        state.calls.append({"url": url, "json": json, "timeout": timeout})
        if isinstance(state.response, Exception):
            raise state.response
        return state.response

    monkeypatch.setattr(oc.requests, "post", fake_post)
    return state


def client(**kw):
    kw.setdefault("think", None)
    return OllamaClient(url="http://x:1/", num_ctx=4096, temperature=0.1, **kw)


def call(c):
    return c.messages.create(model="m", max_tokens=256, system="SYS",
                             messages=[{"role": "user", "content": "hi"}])


def test_request_shape(post):
    call(client())
    sent = post.calls[0]
    assert sent["url"] == "http://x:1/api/chat"  # trailing slash handled
    body = sent["json"]
    assert body["model"] == "m" and body["stream"] is False
    assert body["messages"][0] == {"role": "system", "content": "SYS"}
    assert body["messages"][1] == {"role": "user", "content": "hi"}
    assert body["options"] == {"num_ctx": 4096, "num_predict": 256, "temperature": 0.1}
    assert "think" not in body


def test_response_matches_anthropic_shape(post):
    out = call(client())
    assert out.content[0].type == "text"
    assert out.content[0].text == "Answer [1]."
    assert out.stop_reason == "end_turn"


def test_think_blocks_are_removed(post):
    post.response = FakeResponse(reply("<think>hmm\nmore</think>\n\nFinal [1]."))
    assert call(client()).content[0].text == "Final [1]."


def test_length_stop_is_reported_as_max_tokens(post):
    post.response = FakeResponse(reply(done_reason="length"))
    assert call(client()).stop_reason == "max_tokens"


def test_think_option_is_sent_when_set(post):
    call(client(think=False))
    assert post.calls[0]["json"]["think"] is False


def test_connection_error_is_friendly(post):
    post.response = requests.ConnectionError("refused")
    with pytest.raises(RuntimeError, match="Is it running"):
        call(client())


def test_timeout_is_friendly(post):
    post.response = requests.Timeout("slow")
    with pytest.raises(RuntimeError, match="didn't answer"):
        call(client())


def test_missing_model_says_how_to_pull_it(post):
    post.response = FakeResponse({"error": "model 'm' not found"}, status=404)
    with pytest.raises(RuntimeError, match="ollama pull m"):
        call(client())


def test_other_errors_include_the_servers_message(post):
    post.response = FakeResponse({"error": "boom"}, status=500)
    with pytest.raises(RuntimeError, match="boom"):
        call(client())


def test_nearly_full_context_warns(post):
    post.response = FakeResponse(reply(prompt_tokens=4096))
    with pytest.warns(UserWarning, match="context"):
        call(client())


def test_works_inside_answerer(post):
    from docqa.generation.answerer import Answerer

    source = SearchResult(text="Isp is 450 s", page_number=1, heading="Data",
                          doc_id="d", chunk_index=0, score=0.5)
    retriever = SimpleNamespace(retrieve=lambda *a, **k: [source])
    out = Answerer(retriever, client(), model="m").answer("What is the Isp?")
    assert out.text == "Answer [1]."
    assert out.cited == [1]