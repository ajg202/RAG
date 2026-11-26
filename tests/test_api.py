from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from docqa.api.main import create_app
from docqa.api.routes import clean_source_name
from docqa.api.services import Services
from docqa.generation.answerer import Answer
from docqa.ingestion.pipeline import IngestResult
from docqa.retrieval.vector_store import SearchResult

ID1, ID2, ID3 = "a" * 16, "b" * 16, "c" * 16
DOCS = [
    {"doc_id": ID1, "source": "mars.pdf", "chunks": 7},
    {"doc_id": ID2, "source": "Another.pdf", "chunks": 3},
]
PDF = b"%PDF-1.4 test"


class FakeStore:
    def __init__(self, docs):
        self.docs = [dict(d) for d in docs]
        self.deleted = []

    def count(self, doc_ids=None):
        return sum(d["chunks"] for d in self.docs)

    def list_documents(self):
        return [dict(d) for d in self.docs]

    def delete_document(self, doc_id):
        match = [d for d in self.docs if d["doc_id"] == doc_id]
        if not match:
            return 0
        self.docs.remove(match[0])
        self.deleted.append(doc_id)
        return match[0]["chunks"]


class FakePipeline:
    def __init__(self):
        self.calls = []
        self.status = "ingested"
        self.error = None

    def ingest(self, path, force=False, source=None):
        if self.error:
            raise self.error
        # Read now: the API deletes the temp file right after this call returns.
        self.calls.append(dict(source=source, force=force, content=Path(path).read_bytes()))
        pages = None if self.status == "unchanged" else 2
        return IngestResult(ID1, source, 3, pages, self.status)


def make_answer():
    sources = [
        SearchResult("Intro text", 1, "Intro", ID1, 0, 0.61234567),
        SearchResult("Isp is 450 s", 5, "Data", ID2, 3, 0.55),
    ]
    return Answer(text="The Isp is 450 s [2].", sources=sources, cited=[2])


class FakeAnswerer:
    def __init__(self):
        self.calls = []
        self.errors = []
        self.result = make_answer()

    def answer(self, question, doc_ids=None, top_k=None, min_score=None):
        self.calls.append(dict(question=question, doc_ids=doc_ids, top_k=top_k, min_score=min_score))
        if self.errors:
            raise self.errors.pop(0)
        return self.result


@pytest.fixture
def env(tmp_path):
    services = Services(
        store=FakeStore(DOCS), pipeline=FakePipeline(), answerer=FakeAnswerer(),
        upload_dir=tmp_path, embedder_name="fake-embedder",
        generation_label="fake:model", max_upload_bytes=1000,
    )
    return services, TestClient(create_app(services))


def upload(client, name="worksheet.pdf", data=PDF, **params):
    return client.post("/documents", files={"file": (name, data, "application/pdf")}, params=params)


# ---- health ------------------------------------------------------------------

def test_health(env):
    _, client = env
    body = client.get("/health").json()
    assert body == {"status": "ok", "embedder": "fake-embedder", "generation": "fake:model", "chunks": 10}


# ---- upload ------------------------------------------------------------------

def test_upload_new_document(env):
    services, client = env
    r = upload(client)
    assert r.status_code == 201
    assert r.json() == {"doc_id": ID1, "source": "worksheet.pdf", "chunks": 3,
                        "pages": 2, "status": "ingested"}
    call = services.pipeline.calls[0]
    assert call["source"] == "worksheet.pdf" and call["content"] == PDF and call["force"] is False


def test_upload_unchanged_is_200(env):
    services, client = env
    services.pipeline.status = "unchanged"
    r = upload(client)
    assert r.status_code == 200
    assert r.json()["status"] == "unchanged" and r.json()["pages"] is None


def test_upload_force_is_passed_through(env):
    services, client = env
    services.pipeline.status = "replaced"
    r = upload(client, force="true")
    assert r.status_code == 200
    assert services.pipeline.calls[0]["force"] is True


def test_upload_not_a_pdf_is_422(env):
    services, client = env
    services.pipeline.error = ValueError("Not a PDF: notes.txt")
    r = upload(client, name="notes.txt")
    assert r.status_code == 422
    assert "Not a PDF" in r.json()["detail"]


def test_upload_too_large_is_413(env):
    services, client = env
    r = upload(client, data=b"%PDF-" + b"x" * 2000)
    assert r.status_code == 413
    assert services.pipeline.calls == []


def test_upload_without_file_is_422(env):
    _, client = env
    assert client.post("/documents").status_code == 422


@pytest.mark.parametrize("raw,expected", [
    ("report.pdf", "report.pdf"),
    ("../../secret/report.pdf", "report.pdf"),
    ("C:\\Users\\me\\report.pdf", "report.pdf"),
    ("", "upload.pdf"),
    (None, "upload.pdf"),
    ("bad\x00name\n.pdf", "badname.pdf"),
])
def test_clean_source_name(raw, expected):
    assert clean_source_name(raw) == expected


def test_upload_never_uses_client_path(env):
    services, client = env
    upload(client, name="../../evil/report.pdf")
    assert services.pipeline.calls[0]["source"] == "report.pdf"


# ---- list and delete ------------------------------------------------------------

def test_list_documents_sorted_by_name(env):
    _, client = env
    names = [d["source"] for d in client.get("/documents").json()]
    assert names == ["Another.pdf", "mars.pdf"]


def test_delete_removes_chunks_and_saved_copy(env, tmp_path):
    services, client = env
    saved = tmp_path / f"{ID1}.pdf"
    saved.write_bytes(b"x")
    assert client.delete(f"/documents/{ID1}").status_code == 204
    assert services.store.deleted == [ID1]
    assert not saved.exists()


def test_delete_unknown_document_is_404(env):
    _, client = env
    assert client.delete(f"/documents/{ID3}").status_code == 404


def test_delete_with_malformed_id_is_422(env):
    _, client = env
    assert client.delete("/documents/not-an-id").status_code == 422


# ---- ask ------------------------------------------------------------------------

def test_ask_returns_answer_and_cited_sources(env):
    services, client = env
    r = client.post("/ask", json={"question": "  What is the Isp?  "})
    assert r.status_code == 200
    body = r.json()
    assert body["answer"] == "The Isp is 450 s [2]."
    assert body["retrieved"] == 2 and body["truncated"] is False
    assert body["cited"] == [{
        "n": 2, "doc_id": ID2, "source": "Another.pdf", "page_number": 5,
        "heading": "Data", "score": 0.55, "text": "Isp is 450 s",
    }]
    assert services.answerer.calls[0] == dict(
        question="What is the Isp?", doc_ids=None, top_k=None, min_score=None)


def test_ask_passes_options_and_doc_filter(env):
    services, client = env
    client.post("/ask", json={"question": "q", "doc_ids": [ID1], "top_k": 3, "min_score": 0.2})
    assert services.answerer.calls[0] == dict(question="q", doc_ids=[ID1], top_k=3, min_score=0.2)


def test_ask_one_document_route(env):
    services, client = env
    assert client.post(f"/documents/{ID2}/ask", json={"question": "q"}).status_code == 200
    assert services.answerer.calls[0]["doc_ids"] == [ID2]


def test_ask_unknown_document_is_404_and_skips_the_model(env):
    services, client = env
    assert client.post("/ask", json={"question": "q", "doc_ids": [ID1, ID3]}).status_code == 404
    assert client.post(f"/documents/{ID3}/ask", json={"question": "q"}).status_code == 404
    assert services.answerer.calls == []


def test_ask_with_no_documents_is_409(env):
    services, client = env
    services.store.docs.clear()
    r = client.post("/ask", json={"question": "q"})
    assert r.status_code == 409
    assert services.answerer.calls == []


def test_ask_not_found_answer_has_no_sources(env):
    services, client = env
    services.answerer.result = Answer(text="I couldn't find anything.")
    body = client.post("/ask", json={"question": "q"}).json()
    assert body["cited"] == [] and body["retrieved"] == 0


def test_ask_reports_truncation(env):
    services, client = env
    services.answerer.result.truncated = True
    assert client.post("/ask", json={"question": "q"}).json()["truncated"] is True


@pytest.mark.parametrize("body", [
    {"question": ""},
    {"question": "   "},
    {"question": "q", "top_k": 0},
    {"question": "q", "top_k": 100},
    {"question": "q", "min_score": 2},
    {"question": "q", "doc_ids": []},
    {},
])
def test_ask_rejects_invalid_requests(env, body):
    services, client = env
    assert client.post("/ask", json=body).status_code == 422
    assert services.answerer.calls == []


def test_backend_failure_is_503_and_server_recovers(env):
    services, client = env
    services.answerer.errors = [RuntimeError("Can't reach Ollama at http://localhost:11434.")]
    r = client.post("/ask", json={"question": "q"})
    assert r.status_code == 503
    assert "Ollama" in r.json()["detail"]
    # the work lock must have been released, or this would hang
    assert client.post("/ask", json={"question": "q"}).status_code == 200