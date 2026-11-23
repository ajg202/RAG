from pathlib import Path

import pytest

from docqa.cli import NO_DOCS, format_answer, main, resolve_docs, CliError
from docqa.generation.answerer import Answer
from docqa.ingestion.pipeline import IngestResult
from docqa.retrieval.vector_store import SearchResult

DOCS = [
    {"doc_id": "abc123", "source": "mars.pdf", "chunks": 7},
    {"doc_id": "def456", "source": "other.pdf", "chunks": 3},
]


class FakeStore:
    def __init__(self, docs=None):
        self.docs = list(docs or [])
        self.deleted = []

    def count(self, doc_ids=None):
        return sum(d["chunks"] for d in self.docs)

    def list_documents(self):
        return list(self.docs)

    def delete_document(self, doc_id):
        self.deleted.append(doc_id)
        return next(d["chunks"] for d in self.docs if d["doc_id"] == doc_id)


class FakePipeline:
    def __init__(self):
        self.calls = []
        self.fail = {}

    def ingest(self, path, force=False):
        path = Path(path)
        self.calls.append((path.name, force))
        if path.name in self.fail:
            raise self.fail[path.name]
        return IngestResult("id-" + path.stem, path.name, 3, 2, "ingested")


def make_answer():
    src = SearchResult(text="Isp 450", page_number=2, heading="Vehicle data",
                       doc_id="abc123", chunk_index=0, score=0.5)
    return Answer(text="The Isp is 450 s [1].", sources=[src], cited=[1])


class FakeAnswerer:
    def __init__(self, errors=None):
        self.calls = []
        self.errors = list(errors or [])

    def answer(self, question, doc_ids=None, top_k=None, min_score=None):
        self.calls.append(dict(question=question, doc_ids=doc_ids, top_k=top_k, min_score=min_score))
        if self.errors:
            raise self.errors.pop(0)
        return make_answer()


class FakeServices:
    def __init__(self, docs=DOCS, upload_dir=None, answerer=None):
        self._store = FakeStore(docs)
        self._pipeline = FakePipeline()
        self._answerer = answerer or FakeAnswerer()
        self.upload_dir = upload_dir or Path("unused")
        self.built = []

    def store(self):
        return self._store

    def pipeline(self, device=None):
        self.built.append(("pipeline", device))
        return self._pipeline

    def answerer(self, device=None):
        self.built.append(("answerer", device))
        return self._answerer


# ---- format_answer -----------------------------------------------------------------

def test_format_answer_lists_cited_sources():
    text = format_answer(make_answer(), {"abc123": "mars.pdf"})
    assert "The Isp is 450 s [1]." in text
    assert "[1] mars.pdf, page 2 - Vehicle data" in text


def test_format_answer_without_citations_has_no_sources_section():
    assert "Sources" not in format_answer(Answer(text="Not in the documents."), {})


def test_format_answer_omits_dash_for_empty_heading():
    answer = make_answer()
    answer.sources[0].heading = ""
    assert "page 2\n" in format_answer(answer, {}) + "\n"
    assert " - " not in format_answer(answer, {}).split("Sources:")[1]


def test_format_answer_notes_truncation():
    answer = make_answer()
    answer.truncated = True
    assert "cut off" in format_answer(answer, {})


# ---- resolve_docs ------------------------------------------------------------------

def test_resolve_by_id_prefix_and_filename():
    store = FakeStore(DOCS)
    assert resolve_docs(store, ["abc123"]) == ["abc123"]
    assert resolve_docs(store, ["def"]) == ["def456"]
    assert resolve_docs(store, ["mars.pdf"]) == ["abc123"]


def test_resolve_unknown_raises():
    with pytest.raises(CliError, match="No stored document"):
        resolve_docs(FakeStore(DOCS), ["nope"])


def test_resolve_ambiguous_prefix_raises():
    store = FakeStore([{"doc_id": "abc123", "source": "a.pdf", "chunks": 1},
                       {"doc_id": "abc999", "source": "b.pdf", "chunks": 1}])
    with pytest.raises(CliError, match="several"):
        resolve_docs(store, ["abc"])


# ---- ingest ------------------------------------------------------------------------

def test_ingest_reports_each_file(capsys):
    services = FakeServices()
    assert main(["ingest", "a.pdf", "b.pdf"], services=services) == 0
    out = capsys.readouterr().out
    assert "a.pdf" in out and "b.pdf" in out and "ingested" in out


def test_ingest_expands_folders_to_pdfs_only(tmp_path):
    folder = tmp_path / "pdfs"
    folder.mkdir()
    for name in ("b.pdf", "a.pdf", "notes.txt"):
        (folder / name).write_bytes(b"x")
    services = FakeServices()
    main(["ingest", str(folder)], services=services)
    assert [name for name, _ in services._pipeline.calls] == ["a.pdf", "b.pdf"]


def test_ingest_empty_folder_is_an_error(tmp_path, capsys):
    assert main(["ingest", str(tmp_path)], services=FakeServices()) == 1
    assert "No PDF files" in capsys.readouterr().err


def test_ingest_passes_force_and_device():
    services = FakeServices()
    main(["ingest", "a.pdf", "--force", "--device", "cuda"], services=services)
    assert services._pipeline.calls == [("a.pdf", True)]
    assert services.built == [("pipeline", "cuda")]


def test_ingest_continues_after_a_bad_file(capsys):
    services = FakeServices()
    services._pipeline.fail = {"bad.pdf": ValueError("Not a PDF: bad.pdf")}
    assert main(["ingest", "good.pdf", "bad.pdf", "also.pdf"], services=services) == 1
    assert [n for n, _ in services._pipeline.calls] == ["good.pdf", "bad.pdf", "also.pdf"]
    assert "Not a PDF" in capsys.readouterr().err


# ---- ask ---------------------------------------------------------------------------

def test_ask_prints_answer_and_sources(capsys):
    services = FakeServices()
    assert main(["ask", "What", "is", "the", "Isp?"], services=services) == 0
    out = capsys.readouterr().out
    assert "The Isp is 450 s [1]." in out
    assert "[1] mars.pdf, page 2 - Vehicle data" in out
    assert services._answerer.calls[0]["question"] == "What is the Isp?"
    assert services._answerer.calls[0]["doc_ids"] is None


def test_ask_with_no_documents_stops_before_loading_models(capsys):
    services = FakeServices(docs=[])
    assert main(["ask", "anything"], services=services) == 1
    assert NO_DOCS in capsys.readouterr().err
    assert services.built == []


def test_ask_resolves_doc_filter_and_passes_options():
    services = FakeServices()
    main(["ask", "q", "--doc", "mars.pdf", "--top-k", "3", "--min-score", "0.2",
          "--device", "cpu"], services=services)
    assert services._answerer.calls[0] == dict(question="q", doc_ids=["abc123"], top_k=3, min_score=0.2)
    assert services.built == [("answerer", "cpu")]


def test_ask_unknown_doc_is_an_error(capsys):
    services = FakeServices()
    assert main(["ask", "q", "--doc", "nope.pdf"], services=services) == 1
    assert "No stored document" in capsys.readouterr().err
    assert services._answerer.calls == []


def test_ask_backend_error_is_friendly(capsys):
    services = FakeServices(answerer=FakeAnswerer(errors=[RuntimeError("Can't reach Ollama")]))
    assert main(["ask", "q"], services=services) == 1
    err = capsys.readouterr().err
    assert "Can't reach Ollama" in err and "Traceback" not in err


def test_debug_flag_shows_the_real_exception():
    services = FakeServices(answerer=FakeAnswerer(errors=[RuntimeError("boom")]))
    with pytest.raises(RuntimeError, match="boom"):
        main(["--debug", "ask", "q"], services=services)


# ---- chat --------------------------------------------------------------------------

def feed(monkeypatch, lines):
    it = iter(lines)

    def fake_input(prompt=""):
        item = next(it)
        if isinstance(item, Exception):
            raise item
        return item

    monkeypatch.setattr("builtins.input", fake_input)


def test_chat_answers_until_exit(monkeypatch, capsys):
    feed(monkeypatch, ["What is the Isp?", "", "exit"])
    services = FakeServices()
    assert main(["chat"], services=services) == 0
    assert len(services._answerer.calls) == 1  # the blank line was skipped
    assert "The Isp is 450 s [1]." in capsys.readouterr().out


def test_chat_survives_a_failed_question(monkeypatch, capsys):
    feed(monkeypatch, ["first", "second", "exit"])
    services = FakeServices(answerer=FakeAnswerer(errors=[RuntimeError("boom")]))
    assert main(["chat"], services=services) == 0
    captured = capsys.readouterr()
    assert "boom" in captured.err
    assert "The Isp is 450 s [1]." in captured.out  # the second question still worked


def test_chat_ends_on_eof(monkeypatch):
    feed(monkeypatch, [EOFError()])
    assert main(["chat"], services=FakeServices()) == 0


def test_chat_with_no_documents(capsys):
    assert main(["chat"], services=FakeServices(docs=[])) == 1
    assert NO_DOCS in capsys.readouterr().err


# ---- list and delete ---------------------------------------------------------------

def test_list_shows_documents(capsys):
    assert main(["list"], services=FakeServices()) == 0
    out = capsys.readouterr().out
    assert "abc123" in out and "mars.pdf" in out and "other.pdf" in out


def test_list_when_empty(capsys):
    main(["list"], services=FakeServices(docs=[]))
    assert "No documents" in capsys.readouterr().out


def test_delete_removes_chunks_and_saved_copy(tmp_path, capsys):
    saved = tmp_path / "abc123.pdf"
    saved.write_bytes(b"x")
    services = FakeServices(upload_dir=tmp_path)
    assert main(["delete", "mars.pdf"], services=services) == 0
    assert services._store.deleted == ["abc123"]
    assert not saved.exists()
    assert "deleted abc123" in capsys.readouterr().out


def test_delete_unknown_document_deletes_nothing(capsys):
    services = FakeServices()
    assert main(["delete", "mars.pdf", "nope.pdf"], services=services) == 1
    assert services._store.deleted == []  # nothing is removed if any name is wrong


def test_command_is_required():
    with pytest.raises(SystemExit):
        main([], services=FakeServices())