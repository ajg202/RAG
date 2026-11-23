"""Command line: ingest PDFs, ask questions, chat, list and delete documents."""
import argparse
import sys
from pathlib import Path

from docqa.config import settings
from docqa.generation.answerer import Answer, get_answerer
from docqa.ingestion.pipeline import get_pipeline
from docqa.retrieval.embeddings import get_embedder_name
from docqa.retrieval.vector_store import VectorStore

NO_DOCS = "No documents have been ingested yet. Run: python -m docqa.cli ingest <file.pdf>"


class CliError(Exception):
    """A problem the user can fix; shown without a traceback."""


EXPECTED_ERRORS = (CliError, FileNotFoundError, ValueError, RuntimeError)


class Services:
    """Builds the real components on demand, so `list` and `delete` never load a model."""

    upload_dir = settings.upload_dir

    def store(self) -> VectorStore:
        return VectorStore(get_embedder_name())

    def pipeline(self, device=None):
        return get_pipeline(device)

    def answerer(self, device=None):
        return get_answerer(device)


# ---- helpers ----------------------------------------------------------------

def expand_paths(paths: list[str]) -> list[Path]:
    """Files stay as given; a folder becomes the PDFs directly inside it."""
    files: list[Path] = []
    for raw in paths:
        p = Path(raw)
        if p.is_dir():
            found = sorted(p.glob("*.pdf"))
            if not found:
                raise CliError(f"No PDF files found in {p}")
            files.extend(found)
        else:
            files.append(p)
    return files


def resolve_docs(store, names: list[str]) -> list[str]:
    """Turn file names or (prefixes of) document IDs into exact IDs."""
    docs = store.list_documents()
    ids: list[str] = []
    for name in names:
        matches = [d for d in docs if d["doc_id"] == name or d["source"] == name]
        if not matches:
            matches = [d for d in docs if d["doc_id"].startswith(name)]
        if not matches:
            raise CliError(f"No stored document matches {name!r}. Run `list` to see them.")
        if len(matches) > 1:
            listing = ", ".join(f"{d['doc_id']} ({d['source']})" for d in matches)
            raise CliError(f"{name!r} matches several documents: {listing}. Use a longer ID.")
        ids.append(matches[0]["doc_id"])
    return ids


def format_answer(answer: Answer, doc_names: dict[str, str]) -> str:
    lines = [answer.text]
    if answer.truncated:
        lines += ["", "(The answer was cut off at the token limit.)"]
    if answer.cited:
        lines += ["", "Sources:"]
        for n in answer.cited:
            s = answer.sources[n - 1]
            name = doc_names.get(s.doc_id, s.doc_id)
            heading = f" - {s.heading}" if s.heading else ""
            lines.append(f"  [{n}] {name}, page {s.page_number}{heading}")
    return "\n".join(lines)


def _scope(args, store) -> tuple[list[str] | None, dict[str, str]]:
    """Check there is something to search; work out which documents to restrict to."""
    if store.count() == 0:
        raise CliError(NO_DOCS)
    docs = store.list_documents()
    names = {d["doc_id"]: d["source"] for d in docs}
    doc_ids = resolve_docs(store, args.doc) if args.doc else None
    return doc_ids, names


# ---- commands ------------------------------------------------------------------

def cmd_ingest(args, services) -> int:
    files = expand_paths(args.paths)
    pipeline = services.pipeline(args.device)
    failed = 0
    for path in files:
        try:
            r = pipeline.ingest(path, force=args.force)
        except (FileNotFoundError, ValueError) as e:
            print(f"  skipped {path.name}: {e}", file=sys.stderr)
            failed += 1
            continue
        print(f"  {r.status:<9} {r.source} ({r.chunks} chunks) id={r.doc_id}")
    return 1 if failed else 0


def cmd_ask(args, services) -> int:
    doc_ids, names = _scope(args, services.store())
    answerer = services.answerer(args.device)
    result = answerer.answer(
        " ".join(args.question), doc_ids=doc_ids, top_k=args.top_k, min_score=args.min_score
    )
    print(format_answer(result, names))
    return 0


def cmd_chat(args, services) -> int:
    doc_ids, names = _scope(args, services.store())
    answerer = services.answerer(args.device)
    print("Ask questions about your documents. Type 'exit' to quit.")
    print("Each question is answered on its own; earlier questions aren't remembered.")
    while True:
        try:
            question = input("\n> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not question:
            continue
        if question.lower() in ("exit", "quit", "/quit"):
            break
        try:
            result = answerer.answer(
                question, doc_ids=doc_ids, top_k=args.top_k, min_score=args.min_score
            )
        except Exception as e:  # a failed question shouldn't end the session
            if args.debug:
                raise
            print(f"error: {e}", file=sys.stderr)
            continue
        print()
        print(format_answer(result, names))
    return 0


def cmd_list(args, services) -> int:
    docs = services.store().list_documents()
    if not docs:
        print("No documents ingested yet.")
        return 0
    for d in sorted(docs, key=lambda d: d["source"].lower()):
        print(f"{d['doc_id']}  {d['chunks']:>4} chunks  {d['source']}")
    return 0


def cmd_delete(args, services) -> int:
    store = services.store()
    for doc_id in resolve_docs(store, args.docs):  # resolve everything before deleting anything
        removed = store.delete_document(doc_id)
        (Path(services.upload_dir) / f"{doc_id}.pdf").unlink(missing_ok=True)
        print(f"deleted {doc_id} ({removed} chunks)")
    return 0


COMMANDS = {
    "ingest": cmd_ingest,
    "ask": cmd_ask,
    "chat": cmd_chat,
    "list": cmd_list,
    "delete": cmd_delete,
}


# ---- entry point -----------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="docqa", description="Ask questions about your PDFs.")
    p.add_argument("--debug", action="store_true", help="show full tracebacks (put before the command)")
    sub = p.add_subparsers(dest="command", required=True)

    def add_device(sp):
        sp.add_argument("--device", help="embedder device: cuda or cpu (default: EMBEDDING_DEVICE)")

    def add_query_options(sp):
        sp.add_argument("--doc", action="append", default=[], metavar="ID_OR_FILE",
                        help="only search this document (repeatable)")
        sp.add_argument("--top-k", type=int, help="how many passages to retrieve")
        sp.add_argument("--min-score", type=float, help="drop passages scoring below this")
        add_device(sp)

    ing = sub.add_parser("ingest", help="parse, embed and store PDFs (files or folders)")
    ing.add_argument("paths", nargs="+")
    ing.add_argument("--force", action="store_true", help="re-ingest even if already stored")
    add_device(ing)

    ask = sub.add_parser("ask", help="ask one question")
    ask.add_argument("question", nargs="+")
    add_query_options(ask)

    chat = sub.add_parser("chat", help="ask questions interactively")
    add_query_options(chat)

    sub.add_parser("list", help="show stored documents")

    dele = sub.add_parser("delete", help="remove documents")
    dele.add_argument("docs", nargs="+", metavar="ID_OR_FILE")
    return p


def main(argv: list[str] | None = None, services=None) -> int:
    args = build_parser().parse_args(argv)
    services = services or Services()
    try:
        return COMMANDS[args.command](args, services)
    except KeyboardInterrupt:
        print(file=sys.stderr)
        return 130
    except Exception as e:
        if args.debug:
            raise
        if isinstance(e, EXPECTED_ERRORS):
            print(f"error: {e}", file=sys.stderr)
        else:
            print(f"error ({type(e).__name__}): {e}\n(run with --debug for the full traceback)",
                  file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())