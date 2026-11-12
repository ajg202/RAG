import hashlib
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Literal

from docqa.config import settings
from docqa.ingestion.chunker import chunk_pages
from docqa.ingestion.parser import Page, parse_pdf
from docqa.retrieval.embeddings import Embedder, get_embedder
from docqa.retrieval.vector_store import VectorStore

EMBED_BATCH = 64  #texts per embedding call; hosted APIs cap batch sizes

#dataclass is a simple data structure to hold the result of an ingestion operation. It includes the document ID, source file name, number of chunks stored, number of pages parsed, and the status of the ingestion (ingested, replaced, or unchanged).
@dataclass
class IngestResult:
    doc_id: str
    source: str                 #original file name
    chunks: int                 #chunks stored for this document
    pages: int | None           #none when parsing was skipped
    status: Literal["ingested", "replaced", "unchanged"]

#compute_doc_id computes a unique document ID for a given file path by hashing its contents using SHA-256 and returning the first 16 characters of the hexadecimal digest. This ensures that each document can be uniquely identified based on its content, which is useful for avoiding duplicate ingestion of the same file.
def compute_doc_id(path: Path | str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()[:16]

#_looks_like_pdf checks if a given file path points to a PDF file by reading the first 1024 bytes of the file and looking for the "%PDF-" signature. This is a simple heuristic to determine if the file is likely a PDF, which is important for ensuring that only valid PDF files are processed in the ingestion pipeline.
def _looks_like_pdf(path: Path) -> bool:
    with open(path, "rb") as f:
        return b"%PDF-" in f.read(1024)

#IngestionPipeline is a class that encapsulates the entire process of ingesting a PDF document. It handles parsing the PDF, chunking the text, embedding the chunks, and storing them in a vector store. The pipeline can be configured with different embedders, vector stores, and parsing functions, and it supports options for maximum chunk size and overlap between chunks. The ingest method performs the ingestion process and returns an IngestResult indicating the outcome.
class IngestionPipeline:
    def __init__(
        self,
        embedder: Embedder,
        store: VectorStore,
        parse: Callable[[Path], list[Page]] = parse_pdf,
        upload_dir: Path | str | None = None,
        max_chars: int | None = None,
        overlap: int | None = None,
    ) -> None:
        self._embedder = embedder
        self._store = store
        self._parse = parse
        self._upload_dir = Path(upload_dir) if upload_dir else settings.upload_dir
        self._max_chars = settings.chunk_max_chars if max_chars is None else max_chars
        self._overlap = settings.chunk_overlap if overlap is None else overlap

    #ingest takes a PDF file path and processes it through the ingestion pipeline. It checks if the file is a valid PDF, computes a unique document ID, and determines if the document has already been ingested. If the document is new or if force ingestion is requested, it parses the PDF into pages, chunks the text, embeds the chunks, and stores them in the vector store. The method returns an IngestResult indicating whether the document was ingested, replaced, or unchanged.
    def ingest(self, path: Path | str, force: bool = False) -> IngestResult:
        """Parse, chunk, embed and store one PDF.

        An already-stored file is skipped (OCR and embedding are slow) unless
        force=True, which you need after changing chunk settings.
        """
        path = Path(path)
        if not path.is_file():
            raise FileNotFoundError(f"No such file: {path}")
        if not _looks_like_pdf(path):
            raise ValueError(f"Not a PDF: {path.name}")

        doc_id = compute_doc_id(path)
        existing = self._store.count([doc_id])
        if existing and not force:
            return IngestResult(doc_id, path.name, existing, None, "unchanged")

        pages = self._parse(path)
        chunks = chunk_pages(pages, max_chars=self._max_chars, overlap=self._overlap)
        if not chunks:
            raise ValueError(f"No text could be extracted from {path.name}")

        # Embed first and touch the store last, so a failure here
        # never leaves a half-written or deleted document behind.
        vectors = self._embed([c.text for c in chunks])
        self._store.add_document(doc_id, chunks, vectors, source=path.name)
        self._save_copy(path, doc_id)

        return IngestResult(
            doc_id=doc_id,
            source=path.name,
            chunks=len(chunks),
            pages=len(pages),
            status="replaced" if existing else "ingested",
        )

    #_embed embeds a list of text chunks using the configured embedder. It processes the texts in batches to avoid exceeding API limits or memory constraints. The method returns a list of vectors corresponding to the embedded representations of the input texts.
    def _embed(self, texts: list[str]) -> list[list[float]]:
        vectors: list[list[float]] = []
        for i in range(0, len(texts), EMBED_BATCH):
            vectors.extend(self._embedder.embed_documents(texts[i : i + EMBED_BATCH]))
        return vectors

    #_save_copy saves a copy of the ingested PDF file to the configured upload directory. It creates the necessary directories if they do not exist and only copies the file if it does not already exist in the destination. This ensures that a backup of the original PDF is kept for reference or future use.
    def _save_copy(self, path: Path, doc_id: str) -> None:
        dest = self._upload_dir / f"{doc_id}.pdf"
        dest.parent.mkdir(parents=True, exist_ok=True)
        if not dest.exists():
            shutil.copyfile(path, dest)

#get_pipeline builds an ingestion pipeline using the configured embedder and the default vector store. It returns an instance of IngestionPipeline that can be used to ingest PDF documents into the system.
def get_pipeline() -> IngestionPipeline:
    """Build a pipeline from the configured embedder and the default database."""
    embedder = get_embedder()
    return IngestionPipeline(embedder, VectorStore(embedder.name))

#__main__ block allows the script to be run from the command line. It parses command-line arguments for a PDF file and optional questions, ingests the PDF using the ingestion pipeline, and then retrieves and prints the most relevant chunks for each question using the retriever. This provides a simple interface for testing the ingestion and retrieval functionality directly from the command line.
if __name__ == "__main__":
    import argparse

    from docqa.retrieval.retriever import Retriever

    ap = argparse.ArgumentParser(description="Ingest a PDF, then try some questions.")
    ap.add_argument("pdf")
    ap.add_argument("questions", nargs="*")
    ap.add_argument("--force", action="store_true", help="re-ingest even if already stored")
    args = ap.parse_args()

    embedder = get_embedder()
    store = VectorStore(embedder.name)
    print(IngestionPipeline(embedder, store).ingest(args.pdf, force=args.force))

    retriever = Retriever(embedder, store)
    for question in args.questions:
        print(f"\nQ: {question}")
        for r in retriever.retrieve(question, top_k=3):
            print(f"  {r.score:.3f} | page {r.page_number} | {r.heading!r} | {r.text[:70]!r}")