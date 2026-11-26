import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from docqa.config import settings
from docqa.generation.answerer import Answerer, get_generation_client
from docqa.ingestion.pipeline import IngestionPipeline
from docqa.retrieval.embeddings import get_embedder
from docqa.retrieval.retriever import Retriever
from docqa.retrieval.vector_store import VectorStore

#services.py is responsible for creating and managing the core services used by the Doc Q&A application. It defines a `Services` dataclass that encapsulates the various components needed for document ingestion, retrieval, and question answering. The `build_services` function initializes these components, ensuring they are properly configured and ready for use when the application starts.
@dataclass
class Services:
    store: VectorStore
    pipeline: IngestionPipeline
    answerer: Answerer
    upload_dir: Path
    embedder_name: str = ""
    generation_label: str = ""
    max_upload_bytes: int = 50 * 1024 * 1024
    #one job at a time. A single small GPU can't safely run several embedding or LLM jobs at once (out-of-memory errors).
    work_lock: Any = field(default_factory=threading.Lock)

#build_services initializes the core services required for the Doc Q&A application. It sets up the generation client, embedder, vector store, ingestion pipeline, and answerer. The function ensures that these components are properly configured and ready for use when the application starts. If any misconfiguration is detected, it raises an error at startup.
def build_services() -> Services:
    """Create the real components once. Raises at startup if the backend is misconfigured."""
    client, model = get_generation_client()
    embedder = get_embedder()  #one model, shared by ingestion and questions
    store = VectorStore(embedder.name)
    return Services(
        store=store,
        pipeline=IngestionPipeline(embedder, store),
        answerer=Answerer(Retriever(embedder, store), client, model=model),
        upload_dir=settings.upload_dir,
        embedder_name=embedder.name,
        generation_label=f"{settings.generation_provider}:{model}",
        max_upload_bytes=settings.max_upload_mb * 1024 * 1024,
    )