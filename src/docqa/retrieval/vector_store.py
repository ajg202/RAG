import re
from dataclasses import dataclass
from pathlib import Path

import chromadb
from chromadb.config import Settings as ChromaSettings

from docqa.config import settings
from docqa.ingestion.chunker import Chunk

#chroma has a limit on how many items can be added in one call, so we batch them.
BATCH_SIZE = 500

#class to hold the results of a vector search, including the text, page number, heading, document ID, chunk index, and score.
@dataclass
class SearchResult:
    text: str
    page_number: int
    heading: str
    doc_id: str
    chunk_index: int
    score: float  

#class to manage a vector store using ChromaDB, allowing for adding, querying, and deleting documents and their associated chunks and embeddings.
class VectorStore:
    def __init__(self, embedder_name: str, path: Path | str | None = None) -> None:
        path = Path(path) if path else settings.chroma_dir
        path.mkdir(parents=True, exist_ok=True)
        self._client = chromadb.PersistentClient(
            path=str(path),
            settings=ChromaSettings(anonymized_telemetry=False),
        )
        #one collection per embedding model, so vectors from different models are never compared with each other.
        safe_name = re.sub(r"[^a-zA-Z0-9_-]", "-", embedder_name)
        self._col = self._client.get_or_create_collection(
            name=f"docs-{safe_name}",
            metadata={"hnsw:space": "cosine"},
            embedding_function=None,
        )

    #static method to create a query filter for document IDs, returning None if no filter is needed, a single ID if only one is provided, or a list of IDs if multiple are provided.
    @staticmethod
    def _where(doc_ids: list[str] | None) -> dict | None:
        if doc_ids is None:
            return None
        if len(doc_ids) == 1:
            return {"doc_id": doc_ids[0]}
        return {"doc_id": {"$in": list(doc_ids)}}

    #def method to count the number of stored chunks, optionally limited to some documents. If no document IDs are provided, it counts all chunks. If an empty list is provided, it returns 0. Otherwise, it counts the chunks associated with the specified document IDs.
    def count(self, doc_ids: list[str] | None = None) -> int:
        if doc_ids is None:
            return self._col.count()
        if not doc_ids:
            return 0
        return len(self._col.get(where=self._where(doc_ids), include=[])["ids"])

    #def method to add a document's chunks and their embeddings to the vector store. If the same document ID is added again, it replaces the existing entry. The method checks that the number of chunks matches the number of embeddings, deletes any existing document with the same ID, and then adds the new chunks in batches to avoid exceeding ChromaDB's limits.
    def add_document(
        self,
        doc_id: str,
        chunks: list[Chunk],
        embeddings: list[list[float]],
        source: str = "",
    ) -> int:
        if len(chunks) != len(embeddings):
            raise ValueError(f"{len(chunks)} chunks but {len(embeddings)} embeddings")
        self.delete_document(doc_id)
        for start in range(0, len(chunks), BATCH_SIZE):
            batch = range(start, min(start + BATCH_SIZE, len(chunks)))
            self._col.add(
                ids=[f"{doc_id}-{i:05d}" for i in batch],
                documents=[chunks[i].text for i in batch],
                embeddings=[embeddings[i] for i in batch],
                metadatas=[
                    {
                        "doc_id": doc_id,
                        "source": source,
                        "page_number": chunks[i].page_number,
                        "heading": chunks[i].heading,
                        "chunk_index": i,
                    }
                    for i in batch
                ],
            )
        return len(chunks)

    #def method to query the vector store for the closest chunks to a given embedding. It returns a list of SearchResult objects, sorted by similarity score. The method allows filtering by document IDs and limits the number of results returned based on the top_k parameter or a default setting.
    def query(
        self,
        embedding: list[float],
        top_k: int | None = None,
        doc_ids: list[str] | None = None,
    ) -> list[SearchResult]:
        top_k = settings.top_k if top_k is None else top_k
        available = self.count(doc_ids)
        if available == 0 or top_k <= 0:
            return []
        res = self._col.query(
            query_embeddings=[embedding],
            n_results=min(top_k, available),
            where=self._where(doc_ids),
            include=["documents", "metadatas", "distances"],
        )
        return [
            SearchResult(
                text=text,
                page_number=int(meta["page_number"]),
                heading=meta["heading"],
                doc_id=meta["doc_id"],
                chunk_index=int(meta["chunk_index"]),
                score=1.0 - dist,
            )
            for text, meta, dist in zip(
                res["documents"][0], res["metadatas"][0], res["distances"][0]
            )
        ]

    #def delete_documents(self, doc_ids: list[str]) -> int:
    def delete_document(self, doc_id: str) -> int:
        ids = self._col.get(where={"doc_id": doc_id}, include=[])["ids"]
        if ids:
            self._col.delete(ids=ids)
        return len(ids)

    #def method to list all stored documents, returning a list of dictionaries with each document's ID, source, and the number of chunks it contains. It aggregates the metadata from the stored chunks to provide a summary of each document.
    def list_documents(self) -> list[dict]:
        """One entry per stored document: {'doc_id', 'source', 'chunks'}."""
        docs: dict[str, dict] = {}
        for meta in self._col.get(include=["metadatas"])["metadatas"]:
            d = docs.setdefault(
                meta["doc_id"],
                {"doc_id": meta["doc_id"], "source": meta["source"], "chunks": 0},
            )
            d["chunks"] += 1
        return list(docs.values())