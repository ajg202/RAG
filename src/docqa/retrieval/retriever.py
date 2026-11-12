from docqa.retrieval.embeddings import Embedder, get_embedder
from docqa.retrieval.vector_store import SearchResult, VectorStore

#retriever is responsible for taking a question, embedding it, and retrieving the most relevant chunks from the vector store.
class Retriever:
    def __init__(self, embedder: Embedder, store: VectorStore) -> None:
        self._embedder = embedder
        self._store = store

    #retrieve the most relevant chunks for a given question.
    def retrieve(
        self,
        question: str,
        top_k: int | None = None,
        doc_ids: list[str] | None = None,
        min_score: float | None = None,
    ) -> list[SearchResult]:
        question = question.strip()
        if not question:
            return []
        if doc_ids is not None and not doc_ids:
            return []

        vector = self._embedder.embed_query(question)
        results = self._store.query(vector, top_k=top_k, doc_ids=doc_ids)
        if min_score is not None:
            results = [r for r in results if r.score >= min_score]
        return results

#get_retriever builds a retriever from the configured embedder and the default database.
def get_retriever() -> Retriever:
    embedder = get_embedder()
    return Retriever(embedder, VectorStore(embedder.name))