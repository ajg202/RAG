from typing import Protocol

from docqa.config import settings

#embedding provider interface and implementation
class Embedder(Protocol):
    name: str  #used to keep vectors from different models apart

    def embed_documents(self, texts: list[str]) -> list[list[float]]: ...
    def embed_query(self, text: str) -> list[float]: ...

#local embedding provider implementation using Chroma's DefaultEmbeddingFunction
class LocalEmbedder:
    name = "local-minilm"

    def __init__(self) -> None:
        from chromadb.utils.embedding_functions import DefaultEmbeddingFunction

        self._fn = DefaultEmbeddingFunction()

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        return [[float(x) for x in vec] for vec in self._fn(texts)]

    def embed_query(self, text: str) -> list[float]:
        return self.embed_documents([text])[0]

#returns the appropriate embedder based on the configuration settings
def get_embedder() -> Embedder:
    if settings.embedding_provider == "local":
        return LocalEmbedder()
    raise ValueError(f"Unknown EMBEDDING_PROVIDER: {settings.embedding_provider!r}")

#initial test code to run the embedder from the command line
# if __name__ == "__main__":
#     import sys

#     if len(sys.argv) < 2:
#         raise SystemExit('Usage: python -m docqa.retrieval.embeddings "text to embed"')

#     embedder = get_embedder()
#     text = " ".join(sys.argv[1:])
#     vector = embedder.embed_query(text)

#     print(f"--- {embedder.name} ---")
#     print(vector)