from typing import Protocol

from docqa.config import settings

import warnings

QUERY_TASK = "Given a question about a document, retrieve the passages that answer it"

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

#Qwen3 embedding provider implementation using sentence-transformers. It normalizes vectors and adds an instruction prefix for queries.
class Qwen3Embedder:
    name = "qwen3-embedding-0.6b"

    def __init__(self, batch_size: int = 16, max_seq_length: int = 1024) -> None:
        import torch
        from sentence_transformers import SentenceTransformer

        device = settings.embedding_device
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
            if device == "cpu":
                warnings.warn("CUDA not available; embedding on CPU will be slow.")
        dtype = torch.bfloat16 if device == "cuda" else torch.float32

        self._batch_size = batch_size
        self._model = SentenceTransformer(
            settings.embedding_model,
            device=device,
            model_kwargs={"torch_dtype": dtype},
            processor_kwargs={"padding_side": "left"},
        )
        self._model.max_seq_length = max_seq_length

    def _encode(self, texts: list[str], **kwargs) -> list[list[float]]:
        vecs = self._model.encode(
            texts,
            batch_size=self._batch_size,
            normalize_embeddings=True,
            show_progress_bar=False,
            **kwargs,
        )
        return vecs.tolist()

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        return self._encode(texts)

    def embed_query(self, text: str) -> list[float]:
        return self._encode([text], prompt=f"Instruct: {QUERY_TASK}\nQuery:")[0]

#get_embedder returns an instance of the configured embedding provider based on the settings. It supports "qwen3" and "local" providers, raising an error for unknown configurations.
def get_embedder() -> Embedder:
    if settings.embedding_provider == "qwen3":
        return Qwen3Embedder()
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