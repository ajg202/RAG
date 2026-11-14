import re
from dataclasses import dataclass, field

import anthropic

from docqa.generation.ollama_client import OllamaClient

from docqa.config import settings
from docqa.generation.prompts import SYSTEM_PROMPT, build_user_message
from docqa.retrieval.retriever import get_retriever
from docqa.retrieval.vector_store import SearchResult

NOT_FOUND = "I couldn't find anything in the documents that answers that question."

CITATION_RE = re.compile(r"\[(\d+(?:\s*,\s*\d+)*)\]")

#dataclass Answer represents the result of answering a question. It includes the answer text, the sources used, the indices of cited sources, and whether the answer was truncated due to token limits.
@dataclass
class Answer:
    text: str
    sources: list[SearchResult] = field(default_factory=list)  #everything sent to the model; [1] is sources[0]
    cited: list[int] = field(default_factory=list)             #source numbers the answer actually cites
    truncated: bool = False                                    #True if the reply hit max_tokens

    @property
    def cited_sources(self) -> list[SearchResult]:
        return [self.sources[n - 1] for n in self.cited]

#extract_citations finds all source numbers cited in the answer text, ensuring they are valid and unique. It returns a list of integers representing the cited source indices.
def extract_citations(text: str, n_sources: int) -> list[int]:
    """Source numbers cited as [1], [2][3] or [1, 2], in order of first use.
    Numbers that don't match a real source are dropped."""
    found: list[int] = []
    for match in CITATION_RE.finditer(text):
        for num in re.findall(r"\d+", match.group(1)):
            n = int(num)
            if 1 <= n <= n_sources and n not in found:
                found.append(n)
    return found

#Answerer is a class that handles answering questions based on document excerpts. It uses a retriever to find relevant document chunks and an API client to generate answers using a language model. The answer method retrieves sources, constructs a user message, and calls the model to generate an answer, returning an Answer object with the result.
class Answerer:
    def __init__(self, retriever, client, model: str | None = None, max_tokens: int = 1024) -> None:
        self._retriever = retriever
        self._client = client
        self._model = model or settings.claude_model
        self._max_tokens = max_tokens

    #answer takes a question and retrieves relevant document chunks, then generates an answer using a language model. It returns an Answer object containing the answer text, sources, cited source indices, and whether the answer was truncated.
    def answer(
        self,
        question: str,
        doc_ids: list[str] | None = None,
        top_k: int | None = None,
        min_score: float | None = None,
    ) -> Answer:
        question = question.strip()
        if not question:
            raise ValueError("Question is empty")

        sources = self._retriever.retrieve(
            question, top_k=top_k, doc_ids=doc_ids, min_score=min_score
        )
        if not sources:
            # Nothing to answer from, so don't pay for an API call
            return Answer(text=NOT_FOUND)

        response = self._client.messages.create(
            model=self._model,
            max_tokens=self._max_tokens,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": build_user_message(question, sources)}],
        )
        text = "".join(b.text for b in response.content if b.type == "text").strip()
        return Answer(
            text=text,
            sources=sources,
            cited=extract_citations(text, len(sources)),
            truncated=response.stop_reason == "max_tokens",
        )

#get_answerer constructs an Answerer instance based on the configured backend (Ollama or Anthropic). It initializes the appropriate API client and retrieves the retriever for document retrieval, returning a fully configured Answerer ready to answer questions.
def get_answerer() -> Answerer:
    provider = settings.generation_provider
    if provider == "ollama":
        client, model = OllamaClient(), settings.ollama_model
    elif provider == "anthropic":
        if not settings.anthropic_api_key:
            raise RuntimeError(
                "ANTHROPIC_API_KEY is not set. Copy .env.example to .env and add your key."
            )
        import anthropic

        client, model = anthropic.Anthropic(api_key=settings.anthropic_api_key), settings.claude_model
    else:
        raise ValueError(f"Unknown GENERATION_PROVIDER: {provider!r}")
    #backend is validated first: building the retriever loads the embedding model
    return Answerer(get_retriever(), client, model=model)

#_main_ block allows the script to be run from the command line. It takes a question as input, uses the Answerer to generate an answer, and prints the answer along with any cited sources. If the answer is truncated due to token limits, it notifies the user.
if __name__ == "__main__":
    import sys

    result = get_answerer().answer(" ".join(sys.argv[1:]))
    print(result.text)
    if result.truncated:
        print("\n(Answer was cut off at the token limit.)")
    print("\nSources:")
    for n in result.cited:
        s = result.sources[n - 1]
        print(f"  [{n}] page {s.page_number} | {s.heading!r}")