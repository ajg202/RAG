from typing import Annotated, Literal

from pydantic import BaseModel, Field, StringConstraints

Question = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=2000)]

#DocumentOut is the response model for a document upload, containing the document ID, source name, and number of chunks. IngestOut extends DocumentOut to include the number of pages and the status of the ingestion process. AskIn represents a question input with optional parameters for top_k and min_score. AskAllIn extends AskIn to include a list of document IDs. SourceOut provides details about a specific source passage, including its score and text. AskOut is the response model for an answer, containing the answer text, cited sources, number of retrieved passages, and whether the reply was truncated. HealthOut provides health check information about the service, including status, embedder name, generation label, and chunk count.
class DocumentOut(BaseModel):
    doc_id: str
    source: str
    chunks: int

#IngestOut extends DocumentOut to include the number of pages and the status of the ingestion process.
class IngestOut(DocumentOut):
    pages: int | None  #None when the file was already stored and parsing was skipped
    status: Literal["ingested", "replaced", "unchanged"]

#AskIn represents a question input with optional parameters for top_k and min_score.
class AskIn(BaseModel):
    question: Question
    top_k: int | None = Field(None, ge=1, le=20)
    min_score: float | None = Field(None, ge=-1.0, le=1.0)

#AskAllIn extends AskIn to include a list of document IDs.
class AskAllIn(AskIn):
    doc_ids: list[str] | None = Field(None, min_length=1, max_length=50)

#SourceOut provides details about a specific source passage, including its score and text.
class SourceOut(BaseModel):
    n: int  #the [n] marker used in the answer text
    doc_id: str
    source: str
    page_number: int
    heading: str
    score: float
    text: str

#AskOut is the response model for an answer, containing the answer text, cited sources, number of retrieved passages, and whether the reply was truncated.
class AskOut(BaseModel):
    answer: str
    cited: list[SourceOut]  # only the passages the answer cites
    retrieved: int          # how many passages were sent to the model
    truncated: bool         # True if the reply hit the token limit

#HealthOut provides health check information about the service, including status, embedder name, generation label, and chunk count.
class HealthOut(BaseModel):
    status: str
    embedder: str
    generation: str
    chunks: int