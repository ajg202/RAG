import re
import tempfile
from dataclasses import asdict
from pathlib import Path
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Request, Response, UploadFile
from fastapi import Path as PathParam

from docqa.api.schemas import (
    AskAllIn, AskIn, AskOut, DocumentOut, HealthOut, IngestOut, SourceOut,
)
from docqa.api.services import Services

router = APIRouter()

DocId = Annotated[str, PathParam(pattern=r"^[0-9a-f]{16}$")]


def get_services(request: Request) -> Services:
    return request.app.state.services


ServicesDep = Annotated[Services, Depends(get_services)]

#clean_source_name sanitizes the uploaded file name by removing folder paths, control characters, and limiting its length to 200 characters. If the resulting name is empty, it defaults to "upload.pdf".
def clean_source_name(filename: str | None) -> str:
    """Display name for an upload: no folders, no control characters, bounded length."""
    name = (filename or "").replace("\\", "/").rsplit("/", 1)[-1]
    name = re.sub(r"[\x00-\x1f\x7f]", "", name).strip()
    return name[:200] or "upload.pdf"

#save_upload saves the uploaded file to the specified destination path while enforcing a maximum file size limit. It reads the file in chunks and raises an HTTPException with status code 413 if the file exceeds the specified maximum size.
def save_upload(upload: UploadFile, dest: Path, max_bytes: int) -> None:
    size = 0
    with open(dest, "wb") as out:
        while block := upload.file.read(1 << 20):
            size += len(block)
            if size > max_bytes:
                raise HTTPException(413, f"File is too large (limit {max_bytes / 1_048_576:.0f} MB).")
            out.write(block)

#router.get("/health", response_model=HealthOut) is an endpoint that returns the health status of the service, including the embedder name, generation label, and the number of chunks in the vector store. It uses the ServicesDep dependency to access the core services.
@router.get("/health", response_model=HealthOut)
def health(services: ServicesDep):
    return HealthOut(
        status="ok",
        embedder=services.embedder_name,
        generation=services.generation_label,
        chunks=services.store.count(),
    )

#router.post("/documents", response_model=IngestOut) is an endpoint that handles document uploads. It saves the uploaded file to a temporary directory, processes it through the ingestion pipeline, and returns the ingestion result. It also enforces a maximum file size limit and handles errors related to file size and ingestion.
@router.post("/documents", response_model=IngestOut)
def upload_document(
    services: ServicesDep, response: Response, file: UploadFile, force: bool = False
):
    source = clean_source_name(file.filename)
    # The upload is saved under a fixed name: the client's file name is never used as a path.
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        path = Path(tmp) / "upload.pdf"
        save_upload(file, path, services.max_upload_bytes)
        try:
            with services.work_lock:
                result = services.pipeline.ingest(path, force=force, source=source)
        except ValueError as e:
            raise HTTPException(status_code=422, detail=str(e)) from e
    response.status_code = 201 if result.status == "ingested" else 200
    return IngestOut(**asdict(result))

#router.get("/documents", response_model=list[DocumentOut]) is an endpoint that lists all ingested documents. It retrieves the list of documents from the vector store and returns them sorted by source name.
@router.get("/documents", response_model=list[DocumentOut])
def list_documents(services: ServicesDep):
    return sorted(services.store.list_documents(), key=lambda d: d["source"].lower())

#router.delete("/documents/{doc_id}", status_code=204) is an endpoint that deletes a specific document by its ID. It uses the work_lock to ensure thread safety during the deletion process. If the document is not found, it raises an HTTPException with status code 404. It also removes the corresponding PDF file from the upload directory if it exists.
@router.delete("/documents/{doc_id}", status_code=204)
def delete_document(doc_id: DocId, services: ServicesDep):
    with services.work_lock:
        removed = services.store.delete_document(doc_id)
    if not removed:
        raise HTTPException(404, f"No stored document with id {doc_id}")
    (Path(services.upload_dir) / f"{doc_id}.pdf").unlink(missing_ok=True)

#_ask is a helper function that processes a question input and retrieves an answer from the answerer service. It checks for the existence of documents, validates document IDs, and handles potential runtime errors. The function returns an AskOut response containing the answer text, cited sources, number of retrieved passages, and whether the reply was truncated.
def _ask(services: Services, body: AskIn, doc_ids: list[str] | None) -> AskOut:
    names = {d["doc_id"]: d["source"] for d in services.store.list_documents()}
    if not names:
        raise HTTPException(409, "No documents have been ingested yet. Upload a PDF first.")
    missing = [d for d in doc_ids or [] if d not in names]
    if missing:
        raise HTTPException(404, f"Unknown document id(s): {', '.join(missing)}")

    try:
        with services.work_lock:
            answer = services.answerer.answer(
                body.question, doc_ids=doc_ids, top_k=body.top_k, min_score=body.min_score
            )
    except RuntimeError as e:  # Ollama not running, model missing, no API key...
        raise HTTPException(503, str(e)) from e

    cited = []
    for n in answer.cited:
        s = answer.sources[n - 1]
        cited.append(SourceOut(
            n=n, doc_id=s.doc_id, source=names.get(s.doc_id, s.doc_id),
            page_number=s.page_number, heading=s.heading,
            score=round(s.score, 4), text=s.text,
        ))
    return AskOut(
        answer=answer.text, cited=cited, retrieved=len(answer.sources), truncated=answer.truncated
    )

#router.post("/ask", response_model=AskOut) is an endpoint that handles question answering across all ingested documents. It accepts a question input and optional parameters for top_k and min_score. The endpoint uses the _ask helper function to retrieve an answer and returns the result in an AskOut response model.
@router.post("/ask", response_model=AskOut)
def ask(body: AskAllIn, services: ServicesDep):
    return _ask(services, body, body.doc_ids)

#router.post("/documents/{doc_id}/ask", response_model=AskOut) is an endpoint that handles question answering for a specific document. It accepts a question input and optional parameters for top_k and min_score. The endpoint uses the _ask helper function to retrieve an answer for the specified document ID and returns the result in an AskOut response model.
@router.post("/documents/{doc_id}/ask", response_model=AskOut)
def ask_document(doc_id: DocId, body: AskIn, services: ServicesDep):
    return _ask(services, body, [doc_id])