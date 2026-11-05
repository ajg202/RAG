from dataclasses import dataclass
from docqa.ingestion.parser import Page
import re

#reusable regex pattern object
HEADING_RE = re.compile(r"^#{1,6}\s+(.*)$")

#defines the chunks and their breakdown
@dataclass
class Chunk:
    text: str
    page_number: int
    heading: str  #nearest heading above this chunk or "" if none

#splits the pages into sections based on headings, yielding (page_number, heading, text)
def split_sections(pages: list[Page]):
    """Yield (page_number, heading, text), starting a new section at each heading."""
    heading = ""
    for page in pages:
        buf: list[str] = []
        for line in page.text.splitlines():
            m = HEADING_RE.match(line)
            if m:
                if "".join(buf).strip():
                    yield page.page_number, heading, "\n".join(buf).strip()
                buf = []
                heading = m.group(1).strip()
            buf.append(line)
        if "".join(buf).strip():
            yield page.page_number, heading, "\n".join(buf).strip()

#splits a text into pieces of at most max_chars, with overlap between pieces
def split_by_size(text: str, max_chars: int, overlap: int) -> list[str]:
    if len(text) <= max_chars:
        return [text]
    pieces = []
    start = 0
    half = max_chars // 2
    while start < len(text):
        end = min(start + max_chars, len(text))
        if end < len(text):
            #prefer breaking at a paragraph, then a line, then a sentence
            for sep in ("\n\n", "\n", ". "):
                cut = text.rfind(sep, start, end)
                if cut > start + half:
                    end = cut + len(sep)
                    break
        piece = text[start:end].strip()
        if piece:
            pieces.append(piece)
        if end >= len(text):
            break
        start = max(end - overlap, start + 1)
    return pieces

#chunks the pages into pieces of at most max_chars, with overlap between pieces
def chunk_pages(pages: list[Page], max_chars: int = 2000, overlap: int = 200) -> list[Chunk]:
    chunks = []
    for page_number, heading, text in split_sections(pages):
        for piece in split_by_size(text, max_chars, overlap):
            chunks.append(Chunk(text=piece, page_number=page_number, heading=heading))
    return chunks

#runs the chunker on a PDF file and prints the results to stdout
if __name__ == "__main__":
    import sys
    from docqa.ingestion.parser import parse_pdf

    for i, c in enumerate(chunk_pages(parse_pdf(sys.argv[1]))):
        print(f"=== chunk {i} | page {c.page_number} | {c.heading!r} | {len(c.text)} chars ===")
        print(c.text, "\n")