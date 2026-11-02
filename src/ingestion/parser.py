from dataclasses import dataclass
from importlib import import_module
from pathlib import Path

#defines the pages and their breakdown
@dataclass
class Page:
    page_number: int    #1-based, page numbers for reference
    text: str           #text for the page

#parses the pdf into its respective sections
def parse_pdf(path: str | Path) -> list[Page]:
    pymupdf4llm = import_module("pymupdf4llm")
    chunks = pymupdf4llm.to_markdown(str(path), page_chunks=True)
    return [Page(page_number=i, text=c["text"]) for i, c in enumerate(chunks, start=1)]

if __name__ == "__main__":
    import sys
    for p in parse_pdf(sys.argv[1]):
        print(f"--- page {p.page_number} ---\n{p.text}")