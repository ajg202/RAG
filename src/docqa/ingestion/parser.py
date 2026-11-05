from dataclasses import dataclass
from importlib import import_module
from pathlib import Path

import re

#reusable regex pattern object
PAGE_NUM_RE = re.compile(r"^\s*\d{1,3}\s*$")

#defines the pages and their breakdown
@dataclass
class Page:
    page_number: int    #1-based, page numbers for reference
    text: str           #text for the page

#parses the pdf into its respective sections
def parse_pdf(path: str | Path) -> list[Page]:
    pymupdf4llm = import_module("pymupdf4llm")
    chunks = pymupdf4llm.to_markdown(str(path), page_chunks=True)
    return [Page(page_number=i, text=clean_text(c["text"])) for i, c in enumerate(chunks, start=1)]

#cleans the text of a page, removing trailing page numbers and underlines
def clean_text(text: str) -> str:
    lines = text.splitlines()
    while lines and not lines[-1].strip():
        lines.pop()
    if lines and PAGE_NUM_RE.match(lines[-1]):  #bare page number at the end
        lines.pop()
    text = "\n".join(lines)
    text = re.sub(r"</?u>", "", text)
    return text.strip()

#runs the parser on a PDF file and prints the results to stdout
if __name__ == "__main__":
    import sys
    for p in parse_pdf(sys.argv[1]):
        print(f"--- page {p.page_number} ---\n{p.text}")