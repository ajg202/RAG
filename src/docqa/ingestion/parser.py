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

#decides whether OCR is needed: True if the PDF has a real text layer. Scanned (image-only) PDFs have almost no extractable text, so they return False.
def _has_text_layer(path: str | Path) -> bool:
    import pymupdf

    with pymupdf.open(str(path)) as doc:
        total = sum(len(page.get_text().strip()) for page in doc)
        return total >= 50 * max(len(doc), 1)

#parse_pdf uses the pymupdf4llm library to parse a PDF file into a list of Page objects. It checks if the PDF has a text layer and uses OCR if necessary. The text is cleaned to remove trailing page numbers and underlines.
def parse_pdf(path: str | Path) -> list[Page]:
    pymupdf4llm = import_module("pymupdf4llm")
    chunks = pymupdf4llm.to_markdown(
        str(path),
        page_chunks=True,
        use_ocr=not _has_text_layer(path),
    )
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
# if __name__ == "__main__":
#     import sys
#     for p in parse_pdf(sys.argv[1]):
#         print(f"--- page {p.page_number} ---\n{p.text}")