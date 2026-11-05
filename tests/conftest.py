from pathlib import Path

import pytest

from docqa.ingestion.parser import parse_pdf

FIXTURE_PDF = Path(__file__).parent / "fixtures" / "sample.pdf"


@pytest.fixture(scope="session")
def sample_pages():
    return parse_pdf(FIXTURE_PDF)