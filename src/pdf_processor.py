"""
pdf_processor.py
----------------
Responsible ONLY for turning a raw PDF file into clean, page-tagged text.
Keeping this separate from chunking/embedding satisfies the
"separation of concerns" requirement.
"""

import re
from dataclasses import dataclass
from typing import List

from pypdf import PdfReader
from pypdf.errors import PdfReadError


class InvalidPDFError(Exception):
    """Raised when the uploaded file is not a readable PDF."""
    pass


@dataclass
class PageText:
    page_number: int   # 1-indexed, human friendly
    text: str


MAX_FILE_SIZE_MB = 20


def validate_pdf_size(file_bytes: bytes):
    size_mb = len(file_bytes) / (1024 * 1024)
    if size_mb > MAX_FILE_SIZE_MB:
        raise InvalidPDFError(
            f"File is {size_mb:.1f} MB, which exceeds the {MAX_FILE_SIZE_MB} MB limit."
        )


def clean_text(raw: str) -> str:
    """Remove common PDF-extraction artifacts."""
    if not raw:
        return ""
    text = raw.replace("\x00", " ")
    # Collapse hyphenated line-break words: "exam-\nple" -> "example"
    text = re.sub(r"-\n(?=[a-z])", "", text)
    # Normalise whitespace/newlines
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{2,}", "\n", text)
    text = re.sub(r"\n", " ", text)
    return text.strip()


def extract_pages(file_bytes: bytes, filename: str) -> List[PageText]:
    """
    Extract text from a PDF, page by page.
    Raises InvalidPDFError on corrupt/unsupported files or empty output.
    """
    validate_pdf_size(file_bytes)

    try:
        import io
        reader = PdfReader(io.BytesIO(file_bytes))
    except PdfReadError as e:
        raise InvalidPDFError(f"Could not read '{filename}': {e}")
    except Exception as e:
        raise InvalidPDFError(f"Unexpected error opening '{filename}': {e}")

    if reader.is_encrypted:
        try:
            reader.decrypt("")  # try empty password
        except Exception:
            raise InvalidPDFError(f"'{filename}' is password-protected.")

    pages = []
    for i, page in enumerate(reader.pages, start=1):
        raw = page.extract_text() or ""
        cleaned = clean_text(raw)
        if cleaned:
            pages.append(PageText(page_number=i, text=cleaned))

    if not pages:
        raise InvalidPDFError(
            f"No extractable text found in '{filename}'. "
            f"It may be a scanned/image-only PDF (would need OCR, out of scope here)."
        )

    return pages
