"""
chunker.py
----------
Splits page-level text into overlapping chunks suitable for embedding.

Design decisions:
1. Sentence-aware sliding window chunking.
   - Splitting purely by character count can cut sentences in half, hurting
     retrieval quality.
   - Splitting purely by sentence can create wildly uneven chunk sizes.
   This implementation packs whole sentences into a window up to `chunk_size`
   characters, then overlaps the last `chunk_overlap` characters worth of
   sentences into the next chunk so context isn't lost at chunk boundaries.
2. Section-heading-aware breaks.
   - Many documents (reports, proposals) use numbered headings like
     "2. Group Members". If a short section like that gets packed into a
     500-character chunk alongside a lot of unrelated surrounding text,
     its specific content (e.g. two names) gets semantically diluted when
     the whole chunk is embedded as one vector -- hurting retrieval for
     queries specifically about that section. Detecting heading lines and
     forcing a new chunk to start there keeps short, distinct sections
     as their own concentrated, well-represented chunks.
"""

import re
import uuid
from dataclasses import dataclass, field
from typing import List

from src.pdf_processor import PageText


@dataclass
class Chunk:
    chunk_id: str
    text: str
    page_number: int
    doc_name: str
    chunk_index: int


_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")
# Matches numbered section headings like "2. Group Members" or "10. Conclusion"
_HEADING_RE = re.compile(r"^\d{1,2}\.\s+[A-Z]")


def _split_sentences(text: str) -> List[str]:
    sentences = _SENTENCE_SPLIT_RE.split(text)
    return [s.strip() for s in sentences if s.strip()]


def _is_heading(sentence: str) -> bool:
    return bool(_HEADING_RE.match(sentence))


def chunk_document(
    pages: List[PageText],
    doc_name: str,
    chunk_size: int = 500,
    chunk_overlap: int = 50,
) -> List[Chunk]:
    """
    Turn a list of PageText into a list of Chunks with metadata.
    chunk_size / chunk_overlap are in characters (adjustable from the UI).
    """
    chunks: List[Chunk] = []
    chunk_index = 0

    def flush(current_text: str):
        nonlocal chunk_index
        if not current_text.strip():
            return
        chunks.append(
            Chunk(
                chunk_id=str(uuid.uuid4()),
                text=current_text.strip(),
                page_number=page.page_number,
                doc_name=doc_name,
                chunk_index=chunk_index,
            )
        )
        chunk_index += 1

    for page in pages:
        sentences = _split_sentences(page.text)
        current = ""

        for sentence in sentences:
            # Force a fresh chunk boundary at section headings, so short
            # sections (e.g. "Group Members") don't get diluted inside a
            # large chunk full of unrelated surrounding text.
            if _is_heading(sentence) and current.strip():
                flush(current)
                current = ""

            candidate = f"{current} {sentence}".strip() if current else sentence

            if len(candidate) <= chunk_size or not current:
                # Either it fits, or the buffer is empty (in which case we
                # accept it even if it's larger than chunk_size -- a single
                # oversized sentence becomes its own chunk rather than
                # looping forever trying to shrink it).
                current = candidate
            else:
                flush(current)
                overlap_text = current[-chunk_overlap:] if chunk_overlap > 0 else ""
                # Start the new buffer with the overlap AND this sentence
                # immediately -- never deferred to a future loop pass, so
                # every sentence is guaranteed to be consumed exactly once.
                current = f"{overlap_text} {sentence}".strip() if overlap_text else sentence

        flush(current)

    return chunks