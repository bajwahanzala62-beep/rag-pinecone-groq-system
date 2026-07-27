"""
chunker.py
----------
Splits page-level text into overlapping chunks suitable for embedding.

Design decision: sentence-aware sliding window chunking.
- Splitting purely by character count can cut sentences in half, hurting
  retrieval quality.
- Splitting purely by sentence can create wildly uneven chunk sizes.
This implementation packs whole sentences into a window up to `chunk_size`
characters, then overlaps the last `chunk_overlap` characters worth of
sentences into the next chunk so context isn't lost at chunk boundaries.
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


def _split_sentences(text: str) -> List[str]:
    sentences = _SENTENCE_SPLIT_RE.split(text)
    return [s.strip() for s in sentences if s.strip()]


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

    for page in pages:
        sentences = _split_sentences(page.text)
        current = ""
        i = 0
        while i < len(sentences):
            sentence = sentences[i]
            if len(current) + len(sentence) + 1 <= chunk_size or not current:
                current = f"{current} {sentence}".strip()
                i += 1
            else:
                chunks.append(
                    Chunk(
                        chunk_id=str(uuid.uuid4()),
                        text=current,
                        page_number=page.page_number,
                        doc_name=doc_name,
                        chunk_index=chunk_index,
                    )
                )
                chunk_index += 1
                # build overlap: take trailing characters of current chunk
                overlap_text = current[-chunk_overlap:] if chunk_overlap > 0 else ""
                current = overlap_text.strip()

        if current:
            chunks.append(
                Chunk(
                    chunk_id=str(uuid.uuid4()),
                    text=current,
                    page_number=page.page_number,
                    doc_name=doc_name,
                    chunk_index=chunk_index,
                )
            )
            chunk_index += 1

    return chunks
