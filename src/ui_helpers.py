"""
ui_helpers.py
-------------
Small presentation-layer helpers used by app.py:
- confidence_label(): turns a raw cosine similarity score into a
  human-readable badge (High / Medium / Low) instead of a bare float.
- build_report(): assembles a downloadable text report of a single
  Q&A interaction, including sources, for traceability/demo purposes.
"""

from datetime import datetime
from typing import List

from src.retriever import RetrievedChunk

APP_AUTHOR = "Hanzala Bajwa"

HIGH_THRESHOLD = 0.35
MEDIUM_THRESHOLD = 0.20


def confidence_label(score: float) -> str:
    """Map a cosine similarity score to a readable confidence badge."""
    if score >= HIGH_THRESHOLD:
        return "🟢 High"
    elif score >= MEDIUM_THRESHOLD:
        return "🟡 Medium"
    else:
        return "🔴 Low"


def overall_confidence(chunks: List[RetrievedChunk]) -> str:
    """Overall confidence for the answer, based on the best matching chunk."""
    if not chunks:
        return "🔴 Low"
    best_score = max(c.score for c in chunks)
    return confidence_label(best_score)


def build_report(
    question: str,
    answer: str,
    chunks: List[RetrievedChunk],
    doc_names: List[str],
) -> str:
    """Build a plain-text report of one Q&A exchange for download."""
    lines = []
    lines.append("=" * 60)
    lines.append("RAG SYSTEM - QUESTION & ANSWER REPORT")
    lines.append("=" * 60)
    lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"Documents searched: {', '.join(doc_names)}")
    lines.append(f"Overall confidence: {overall_confidence(chunks)}")
    lines.append("-" * 60)
    lines.append("QUESTION:")
    lines.append(question)
    lines.append("-" * 60)
    lines.append("ANSWER:")
    lines.append(answer)
    lines.append("-" * 60)
    lines.append(f"SOURCES USED ({len(chunks)}):")
    if not chunks:
        lines.append("  (none passed the similarity threshold)")
    for i, c in enumerate(chunks, start=1):
        lines.append("")
        lines.append(f"  [{i}] Document: {c.doc_name}")
        lines.append(f"      Page: {c.page_number}")
        lines.append(f"      Similarity score: {c.score:.3f}  ({confidence_label(c.score)})")
        lines.append(f"      Excerpt: {c.text[:400]}{'...' if len(c.text) > 400 else ''}")
    lines.append("-" * 60)
    lines.append(f"Developed by: {APP_AUTHOR}")
    lines.append("=" * 60)
    return "\n".join(lines)