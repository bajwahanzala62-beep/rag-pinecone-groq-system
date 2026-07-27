"""
retriever.py
------------
Sits between the raw Pinecone query and the LLM generator.
Applies the similarity threshold and formats results for
both the prompt and the on-screen source attribution.
"""

from dataclasses import dataclass
from typing import List, Optional

from src.embeddings import embed_query
from src.vector_store import VectorStore


@dataclass
class RetrievedChunk:
    text: str
    page_number: int
    doc_name: str
    score: float


def retrieve(
    query: str,
    vector_store: VectorStore,
    doc_names: List[str],
    top_k: int = 4,
    similarity_threshold: float = 0.35,
    page_filter: Optional[int] = None,
) -> List[RetrievedChunk]:
    if not query or not query.strip():
        raise ValueError("Query cannot be empty.")

    query_vector = embed_query(query)
    raw_results = vector_store.query(
        query_vector=query_vector,
        doc_names=doc_names,
        top_k=top_k,
        page_filter=page_filter,
    )

    # Hallucination-prevention gate #1: drop chunks below the similarity
    # threshold BEFORE they ever reach the LLM prompt.
    filtered = [r for r in raw_results if r["score"] >= similarity_threshold]

    return [
        RetrievedChunk(
            text=r["text"],
            page_number=r["page_number"],
            doc_name=r["doc_name"],
            score=r["score"],
        )
        for r in filtered
    ]
