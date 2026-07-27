"""
embeddings.py
-------------
Uses Pinecone's hosted Inference API to generate embeddings, instead of
loading a local sentence-transformers/PyTorch model into the app's own
process.

Why: local embedding models (PyTorch + sentence-transformers) have a
large memory footprint (several hundred MB just to import/load), which
was causing out-of-memory crashes on free-tier hosting (e.g. Streamlit
Community Cloud's ~1GB RAM limit). Pinecone computes the embeddings on
its own servers instead -- this app just sends text and receives
vectors back, keeping our own memory usage minimal.

Model choice: multilingual-e5-large (1024 dimensions)
- Hosted by Pinecone, no local model weights to download/load
- Strong general-purpose semantic similarity performance
"""

from typing import List

from pinecone import Pinecone

from src.config import Config

_BATCH_SIZE = 90  # Pinecone's inference API caps inputs per request


def _get_client() -> Pinecone:
    return Pinecone(api_key=Config.PINECONE_API_KEY)


def _extract_vector(item) -> List[float]:
    """Handle both dict-like and attribute-style response items."""
    if isinstance(item, dict):
        return item["values"]
    return item.values


def embed_texts(texts: List[str]) -> List[List[float]]:
    """Embed a batch of texts (document chunks). Returns list of float vectors."""
    if not texts:
        return []
    pc = _get_client()
    vectors: List[List[float]] = []
    try:
        for i in range(0, len(texts), _BATCH_SIZE):
            batch = texts[i:i + _BATCH_SIZE]
            result = pc.inference.embed(
                model=Config.EMBEDDING_MODEL,
                inputs=batch,
                parameters={"input_type": "passage", "truncate": "END"},
            )
            vectors.extend(_extract_vector(item) for item in result)
    except Exception as e:
        raise RuntimeError(f"Embedding generation failed (Pinecone Inference API): {e}")
    return vectors


def embed_query(query: str) -> List[float]:
    """Embed a single user query the same way chunks were embedded."""
    pc = _get_client()
    try:
        result = pc.inference.embed(
            model=Config.EMBEDDING_MODEL,
            inputs=[query],
            parameters={"input_type": "query", "truncate": "END"},
        )
    except Exception as e:
        raise RuntimeError(f"Embedding generation failed (Pinecone Inference API): {e}")
    return _extract_vector(result[0])