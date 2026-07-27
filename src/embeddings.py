"""
embeddings.py
-------------
Wraps the embedding model so the rest of the app never touches
sentence-transformers directly. Swapping models later (e.g. to an
OpenAI embedding API) only requires changes here.

Model choice: all-MiniLM-L6-v2
- 384 dimensions -> cheap to store/query in Pinecone
- Runs on CPU at good speed -> no GPU/API cost required
- Strong general-purpose semantic similarity performance for its size
"""

from functools import lru_cache
from typing import List

import numpy as np
from sentence_transformers import SentenceTransformer

from src.config import Config


@lru_cache(maxsize=1)
def _get_model() -> SentenceTransformer:
    # Cached so the ~80MB model loads only once per process,
    # not on every embedding call (important in a Streamlit app
    # that reruns the script on every interaction).
    return SentenceTransformer(Config.EMBEDDING_MODEL)


def embed_texts(texts: List[str]) -> List[List[float]]:
    """Embed a batch of texts. Returns list of float vectors."""
    if not texts:
        return []
    model = _get_model()
    vectors = model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
    return vectors.tolist()


def embed_query(query: str) -> List[float]:
    """Embed a single user query the same way chunks were embedded."""
    model = _get_model()
    vector = model.encode([query], normalize_embeddings=True, show_progress_bar=False)[0]
    return vector.tolist()
