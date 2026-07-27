"""
src/embeddings.py
-----------------
Handles vector embedding generation using SentenceTransformers.
Uses Streamlit resource caching to ensure the model is only loaded into RAM
once per app session, avoiding high RAM usage and container crashes on Streamlit Cloud.
"""

from typing import List
import streamlit as st
from sentence_transformers import SentenceTransformer
from src.config import Config


@st.cache_resource(show_spinner="Loading embedding model...")
def get_embedding_model() -> SentenceTransformer:
    """
    Loads and caches the SentenceTransformer model in memory.
    """
    model_name = getattr(Config, "EMBEDDING_MODEL", "all-MiniLM-L6-v2")
    return SentenceTransformer(model_name)


def embed_texts(texts: List[str]) -> List[List[float]]:
    """
    Generates vector embeddings for a list of text strings.

    Args:
        texts (List[str]): List of text strings/chunks to embed.

    Returns:
        List[List[float]]: A list of float vector embeddings.
    """
    if not texts:
        return []

    try:
        model = get_embedding_model()
        # Batch processing keeps memory usage controlled on free-tier containers
        embeddings = model.encode(
            texts,
            batch_size=32,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        return embeddings.tolist()
    except Exception as e:
        raise RuntimeError(f"Failed to generate embeddings: {str(e)}") from e