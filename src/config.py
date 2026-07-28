"""
config.py
----------
Centralised configuration and environment variable handling.
Keeping this in one place satisfies the "environment variable handling
for API keys" non-functional requirement and avoids scattering
os.getenv() calls across the codebase.
"""

import os
from dotenv import load_dotenv

load_dotenv()  # reads .env file if present (local dev); on Streamlit Cloud
                # these are set via st.secrets instead (see app.py)


class Config:
    # Pinecone
    PINECONE_API_KEY = os.getenv("PINECONE_API_KEY", "")
    PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "rag-intermediate-index")
    PINECONE_CLOUD = os.getenv("PINECONE_CLOUD", "aws")
    PINECONE_REGION = os.getenv("PINECONE_REGION", "us-east-1")

    # LLM
    GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
    GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")

    # Embeddings (Pinecone-hosted Inference API -- see src/embeddings.py)
    EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "multilingual-e5-large")
    EMBEDDING_DIM = 1024  # fixed dimension for multilingual-e5-large

    # Chunking defaults (overridable from the UI)
    DEFAULT_CHUNK_SIZE = 350       # characters (smaller = more focused chunks)
    DEFAULT_CHUNK_OVERLAP = 60     # characters
    DEFAULT_TOP_K = 6              # cast a wider net before the similarity gate
    DEFAULT_SIM_THRESHOLD = 0.15   # cosine similarity cutoff (tuned for multilingual-e5-large)

    @classmethod
    def validate(cls):
        """Raise a clear error early if required keys are missing."""
        missing = []
        if not cls.PINECONE_API_KEY:
            missing.append("PINECONE_API_KEY")
        if not cls.GROQ_API_KEY:
            missing.append("GROQ_API_KEY")
        if missing:
            raise EnvironmentError(
                f"Missing required environment variables: {', '.join(missing)}. "
                f"Add them to your .env file or Streamlit secrets."
            )