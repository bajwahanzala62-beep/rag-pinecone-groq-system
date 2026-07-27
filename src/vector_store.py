"""
vector_store.py
----------------
All Pinecone-specific logic lives here: index creation, upserting,
querying, metadata filtering, and namespace management.

Namespace design decision:
Each uploaded document gets its OWN namespace (derived from a hash of
the filename). This gives us free "multi-document support" and lets us
scope a query to one document, several documents, or the whole index,
without needing separate indexes per file.
"""

import hashlib
import time
from typing import List, Dict, Optional

from pinecone import Pinecone, ServerlessSpec

from src.config import Config
from src.chunker import Chunk


class PineconeConnectionError(Exception):
    pass


def _namespace_for(doc_name: str) -> str:
    """Deterministic, filesystem-safe namespace per document."""
    digest = hashlib.md5(doc_name.encode()).hexdigest()[:10]
    return f"doc-{digest}"


class VectorStore:
    def __init__(self):
        try:
            self.pc = Pinecone(api_key=Config.PINECONE_API_KEY)
        except Exception as e:
            raise PineconeConnectionError(f"Could not initialise Pinecone client: {e}")

        self.index_name = Config.PINECONE_INDEX_NAME
        self._ensure_index_exists()
        self.index = self.pc.Index(self.index_name)

    def _ensure_index_exists(self):
        try:
            existing = [idx["name"] for idx in self.pc.list_indexes()]
        except Exception as e:
            raise PineconeConnectionError(f"Could not reach Pinecone: {e}")

        if self.index_name not in existing:
            self.pc.create_index(
                name=self.index_name,
                dimension=Config.EMBEDDING_DIM,
                metric="cosine",
                spec=ServerlessSpec(
                    cloud=Config.PINECONE_CLOUD,
                    region=Config.PINECONE_REGION,
                ),
            )
            # Wait for the index to be ready before returning
            while not self.pc.describe_index(self.index_name).status["ready"]:
                time.sleep(1)

    def upsert_chunks(self, chunks: List[Chunk], vectors: List[List[float]]):
        """Upsert chunk vectors + metadata into the document's namespace."""
        if len(chunks) != len(vectors):
            raise ValueError("chunks and vectors length mismatch")
        if not chunks:
            return

        namespace = _namespace_for(chunks[0].doc_name)
        payload = []
        for chunk, vector in zip(chunks, vectors):
            payload.append({
                "id": chunk.chunk_id,
                "values": vector,
                "metadata": {
                    "text": chunk.text,
                    "page_number": chunk.page_number,
                    "doc_name": chunk.doc_name,
                    "chunk_index": chunk.chunk_index,
                },
            })

        # Batch upserts (Pinecone recommends <= 100 vectors per request)
        batch_size = 100
        try:
            for i in range(0, len(payload), batch_size):
                self.index.upsert(vectors=payload[i:i + batch_size], namespace=namespace)
        except Exception as e:
            raise PineconeConnectionError(
                f"Failed to upsert vectors to Pinecone (namespace '{namespace}'): {e}"
            )

    def query(
        self,
        query_vector: List[float],
        doc_names: List[str],
        top_k: int = 4,
        page_filter: Optional[int] = None,
    ) -> List[Dict]:
        """
        Query across one or more document namespaces and merge results
        by similarity score (Pinecone free/serverless tier queries one
        namespace per call, so we fan out and merge client-side).
        """
        results = []
        metadata_filter = {"page_number": page_filter} if page_filter else None

        for doc_name in doc_names:
            namespace = _namespace_for(doc_name)
            try:
                response = self.index.query(
                    vector=query_vector,
                    top_k=top_k,
                    namespace=namespace,
                    include_metadata=True,
                    filter=metadata_filter,
                )
            except Exception as e:
                raise PineconeConnectionError(f"Pinecone query failed: {e}")

            for match in response.get("matches", []):
                results.append({
                    "score": match["score"],
                    "text": match["metadata"]["text"],
                    "page_number": match["metadata"]["page_number"],
                    "doc_name": match["metadata"]["doc_name"],
                })

        results.sort(key=lambda r: r["score"], reverse=True)
        return results[:top_k]

    def list_indexed_documents(self) -> List[str]:
        """
        Pinecone doesn't let us list namespaces -> document names directly
        pre-query, so the app tracks uploaded doc names in session_state
        and this store is queried per-document. This helper is kept for
        completeness / future extension (e.g. via describe_index_stats).
        """
        stats = self.index.describe_index_stats()
        return list(stats.get("namespaces", {}).keys())