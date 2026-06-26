"""
vector_store.py
───────────────
Embeds text chunks using sentence-transformers and indexes them with FAISS.
Supports similarity search with metadata retrieval.
"""

import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from typing import List, Tuple
from src.document_processor import Chunk


class VectorStore:
    """
    Wraps FAISS flat L2 index with sentence-transformer embeddings.
    Build once, query many times.
    """

    MODEL_NAME = "all-MiniLM-L6-v2"   # 384-dim, fast, good quality

    def __init__(self):
        self.model = SentenceTransformer(self.MODEL_NAME)
        self.index = None
        self.chunks: List[Chunk] = []
        self.dimension = 384

    # ── Build ─────────────────────────────────────────────────────────────────
    def build(self, chunks: List[Chunk]) -> None:
        """Embed all chunks and build FAISS IndexFlatIP (cosine via normalised L2)."""
        if not chunks:
            raise ValueError("No chunks to index.")

        self.chunks = chunks
        texts = [c.text for c in chunks]

        # Encode in batches
        print(f"[VectorStore] Encoding {len(texts)} chunks...")
        embeddings = self.model.encode(
            texts,
            batch_size=64,
            show_progress_bar=False,
            normalize_embeddings=True   # Enables cosine similarity via dot product
        )
        embeddings = np.array(embeddings, dtype=np.float32)

        # FAISS index
        self.index = faiss.IndexFlatIP(self.dimension)
        self.index.add(embeddings)
        print(f"[VectorStore] Index built: {self.index.ntotal} vectors.")

    # ── Search ────────────────────────────────────────────────────────────────
    def search(self, query: str, top_k: int = 3) -> List[Tuple[Chunk, float]]:
        """Return top-k (chunk, score) tuples for a query string."""
        if self.index is None:
            raise RuntimeError("Vector store not built. Call build() first.")

        query_vec = self.model.encode(
            [query], normalize_embeddings=True
        ).astype(np.float32)

        scores, indices = self.index.search(query_vec, min(top_k, len(self.chunks)))

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0:
                continue
            results.append((self.chunks[idx], float(score)))

        return results

    # ── Persistence (optional) ────────────────────────────────────────────────
    def save(self, path: str) -> None:
        """Save FAISS index to disk."""
        faiss.write_index(self.index, path)

    def load(self, path: str) -> None:
        """Load FAISS index from disk."""
        self.index = faiss.read_index(path)
