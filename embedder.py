# rag_mvp/embedder.py
from __future__ import annotations
from dataclasses import dataclass
from typing import List, Tuple
import numpy as np

from sentence_transformers import SentenceTransformer
from chunker import Chunk

@dataclass
class Payload:
    doc_id: str
    ordinal: int
    text: str
    metadata: dict

class LocalEmbedder:
    """
    Thin wrapper over SentenceTransformer for batch embedding.
    Defaults to 'all-MiniLM-L6-v2' (384-dim, fast, good enough to learn with).
    """
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name, device="cpu")

    def embed_chunks(self, chunks: List[Chunk]) -> Tuple[np.ndarray, List[Payload]]:
        texts = [c.text for c in chunks]
        vecs = self.model.encode(
            texts,
            batch_size=64,
            convert_to_numpy=True,
            normalize_embeddings=True,  # cosine-ready
            show_progress_bar=False,
        )
        payloads = [
            Payload(doc_id=c.doc_id, ordinal=c.ordinal, text=c.text, metadata=c.metadata)
            for c in chunks
        ]
        return vecs, payloads

    def embed_query(self, query: str) -> np.ndarray:
        v = self.model.encode([query], convert_to_numpy=True, normalize_embeddings=True)
        return v[0]
