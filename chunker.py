# rag_mvp/chunker.py
from __future__ import annotations
from dataclasses import dataclass
from typing import List
import re
from loader import RawDoc

@dataclass
class Chunk:
    doc_id: str      # comes from RawDoc.source_id
    ordinal: int     # 0,1,2...
    text: str
    metadata: dict

def normalize(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()

def chunk_text(
    rawdoc: RawDoc,
    chunk_tokens: int = 500,
    overlap_tokens: int = 50,
) -> List[Chunk]:
    """
    Split a RawDoc into chunks using word-based approximation
    (1 token ≈ 0.75 words).
    """
    words = normalize(rawdoc.text).split()
    if not words:
        return []

    window = max(int(chunk_tokens * 0.75), 1)
    overlap = max(int(overlap_tokens * 0.75), 0)

    chunks: List[Chunk] = []
    start = 0
    ordinal = 0

    while start < len(words):
        end = min(start + window, len(words))
        piece = " ".join(words[start:end])
        chunks.append(
            Chunk(
                doc_id=rawdoc.source_id,
                ordinal=ordinal,
                text=piece,
                metadata={
                    "filename": rawdoc.metadata.get("filename"),
                    "relpath": rawdoc.metadata.get("relpath"),
                    "chunk_size": len(piece.split()),
                },
            )
        )
        ordinal += 1
        if end == len(words):
            break
        start = end - overlap if end - overlap > start else end

    return chunks
