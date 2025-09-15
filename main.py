# rag_mvp/main.py
from loader import load_folder
from chunker import chunk_text
from embedder import LocalEmbedder
import numpy as np

if __name__ == "__main__":
    # 1) Load raw docs
    docs = load_folder("data/raw")
    print(f"Loaded {len(docs)} document(s).")

    # 2) Chunk them
    all_chunks = []
    for doc in docs:
        chunks = chunk_text(doc, chunk_tokens=100, overlap_tokens=20)
        all_chunks.extend(chunks)
        print(f"Doc {doc.source_id} → {len(chunks)} chunks")
    print(f"Total chunks: {len(all_chunks)}")

    if not all_chunks:
        print("No chunks — add files to data/raw/ and retry.")
        raise SystemExit(0)

    # 3) Embed
    emb = LocalEmbedder()
    vectors, payloads = emb.embed_chunks(all_chunks)
    print(f"Embeddings shape: {vectors.shape}  (num_chunks x dim)")

    # quick sanity: cosine similarity between first two chunks (if exist)
    if vectors.shape[0] >= 2:
        cos = float(np.dot(vectors[0], vectors[1]))
        print(f"cos(chunk0, chunk1) = {cos:.3f}")

    # quick query test
    q = "demo search query"
    qv = emb.embed_query(q)
    print(f"Query vector dim: {qv.shape[0]}")
