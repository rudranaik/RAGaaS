# rag_mvp/main.py
from loader import load_folder

if __name__ == "__main__":
    docs = load_folder("data/raw")
    print(f"Loaded {len(docs)} document(s).")
    for i, d in enumerate(docs[:5], 1):
        preview = (d.text[:120] + "…") if len(d.text) > 120 else d.text
        print(f"\n[{i}] {d.source_id}  ({len(d.text)} chars)")
        print(preview)
# rag_mvp/main.py
from loader import load_folder
from chunker import chunk_text

if __name__ == "__main__":
    docs = load_folder("data/raw")
    print(f"Loaded {len(docs)} document(s).")

    all_chunks = []
    for doc in docs:
        chunks = chunk_text(doc, chunk_tokens=100, overlap_tokens=20)
        all_chunks.extend(chunks)
        print(f"\nDoc {doc.source_id} → {len(chunks)} chunks")
        for ch in chunks[:3]:  # preview first 3 chunks
            preview = (ch.text[:80] + "…") if len(ch.text) > 80 else ch.text
            print(f"  - Chunk {ch.ordinal}: {preview}")

    print(f"\nTotal chunks: {len(all_chunks)}")
