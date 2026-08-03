"""
build_index.py

The OFFLINE phase: read every file in corpus/, chunk it, then build TWO
indexes from the exact same chunks:
  - a vector index (semantic similarity, via embeddings)
  - a BM25 index (lexical/keyword similarity)

Both get saved to disk so query.py can load them without re-processing the
corpus on every question.

Usage:
    python build_index.py            # uses HashingEmbedder (no setup needed)
    python build_index.py --real     # uses SentenceTransformerEmbedder
"""

import argparse
from pathlib import Path
from chunking import chunk_text
from embeddings import HashingEmbedder, SentenceTransformerEmbedder
from vector_store import VectorStore
from bm25_search import BM25Store

CORPUS_DIR = "corpus"
INDEX_DIR = "index"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--real", action="store_true", help="Use real semantic embeddings")
    args = parser.parse_args()

    embedder = SentenceTransformerEmbedder() if args.real else HashingEmbedder()
    print(f"Using embedder: {type(embedder).__name__} (dimension={embedder.dimension})")

    all_chunks = []
    for file_path in sorted(Path(CORPUS_DIR).glob("*.txt")):
        text = file_path.read_text()
        chunks = chunk_text(text, source=file_path.name, chunk_size=60, overlap=15)
        print(f"  {file_path.name}: split into {len(chunks)} chunk(s)")
        all_chunks.extend(chunks)

    print(f"Total chunks: {len(all_chunks)}")

    texts = [c.text for c in all_chunks]
    metadata = [{"text": c.text, "source": c.source, "chunk_index": c.chunk_index} for c in all_chunks]

    # --- Vector index (semantic) ---
    vectors = embedder.embed(texts)
    vector_store = VectorStore()
    vector_store.add(vectors, metadata)
    vector_store.save(INDEX_DIR)
    print(f"Saved vector index to '{INDEX_DIR}/'")

    # --- BM25 index (lexical) ---
    bm25_store = BM25Store()
    bm25_store.build(texts, metadata)
    bm25_store.save(INDEX_DIR)
    print(f"Saved BM25 index to '{INDEX_DIR}/'")


if __name__ == "__main__":
    main()