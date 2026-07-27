"""
build_index.py

The OFFLINE phase: read every file in corpus/, chunk it, embed every chunk,
and save the resulting vectors + metadata to disk under index/.

Run this once, and again any time the files in corpus/ change. You do NOT
need to re-run this every time you ask a question - that's what query.py is for.

Usage:
    python build_index.py            # uses HashingEmbedder (no setup needed)
    python build_index.py --real     # uses SentenceTransformerEmbedder (real semantic embeddings)
"""

import argparse
from pathlib import Path
from chunking import chunk_text
from embeddings import HashingEmbedder, SentenceTransformerEmbedder
from vector_store import VectorStore

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

    print(f"Total chunks to embed: {len(all_chunks)}")

    texts = [c.text for c in all_chunks]
    vectors = embedder.embed(texts)
    metadata = [{"text": c.text, "source": c.source, "chunk_index": c.chunk_index} for c in all_chunks]

    store = VectorStore()
    store.add(vectors, metadata)
    store.save(INDEX_DIR)

    print(f"Saved index with {len(all_chunks)} chunks to '{INDEX_DIR}/'")


if __name__ == "__main__":
    main()