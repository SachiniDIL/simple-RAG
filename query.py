"""
query.py

The ONLINE phase: load the saved index from disk, embed the user's question
with the SAME embedder used at index time, and retrieve the top-k most
similar chunks.

Usage:
    python query.py "what is chunking?"
    python query.py "what is chunking?" --real     # must match how build_index.py was run

IMPORTANT: the embedder here must match the one used in build_index.py, or
the query vector and stored vectors won't be comparable at all.
"""

import argparse
from embeddings import HashingEmbedder, SentenceTransformerEmbedder
from vector_store import VectorStore

INDEX_DIR = "index"
TOP_K = 3


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("query", help="The question to ask")
    parser.add_argument("--real", action="store_true", help="Use real semantic embeddings (must match build_index.py)")
    args = parser.parse_args()

    embedder = SentenceTransformerEmbedder() if args.real else HashingEmbedder()
    store = VectorStore.load(INDEX_DIR)

    query_vector = embedder.embed([args.query])[0]
    results = store.search(query_vector, top_k=TOP_K)

    print(f"\nTop {len(results)} retrieved chunks for: {args.query!r}\n")
    for i, r in enumerate(results, 1):
        print(f"{i}. score={r['score']:.4f}  source={r['source']} (chunk {r['chunk_index']})")
        print(f"   {r['text'][:160]}...\n")


if __name__ == "__main__":
    main()