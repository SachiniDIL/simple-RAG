"""
query.py

The ONLINE phase: load the saved index from disk, embed the user's question
with the SAME embedder used at index time, retrieve the top-k most similar
chunks, and reject the result entirely if even the best match is below a
minimum relevance threshold.

Usage:
    python query.py "what is chunking?"
    python query.py "what is chunking?" --real
    python query.py "what is chunking?" --real --threshold 0.35

IMPORTANT: the embedder here must match the one used in build_index.py, or
the query vector and stored vectors won't be comparable at all.
"""

import argparse
from embeddings import HashingEmbedder, SentenceTransformerEmbedder
from vector_store import VectorStore

INDEX_DIR = "index"
TOP_K = 3
DEFAULT_THRESHOLD = 0.3  # tuned against SentenceTransformerEmbedder scores - see README


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("query", help="The question to ask")
    parser.add_argument("--real", action="store_true", help="Use real semantic embeddings (must match build_index.py)")
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD,
                         help="Minimum top-score required to trust the retrieval (default: 0.3)")
    args = parser.parse_args()

    embedder = SentenceTransformerEmbedder() if args.real else HashingEmbedder()
    store = VectorStore.load(INDEX_DIR)

    query_vector = embedder.embed([args.query])[0]
    results = store.search(query_vector, top_k=TOP_K)

    print(f"\nQuery: {args.query!r}")

    if not results or results[0]["score"] < args.threshold:
        top_score = results[0]["score"] if results else 0.0
        print(f"\nNo sufficiently relevant information found "
              f"(best match scored {top_score:.4f}, below threshold {args.threshold}).")
        print("Try rephrasing the question, or it may genuinely not be covered in the corpus.\n")
        return

    print(f"Top {len(results)} retrieved chunks:\n")
    for i, r in enumerate(results, 1):
        print(f"{i}. score={r['score']:.4f}  source={r['source']} (chunk {r['chunk_index']})")
        print(f"   {r['text'][:160]}...\n")


if __name__ == "__main__":
    main()