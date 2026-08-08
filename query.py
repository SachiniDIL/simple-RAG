"""
query.py

The ONLINE phase: load the saved index, embed the question, retrieve chunks
(optionally combining semantic + lexical search via hybrid/RRF), reject
retrieval below a relevance threshold, and optionally generate a grounded
answer with a local LLM.

Usage:
    python query.py "what is chunking?" --real
    python query.py "what is chunking?" --real --hybrid
    python query.py "what is chunking?" --real --hybrid --generate
"""

import argparse
import requests
from embeddings import HashingEmbedder, SentenceTransformerEmbedder
from vector_store import VectorStore
from bm25_search import BM25Store

INDEX_DIR = "index"
TOP_K = 3
DEFAULT_THRESHOLD = 0.3  # tuned against semantic (cosine) scores specifically
RRF_K = 60

OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "llama3.2:1b"


def rank_of_each(scores):
    order = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
    ranks = [0] * len(scores)
    for rank, idx in enumerate(order, start=1):
        ranks[idx] = rank
    return ranks


def semantic_search(query_vector, vector_store, top_k):
    scores = vector_store.vectors @ query_vector
    top_indices = scores.argsort()[::-1][:top_k]
    return [
        {**vector_store.metadata[i], "score": float(scores[i])}
        for i in top_indices
    ]


def hybrid_search(query, query_vector, vector_store, bm25_store, top_k):
    semantic_scores = vector_store.vectors @ query_vector
    lexical_scores = bm25_store.score_all(query)
    semantic_ranks = rank_of_each(semantic_scores)
    lexical_ranks = rank_of_each(lexical_scores)

    fused = []
    for i, meta in enumerate(vector_store.metadata):
        rrf_score = 1 / (RRF_K + semantic_ranks[i]) + 1 / (RRF_K + lexical_ranks[i])
        fused.append({
            **meta,
            "score": float(semantic_scores[i]),  # used for the relevance threshold
            "rrf_score": rrf_score,
        })
    fused.sort(key=lambda r: r["rrf_score"], reverse=True)
    return fused[:top_k]


def build_prompt(query: str, chunks: list) -> str:
    context = "\n\n".join(f"[Source: {c['source']}]\n{c['text']}" for c in chunks)
    return (
        "Answer the question using ONLY the context below. "
        "If the context doesn't fully answer it, say what's missing.\n\n"
        f"Context:\n{context}\n\n"
        f"Question: {query}\n"
        "Answer:"
    )


def generate_answer(prompt: str) -> str:
    response = requests.post(
        OLLAMA_URL,
        json={"model": OLLAMA_MODEL, "prompt": prompt, "stream": False},
        timeout=120,
    )
    response.raise_for_status()
    return response.json()["response"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("query", help="The question to ask")
    parser.add_argument("--real", action="store_true", help="Use real semantic embeddings (must match build_index.py)")
    parser.add_argument("--hybrid", action="store_true", help="Combine semantic + BM25 lexical search via RRF")
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD,
                         help="Minimum semantic score required to trust the retrieval (default: 0.3)")
    parser.add_argument("--generate", action="store_true", help="Also call a local LLM to write a grounded answer")
    args = parser.parse_args()

    embedder = SentenceTransformerEmbedder() if args.real else HashingEmbedder()
    vector_store = VectorStore.load(INDEX_DIR)
    query_vector = embedder.embed([args.query])[0]

    if args.hybrid:
        bm25_store = BM25Store.load(INDEX_DIR)
        results = hybrid_search(args.query, query_vector, vector_store, bm25_store, TOP_K)
    else:
        results = semantic_search(query_vector, vector_store, TOP_K)

    print(f"\nQuery: {args.query!r}  (mode: {'hybrid' if args.hybrid else 'semantic-only'})")

    if not results or results[0]["score"] < args.threshold:
        top_score = results[0]["score"] if results else 0.0
        print(f"\nNo sufficiently relevant information found "
              f"(best semantic score {top_score:.4f}, below threshold {args.threshold}).")
        print("Try rephrasing the question, or it may genuinely not be covered in the corpus.\n")
        return

    print(f"Top {len(results)} retrieved chunks:\n")
    for i, r in enumerate(results, 1):
        extra = f"  rrf={r['rrf_score']:.5f}" if args.hybrid else ""
        print(f"{i}. semantic={r['score']:.4f}{extra}  source={r['source']} (chunk {r['chunk_index']})")
        print(f"   {r['text'][:160]}...\n")

    if args.generate:
        prompt = build_prompt(args.query, results)
        print("--- Generated answer (local model, may take a moment on CPU) ---")
        print(generate_answer(prompt))
        print()


if __name__ == "__main__":
    main()