"""
query.py

The ONLINE phase: load the saved index, embed the question, retrieve top-k
chunks above a relevance threshold, and optionally send them to a LOCAL,
open-source LLM (via Ollama) to generate a real, grounded answer.

Usage:
    python query.py "what is chunking?"
    python query.py "what is chunking?" --real
    python query.py "what is chunking?" --real --generate

Requires Ollama running locally (https://ollama.com) with a model pulled, e.g.:
    ollama pull llama3.2:1b
"""

import argparse
import requests
from embeddings import HashingEmbedder, SentenceTransformerEmbedder
from vector_store import VectorStore

INDEX_DIR = "index"
TOP_K = 3
DEFAULT_THRESHOLD = 0.3

OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "llama3.2:1b"


def build_prompt(query: str, chunks: list) -> str:
    """
    This is the "augmented" part of Retrieval-Augmented Generation: stuff the
    retrieved chunks into the prompt and instruct the model to only use them.
    """
    context = "\n\n".join(f"[Source: {c['source']}]\n{c['text']}" for c in chunks)
    return (
        "Answer the question using ONLY the context below. "
        "If the context doesn't fully answer it, say what's missing.\n\n"
        f"Context:\n{context}\n\n"
        f"Question: {query}\n"
        "Answer:"
    )


def generate_answer(prompt: str) -> str:
    """
    Calls a locally running Ollama server - no API key, no internet required
    after the model has been pulled once.
    """
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
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD,
                         help="Minimum top-score required to trust the retrieval (default: 0.3)")
    parser.add_argument("--generate", action="store_true", help="Also call a local LLM to write a grounded answer")
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

    if args.generate:
        prompt = build_prompt(args.query, results)
        print("--- Generated answer (local model, may take a moment on CPU) ---")
        print(generate_answer(prompt))
        print()


if __name__ == "__main__":
    main()