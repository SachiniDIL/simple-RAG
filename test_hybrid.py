"""
test_hybrid.py

Runs semantic search and BM25 search SEPARATELY on the same query, then fuses
their two rankings with Reciprocal Rank Fusion (RRF), and prints all three
side by side - so you can see hybrid search recover from each method's blind
spot.

Requires an index built with --real:
    python build_index.py --real
"""

from embeddings import SentenceTransformerEmbedder
from vector_store import VectorStore
from bm25_search import BM25Store

INDEX_DIR = "index"
RRF_K = 60  # standard constant from the original RRF paper


def rank_of_each(scores):
    """Given a list of scores, return each item's 1-based rank (1 = best)."""
    order = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
    ranks = [0] * len(scores)
    for rank, idx in enumerate(order, start=1):
        ranks[idx] = rank
    return ranks


def hybrid_search(query, embedder, vector_store, bm25_store, top_k=3):
    query_vector = embedder.embed([query])[0]
    semantic_scores = vector_store.vectors @ query_vector
    lexical_scores = bm25_store.score_all(query)

    semantic_ranks = rank_of_each(semantic_scores)
    lexical_ranks = rank_of_each(lexical_scores)

    fused = []
    for i, meta in enumerate(vector_store.metadata):
        rrf_score = 1 / (RRF_K + semantic_ranks[i]) + 1 / (RRF_K + lexical_ranks[i])
        fused.append({
            "text": meta["text"],
            "source": meta["source"],
            "chunk_index": meta["chunk_index"],
            "semantic_score": float(semantic_scores[i]),
            "semantic_rank": semantic_ranks[i],
            "lexical_score": float(lexical_scores[i]),
            "lexical_rank": lexical_ranks[i],
            "rrf_score": rrf_score,
        })

    fused.sort(key=lambda r: r["rrf_score"], reverse=True)
    return fused[:top_k]


def show(query):
    embedder = SentenceTransformerEmbedder()
    vector_store = VectorStore.load(INDEX_DIR)
    bm25_store = BM25Store.load(INDEX_DIR)

    print("=" * 75)
    print(f"QUERY: {query!r}")
    print("=" * 75)

    results = hybrid_search(query, embedder, vector_store, bm25_store)
    for i, r in enumerate(results, 1):
        preview = r["text"][:70].replace("\n", " ")
        print(f"\n#{i}  {r['source']} (chunk {r['chunk_index']})")
        print(f"    semantic: score={r['semantic_score']:.4f}  rank={r['semantic_rank']}")
        print(f"    lexical:  score={r['lexical_score']:.4f}  rank={r['lexical_rank']}")
        print(f"    fused RRF score: {r['rrf_score']:.5f}")
        print(f"    \"{preview}...\"")
    print()


# Case 1: an exact, rare term (BM25's strength - semantic underweighted this earlier)
show("what does BM25 have to do with retrieval?")

# Case 2: a full paraphrase with almost no shared vocabulary (semantic's strength)
show("how does it feel when the model already has proof in front of it instead of guessing from memory?")