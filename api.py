"""
api.py

FastAPI wrapper around the RAG pipeline. Loads the embedder and both indexes
ONCE at startup - not per request - since loading an embedding model is slow
and it never changes between queries.

Run with:
    uvicorn api:app --reload
Then open http://127.0.0.1:8000/docs to test it directly, before any frontend exists.
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from embeddings import SentenceTransformerEmbedder
from vector_store import VectorStore
from bm25_search import BM25Store
from query import rank_of_each, RRF_K, build_prompt, generate_answer

INDEX_DIR = "index"
TOP_K = 3
DEFAULT_THRESHOLD = 0.3

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # Next.js dev server
    allow_methods=["*"],
    allow_headers=["*"],
)

print("Loading embedder and indexes (happens once, at server startup)...")
embedder = SentenceTransformerEmbedder()
vector_store = VectorStore.load(INDEX_DIR)
bm25_store = BM25Store.load(INDEX_DIR)
print("Ready.")


class QueryRequest(BaseModel):
    query: str
    hybrid: bool = False
    generate: bool = False
    threshold: float = DEFAULT_THRESHOLD


class Chunk(BaseModel):
    text: str
    source: str
    chunk_index: int
    score: float
    rrf_score: float | None = None


class QueryResponse(BaseModel):
    query: str
    mode: str
    rejected: bool
    message: str | None = None
    results: list[Chunk] = []
    answer: str | None = None


def semantic_search(query_vector, top_k):
    scores = vector_store.vectors @ query_vector
    top_indices = scores.argsort()[::-1][:top_k]
    return [{**vector_store.metadata[i], "score": float(scores[i])} for i in top_indices]


def hybrid_search(query, query_vector, top_k):
    semantic_scores = vector_store.vectors @ query_vector
    lexical_scores = bm25_store.score_all(query)
    semantic_ranks = rank_of_each(semantic_scores)
    lexical_ranks = rank_of_each(lexical_scores)

    fused = []
    for i, meta in enumerate(vector_store.metadata):
        rrf_score = 1 / (RRF_K + semantic_ranks[i]) + 1 / (RRF_K + lexical_ranks[i])
        fused.append({**meta, "score": float(semantic_scores[i]), "rrf_score": rrf_score})
    fused.sort(key=lambda r: r["rrf_score"], reverse=True)
    return fused[:top_k]


@app.post("/query", response_model=QueryResponse)
def query_endpoint(req: QueryRequest):
    query_vector = embedder.embed([req.query])[0]

    if req.hybrid:
        results = hybrid_search(req.query, query_vector, TOP_K)
        mode = "hybrid"
    else:
        results = semantic_search(query_vector, TOP_K)
        mode = "semantic-only"

    if not results or results[0]["score"] < req.threshold:
        top_score = results[0]["score"] if results else 0.0
        return QueryResponse(
            query=req.query,
            mode=mode,
            rejected=True,
            message=f"No sufficiently relevant information found "
                    f"(best score {top_score:.4f}, below threshold {req.threshold}).",
        )

    answer = None
    if req.generate:
        prompt = build_prompt(req.query, results)
        answer = generate_answer(prompt)

    return QueryResponse(
        query=req.query,
        mode=mode,
        rejected=False,
        results=[Chunk(**r) for r in results],
        answer=answer,
    )