# simple-rag

A full-stack Retrieval-Augmented Generation (RAG) system, built from scratch to learn how
embeddings, vector stores, retrieval, and generation actually work — no black-box
frameworks for the core pipeline, no paid APIs anywhere in the stack.

## What is RAG?

Retrieval-Augmented Generation combines two steps:

1. **Retrieval** — given a question, search a collection of documents and pull out the
   most relevant pieces.
2. **Generation** — hand those pieces to an LLM along with the question, so it answers
   using real source text instead of relying purely on what it memorized during training.

RAG runs in two phases that happen at different times:
- **Indexing (offline)** — read documents → split into chunks → embed each chunk → store
  the vectors. Runs once, and again whenever the source documents change.
- **Querying (online)** — embed the user's question → find the closest stored chunks →
  optionally generate an answer from them. Runs on every question.

## Project status

**Backend (Python)**
- [x] `corpus/` — sample source documents
- [x] `chunking.py` — splits documents into overlapping chunks
- [x] `embeddings.py` — two swappable embedders (`HashingEmbedder` offline stand-in,
      `SentenceTransformerEmbedder` for real semantic embeddings)
- [x] `vector_store.py` — cosine similarity search over stored vectors, save/load
- [x] `bm25_search.py` — lexical (keyword) search via BM25, for hybrid search
- [x] `build_index.py` — offline indexing: builds both a vector index and a BM25 index
- [x] `query.py` — CLI: semantic or hybrid retrieval, relevance threshold, local
      generation via Ollama
- [x] `api.py` — FastAPI wrapper exposing the same pipeline over HTTP

**Frontend (Next.js)**
- [x] `frontend/` — single-page UI: question input, hybrid/generate toggles, results
      display, built with TanStack Query against the FastAPI backend

## Architecture

```
┌─────────────────┐         ┌──────────────────┐         ┌─────────────────────┐
│  Next.js (3000)  │  HTTP   │  FastAPI (8000)   │         │  Local files         │
│  frontend/       │ ──────> │  api.py           │ ──────> │  index/ (vectors,     │
│  question form,  │  POST   │  loads embedder +  │         │  BM25) built by       │
│  results display │  /query │  indexes once at   │         │  build_index.py       │
└─────────────────┘         │  startup            │         └─────────────────────┘
                              └──────────────────┘
                                       │
                                       │ optional, if generate=true
                                       ▼
                              ┌──────────────────┐
                              │  Ollama (11434)   │
                              │  local LLM         │
                              │  (llama3.2:1b)     │
                              └──────────────────┘
```

## Structure

```
simple-RAG/
├── corpus/               # source .txt files to index
├── chunking.py            # splits text into overlapping chunks
├── embeddings.py           # text -> vector (HashingEmbedder / SentenceTransformerEmbedder)
├── vector_store.py         # cosine similarity search, save/load
├── bm25_search.py           # lexical/keyword search via BM25, save/load
├── build_index.py           # offline: builds vector + BM25 indexes from corpus/
├── query.py                 # CLI: retrieval (semantic or hybrid) + optional generation
├── api.py                   # FastAPI wrapper around the same pipeline
├── test_chunking.py         # standalone script visualizing chunking step by step
├── test_embeddings.py       # standalone script comparing embedding similarity
├── test_vector_store.py     # standalone script showing full similarity ranking
├── test_hybrid.py           # standalone script comparing semantic vs lexical vs fused ranking
├── requirements.txt
├── frontend/                # Next.js UI
│   ├── app/
│   │   ├── page.tsx          # main query UI
│   │   ├── providers.tsx      # TanStack Query provider
│   │   └── layout.tsx
│   ├── .env.local.example
│   └── README.md              # frontend-specific setup notes
└── README.md                  # this file
```

## Corpus

Five short `.txt` files, each covering one distinct RAG-related concept (embeddings,
vector stores, chunking, retrieval, generation) — kept topically separate on purpose, so
it's easy to tell whether retrieval pulled the *right* chunk for a given question.

## Setup

**Backend:**
```
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

For local generation, also install [Ollama](https://ollama.com) and pull a model:
```
ollama pull llama3.2:1b
```

Build the index (once, or whenever corpus/ changes):
```
python build_index.py --real
```

**Frontend:**
```
cd frontend
npm install
copy .env.local.example .env.local
```

## Running the full stack

Two terminals, both running at the same time:

```
# Terminal 1 - backend
venv\Scripts\activate
python -m uvicorn api:app --reload

# Terminal 2 - frontend
cd frontend
npm run dev
```

Open `http://localhost:3000`.

## Running the backend alone (CLI, no server)

```
python query.py "what is chunking?" --real
python query.py "what is chunking?" --real --hybrid
python query.py "what is chunking?" --real --hybrid --generate
```

**Important:** `HashingEmbedder` and `SentenceTransformerEmbedder` produce vectors of
different sizes (256 vs 384 dimensions) and live in different vector spaces. Keep separate
index folders for each (e.g. `index_hashing/`) so you don't accidentally query one
embedder's index with the other's query vector.

## What each piece actually does

| Concept | Where | What it solves |
|---|---|---|
| Chunking | `chunking.py` | Splits long text into overlapping pieces so a sentence at a boundary isn't cut in half and orphaned from its context |
| Semantic embeddings | `embeddings.py` | Represents text as a vector capturing *meaning*, so paraphrased questions still match the right chunk |
| Vector store | `vector_store.py` | Finds the closest stored vectors to a query vector (brute-force cosine similarity) |
| BM25 / hybrid search | `bm25_search.py`, RRF in `query.py`/`api.py` | Catches exact rare terms that semantic search sometimes underweights, without losing semantic search's strength at paraphrasing |
| Relevance threshold | `query.py`/`api.py` | Rejects retrieval outright when nothing in the corpus is actually relevant, instead of confidently returning irrelevant chunks |
| Generation | `query.py --generate`, `api.py` | Sends retrieved chunks + question to a local LLM so the answer is grounded in real source text |
| API layer | `api.py` | Loads the embedder and indexes once at startup (not per request) and exposes retrieval + generation over HTTP |
| Frontend | `frontend/` | Question input, hybrid/generate toggles, results and generated-answer display, calling the API with TanStack Query |

## Hashing vs real embeddings (measured on this corpus)

| Query | HashingEmbedder | SentenceTransformerEmbedder |
|---|---|---|
| Genuine question about the corpus | scores ~0.2–0.3 | scores ~0.45–0.65 |
| Complete nonsense ("what is a mango") | scores ~0.28–0.45 (false confidence) | scores ~0.07–0.11 (correctly low) |
| Exact-term match buried in a paraphrased question | missed entirely | found, though not always ranked first |

Hashing only counts shared words, with no down-weighting of common words like "the" or
"is" — so nonsense questions can score *higher* than legitimate ones just by sharing
filler words with the corpus. A real embedding model captures meaning instead.

## Why this exists

Built as a hands-on way to actually understand — rather than just use — the concepts
behind modern AI search: chunking strategy, what an embedding vector represents, cosine
similarity, lexical vs semantic search, hybrid retrieval via RRF, relevance thresholds,
how retrieved context grounds an LLM's answer, and wrapping the whole pipeline behind a
real API and UI. Runs entirely on free, open-source tools — no paid API required anywhere
in the stack.