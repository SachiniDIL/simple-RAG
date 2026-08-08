# simple-rag

A small, from-scratch Retrieval-Augmented Generation (RAG) system, built step by step to
learn how embeddings, vector stores, retrieval, and generation actually work under the
hood — no frameworks, no black boxes, no paid APIs.

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

- [x] `corpus/` — sample source documents
- [x] `chunking.py` — splits documents into overlapping chunks
- [x] `embeddings.py` — two swappable embedders:
  - `HashingEmbedder` — offline, deterministic, word-overlap only (no setup needed)
  - `SentenceTransformerEmbedder` — real semantic embeddings via `all-MiniLM-L6-v2`
- [x] `vector_store.py` — stores vectors, searches by cosine similarity, saves/loads from disk
- [x] `bm25_search.py` — lexical (keyword) search via BM25, for hybrid search
- [x] `build_index.py` — offline indexing script; builds BOTH a vector index and a BM25 index
- [x] `query.py` — online query script, supports:
  - `--real` — real semantic embeddings vs the offline hashing embedder
  - `--hybrid` — combine semantic + BM25 rankings via Reciprocal Rank Fusion (RRF)
  - `--threshold` — reject retrieval below a minimum relevance score instead of
    always returning top-k regardless of actual relevance
  - `--generate` — send retrieved chunks to a local, open-source LLM (via Ollama) for
    a real, grounded answer
- [ ] Wrap this in a FastAPI + Next.js app (potential next step)

## Structure

```
simple-rag/
├── corpus/            # source .txt files to index
├── chunking.py         # splits text into overlapping chunks
├── embeddings.py        # text -> vector (HashingEmbedder / SentenceTransformerEmbedder)
├── vector_store.py      # stores vectors, cosine similarity search, save/load
├── bm25_search.py        # lexical/keyword search via BM25, save/load
├── build_index.py        # offline: builds vector + BM25 indexes from corpus/
├── query.py              # online: retrieval (semantic or hybrid) + optional generation
├── test_chunking.py      # standalone script to visualize chunking step by step
├── test_embeddings.py    # standalone script comparing embedding similarity on word pairs
├── test_vector_store.py  # standalone script showing full similarity ranking
├── test_hybrid.py        # standalone script comparing semantic vs lexical vs fused ranking
└── requirements.txt
```

## Corpus

Five short `.txt` files, each covering one distinct RAG-related concept (embeddings,
vector stores, chunking, retrieval, generation) — kept topically separate on purpose, so
it's easy to tell whether retrieval pulled the *right* chunk for a given question.

## Setup

```
python -m venv venv
venv\Scripts\activate                # Windows
pip install -r requirements.txt
```

For local generation (`--generate`), also install [Ollama](https://ollama.com) and pull a
small open-source model:
```
ollama pull llama3.2:1b
```

## Usage

Build the index (run once, or whenever corpus/ changes — builds both a vector index and a
BM25 index):

```
python build_index.py            # HashingEmbedder - fast, offline, word-overlap only
python build_index.py --real     # SentenceTransformerEmbedder - real semantic embeddings
```

Query it:

```
python query.py "what is chunking?" --real
python query.py "what is chunking?" --real --hybrid
python query.py "what is chunking?" --real --hybrid --generate
```

**Important:** `HashingEmbedder` and `SentenceTransformerEmbedder` produce vectors of
different sizes (256 vs 384 dimensions) and live in different vector spaces entirely.
Keep separate index folders for each (e.g. rename one to `index_hashing/`) so you don't
accidentally query one embedder's index with the other's query vector — mixing them
causes a shape-mismatch error.

## What each piece actually does

| Concept | Where | What it solves |
|---|---|---|
| Chunking | `chunking.py` | Splits long text into overlapping pieces so a sentence at a boundary isn't cut in half and orphaned from its context |
| Semantic embeddings | `embeddings.py` | Represents text as a vector capturing *meaning*, so paraphrased questions still match the right chunk |
| Vector store | `vector_store.py` | Finds the closest stored vectors to a query vector (brute-force cosine similarity) |
| BM25 / hybrid search | `bm25_search.py`, RRF in `query.py` | Catches exact rare terms that semantic search sometimes underweights, without losing semantic search's strength at paraphrasing |
| Relevance threshold | `query.py` | Rejects retrieval outright when nothing in the corpus is actually relevant, instead of confidently returning irrelevant chunks |
| Generation | `query.py --generate` | Sends retrieved chunks + question to a local LLM so the answer is grounded in real source text |

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
and how retrieved context grounds an LLM's answer. Runs entirely on free, open-source
tools — no paid API required.