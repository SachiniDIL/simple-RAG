# simple-rag

A small, from-scratch Retrieval-Augmented Generation (RAG) system, built step by step to
learn how embeddings, vector stores, and retrieval actually work under the hood — no
frameworks, no black boxes.

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
- [x] `build_index.py` — offline indexing script (`--real` flag toggles embedder)
- [x] `query.py` — online query script (`--real` flag toggles embedder, must match index)
- [ ] Relevance threshold — reject/flag low-confidence retrievals instead of always
      returning top-k regardless of actual relevance
- [ ] Generation step — send retrieved chunks + question to an LLM for a real answer

## Structure

```
simple-rag/
├── corpus/            # source .txt files to index
├── chunking.py         # splits text into overlapping chunks
├── embeddings.py        # text -> vector (HashingEmbedder / SentenceTransformerEmbedder)
├── vector_store.py      # stores vectors, finds nearest matches, save/load
├── build_index.py       # offline: builds the index from corpus/
├── query.py             # online: retrieves top-k chunks for a question
├── test_chunking.py     # standalone script to visualize chunking step by step
├── test_embeddings.py   # standalone script comparing embedding similarity on word pairs
├── test_vector_store.py # standalone script showing full similarity ranking
└── requirements.txt
```

## Corpus

Five short `.txt` files, each covering one distinct RAG-related concept (embeddings,
vector stores, chunking, retrieval, generation) — kept topically separate on purpose, so
it's easy to tell whether retrieval pulled the *right* chunk for a given question.

## Setup

```
python -m venv venv
venv\Scripts\activate      # Windows
pip install numpy
pip install sentence-transformers   # only needed for --real mode
```

## Usage

Build the index (run once, or whenever corpus/ changes):

```
python build_index.py            # HashingEmbedder - fast, offline, word-overlap only
python build_index.py --real     # SentenceTransformerEmbedder - real semantic embeddings
```

Query it:

```
python query.py "what is chunking?"          # must match the embedder used to build the index
python query.py "what is chunking?" --real
```

**Important:** `HashingEmbedder` and `SentenceTransformerEmbedder` produce vectors of
different sizes (256 vs 384 dimensions) and live in different vector spaces entirely.
Keep separate index folders for each (e.g. rename one to `index_hashing/`) so you don't
accidentally query one embedder's index with the other's query vector — mixing them
causes a shape-mismatch error.

## What real embeddings actually buy you

Tested side by side on the same questions:

| Query | HashingEmbedder | SentenceTransformerEmbedder |
|---|---|---|
| Genuine question about the corpus | scores ~0.2–0.3 | scores ~0.45–0.56 |
| Complete nonsense ("what is a mango") | scores ~0.28–0.45 (false confidence) | scores ~0.07–0.11 (correctly low) |
| Exact-term match buried in a paraphrased question | missed entirely | correctly surfaced in top-3 |

Hashing only counts shared words, with no down-weighting of common words like "the" or
"is" — so nonsense questions can score *higher* than legitimate ones just by sharing
filler words with the corpus. A real embedding model captures meaning instead, giving a
usable gap between relevant and irrelevant results.

## Why this exists

Built as a hands-on way to actually understand — rather than just use — the concepts
behind modern AI search: chunking strategy, what an embedding vector represents, cosine
similarity, and how a vector store answers "which of these is closest to my query."