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
- [ ] `chunking.py` — split documents into overlapping pieces
- [ ] `embeddings.py` — turn chunks into vectors
- [ ] `vector_store.py` — store vectors and search by similarity
- [ ] `build_index.py` — offline indexing script
- [ ] `query.py` — online query + optional generation script

## Structure

```
simple-rag/
├── corpus/          # source .txt files to index
├── chunking.py       # (coming up) splits text into overlapping chunks
├── embeddings.py      # (coming up) text -> vector
├── vector_store.py    # (coming up) stores vectors, finds nearest matches
├── build_index.py     # (coming up) offline: builds the index from corpus/
├── query.py           # (coming up) online: answers a question using the index
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
pip install -r requirements.txt
```

## Usage

_(to be filled in as build_index.py and query.py are built)_

## Why this exists

Built as a hands-on way to actually understand — rather than just use — the concepts
behind modern AI search: chunking strategy, what an embedding vector represents, cosine
similarity, and how a vector store answers "which of these is closest to my query."