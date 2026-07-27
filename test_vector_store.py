"""
test_vector_store.py

Builds a tiny index from every file in corpus/, then runs one query and
prints the similarity score against EVERY stored chunk (not just top-k),
so you can see the full ranking before it gets cut down.
"""

from pathlib import Path
from chunking import chunk_text
from embeddings import HashingEmbedder
from vector_store import VectorStore

embedder = HashingEmbedder(dimension=256)
store = VectorStore()

# --- Build the index (this is the "offline indexing" phase) ---
all_chunks = []
for file_path in sorted(Path("corpus").glob("*.txt")):
    text = file_path.read_text()
    chunks = chunk_text(text, source=file_path.name, chunk_size=60, overlap=15)
    all_chunks.extend(chunks)

print(f"Indexed {len(all_chunks)} chunks from {len(list(Path('corpus').glob('*.txt')))} files\n")

texts = [c.text for c in all_chunks]
vectors = embedder.embed(texts)
metadata = [{"text": c.text, "source": c.source, "chunk_index": c.chunk_index} for c in all_chunks]
store.add(vectors, metadata)

# --- Query it (this is the "online query" phase) ---
query = "why do we split documents before embedding them?"
query_vector = embedder.embed([query])[0]

print(f"Query: {query!r}\n")
print("=" * 70)
print("FULL RANKING (every stored chunk, sorted by similarity)")
print("=" * 70)

all_scores = store.vectors @ query_vector
ranking = sorted(
    zip(all_scores, store.metadata),
    key=lambda pair: pair[0],
    reverse=True,
)
for rank, (score, meta) in enumerate(ranking, 1):
    preview = meta["text"][:70].replace("\n", " ")
    print(f"{rank:2d}. score={score:.4f}  {meta['source']} (chunk {meta['chunk_index']})  \"{preview}...\"")

print("\n" + "=" * 70)
print("TOP-3 ONLY (what search() actually returns)")
print("=" * 70)
top_results = store.search(query_vector, top_k=3)
for i, r in enumerate(top_results, 1):
    print(f"{i}. score={r['score']:.4f}  {r['source']} (chunk {r['chunk_index']})")
    print(f"   {r['text'][:120]}...\n")