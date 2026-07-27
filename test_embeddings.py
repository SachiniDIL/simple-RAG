"""
test_embeddings.py

Prints the actual vectors produced by HashingEmbedder, then manually walks
through the cosine similarity calculation for two pairs of sentences:

Pair A: shares several words       -> expect a HIGH score
Pair B: shares almost no words,
        but means almost the same  -> expect a LOW score (this is the point)
"""

import numpy as np
from embeddings import HashingEmbedder

embedder = HashingEmbedder(dimension=256)

pair_a = (
    "the deadline for the assignment was extended",
    "the assignment deadline has been extended by two days",
)

pair_b = (
    "the deadline for the assignment was extended",
    "they pushed back the due date",
)


def show_pair(label, text1, text2):
    print("=" * 60)
    print(label)
    print("=" * 60)
    print(f"Text 1: {text1}")
    print(f"Text 2: {text2}\n")

    vectors = embedder.embed([text1, text2])
    v1, v2 = vectors[0], vectors[1]

    print(f"Vector shape: {v1.shape}")
    print(f"Vector 1 (first 10 values): {v1[:10]}")
    print(f"Vector 2 (first 10 values): {v2[:10]}")
    print(f"Nonzero buckets in vector 1: {(v1 != 0).sum()} / {embedder.dimension}")
    print(f"Nonzero buckets in vector 2: {(v2 != 0).sum()} / {embedder.dimension}\n")

    nonzero_indices = np.nonzero(v1)[0]
    print(f"Vector 1 nonzero slots: {list(zip(nonzero_indices, v1[nonzero_indices]))}")
    
    # Since embed() already normalizes each vector to length 1, the dot
    # product IS the cosine similarity - no extra division needed.
    dot_product = float(np.dot(v1, v2))
    print(f"Cosine similarity (dot product of normalized vectors): {dot_product:.4f}\n")


show_pair("PAIR A: shares vocabulary", *pair_a)
show_pair("PAIR B: same meaning, different words", *pair_b)

print("=" * 60)
print("TAKEAWAY")
print("=" * 60)
print("Pair A scores high because both sentences literally share words")
print("like 'deadline', 'assignment', 'extended'.")
print("Pair B means almost the same thing but shares almost no words -")
print("HashingEmbedder can't see that. A real semantic embedding model")
print("would score Pair B highly too, because it understands MEANING,")
print("not just vocabulary overlap.")