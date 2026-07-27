"""
vector_store.py

The simplest possible vector store: a numpy array of vectors, plus a parallel
list of metadata (the chunk text and which file it came from). Search compares
the query against EVERY stored vector (brute-force / "flat" search) - no
approximate index, since our corpus is tiny.
"""

import numpy as np

class VectorStore:
    def __init__(self):
        self.vectors = None   # shape: (num_chunks, dimension)
        self.metadata = []    # one dict per stored chunk

    def add(self, vectors: np.ndarray, metadata: list[dict]) -> None:
        if self.vectors is None:
            self.vectors = vectors
        else:
            self.vectors = np.vstack([self.vectors, vectors])
        self.metadata.extend(metadata)

    def search(self, query_vector: np.ndarray, top_k: int = 3) -> list[dict]:
        """
        Return the top_k stored chunks most similar to query_vector.
        """
        if self.vectors is None:
            return []

        # Since every vector is already normalized (length 1), the dot product
        # of the query against ALL stored vectors at once gives us every
        # cosine similarity score in a single matrix multiply.
        scores = self.vectors @ query_vector  # shape: (num_chunks,)

        top_k = min(top_k, len(scores))
        top_indices = np.argsort(-scores)[:top_k]  # highest scores first

        results = []
        for idx in top_indices:
            entry = dict(self.metadata[idx])
            entry["score"] = float(scores[idx])
            results.append(entry)
        return results