"""
vector_store.py

The simplest possible vector store: a numpy array of vectors, plus a parallel
list of metadata (the chunk text and which file it came from). Search compares
the query against EVERY stored vector (brute-force / "flat" search) - no
approximate index, since our corpus is tiny.
"""

import json
import numpy as np
from pathlib import Path


class VectorStore:
    def __init__(self):
        self.vectors = None
        self.metadata = []

    def add(self, vectors: np.ndarray, metadata: list) -> None:
        if self.vectors is None:
            self.vectors = vectors
        else:
            self.vectors = np.vstack([self.vectors, vectors])
        self.metadata.extend(metadata)

    def search(self, query_vector: np.ndarray, top_k: int = 3) -> list:
        if self.vectors is None:
            return []
        scores = self.vectors @ query_vector
        top_k = min(top_k, len(scores))
        top_indices = np.argsort(-scores)[:top_k]
        results = []
        for idx in top_indices:
            entry = dict(self.metadata[idx])
            entry["score"] = float(scores[idx])
            results.append(entry)
        return results

    def save(self, directory: str) -> None:
        path = Path(directory)
        path.mkdir(parents=True, exist_ok=True)
        np.save(path / "vectors.npy", self.vectors)
        with open(path / "metadata.json", "w") as f:
            json.dump(self.metadata, f)

    @classmethod
    def load(cls, directory: str) -> "VectorStore":
        path = Path(directory)
        store = cls()
        store.vectors = np.load(path / "vectors.npy")
        with open(path / "metadata.json") as f:
            store.metadata = json.load(f)
        return store