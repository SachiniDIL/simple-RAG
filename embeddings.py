"""
embeddings.py

A deterministic, dependency-free "embedder" for learning the RAG pipeline
without needing internet access or a model download.

How it works: each word is hashed into one of `dimension` buckets, and we
count how often each bucket gets hit. The resulting vector is normalized
(divided by its own length) so cosine similarity behaves correctly.

IMPORTANT LIMITATION: this only recognizes shared WORDS, not shared MEANING —
same blind spot as the TF-IDF/Jaccard tools you built before. Two sentences
that mean the same thing but share no vocabulary will score low here. We'll
swap in a real semantic model later specifically to see that gap close.
"""

import hashlib
import numpy as np


class HashingEmbedder:
    def __init__(self, dimension: int = 256):
        self.dimension = dimension

    def _embed_one(self, text: str) -> np.ndarray:
        vec = np.zeros(self.dimension, dtype=np.float32)
        for word in text.lower().split():
            h = int(hashlib.md5(word.encode("utf-8")).hexdigest(), 16)
            vec[h % self.dimension] += 1.0
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm
        return vec

    def embed(self, texts: list[str]) -> np.ndarray:
        return np.vstack([self._embed_one(t) for t in texts])