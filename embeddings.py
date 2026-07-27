"""
embeddings.py

HashingEmbedder: deterministic, offline, word-overlap only (Step 3).
SentenceTransformerEmbedder: a real, pretrained semantic model. Understands
meaning, not just word overlap. Downloads model weights from the internet
the first time it runs, then caches them locally.
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


class SentenceTransformerEmbedder:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        from sentence_transformers import SentenceTransformer
        print(f"Loading model '{model_name}' (first run downloads it - may take a minute)...")
        self.model = SentenceTransformer(model_name)
        self.dimension = self.model.get_sentence_embedding_dimension()

    def embed(self, texts: list[str]) -> np.ndarray:
        vectors = self.model.encode(texts, normalize_embeddings=True)
        return np.asarray(vectors, dtype=np.float32)