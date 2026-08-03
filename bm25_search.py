"""
bm25_search.py

Lexical (keyword-based) search using BM25 - a more refined relative of TF-IDF
(same family of tool as the plagiarism-detection engine you built before this
project). Unlike embeddings, BM25 doesn't understand meaning - it only cares
whether the same WORDS appear - but unlike our old HashingEmbedder, it DOES
downweight common words and reward rare, specific terms. That's why it's
strong at exact-term matches (codes, names, acronyms - like the literal word
"BM25" itself) even when a semantic model underweights them.
"""

import json
import pickle
from pathlib import Path
from rank_bm25 import BM25Okapi


class BM25Store:
    def __init__(self):
        self.bm25 = None
        self.metadata = []

    def build(self, texts: list[str], metadata: list[dict]) -> None:
        tokenized_corpus = [t.lower().split() for t in texts]
        self.bm25 = BM25Okapi(tokenized_corpus)
        self.metadata = metadata

    def score_all(self, query: str):
        """Return a BM25 score for EVERY stored chunk (not just top-k)."""
        tokenized_query = query.lower().split()
        return self.bm25.get_scores(tokenized_query)

    def save(self, directory: str) -> None:
        path = Path(directory)
        path.mkdir(parents=True, exist_ok=True)
        with open(path / "bm25.pkl", "wb") as f:
            pickle.dump(self.bm25, f)
        with open(path / "bm25_metadata.json", "w") as f:
            json.dump(self.metadata, f)

    @classmethod
    def load(cls, directory: str) -> "BM25Store":
        path = Path(directory)
        store = cls()
        with open(path / "bm25.pkl", "rb") as f:
            store.bm25 = pickle.load(f)
        with open(path / "bm25_metadata.json") as f:
            store.metadata = json.load(f)
        return store