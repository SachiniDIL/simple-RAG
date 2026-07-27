"""
chunking.py

Splits a long piece of text into smaller overlapping chunks before embedding.

Why overlap? If we split with zero overlap, a sentence that straddles the boundary
between chunk N and chunk N+1 gets torn in half, and neither half makes full sense
on its own. A small overlap (e.g. 50 words) means that sentence usually survives
intact in at least one chunk.

This is a simple word-count based splitter. Real systems often split on paragraph
or sentence boundaries first, then group those into chunks close to a target size -
but word-count chunking is enough to see the RAG mechanics clearly.
"""

from dataclasses import dataclass


@dataclass
class Chunk:
    text: str
    source: str      # which file this chunk came from
    chunk_index: int  # position of this chunk within that file


def chunk_text(text: str, source: str, chunk_size: int = 120, overlap: int = 30,
               verbose: bool = False) -> list[Chunk]:
    """
    Split `text` into chunks of roughly `chunk_size` words, each chunk overlapping
    the previous one by `overlap` words.

    If verbose=True, prints each step of the sliding window as it happens:
    where the window starts/ends, and the resulting chunk text.
    """
    words = text.split()
    if not words:
        return []

    chunks = []
    start = 0
    index = 0
    step = max(chunk_size - overlap, 1)  # how far the window slides each time

    if verbose:
        print(f"Total words in '{source}': {len(words)}")
        print(f"chunk_size={chunk_size}, overlap={overlap}, step={step}\n")

    while start < len(words):
        end = min(start + chunk_size, len(words))
        chunk_words = words[start:end]
        chunk_str = " ".join(chunk_words)

        if verbose:
            print(f"--- chunk {index} ---")
            print(f"word window: [{start}:{end}]  ({len(chunk_words)} words)")
            print(f"text: {chunk_str}\n")

        chunks.append(Chunk(text=chunk_str, source=source, chunk_index=index))
        index += 1
        if end == len(words):
            break
        start += step

    if verbose:
        print(f"Total chunks produced: {len(chunks)}")

    return chunks