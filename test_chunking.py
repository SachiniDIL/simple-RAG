"""
test_chunking.py

Run this to SEE chunking happen: prints each chunk as it's created (via
verbose=True), then prints the overlap between consecutive chunks so you can
confirm the boundary-splitting problem is actually being fixed.
"""

from chunking import chunk_text

with open("corpus/chunking.txt") as f:
    text = f.read()

print("=" * 60)
print("STEP-BY-STEP CHUNKING PROCESS")
print("=" * 60)
chunks = chunk_text(text, source="chunking.txt", chunk_size=60, overlap=15, verbose=True)

print("\n" + "=" * 60)
print("CHECKING OVERLAP BETWEEN CONSECUTIVE CHUNKS")
print("=" * 60)
for i in range(len(chunks) - 1):
    words_a = chunks[i].text.split()
    words_b = chunks[i + 1].text.split()
    # The last `overlap` words of chunk i should equal the first `overlap` words of chunk i+1
    tail_of_a = words_a[-15:]
    head_of_b = words_b[:15]
    print(f"\nEnd of chunk {i}:   ...{' '.join(tail_of_a)}")
    print(f"Start of chunk {i+1}: {' '.join(head_of_b)}...")
    print(f"Match: {tail_of_a == head_of_b}")