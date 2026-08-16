"""
dump_chunks.py — export all stored chunks from the Chroma store to JSON.

Assumes your retriever code lives in src/retriever.py and exposes load_store()
(which opens the persisted Chroma store without rebuilding).

Run from the project root:
    python dump_chunks.py

Produces chunks_dump.json — every chunk's id, text, and metadata — so you can
scan them (e.g. to hand-pick ideal_context for the faithfulness dataset).
"""

import json
from collections import Counter
import sys
from pathlib import Path

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.retriever import load_store

# Open the already-built store (no re-embedding)
store = load_store()

# Pull every stored chunk. include=["documents","metadatas"] is required —
# by default Chroma returns ids + metadata but NOT the chunk text.
data = store._collection.get(include=["documents", "metadatas"])

# Zip the parallel lists into one record per chunk
dump = [
    {"id": i, "text": d, "meta": m}
    for i, d, m in zip(data["ids"], data["documents"], data["metadatas"])
]

# Sort by source or session then by id so related chunks sit together (easier to scan)
dump.sort(key=lambda c: (str(c["meta"].get("source", c["meta"].get("session", ""))), c["id"]))

output_file = PROJECT_ROOT / "chunks_dump.json"

with open(output_file, "w", encoding="utf-8") as f:
    json.dump(dump, f, indent=2, ensure_ascii=False)

print(f"Dumped {len(dump)} chunks to {output_file.name}")

# Quick per-source/session count so you can see the spread
counts = Counter(str(c["meta"].get("source", c["meta"].get("session", "?"))) for c in dump)
for key in sorted(counts):
    print(f"  {key}: {counts[key]} chunks")
