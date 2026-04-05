"""
rag/embedder.py

Embeds clinical guideline chunks into ChromaDB using sentence-transformers.

Key behaviours:
  - Only re-embeds chunks from sources that changed (hash-based)
  - Deletes old chunks for a source before inserting new ones
  - Tags every chunk with source metadata for citations
  - Uses a free local embedding model (no API cost)
  - Persists to disk so the vector store survives restarts
"""
import os
from datetime import datetime
from pathlib import Path
from typing import Optional

import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

load_dotenv()

# ── Config ────────────────────────────────────────────────────────────────────

CHROMA_DIR = Path(__file__).parent.parent / "data" / "chroma_guidelines"
CHROMA_DIR.mkdir(parents=True, exist_ok=True)

COLLECTION_NAME = "clinical_guidelines"

# Free, fast, good quality for medical text
# Runs locally — no API calls, no cost
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


# ── Clients (lazy init) ───────────────────────────────────────────────────────

_chroma_client: Optional[chromadb.PersistentClient] = None
_collection = None
_embedder: Optional[SentenceTransformer] = None


def get_chroma():
    global _chroma_client, _collection
    if _chroma_client is None:
        _chroma_client = chromadb.PersistentClient(
            path=str(CHROMA_DIR),
            settings=Settings(anonymized_telemetry=False)
        )
        _collection = _chroma_client.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"}
        )
    return _chroma_client, _collection


def get_embedder() -> SentenceTransformer:
    global _embedder
    if _embedder is None:
        print(f"[Embedder] Loading model: {EMBEDDING_MODEL}")
        _embedder = SentenceTransformer(EMBEDDING_MODEL)
    return _embedder


# ── Core embedding function ───────────────────────────────────────────────────

def embed_source(scraped: dict) -> dict:
    """
    Embed all chunks from a scraped guideline source into ChromaDB.
    Deletes previous chunks for this source before inserting new ones.

    Args:
        scraped: result dict from scraper.scrape_guideline()

    Returns:
        dict with embedding stats
    """
    source_id = scraped["source_id"]
    chunks = scraped.get("chunks", [])

    if not chunks:
        return {
            "source_id": source_id,
            "status": "skipped",
            "reason": scraped.get("error", "No chunks"),
            "chunks_embedded": 0
        }

    if not scraped.get("changed", True):
        return {
            "source_id": source_id,
            "status": "unchanged",
            "reason": "Content hash unchanged — skipping re-embed",
            "chunks_embedded": 0
        }

    print(f"  [Embedder] Embedding {len(chunks)} chunks for {source_id}...")

    _, collection = get_chroma()
    model = get_embedder()

    # Delete existing chunks for this source
    try:
        existing = collection.get(where={"source_id": source_id})
        if existing["ids"]:
            collection.delete(ids=existing["ids"])
            print(f"  [Embedder] Deleted {len(existing['ids'])} old chunks for {source_id}")
    except Exception as e:
        print(f"  [Embedder] Warning: could not delete old chunks: {e}")

    # Generate embeddings
    embeddings = model.encode(chunks, show_progress_bar=False).tolist()

    # Build IDs and metadata
    ids = [f"{source_id}_chunk_{i}" for i in range(len(chunks))]
    metadatas = [
        {
            "source_id": source_id,
            "source_name": scraped["name"],
            "url": scraped["url"],
            "category": scraped["category"],
            "chunk_index": i,
            "total_chunks": len(chunks),
            "scraped_at": scraped["scraped_at"],
            "content_hash": scraped.get("hash", ""),
        }
        for i in range(len(chunks))
    ]

    # Insert in batches of 100 to avoid memory issues
    batch_size = 100
    for batch_start in range(0, len(chunks), batch_size):
        batch_end = min(batch_start + batch_size, len(chunks))
        collection.add(
            ids=ids[batch_start:batch_end],
            embeddings=embeddings[batch_start:batch_end],
            documents=chunks[batch_start:batch_end],
            metadatas=metadatas[batch_start:batch_end],
        )

    print(f"  [Embedder] ✓ {len(chunks)} chunks embedded for {source_id}")

    return {
        "source_id": source_id,
        "status": "embedded",
        "chunks_embedded": len(chunks),
        "scraped_at": scraped["scraped_at"]
    }


def embed_all(scraped_results: list[dict]) -> list[dict]:
    """
    Embed all scraped guideline results. Only processes changed sources.
    """
    stats = []
    changed = [r for r in scraped_results if r.get("changed", False)]
    unchanged = [r for r in scraped_results if not r.get("changed", False)]

    print(f"\n[Embedder] {len(changed)} sources to embed, {len(unchanged)} unchanged.")

    for result in scraped_results:
        stat = embed_source(result)
        stats.append(stat)

    embedded = sum(1 for s in stats if s["status"] == "embedded")
    skipped = sum(1 for s in stats if s["status"] in ("unchanged", "skipped"))
    total_chunks = sum(s.get("chunks_embedded", 0) for s in stats)

    print(f"\n[Embedder] Done. {embedded} sources embedded ({total_chunks} total chunks), {skipped} skipped.")
    return stats


# ── Stats / inspection ────────────────────────────────────────────────────────

def get_collection_stats() -> dict:
    """Return stats about the current vector store."""
    _, collection = get_chroma()
    count = collection.count()

    # Get unique source IDs
    if count == 0:
        return {"total_chunks": 0, "sources": [], "last_updated": None}

    results = collection.get(include=["metadatas"])
    source_ids = list(set(m["source_id"] for m in results["metadatas"]))
    last_dates = {}
    for m in results["metadatas"]:
        sid = m["source_id"]
        date = m.get("scraped_at", "")
        if sid not in last_dates or date > last_dates[sid]:
            last_dates[sid] = date

    return {
        "total_chunks": count,
        "total_sources": len(source_ids),
        "sources": [
            {
                "source_id": sid,
                "last_updated": last_dates.get(sid, "unknown")
            }
            for sid in sorted(source_ids)
        ],
        "collection_name": COLLECTION_NAME,
        "storage_path": str(CHROMA_DIR)
    }


def clear_source(source_id: str) -> int:
    """Remove all chunks for a specific source. Used before re-embedding."""
    _, collection = get_chroma()
    existing = collection.get(where={"source_id": source_id})
    if existing["ids"]:
        collection.delete(ids=existing["ids"])
        return len(existing["ids"])
    return 0


# ── CLI test ──────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    stats = get_collection_stats()
    print(f"\n=== ChromaDB Stats ===")
    print(f"Total chunks: {stats['total_chunks']}")
    print(f"Total sources: {stats['total_sources']}")
    for s in stats["sources"]:
        print(f"  {s['source_id']}: last updated {s['last_updated'][:10]}")
