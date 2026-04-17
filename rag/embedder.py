"""
rag/embedder.py

Clinical guideline embedder — now using OpenAI's text-embedding-3-small
instead of local sentence-transformers.

Why this change:
  Railway's free trial tier limits us to 1 GB RAM. Loading torch (~300 MB),
  sentence-transformers (~90 MB), and a cross-encoder (~85 MB) plus
  transformers tokenizers (~200 MB) consumed ~700 MB just at idle —
  the OOM killer was killing the process during agent runs.

  OpenAI's text-embedding-3-small:
    - 1536-dimensional embeddings (matches ChromaDB cosine space)
    - $0.00002 per 1K tokens = ~$0.001 for all 20 seed guidelines combined
    - Zero local memory footprint (network call)
    - No model download, no startup time
    - MTEB score comparable to all-MiniLM-L6-v2 on retrieval tasks

  This keeps the exact same ChromaDB persistence and retrieval interface —
  only the embedding function changes.
"""
import logging
import os
import warnings
from datetime import datetime
from pathlib import Path

os.environ["ANONYMIZED_TELEMETRY"] = "false"
os.environ["CHROMA_TELEMETRY"] = "false"
logging.getLogger("chromadb").setLevel(logging.ERROR)
logging.getLogger("httpx").setLevel(logging.ERROR)
warnings.filterwarnings("ignore")

from dotenv import load_dotenv

load_dotenv()

CHROMA_DIR = Path(__file__).parent.parent / "data" / "chroma_guidelines"
CHROMA_DIR.mkdir(parents=True, exist_ok=True)

COLLECTION_NAME = "clinical_guidelines"

# OpenAI embedding model — smallest/cheapest, still strong on retrieval
EMBEDDING_MODEL = "text-embedding-3-small"
EMBEDDING_DIMS = 1536

_chroma_client = None
_collection = None
_openai_client = None


def get_chroma():
    """Get or create the ChromaDB collection (persistent, filesystem-backed)."""
    global _chroma_client, _collection
    if _chroma_client is None:
        import chromadb
        from chromadb.config import Settings
        _chroma_client = chromadb.PersistentClient(
            path=str(CHROMA_DIR),
            settings=Settings(anonymized_telemetry=False)
        )
        _collection = _chroma_client.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"}
        )
    return _chroma_client, _collection


def get_openai_client():
    """Lazy-load the OpenAI client — only created when first embedding happens."""
    global _openai_client
    if _openai_client is None:
        from openai import OpenAI
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError(
                "OPENAI_API_KEY not set. Required for guideline embeddings. "
                "Add it to Railway Variables / Render Environment."
            )
        _openai_client = OpenAI(api_key=api_key)
    return _openai_client


def embed_texts(texts: list[str], batch_size: int = 100) -> list[list[float]]:
    """
    Embed a list of texts using OpenAI's text-embedding-3-small.
    Batches up to 100 at a time (OpenAI's max per request).

    Returns list of embeddings (each 1536 floats), one per input text.
    """
    client = get_openai_client()
    all_embeddings: list[list[float]] = []

    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        # OpenAI auto-truncates to 8191 tokens per item; our chunks are ~400 words
        # so we never hit that limit.
        response = client.embeddings.create(
            model=EMBEDDING_MODEL,
            input=batch
        )
        # Responses come back in the same order as input
        batch_embeddings = [item.embedding for item in response.data]
        all_embeddings.extend(batch_embeddings)

    return all_embeddings


def embed_source(scraped: dict) -> dict:
    """
    Embed all chunks from a scraped guideline source into ChromaDB.
    Deletes previous chunks for this source before inserting new ones.
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

    # Delete existing chunks for this source
    try:
        existing = collection.get(where={"source_id": source_id})
        if existing["ids"]:
            collection.delete(ids=existing["ids"])
            print(f"  [Embedder] Deleted {len(existing['ids'])} old chunks for {source_id}")
    except Exception as e:
        print(f"  [Embedder] Warning: could not delete old chunks: {e}")

    # Generate embeddings via OpenAI
    try:
        embeddings = embed_texts(chunks)
    except Exception as e:
        print(f"  [Embedder] ❌ OpenAI embedding call failed for {source_id}: {e}")
        return {
            "source_id": source_id,
            "status": "skipped",
            "reason": f"OpenAI API error: {e}",
            "chunks_embedded": 0
        }

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

    # Insert in smaller batches to avoid ChromaDB transaction size limits
    batch_size = 50
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
    """Embed all scraped guideline results. Only processes changed sources."""
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


def get_collection_stats() -> dict:
    """Return stats about the current vector store."""
    _, collection = get_chroma()
    count = collection.count()

    if count == 0:
        return {"total_chunks": 0, "total_sources": 0, "sources": [], "last_updated": None}

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
            {"source_id": sid, "last_updated": last_dates.get(sid, "unknown")}
            for sid in sorted(source_ids)
        ],
        "collection_name": COLLECTION_NAME,
        "storage_path": str(CHROMA_DIR)
    }


def clear_source(source_id: str) -> int:
    """Remove all chunks for a specific source."""
    _, collection = get_chroma()
    existing = collection.get(where={"source_id": source_id})
    if existing["ids"]:
        collection.delete(ids=existing["ids"])
        return len(existing["ids"])
    return 0


if __name__ == "__main__":
    stats = get_collection_stats()
    print(f"\n=== ChromaDB Stats ===")
    print(f"Total chunks: {stats['total_chunks']}")
    print(f"Total sources: {stats['total_sources']}")
    for s in stats["sources"]:
        print(f"  {s['source_id']}: last updated {s['last_updated'][:10]}")