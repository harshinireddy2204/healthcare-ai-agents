"""
rag/scraper.py

Fetches clinical guideline content from source URLs, parses it into
clean text chunks, and detects changes using content hashing.

Key design decisions:
  - Hash-based change detection: only re-embed chunks that actually changed
  - Graceful fallback: if a URL fails, use the last successful version
  - Chunk overlap: 50-token overlap so context isn't lost at boundaries
  - Source tagging: every chunk carries source_id, url, date for citations
"""
import hashlib
import json
import os
import re
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

import httpx
from dotenv import load_dotenv

load_dotenv()

# Directory to cache raw guideline content and hashes
CACHE_DIR = Path(__file__).parent.parent / "data" / "guideline_cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

HASH_FILE = CACHE_DIR / "content_hashes.json"
CHUNK_SIZE = 400       # tokens (~300 words per chunk)
CHUNK_OVERLAP = 50     # overlap between adjacent chunks


# ── Hash helpers ──────────────────────────────────────────────────────────────

def load_hashes() -> dict:
    if HASH_FILE.exists():
        with open(HASH_FILE) as f:
            return json.load(f)
    return {}


def save_hashes(hashes: dict):
    with open(HASH_FILE, "w") as f:
        json.dump(hashes, f, indent=2)


def content_hash(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()[:16]


# ── Text fetching ─────────────────────────────────────────────────────────────

def fetch_url(url: str, timeout: int = 20) -> Optional[str]:
    """
    Fetch a URL and return clean text content.
    Returns None on failure (caller uses cached version).
    """
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (compatible; HealthcareAI-GuidelineBot/1.0; "
            "+https://github.com/harshinireddy2204/healthcare-ai-agents)"
        ),
        "Accept": "text/html,application/xhtml+xml,text/plain",
    }

    try:
        with httpx.Client(follow_redirects=True, timeout=timeout) as client:
            response = client.get(url, headers=headers)
            response.raise_for_status()
            return response.text
    except Exception as e:
        print(f"  [Scraper] Failed to fetch {url}: {e}")
        return None


def clean_html(raw: str) -> str:
    """Strip HTML tags and normalize whitespace."""
    # Remove scripts and styles
    raw = re.sub(r"<script[^>]*>.*?</script>", " ", raw, flags=re.DOTALL | re.IGNORECASE)
    raw = re.sub(r"<style[^>]*>.*?</style>", " ", raw, flags=re.DOTALL | re.IGNORECASE)
    # Remove HTML tags
    raw = re.sub(r"<[^>]+>", " ", raw)
    # Normalize whitespace
    raw = re.sub(r"\s+", " ", raw).strip()
    # Decode common HTML entities
    raw = raw.replace("&amp;", "&").replace("&lt;", "<").replace("&gt;", ">")
    raw = raw.replace("&nbsp;", " ").replace("&#39;", "'").replace("&quot;", '"')
    return raw


# ── Chunking ──────────────────────────────────────────────────────────────────

def chunk_text(text: str, chunk_size: int = CHUNK_SIZE,
               overlap: int = CHUNK_OVERLAP) -> list[str]:
    """
    Split text into overlapping chunks by word count.
    Tries to break at sentence boundaries for readability.
    """
    # Split into sentences first
    sentences = re.split(r"(?<=[.!?])\s+", text)

    chunks = []
    current_words = []
    current_size = 0

    for sentence in sentences:
        words = sentence.split()
        word_count = len(words)

        if current_size + word_count > chunk_size and current_words:
            # Save current chunk
            chunks.append(" ".join(current_words))
            # Keep overlap
            current_words = current_words[-overlap:] if overlap > 0 else []
            current_size = len(current_words)

        current_words.extend(words)
        current_size += word_count

    # Don't lose the last chunk
    if current_words:
        chunks.append(" ".join(current_words))

    # Filter out very short chunks (navigation text, headers etc.)
    return [c for c in chunks if len(c.split()) > 20]


# ── Cache helpers ─────────────────────────────────────────────────────────────

def get_cache_path(source_id: str) -> Path:
    return CACHE_DIR / f"{source_id}.txt"


def load_cached(source_id: str) -> Optional[str]:
    path = get_cache_path(source_id)
    if path.exists():
        return path.read_text(encoding="utf-8")
    return None


def save_cache(source_id: str, text: str):
    get_cache_path(source_id).write_text(text, encoding="utf-8")


# ── Main scrape function ──────────────────────────────────────────────────────

def scrape_guideline(source: dict) -> dict:
    """
    Scrape a single guideline source and return structured result.

    Returns:
        {
            "source_id": str,
            "name": str,
            "url": str,
            "category": str,
            "chunks": list[str],
            "changed": bool,         # True if content changed since last scrape
            "hash": str,
            "scraped_at": str,
            "error": str | None
        }
    """
    source_id = source["id"]
    print(f"  [Scraper] Fetching: {source['name']}")

    hashes = load_hashes()

    # Try to fetch fresh content
    raw = fetch_url(source["url"])

    if raw is None:
        # Fall back to cache
        cached = load_cached(source_id)
        if cached:
            print(f"  [Scraper] Using cached version for {source_id}")
            chunks = chunk_text(cached)
            return {
                "source_id": source_id,
                "name": source["name"],
                "url": source["url"],
                "category": source["category"],
                "chunks": chunks,
                "changed": False,
                "hash": hashes.get(source_id, ""),
                "scraped_at": datetime.utcnow().isoformat(),
                "error": "Fetch failed — using cached version"
            }
        else:
            return {
                "source_id": source_id,
                "name": source["name"],
                "url": source["url"],
                "category": source["category"],
                "chunks": [],
                "changed": False,
                "hash": "",
                "scraped_at": datetime.utcnow().isoformat(),
                "error": "Fetch failed — no cache available"
            }

    # Clean and chunk
    clean = clean_html(raw)
    new_hash = content_hash(clean)
    old_hash = hashes.get(source_id, "")
    changed = new_hash != old_hash

    if changed:
        print(f"  [Scraper] Content CHANGED for {source_id} (hash: {old_hash[:8]} → {new_hash[:8]})")
        save_cache(source_id, clean)
        hashes[source_id] = new_hash
        save_hashes(hashes)
    else:
        print(f"  [Scraper] No change for {source_id}")
        clean = load_cached(source_id) or clean

    chunks = chunk_text(clean)

    # Rate limit between requests
    time.sleep(1)

    return {
        "source_id": source_id,
        "name": source["name"],
        "url": source["url"],
        "category": source["category"],
        "keywords": source.get("keywords", []),
        "chunks": chunks,
        "changed": changed,
        "hash": new_hash,
        "scraped_at": datetime.utcnow().isoformat(),
        "error": None
    }


def scrape_all(sources: list[dict], force: bool = False) -> list[dict]:
    """
    Scrape all provided sources.
    If force=True, re-embed even unchanged content.
    """
    results = []
    print(f"\n[Scraper] Starting scrape of {len(sources)} sources...")

    for i, source in enumerate(sources, 1):
        print(f"\n[Scraper] [{i}/{len(sources)}] {source['id']}")
        result = scrape_guideline(source)
        if force:
            result["changed"] = True  # force re-embed
        results.append(result)

    changed = sum(1 for r in results if r["changed"])
    errors = sum(1 for r in results if r["error"])
    print(f"\n[Scraper] Done. {changed} changed, {errors} errors out of {len(sources)} sources.")

    return results


# ── CLI test ──────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    from rag.guideline_sources import WEEKLY_SOURCES
    results = scrape_all(WEEKLY_SOURCES[:2])  # test with first 2
    for r in results:
        print(f"\n{r['source_id']}: {len(r['chunks'])} chunks, changed={r['changed']}")
        if r["chunks"]:
            print(f"  First chunk preview: {r['chunks'][0][:200]}...")
