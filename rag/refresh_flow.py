"""
rag/refresh_flow.py

Prefect flow for refreshing clinical guidelines in ChromaDB.

Two modes:
  1. Automatic weekly refresh — runs every Sunday at 2 AM
     Only scrapes "weekly" sources (USPSTF, CDC immunizations)
     Only re-embeds sources whose content actually changed

  2. Manual full refresh — triggered via POST /refresh-guidelines API
     Scrapes ALL sources (weekly + manual)
     Used when a major guideline update is published (e.g. ADA Standards Jan 2026)

Run manually:
    python rag/refresh_flow.py

Run with force (re-embed everything):
    python rag/refresh_flow.py --force
"""

import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional, List

from prefect import flow, task, get_run_logger
from prefect.server.schemas.schedules import CronSchedule
from dotenv import load_dotenv

load_dotenv()

REFRESH_LOG = Path(__file__).parent.parent / "data" / "guideline_cache" / "refresh_log.json"


# ── Tasks ─────────────────────────────────────────────────────────────────────

@task(
    name="scrape-guideline-sources",
    description="Fetch and parse all target guideline sources.",
    retries=2,
    retry_delay_seconds=30,
)
def scrape_sources_task(source_ids: Optional[List[str]] = None, force: bool = False) -> list[dict]:
    logger = get_run_logger()
    from rag.guideline_sources import GUIDELINE_SOURCES, SOURCES_BY_ID
    from rag.scraper import scrape_all

    if source_ids:
        sources = [SOURCES_BY_ID[sid] for sid in source_ids if sid in SOURCES_BY_ID]
        logger.info(f"Scraping {len(sources)} specific sources: {source_ids}")
    else:
        sources = GUIDELINE_SOURCES
        logger.info(f"Scraping all {len(sources)} sources")

    results = scrape_all(sources, force=force)

    changed = sum(1 for r in results if r.get("changed"))
    errors = sum(1 for r in results if r.get("error"))
    logger.info(f"Scrape complete: {changed} changed, {errors} errors")

    return results


@task(
    name="embed-changed-sources",
    description="Embed changed guideline chunks into ChromaDB.",
    retries=1,
)
def embed_sources_task(scraped_results: list[dict]) -> list[dict]:
    logger = get_run_logger()
    from rag.embedder import embed_all

    to_embed = sum(1 for r in scraped_results if r.get("changed"))
    logger.info(f"Embedding {to_embed} changed sources...")

    stats = embed_all(scraped_results)

    total_chunks = sum(s.get("chunks_embedded", 0) for s in stats)
    logger.info(f"Embedding complete: {total_chunks} total chunks embedded")

    return stats


@task(
    name="validate-collection",
    description="Verify the ChromaDB collection is healthy after refresh.",
)
def validate_collection_task() -> dict:
    logger = get_run_logger()
    from rag.embedder import get_collection_stats
    from rag.retriever import retrieve_guidelines

    stats = get_collection_stats()
    logger.info(f"Collection has {stats['total_chunks']} chunks across {stats['total_sources']} sources")

    if stats["total_chunks"] > 0:
        results = retrieve_guidelines("mammogram screening women", n_results=1)
        if results and results[0].get("relevance_score", 0) > 0.1:
            logger.info(f"Validation query OK: returned '{results[0]['source_name']}'")
            stats["validation"] = "PASS"
        else:
            logger.warning("Validation query returned no results")
            stats["validation"] = "WARN — no results"
    else:
        stats["validation"] = "SKIP — collection empty"

    return stats


@task(
    name="write-refresh-log",
    description="Write refresh summary to the audit log.",
)
def write_refresh_log_task(embed_stats: list[dict], collection_stats: dict,
                            mode: str, triggered_by: str = "prefect") -> dict:
    logger = get_run_logger()

    REFRESH_LOG.parent.mkdir(parents=True, exist_ok=True)

    existing = []
    if REFRESH_LOG.exists():
        try:
            with open(REFRESH_LOG) as f:
                existing = json.load(f)
        except Exception:
            existing = []

    entry = {
        "timestamp": datetime.utcnow().isoformat(),
        "mode": mode,
        "triggered_by": triggered_by,
        "sources_embedded": sum(1 for s in embed_stats if s["status"] == "embedded"),
        "sources_unchanged": sum(1 for s in embed_stats if s["status"] == "unchanged"),
        "sources_skipped": sum(1 for s in embed_stats if s["status"] == "skipped"),
        "total_chunks": collection_stats.get("total_chunks", 0),
        "total_sources": collection_stats.get("total_sources", 0),
        "validation": collection_stats.get("validation", "UNKNOWN"),
        "embed_details": embed_stats
    }

    existing.insert(0, entry)
    existing = existing[:50]

    with open(REFRESH_LOG, "w") as f:
        json.dump(existing, f, indent=2)

    logger.info(f"Refresh log written: {entry['sources_embedded']} sources updated")
    return entry


# ── Flows ─────────────────────────────────────────────────────────────────────

@flow(name="weekly-guidelines-refresh", log_prints=True)
def weekly_refresh_flow() -> dict:
    logger = get_run_logger()
    logger.info("=== Weekly Guidelines Refresh ===")

    from rag.guideline_sources import WEEKLY_SOURCES
    weekly_ids = [s["id"] for s in WEEKLY_SOURCES]

    scraped = scrape_sources_task(source_ids=weekly_ids)
    embed_stats = embed_sources_task(scraped)
    collection_stats = validate_collection_task()
    log_entry = write_refresh_log_task(embed_stats, collection_stats,
                                        mode="weekly", triggered_by="prefect_schedule")

    return log_entry


@flow(name="manual-guidelines-refresh", log_prints=True)
def manual_refresh_flow(
    source_ids: Optional[List[str]] = None,
    force: bool = False,
    triggered_by: str = "api"
) -> dict:
    logger = get_run_logger()

    source_ids = source_ids or []

    mode = "manual_full" if not source_ids else f"manual_targeted:{','.join(source_ids)}"
    logger.info(f"=== Manual Guidelines Refresh ({mode}) ===")

    if force:
        logger.info("FORCE mode enabled")

    scraped = scrape_sources_task(source_ids=source_ids, force=force)
    embed_stats = embed_sources_task(scraped)
    collection_stats = validate_collection_task()
    log_entry = write_refresh_log_task(embed_stats, collection_stats,
                                        mode=mode, triggered_by=triggered_by)

    return log_entry


# ── CLI entry ─────────────────────────────────────────────────────────

if __name__ == "__main__":
    force = "--force" in sys.argv
    target_ids = None

    for arg in sys.argv[1:]:
        if arg.startswith("--sources="):
            target_ids = arg.replace("--sources=", "").split(",")

    print(f"\nRunning manual refresh (force={force}, sources={target_ids or 'ALL'})")

    result = manual_refresh_flow(
        source_ids=target_ids or [],
        force=force,
        triggered_by="cli"
    )

    print("\n=== Refresh Complete ===")
    print(f"Sources embedded: {result['sources_embedded']}")
    print(f"Sources unchanged: {result['sources_unchanged']}")
    print(f"Total chunks: {result['total_chunks']}")
    print(f"Validation: {result['validation']}")