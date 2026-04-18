"""
api/main.py

FastAPI service — REST interface for the healthcare AI agent system.

Two-mode access control:
  Public endpoints (no auth): health, audit-log, pending-reviews, workflow-status,
                              analytics, guidelines-status, guidelines-search,
                              diagnostics, resolve-review
  Passcode-gated endpoints:   /process-patient, /refresh-guidelines
                              (these consume OpenAI tokens)

Set DEMO_PASSCODE env var to enable live runs. If unset, live runs are rejected
with HTTP 403 and the app runs in demo-only mode (cached data + free APIs only).
"""
import json
import os
from datetime import datetime
from typing import Optional, Literal

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel
from sqlalchemy import create_engine, text

load_dotenv()

app = FastAPI(
    title="Healthcare AI Multi-Agent System",
    description=(
        "Prior authorization, care gap detection, and risk triage "
        "powered by LangGraph + CrewAI agents."
    ),
    version="1.0.0",
)

DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./healthcare_agents.db")
engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})


# ── Passcode gate for token-burning endpoints ─────────────────────────────────
#
# Only /process-patient and /refresh-guidelines consume OpenAI tokens per call.
# Everything else (SQLite reads, OpenFDA, FHIR, Knowledge Graph, guideline
# SEARCH) is free or negligible. We gate those two endpoints behind a passcode
# so the shared OpenAI budget isn't drained by drive-by traffic.
#
# The automatic startup seed embedding (~$0.001 per deploy) is NOT gated — it
# runs once on boot without a passcode because without it, RAG is dead.
# Similarly, the frontend's empty-ChromaDB recovery fallback is not gated.

DEMO_PASSCODE = os.getenv("DEMO_PASSCODE", "").strip()


def _check_passcode(provided: Optional[str]) -> tuple[bool, str]:
    """
    Validate a provided passcode against DEMO_PASSCODE.
    Returns (is_valid, error_message). If DEMO_PASSCODE is unset, always rejects
    (demo-only mode). Uses constant-time comparison to prevent timing attacks.
    """
    if not DEMO_PASSCODE:
        return False, (
            "Live agent workflows are disabled on this deployment "
            "(DEMO_PASSCODE env var not set). Explore the cached demo runs "
            "in 'Example workflow outputs' and all other dashboard pages — "
            "they're fully functional."
        )

    if not provided:
        return False, "Passcode required. Contact the deployment owner for access."

    # Constant-time comparison
    import hmac
    if not hmac.compare_digest(provided.strip(), DEMO_PASSCODE):
        return False, "Incorrect passcode."

    return True, ""


# ── Demo workflow cache ────────────────────────────────────────────────────────
# Pre-recorded workflow outputs served from data/demo_runs.json so visitors
# can see complete agent output without burning tokens. Regenerate with
# `python scripts/generate_demo_runs.py`.

_DEMO_RUNS: dict = {}


def _load_demo_runs():
    """Load pre-recorded workflow outputs at startup."""
    global _DEMO_RUNS
    from pathlib import Path
    demo_path = Path("data/demo_runs.json")
    if demo_path.exists():
        try:
            with open(demo_path) as f:
                _DEMO_RUNS = json.load(f)
            print(f"[Startup] ✅ Loaded {len(_DEMO_RUNS)} cached demo workflow runs")
        except Exception as e:
            print(f"[Startup] Could not load demo_runs.json: {e}")
            _DEMO_RUNS = {}
    else:
        print("[Startup] data/demo_runs.json not found — cached demo runs unavailable")
        _DEMO_RUNS = {}


# ── Database setup ────────────────────────────────────────────────────────────

def init_db():
    with engine.connect() as conn:
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS review_queue (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                patient_id  TEXT NOT NULL,
                created_at  TEXT NOT NULL,
                status      TEXT NOT NULL DEFAULT 'PENDING',
                agent_output TEXT,
                auth_results TEXT,
                reason      TEXT,
                resolved_by TEXT,
                resolved_at TEXT,
                resolution  TEXT,
                notes       TEXT
            )
        """))
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS audit_log (
                id           INTEGER PRIMARY KEY AUTOINCREMENT,
                patient_id   TEXT NOT NULL,
                processed_at TEXT NOT NULL,
                status       TEXT NOT NULL,
                result_json  TEXT
            )
        """))
        conn.commit()


def _seed_guidelines_if_empty():
    """
    Embed hardcoded seed guidelines into ChromaDB if the collection is empty.
    Runs once on startup. Costs ~$0.001 per deploy — essential for RAG to work.
    This is NOT passcode-gated because agents need it to function.
    """
    try:
        from rag.embedder import get_collection_stats, embed_source
        stats = get_collection_stats()

        if stats.get("total_chunks", 0) > 0:
            print(f"[Startup] Guidelines already loaded: {stats['total_chunks']} chunks across {stats['total_sources']} sources")
            return

        print("[Startup] ChromaDB is empty — embedding seed guidelines...")
        from rag.guideline_seed import get_seed_as_scraped_format, get_seed_count
        seed_data = get_seed_as_scraped_format()

        embedded_count = 0
        for source in seed_data:
            result = embed_source(source)
            if result.get("status") == "embedded":
                embedded_count += result.get("chunks_embedded", 0)

        print(f"[Startup] ✅ Seed guidelines embedded: {embedded_count} chunks across {len(seed_data)} sources")
        print("[Startup] Agents now have RAG access. Run /refresh-guidelines for full 63-source load.")

    except ImportError as e:
        print(f"[Startup] RAG dependencies not available (OK for lightweight deploys): {e}")
    except Exception as e:
        print(f"[Startup] Warning: could not seed guidelines: {e}")
        import traceback
        traceback.print_exc()


@app.on_event("startup")
def on_startup():
    init_db()

    # Load demo runs cache
    _load_demo_runs()

    # Report passcode mode
    if DEMO_PASSCODE:
        print(f"[Startup] 🔐 Passcode gating ENABLED on /process-patient and /refresh-guidelines")
    else:
        print(f"[Startup] ⚠️  DEMO_PASSCODE not set — live agent runs are DISABLED (demo-only mode)")

    # Seed demo data if audit_log is empty (fresh deploy / Railway restart).
    try:
        with engine.connect() as conn:
            row_count = conn.execute(text("SELECT COUNT(*) FROM audit_log")).scalar() or 0
        if row_count == 0:
            print("[Startup] audit_log is empty — seeding demo data...")
            from scripts.reset_demo_data import reset_demo_data
            reset_demo_data()
        else:
            print(f"[Startup] audit_log has {row_count} rows — skipping demo seed")
    except Exception as e:
        print(f"[Startup] Warning: could not seed demo data: {e}")

    # Validate OpenAI credentials
    try:
        from utils.llm_utils import check_openai_credentials
        cred_check = check_openai_credentials()
        if cred_check["ok"]:
            print(f"[Startup] ✅ {cred_check['message']}")
        else:
            print(f"[Startup] ⚠️  OpenAI credential check FAILED: {cred_check['message']}")
    except Exception as e:
        print(f"[Startup] Could not validate OpenAI credentials: {e}")

    # Seed guidelines in background
    import threading
    t = threading.Thread(target=_seed_guidelines_if_empty, daemon=True)
    t.start()
    print("[Startup] API ready. Guidelines seeding in background...")


# ── Pydantic models ───────────────────────────────────────────────────────────

class ProcessPatientRequest(BaseModel):
    patient_id: str
    mode: Literal["full", "auth_only", "care_gap_only"] = "full"
    passcode: Optional[str] = None  # Required to trigger live runs


class ResolveReviewRequest(BaseModel):
    resolution: Literal["APPROVED", "REJECTED", "MODIFIED"]
    resolved_by: str
    notes: Optional[str] = None


class PatientSummary(BaseModel):
    patient_id: str
    status: str
    message: str
    workflow_triggered: str


# ── Audit log helpers ─────────────────────────────────────────────────────────

def write_audit(patient_id: str, status: str, result: dict):
    try:
        with engine.connect() as conn:
            conn.execute(text("""
                INSERT INTO audit_log (patient_id, processed_at, status, result_json)
                VALUES (:pid, :ts, :status, :result)
            """), {
                "pid": patient_id,
                "ts": datetime.utcnow().isoformat(),
                "status": status,
                "result": json.dumps(result)
            })
            conn.commit()
        print(f"[Audit] {patient_id} → {status}")
    except Exception as e:
        print(f"[Audit] Failed to write log for {patient_id}: {e}")


# ── Global workflow concurrency control ──────────────────────────────────────

import threading

_workflow_lock = threading.Lock()
_current_workflow: dict = {"patient_id": None, "mode": None, "started_at": None}


def is_workflow_running() -> dict:
    if _workflow_lock.locked() and _current_workflow["patient_id"]:
        elapsed = (datetime.utcnow() - _current_workflow["started_at"]).total_seconds() \
            if _current_workflow["started_at"] else 0
        return {
            "running": True,
            "patient_id": _current_workflow["patient_id"],
            "mode": _current_workflow["mode"],
            "elapsed_seconds": int(elapsed),
            "estimated_remaining_seconds": max(0, 180 - int(elapsed)),
        }
    return {"running": False}


# ── Background task ───────────────────────────────────────────────────────────

def run_triage_background(patient_id: str, mode: str):
    print(f"\n[Background] Starting {mode} workflow for {patient_id}")
    _current_workflow["patient_id"] = patient_id
    _current_workflow["mode"] = mode
    _current_workflow["started_at"] = datetime.utcnow()

    def _with_retry(fn, *args, max_retries=3, base_delay=15):
        import time
        for attempt in range(max_retries):
            try:
                return fn(*args)
            except Exception as e:
                err_str = str(e)
                is_rate_limit = "429" in err_str or "rate_limit" in err_str.lower()
                is_connection = "connection" in err_str.lower() or "APIConnectionError" in err_str
                if is_rate_limit or is_connection:
                    wait = base_delay * (2 ** attempt)
                    kind = "Rate limit" if is_rate_limit else "Connection error"
                    print(f"[Background] {kind} — waiting {wait}s before retry {attempt+1}/{max_retries}")
                    import time as t; t.sleep(wait)
                else:
                    raise
        raise RuntimeError(f"Max retries exceeded for {patient_id}")

    try:
        if mode == "auth_only":
            from agents.prior_auth_agent import run_prior_auth
            from tools.ehr_tools import get_pending_auth_requests

            requests = get_pending_auth_requests.invoke({"patient_id": patient_id})
            results = []
            for req in (requests or []):
                if isinstance(req, dict) and "request_id" in req:
                    r = _with_retry(run_prior_auth, patient_id, req["request_id"], req["item"])
                    results.append(r)
                    print(f"[Background] Auth result: {r.get('decision')} ({r.get('confidence', 0):.0%})")

            write_audit(patient_id, "COMPLETED", {
                "mode": "auth_only",
                "auth_results": results
            })

        elif mode == "care_gap_only":
            from agents.care_gap_agent import run_care_gap_review

            print(f"[Background] Running care gap review for {patient_id}...")
            result = _with_retry(run_care_gap_review, patient_id)
            print(f"[Background] Care gap complete: {result['steps_executed']} steps")

            write_audit(patient_id, "COMPLETED", {
                "mode": "care_gap_only",
                "steps_executed": result["steps_executed"],
                "final_report": result["final_report"]
            })

        else:  # full
            from agents.triage_supervisor import run_triage

            result = _with_retry(run_triage, patient_id)
            result["mode"] = "full"
            status = result.get("status", "COMPLETED")
            write_audit(patient_id, status, result)

        print(f"[Background] {patient_id} workflow finished successfully")

    except Exception as e:
        print(f"[Background] ERROR for {patient_id}: {e}")
        import traceback
        traceback.print_exc()
        write_audit(patient_id, "FAILED", {"error": str(e), "mode": mode})

    finally:
        _current_workflow["patient_id"] = None
        _current_workflow["mode"] = None
        _current_workflow["started_at"] = None
        if _workflow_lock.locked():
            _workflow_lock.release()


# ── Public endpoints ──────────────────────────────────────────────────────────

@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "version": "1.0.0",
        "demo_mode": not bool(DEMO_PASSCODE),
        "live_runs_enabled": bool(DEMO_PASSCODE),
    }


@app.get("/diagnostics")
def diagnostics():
    results = {"timestamp": datetime.utcnow().isoformat()}

    try:
        from utils.llm_utils import check_openai_credentials
        results["openai"] = check_openai_credentials()
    except Exception as e:
        results["openai"] = {"ok": False, "message": f"Check failed: {e}"}

    env_status = {}
    for key in ["OPENAI_API_KEY", "MODEL_NAME", "DATABASE_URL", "LANGCHAIN_API_KEY", "USE_FHIR", "DEMO_PASSCODE"]:
        val = os.getenv(key, "")
        if key == "OPENAI_API_KEY":
            env_status[key] = f"set (length={len(val)}, starts with '{val[:3]}')" if val else "NOT SET"
        elif key == "LANGCHAIN_API_KEY":
            env_status[key] = "set" if val else "not set (optional)"
        elif key == "DEMO_PASSCODE":
            env_status[key] = "set (live runs enabled)" if val else "not set (demo-only mode)"
        else:
            env_status[key] = val or "not set"
    results["environment"] = env_status

    try:
        import httpx
        with httpx.Client(timeout=5) as client:
            r = client.get("https://api.fda.gov/drug/label.json", params={"limit": 1})
            results["openfda"] = {
                "ok": r.status_code == 200,
                "status_code": r.status_code,
                "message": "OpenFDA API reachable" if r.status_code == 200 else f"HTTP {r.status_code}"
            }
    except Exception as e:
        results["openfda"] = {"ok": False, "message": f"Cannot reach OpenFDA: {type(e).__name__}: {e}"}

    try:
        from rag.embedder import get_collection_stats
        stats = get_collection_stats()
        results["guidelines"] = {
            "ok": stats.get("total_chunks", 0) > 0,
            "total_chunks": stats.get("total_chunks", 0),
            "total_sources": stats.get("total_sources", 0),
        }
    except Exception as e:
        results["guidelines"] = {"ok": False, "message": f"ChromaDB error: {e}"}

    results["overall_ok"] = all([
        results["openai"].get("ok", False),
        results["openfda"].get("ok", False),
    ])

    return results


@app.get("/demo-runs")
def list_demo_runs():
    """Return index of pre-recorded workflow runs (public, no auth)."""
    return {
        "count": len(_DEMO_RUNS),
        "available_patients": list(_DEMO_RUNS.keys()),
        "runs": [
            {
                "patient_id": pid,
                "complexity_tier": data.get("complexity_tier", ""),
                "status": data.get("status", ""),
                "mode": data.get("mode", "full"),
                "generated_at": data.get("_generated_at", ""),
            }
            for pid, data in _DEMO_RUNS.items()
        ]
    }


@app.get("/demo-runs/{patient_id}")
def get_demo_run(patient_id: str):
    """Return the full pre-recorded workflow output for one patient (public)."""
    if patient_id not in _DEMO_RUNS:
        raise HTTPException(
            status_code=404,
            detail=f"No cached demo run for patient {patient_id}. "
                   f"Available: {list(_DEMO_RUNS.keys())}"
        )
    return _DEMO_RUNS[patient_id]


# ── 🔐 PASSCODE-GATED: Live agent execution ───────────────────────────────────

@app.post("/process-patient", response_model=PatientSummary)
def process_patient(
    request: ProcessPatientRequest,
    background_tasks: BackgroundTasks
):
    """
    🔐 PASSCODE REQUIRED. Triggers a live agent workflow that consumes OpenAI
    tokens (~$0.04 per run). Set DEMO_PASSCODE env var to enable.
    """
    # Passcode gate
    is_valid, error_msg = _check_passcode(request.passcode)
    if not is_valid:
        raise HTTPException(
            status_code=403,
            detail={
                "error": "passcode_required",
                "message": error_msg,
                "hint": "Browse the 'Example workflow outputs' section to see "
                        "pre-recorded agent runs with complete output — no passcode needed."
            }
        )

    # Workflow concurrency lock
    if not _workflow_lock.acquire(blocking=False):
        current = is_workflow_running()
        raise HTTPException(
            status_code=409,
            detail={
                "error": "workflow_in_progress",
                "message": (
                    f"Another workflow is already running for patient "
                    f"{current.get('patient_id', 'unknown')} "
                    f"(mode: {current.get('mode', 'unknown')}, "
                    f"running {current.get('elapsed_seconds', 0)}s). "
                    f"Please wait ~{current.get('estimated_remaining_seconds', 180)}s and try again."
                ),
                "current_workflow": current
            }
        )

    background_tasks.add_task(
        run_triage_background,
        request.patient_id,
        request.mode
    )

    mode_labels = {
        "full": "Prior auth + care gap + risk triage (CrewAI supervisor)",
        "auth_only": "Prior authorization only",
        "care_gap_only": "Care gap review only"
    }

    return PatientSummary(
        patient_id=request.patient_id,
        status="PROCESSING",
        message="Workflow triggered. Check /audit-log for results in 60-180 seconds.",
        workflow_triggered=mode_labels[request.mode]
    )


@app.get("/workflow-status")
def workflow_status():
    return is_workflow_running()


@app.get("/pending-reviews")
def get_pending_reviews(limit: int = 20):
    with engine.connect() as conn:
        rows = conn.execute(text("""
            SELECT id, patient_id, created_at, status, reason, agent_output
            FROM review_queue
            WHERE status = 'PENDING'
            ORDER BY created_at DESC
            LIMIT :limit
        """), {"limit": limit}).fetchall()

    reviews = []
    for row in rows:
        reviews.append({
            "review_id": row[0],
            "patient_id": row[1],
            "created_at": row[2],
            "status": row[3],
            "reason": row[4],
            "agent_output_preview": (row[5] or "")[:300] + "..." if row[5] and len(row[5]) > 300 else row[5]
        })

    return {"pending_count": len(reviews), "reviews": reviews}


@app.post("/resolve-review/{review_id}")
def resolve_review(review_id: int, request: ResolveReviewRequest):
    with engine.connect() as conn:
        row = conn.execute(
            text("SELECT id, status FROM review_queue WHERE id = :id"),
            {"id": review_id}
        ).fetchone()

        if not row:
            raise HTTPException(status_code=404, detail=f"Review {review_id} not found")
        if row[1] != "PENDING":
            raise HTTPException(status_code=400, detail=f"Review {review_id} is already {row[1]}")

        conn.execute(text("""
            UPDATE review_queue
            SET status = 'RESOLVED',
                resolved_by = :resolved_by,
                resolved_at = :resolved_at,
                resolution = :resolution,
                notes = :notes
            WHERE id = :id
        """), {
            "id": review_id,
            "resolved_by": request.resolved_by,
            "resolved_at": datetime.utcnow().isoformat(),
            "resolution": request.resolution,
            "notes": request.notes
        })
        conn.commit()

    return {
        "review_id": review_id,
        "resolution": request.resolution,
        "resolved_by": request.resolved_by,
        "resolved_at": datetime.utcnow().isoformat(),
        "message": "Review resolved successfully"
    }


@app.get("/audit-log")
def get_audit_log(patient_id: Optional[str] = None, limit: int = 50):
    query = "SELECT * FROM audit_log"
    params: dict = {"limit": limit}

    if patient_id:
        query += " WHERE patient_id = :patient_id"
        params["patient_id"] = patient_id

    query += " ORDER BY processed_at DESC LIMIT :limit"

    with engine.connect() as conn:
        rows = conn.execute(text(query), params).fetchall()

    return {
        "total": len(rows),
        "entries": [
            {
                "id": row[0],
                "patient_id": row[1],
                "processed_at": row[2],
                "status": row[3],
                "result": json.loads(row[4]) if row[4] else {}
            }
            for row in rows
        ]
    }


# ── Guidelines RAG endpoints ──────────────────────────────────────────────────

class RefreshGuidelinesRequest(BaseModel):
    source_ids: Optional[list[str]] = None
    force: bool = False
    triggered_by: str = "api"
    passcode: Optional[str] = None  # Required for dashboard-triggered refreshes


# ── 🔐 PASSCODE-GATED: Guideline refresh ──────────────────────────────────────

@app.post("/refresh-guidelines")
def refresh_guidelines(
    request: RefreshGuidelinesRequest,
    background_tasks: BackgroundTasks
):
    """
    🔐 PASSCODE REQUIRED (with one exception). Scrapes 63 URLs and embeds
    ~700 chunks via OpenAI's embedding API (~$0.10 per full refresh).

    Exception: `triggered_by=frontend_fallback` is allowed without passcode.
    This is the one-time auto-seed that the frontend triggers when ChromaDB
    is empty on fresh deploy — a startup recovery, not a user action.
    """
    AUTO_FALLBACK_TRIGGER = "frontend_fallback"
    if request.triggered_by != AUTO_FALLBACK_TRIGGER:
        is_valid, error_msg = _check_passcode(request.passcode)
        if not is_valid:
            raise HTTPException(
                status_code=403,
                detail={
                    "error": "passcode_required",
                    "message": error_msg + " (Guideline refresh consumes OpenAI embedding tokens.)",
                    "hint": "Searching existing guidelines is free — try the search box above."
                }
            )

    background_tasks.add_task(
        _run_guidelines_refresh_background,
        request.source_ids,
        request.force,
        request.triggered_by
    )

    scope = f"{len(request.source_ids)} specific sources" if request.source_ids else "ALL sources"
    return {
        "status": "REFRESH_STARTED",
        "scope": scope,
        "force": request.force,
        "message": f"Guidelines refresh triggered for {scope}. Check /guidelines-status for progress.",
        "triggered_at": datetime.utcnow().isoformat()
    }


@app.get("/guidelines-status")
def get_guidelines_status():
    try:
        from rag.embedder import get_collection_stats

        stats = get_collection_stats()

        import json as _json
        from pathlib import Path
        log_path = Path("data/guideline_cache/refresh_log.json")
        log = []
        if log_path.exists():
            try:
                with open(log_path) as f:
                    log = _json.load(f)[:5]
            except Exception:
                log = []

        seed_sources = [s["source_id"] for s in stats.get("sources", []) if s["source_id"].startswith("seed_")]
        scraped_sources = [s for s in stats.get("sources", []) if not s["source_id"].startswith("seed_")]

        if stats["total_chunks"] == 0:
            status_msg = "empty — seeding in progress or run /refresh-guidelines"
        elif seed_sources and not scraped_sources:
            status_msg = f"seeded ({len(seed_sources)} core sources, {stats['total_chunks']} chunks) — run /refresh-guidelines for full 63-source load"
        else:
            status_msg = f"healthy ({stats['total_sources']} sources, {stats['total_chunks']} chunks)"

        return {
            "collection": stats,
            "recent_refreshes": log,
            "status": status_msg,
            "seed_sources_count": len(seed_sources),
            "scraped_sources_count": len(scraped_sources),
        }
    except Exception as e:
        return {
            "collection": {"total_chunks": 0, "total_sources": 0},
            "recent_refreshes": [],
            "status": f"error: {e}"
        }


@app.get("/guidelines-search")
def search_guidelines(q: str, category: Optional[str] = None, n: int = 3):
    """Semantic search over guidelines. Public — each query costs ~$0.000002 (negligible)."""
    try:
        from rag.retriever import retrieve_guidelines
        results = retrieve_guidelines(q, category=category, n_results=n)
        return {
            "query": q,
            "category": category,
            "results": results
        }
    except Exception as e:
        return {"error": str(e), "query": q}


# ── Analytics endpoints (all public, SQLite reads only) ───────────────────────

def _safe_analytics(fn, *args, fallback=None):
    try:
        return fn(*args)
    except Exception as e:
        print(f"[Analytics] {fn.__name__} error: {e}")
        return fallback or {"error": str(e)}


@app.get("/analytics/performance")
def analytics_performance(days: int = 30):
    from analytics.queries import get_agent_performance_summary
    return _safe_analytics(get_agent_performance_summary, days, fallback={
        "period_days": days, "total_runs": 0, "completed": 0,
        "escalated_to_human": 0, "failed": 0, "unique_patients": 0,
        "completion_rate": 0, "escalation_rate": 0, "automation_rate": 0
    })


@app.get("/analytics/prior-auth")
def analytics_prior_auth(days: int = 30):
    from analytics.queries import get_prior_auth_metrics
    return _safe_analytics(get_prior_auth_metrics, days, fallback={
        "total_auth_requests": 0, "decisions": {"APPROVE": 0, "DENY": 0, "ESCALATE": 0},
        "approval_rate": 0, "escalation_rate": 0, "avg_confidence": 0,
        "critic_reviewed": 0, "agent_revised": 0, "revision_rate": 0
    })


@app.get("/analytics/care-gaps")
def analytics_care_gaps(days: int = 30):
    from analytics.queries import get_care_gap_metrics
    return _safe_analytics(get_care_gap_metrics, days, fallback={
        "patients_analyzed": 0, "patients_with_gaps": 0,
        "total_gaps_identified": 0, "avg_gaps_per_patient": 0, "gap_frequency": []
    })


@app.get("/analytics/complexity")
def analytics_complexity(days: int = 30):
    from analytics.queries import get_complexity_distribution
    return _safe_analytics(get_complexity_distribution, days, fallback={
        "distribution": {"LOW": 0, "MODERATE": 0, "HIGH": 0},
        "pct_low": 0, "pct_moderate": 0, "pct_high": 0,
        "estimated_tokens_used": 0, "estimated_tokens_saved": 0,
        "estimated_cost_savings_pct": 0
    })


@app.get("/analytics/cohorts")
def analytics_cohorts():
    from analytics.queries import get_patient_cohort_analysis
    return _safe_analytics(get_patient_cohort_analysis, fallback=[])


@app.get("/analytics/sla")
def analytics_sla():
    from analytics.queries import get_review_queue_sla_metrics
    return _safe_analytics(get_review_queue_sla_metrics, fallback={
        "total_cases": 0, "resolved": 0, "pending": 0,
        "resolution_rate": 0, "avg_review_time_hours": 0,
        "resolution_breakdown": {}, "sla_target_hours": 4, "sla_met_pct": 0
    })


@app.post("/analytics/data-quality")
def analytics_data_quality():
    from analytics.data_quality import run_data_quality_check
    return run_data_quality_check()


def _run_guidelines_refresh_background(
    source_ids: Optional[list[str]],
    force: bool,
    triggered_by: str
):
    print(f"\n[Guidelines Refresh] Starting ({triggered_by}, force={force}, sources={source_ids or 'ALL'})")
    try:
        from rag.guideline_sources import GUIDELINE_SOURCES, SOURCES_BY_ID
        from rag.scraper import scrape_all
        from rag.embedder import embed_all, get_collection_stats
        import json as _json
        from pathlib import Path

        if source_ids:
            sources = [SOURCES_BY_ID[sid] for sid in source_ids if sid in SOURCES_BY_ID]
            print(f"[Guidelines Refresh] Targeted refresh: {len(sources)} sources")
        else:
            sources = GUIDELINE_SOURCES
            print(f"[Guidelines Refresh] Full refresh: {len(sources)} sources")

        scraped = scrape_all(sources, force=force)
        changed = sum(1 for r in scraped if r.get("changed"))
        errors = sum(1 for r in scraped if r.get("error") and not r.get("chunks"))
        print(f"[Guidelines Refresh] Scrape complete: {changed} changed, {errors} errors")

        embed_stats = embed_all(scraped)
        total_chunks = sum(s.get("chunks_embedded", 0) for s in embed_stats)
        print(f"[Guidelines Refresh] Embed complete: {total_chunks} chunks")

        stats = get_collection_stats()
        validation = "PASS" if stats.get("total_chunks", 0) > 0 else "WARN — collection empty"

        log_path = Path("data/guideline_cache/refresh_log.json")
        log_path.parent.mkdir(parents=True, exist_ok=True)
        existing = []
        if log_path.exists():
            try:
                with open(log_path) as f:
                    existing = _json.load(f)
            except Exception:
                existing = []

        entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "mode": f"direct_{'full' if not source_ids else 'targeted'}",
            "triggered_by": triggered_by,
            "sources_embedded": sum(1 for s in embed_stats if s["status"] == "embedded"),
            "sources_unchanged": sum(1 for s in embed_stats if s["status"] == "unchanged"),
            "sources_skipped": sum(1 for s in embed_stats if s["status"] == "skipped"),
            "total_chunks": stats.get("total_chunks", 0),
            "total_sources": stats.get("total_sources", 0),
            "validation": validation,
        }
        existing.insert(0, entry)
        with open(log_path, "w") as f:
            _json.dump(existing[:50], f, indent=2)

        print(f"[Guidelines Refresh] ✅ Done — {entry['sources_embedded']} sources, "
              f"{entry['total_chunks']} total chunks, validation: {validation}")

    except Exception as e:
        print(f"[Guidelines Refresh] ERROR: {e}")
        import traceback
        traceback.print_exc()