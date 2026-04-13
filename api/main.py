"""
api/main.py

FastAPI service — REST interface for the healthcare AI agent system.
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


@app.on_event("startup")
def on_startup():
    init_db()


# ── Pydantic models ───────────────────────────────────────────────────────────

class ProcessPatientRequest(BaseModel):
    patient_id: str
    mode: Literal["full", "auth_only", "care_gap_only"] = "full"


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
    """Write a result to the audit log table."""
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


# ── Background task ───────────────────────────────────────────────────────────

def run_triage_background(patient_id: str, mode: str):
    """
    Run agent workflow in background. Logs both success and failure
    to the audit_log table so the result is always visible.
    """
    print(f"\n[Background] Starting {mode} workflow for {patient_id}")

    try:
        if mode == "auth_only":
            from agents.prior_auth_agent import run_prior_auth
            from tools.ehr_tools import get_pending_auth_requests

            requests = get_pending_auth_requests.invoke({"patient_id": patient_id})
            results = []
            for req in (requests or []):
                if isinstance(req, dict) and "request_id" in req:
                    r = run_prior_auth(patient_id, req["request_id"], req["item"])
                    results.append(r)
                    print(f"[Background] Auth result: {r}")

            write_audit(patient_id, "COMPLETED", {
                "mode": mode,
                "auth_results": results
            })

        elif mode == "care_gap_only":
            from agents.care_gap_agent import run_care_gap_review

            print(f"[Background] Running care gap review for {patient_id}...")
            result = run_care_gap_review(patient_id)
            print(f"[Background] Care gap complete: {result['steps_executed']} steps")

            write_audit(patient_id, "COMPLETED", {
                "mode": mode,
                "steps_executed": result["steps_executed"],
                "final_report": result["final_report"]
            })

        else:  # full
            from agents.triage_supervisor import run_triage

            result = run_triage(patient_id)
            status = result.get("status", "COMPLETED")
            write_audit(patient_id, status, result)

        print(f"[Background] {patient_id} workflow finished successfully")

    except Exception as e:
        print(f"[Background] ERROR for {patient_id}: {e}")
        import traceback
        traceback.print_exc()
        write_audit(patient_id, "FAILED", {"error": str(e), "mode": mode})


# ── Endpoints ──────────────────────────────────────────────────────────────────

@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "version": "1.0.0"
    }


@app.post("/process-patient", response_model=PatientSummary)
def process_patient(
    request: ProcessPatientRequest,
    background_tasks: BackgroundTasks
):
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
        message="Workflow triggered. Check /audit-log for results.",
        workflow_triggered=mode_labels[request.mode]
    )


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
    source_ids: Optional[list[str]] = None  # None = refresh ALL sources
    force: bool = False                      # True = re-embed even unchanged
    triggered_by: str = "api"


@app.post("/refresh-guidelines")
def refresh_guidelines(
    request: RefreshGuidelinesRequest,
    background_tasks: BackgroundTasks
):
    """
    Manually trigger a guidelines refresh.
    Use this when a major guideline update is published (e.g. new ADA Standards).
    source_ids=null refreshes ALL sources.
    force=true re-embeds everything even if content hasn't changed.
    """
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
    """
    Return current state of the guidelines vector store
    and the last 5 refresh runs.
    """
    try:
        from rag.embedder import get_collection_stats
        from rag.refresh_flow import get_refresh_log

        stats = get_collection_stats()
        log = get_refresh_log()[:5]  # last 5 runs

        return {
            "collection": stats,
            "recent_refreshes": log,
            "status": "healthy" if stats["total_chunks"] > 0 else "empty — run /refresh-guidelines first"
        }
    except Exception as e:
        return {
            "collection": {"total_chunks": 0, "total_sources": 0},
            "recent_refreshes": [],
            "status": f"error: {e}"
        }


@app.get("/guidelines-search")
def search_guidelines(q: str, category: Optional[str] = None, n: int = 3):
    """
    Test the guidelines search directly.
    Useful for verifying the RAG layer is working before running agents.
    """
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


def _run_guidelines_refresh_background(
    source_ids: Optional[list[str]],
    force: bool,
    triggered_by: str
):
    """Background task that runs the Prefect manual refresh flow."""
    try:
        from rag.refresh_flow import manual_refresh_flow
        manual_refresh_flow(
            source_ids=source_ids,
            force=force,
            triggered_by=triggered_by
        )
    except Exception as e:
        print(f"[Guidelines Refresh] ERROR: {e}")
        import traceback
        traceback.print_exc()