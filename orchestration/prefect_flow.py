"""
orchestration/prefect_flow.py

Prefect 3.x workflow for scheduled batch patient triage.

Features demonstrated:
  - @flow and @task decorators
  - retry logic with exponential backoff
  - task dependencies
  - failure notifications
  - scheduled daily runs

Run locally:
    python orchestration/prefect_flow.py

Deploy to Prefect Cloud:
    prefect deploy orchestration/prefect_flow.py:daily_triage_flow
"""
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
from prefect import flow, task, get_run_logger
from prefect.tasks import task_input_hash
from prefect.notifications import SlackWebhook  # optional: if you add Slack
from datetime import timedelta

load_dotenv()

DATA_PATH = Path(__file__).parent.parent / "data" / "synthetic_patients.json"


# ── Tasks ──────────────────────────────────────────────────────────────────────

@task(
    name="load-patient-batch",
    description="Load the daily patient batch from the data store.",
    retries=3,
    retry_delay_seconds=10,
    cache_key_fn=task_input_hash,
    cache_expiration=timedelta(hours=1),
)
def load_patient_batch(batch_date: Optional[str] = None) -> list[str]:
    """Load patient IDs scheduled for triage today."""
    logger = get_run_logger()

    with open(DATA_PATH) as f:
        patients = json.load(f)

    patient_ids = [p["patient_id"] for p in patients]
    logger.info(f"Loaded {len(patient_ids)} patients for triage on {batch_date or 'today'}")
    return patient_ids


@task(
    name="run-prior-auth",
    description="Run prior authorization agent for pending requests.",
    retries=2,
    retry_delay_seconds=30,
)
def run_prior_auth_task(patient_id: str) -> dict:
    """Run the prior auth ReAct agent for one patient."""
    logger = get_run_logger()
    logger.info(f"[Prior Auth] Processing patient {patient_id}")

    # Import here to avoid circular imports at module level
    from agents.prior_auth_agent import run_prior_auth
    from tools.ehr_tools import get_pending_auth_requests

    # Get pending requests
    requests = get_pending_auth_requests.invoke({"patient_id": patient_id})
    if not requests:
        return {"patient_id": patient_id, "status": "no_pending_requests"}

    results = []
    for req in requests:
        if isinstance(req, dict) and "request_id" in req:
            result = run_prior_auth(
                patient_id=patient_id,
                request_id=req["request_id"],
                item_name=req["item"]
            )
            results.append(result)
            logger.info(f"  Request {req['request_id']}: {result['decision']} "
                        f"(confidence: {result['confidence']:.2f})")

    return {"patient_id": patient_id, "auth_results": results}


@task(
    name="run-care-gap-review",
    description="Run care gap review agent for one patient.",
    retries=2,
    retry_delay_seconds=30,
)
def run_care_gap_task(patient_id: str) -> dict:
    """Run the care gap Plan-and-Execute agent for one patient."""
    logger = get_run_logger()
    logger.info(f"[Care Gap] Processing patient {patient_id}")

    from agents.care_gap_agent import run_care_gap_review
    result = run_care_gap_review(patient_id)

    logger.info(f"  Care gap review complete: {result['steps_executed']} steps")
    return {"patient_id": patient_id, "care_gap_result": result}


@task(
    name="aggregate-results",
    description="Aggregate all agent outputs and write audit log.",
    retries=1,
)
def aggregate_results(patient_id: str, auth_result: dict, care_gap_result: dict) -> dict:
    """Combine auth and care gap results into a single patient summary."""
    logger = get_run_logger()

    summary = {
        "patient_id": patient_id,
        "processed_at": datetime.utcnow().isoformat(),
        "prior_auth": auth_result,
        "care_gaps": care_gap_result,
        "requires_human_review": False,
        "escalation_reasons": []
    }

    # Check if escalation is needed
    auth_results_list = auth_result.get("auth_results", [])
    for r in auth_results_list:
        if r.get("decision") == "ESCALATE":
            summary["requires_human_review"] = True
            summary["escalation_reasons"].append(
                f"Auth request {r.get('request_id')} requires escalation: {r.get('justification')}"
            )
        elif r.get("confidence", 1.0) < float(os.getenv("CONFIDENCE_THRESHOLD", "0.75")):
            summary["requires_human_review"] = True
            summary["escalation_reasons"].append(
                f"Low confidence ({r.get('confidence', 0):.2f}) on auth decision for {r.get('item')}"
            )

    if summary["requires_human_review"]:
        logger.warning(f"  Patient {patient_id} flagged for human review: "
                       f"{summary['escalation_reasons']}")
    else:
        logger.info(f"  Patient {patient_id} fully processed — no escalation needed")

    return summary


@task(
    name="write-audit-log",
    description="Persist all results to the audit log.",
    retries=3,
    retry_delay_seconds=5,
)
def write_audit_log(results: list[dict]) -> str:
    """Write the batch run results to the audit log table."""
    logger = get_run_logger()

    try:
        from sqlalchemy import create_engine, text
        db_url = os.getenv("DATABASE_URL", "sqlite:///./healthcare_agents.db")
        engine = create_engine(db_url)

        with engine.connect() as conn:
            for r in results:
                conn.execute(text("""
                    INSERT OR REPLACE INTO audit_log
                    (patient_id, processed_at, status, result_json)
                    VALUES (:patient_id, :processed_at, :status, :result_json)
                """), {
                    "patient_id": r["patient_id"],
                    "processed_at": r["processed_at"],
                    "status": "PENDING_REVIEW" if r["requires_human_review"] else "COMPLETED",
                    "result_json": json.dumps(r)
                })
            conn.commit()

        logger.info(f"Audit log written for {len(results)} patients")
        return f"Audit log updated: {len(results)} records"

    except Exception as e:
        logger.error(f"Failed to write audit log: {e}")
        raise


@task(
    name="generate-batch-report",
    description="Generate a summary report for this batch run.",
)
def generate_batch_report(results: list[dict]) -> dict:
    """Summarise the batch run stats."""
    total = len(results)
    escalated = sum(1 for r in results if r.get("requires_human_review"))
    completed = total - escalated

    report = {
        "run_date": datetime.utcnow().isoformat(),
        "total_patients": total,
        "completed": completed,
        "escalated_to_human": escalated,
        "escalation_rate": f"{(escalated/total*100):.1f}%" if total > 0 else "0%",
        "escalation_details": [
            {
                "patient_id": r["patient_id"],
                "reasons": r.get("escalation_reasons", [])
            }
            for r in results if r.get("requires_human_review")
        ]
    }

    return report


# ── Main flow ──────────────────────────────────────────────────────────────────

@flow(
    name="daily-healthcare-triage",
    description=(
        "Daily batch workflow: runs prior auth and care gap agents for all "
        "scheduled patients, escalates low-confidence cases to HITL queue."
    ),
    log_prints=True,
)
def daily_triage_flow(batch_date: Optional[str] = None) -> dict:
    """
    Main Prefect flow. Runs sequentially per patient to avoid rate limits.
    In production, switch to concurrent execution for scale.
    """
    logger = get_run_logger()
    run_date = batch_date or datetime.utcnow().strftime("%Y-%m-%d")

    logger.info(f"=== Daily Healthcare Triage — {run_date} ===")

    # Step 1: Load the patient batch
    patient_ids = load_patient_batch(run_date)

    # Step 2: Process each patient
    all_results = []
    for patient_id in patient_ids:
        logger.info(f"\nProcessing patient {patient_id}...")

        # Run both agents (in prod: run concurrently with submit())
        auth_result = run_prior_auth_task(patient_id)
        care_gap_result = run_care_gap_task(patient_id)

        # Aggregate
        summary = aggregate_results(patient_id, auth_result, care_gap_result)
        all_results.append(summary)

    # Step 3: Persist results + generate report
    write_audit_log(all_results)
    report = generate_batch_report(all_results)

    logger.info("\n=== Batch Complete ===")
    logger.info(f"Total: {report['total_patients']} | "
                f"Completed: {report['completed']} | "
                f"Escalated: {report['escalated_to_human']} "
                f"({report['escalation_rate']})")

    return report


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Run once immediately (for testing)
    result = daily_triage_flow()
    print("\n=== Flow Result ===")
    print(json.dumps(result, indent=2))
