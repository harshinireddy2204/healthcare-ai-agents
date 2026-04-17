"""
analytics/queries.py

SQL analytics layer for healthcare AI agent reporting.

Path-safety note:
  Both api/main.py (writer) and this module (reader) use the same
  DATABASE_URL env var with the same default. The get_engine() helper
  uses check_same_thread=False for SQLite so concurrent FastAPI requests
  can all read from the same file.

  On Railway/Render: SQLite is ephemeral (wiped on redeploy). The API
  seeds demo data on startup if audit_log is empty, so analytics pages
  always have content for demo visitors.
"""
import os
import json
from datetime import datetime, timedelta
from typing import Optional

from sqlalchemy import create_engine, text
from dotenv import load_dotenv

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./healthcare_agents.db")


def get_engine():
    """
    Create an engine with SQLite-safe settings matching api/main.py.
    check_same_thread=False is required so FastAPI's threadpool can
    share the connection with background tasks.
    """
    return create_engine(
        DATABASE_URL,
        connect_args={"check_same_thread": False} if "sqlite" in DATABASE_URL else {},
        pool_pre_ping=True,  # detect stale connections on long-running services
    )


# ── Operational metrics ───────────────────────────────────────────────────────

def get_agent_performance_summary(days: int = 30) -> dict:
    """Overall agent performance metrics for the last N days."""
    engine = get_engine()
    since = (datetime.utcnow() - timedelta(days=days)).isoformat()

    with engine.connect() as conn:
        result = conn.execute(text("""
            SELECT
                COUNT(*)                                           AS total_runs,
                SUM(CASE WHEN status = 'COMPLETED' THEN 1 ELSE 0 END)      AS completed,
                SUM(CASE WHEN status = 'PENDING_REVIEW' THEN 1 ELSE 0 END) AS escalated,
                SUM(CASE WHEN status = 'FAILED' THEN 1 ELSE 0 END)         AS failed,
                COUNT(DISTINCT patient_id)                        AS unique_patients
            FROM audit_log
            WHERE processed_at >= :since
        """), {"since": since}).fetchone()

        total = result[0] or 1
        return {
            "period_days": days,
            "total_runs": result[0] or 0,
            "completed": result[1] or 0,
            "escalated_to_human": result[2] or 0,
            "failed": result[3] or 0,
            "unique_patients": result[4] or 0,
            "completion_rate": round((result[1] or 0) / total * 100, 1),
            "escalation_rate": round((result[2] or 0) / total * 100, 1),
            "automation_rate": round(((result[1] or 0)) / total * 100, 1),
        }


def get_prior_auth_metrics(days: int = 30) -> dict:
    """Prior authorization decision metrics."""
    engine = get_engine()
    since = (datetime.utcnow() - timedelta(days=days)).isoformat()

    with engine.connect() as conn:
        rows = conn.execute(text("""
            SELECT result_json
            FROM audit_log
            WHERE status IN ('COMPLETED', 'PENDING_REVIEW')
            AND processed_at >= :since
        """), {"since": since}).fetchall()

    decisions = {"APPROVE": 0, "DENY": 0, "ESCALATE": 0}
    confidences = []
    critic_reviews = 0
    revisions = 0

    for (result_json,) in rows:
        try:
            result = json.loads(result_json) if result_json else {}

            # auth_results can live in two places:
            #  - direct key for auth_only mode
            #  - nested inside workflow_results for full mode
            auth_results = (
                result.get("auth_results") or
                result.get("workflow_results", {}).get("auth_results", [])
            )

            # Also scan crew_output for embedded PRIOR AUTH JSON in full mode
            if not auth_results:
                crew = result.get("crew_output", "")
                if "PRIOR AUTH:" in crew:
                    auth_section = crew.split("PRIOR AUTH:", 1)[1]
                    # Get the JSON array portion before the next section
                    for sep in ["SYNTHESIS:", "KNOWLEDGE GRAPH:", "CARE GAPS:"]:
                        if sep in auth_section:
                            auth_section = auth_section.split(sep, 1)[0]
                    auth_section = auth_section.strip()
                    if auth_section.startswith("["):
                        try:
                            auth_results = json.loads(auth_section)
                        except Exception:
                            pass

            for r in auth_results:
                d = r.get("decision", "")
                if d in decisions:
                    decisions[d] += 1
                conf = r.get("confidence")
                if conf is not None:
                    confidences.append(float(conf))
                if r.get("critic_reviewed"):
                    critic_reviews += 1
                if r.get("was_revised"):
                    revisions += 1
        except Exception:
            pass

    total_auth = sum(decisions.values()) or 1
    return {
        "total_auth_requests": total_auth,
        "decisions": decisions,
        "approval_rate": round(decisions["APPROVE"] / total_auth * 100, 1),
        "escalation_rate": round(decisions["ESCALATE"] / total_auth * 100, 1),
        "avg_confidence": round(sum(confidences) / len(confidences), 2) if confidences else 0,
        "critic_reviewed": critic_reviews,
        "agent_revised": revisions,
        "revision_rate": round(revisions / max(critic_reviews, 1) * 100, 1)
    }


def get_care_gap_metrics(days: int = 30) -> dict:
    """Care gap identification metrics."""
    engine = get_engine()
    since = (datetime.utcnow() - timedelta(days=days)).isoformat()

    with engine.connect() as conn:
        rows = conn.execute(text("""
            SELECT patient_id, result_json
            FROM audit_log
            WHERE status IN ('COMPLETED', 'PENDING_REVIEW')
            AND processed_at >= :since
        """), {"since": since}).fetchall()

    gap_counts = {}
    patients_with_gaps = set()
    total_gaps = 0
    patients_analyzed = set()

    for (patient_id, result_json) in rows:
        patients_analyzed.add(patient_id)
        try:
            result = json.loads(result_json) if result_json else {}

            # Collect all report text — scan both final_report and crew_output
            report_text = result.get("final_report", "") + " " + result.get("crew_output", "")
            # Also check nested workflow_results
            wf = result.get("workflow_results", {})
            report_text += " " + wf.get("care_summary", "")

            if not report_text.strip():
                continue

            gap_keywords = {
                "mammogram": "Breast Cancer Screening",
                "colonoscopy": "Colorectal Screening",
                "flu vaccine": "Influenza Vaccination",
                "hba1c": "HbA1c Monitoring",
                "nephrology": "Nephrology Referral",
                "blood pressure": "Blood Pressure Monitoring",
                "cholesterol": "Lipid Screening",
                "depression": "Depression Screening",
                "bone density": "Bone Density Screening",
                "dexa": "Bone Density Screening",
                "cardiology": "Cardiology Referral",
                "fall risk": "Falls Prevention",
            }

            report_lower = report_text.lower()
            found_gaps = set()
            for keyword, label in gap_keywords.items():
                if keyword in report_lower:
                    found_gaps.add(label)

            for label in found_gaps:
                gap_counts[label] = gap_counts.get(label, 0) + 1
                total_gaps += 1

            if found_gaps:
                patients_with_gaps.add(patient_id)
        except Exception:
            pass

    patients_count = len(patients_analyzed) or 1
    return {
        "patients_analyzed": len(patients_analyzed),
        "patients_with_gaps": len(patients_with_gaps),
        "total_gaps_identified": total_gaps,
        "avg_gaps_per_patient": round(total_gaps / patients_count, 1),
        "gap_frequency": sorted(
            [{"gap": k, "count": v, "pct": round(v / patients_count * 100, 1)}
             for k, v in gap_counts.items()],
            key=lambda x: x["count"],
            reverse=True
        )
    }


def get_complexity_distribution(days: int = 30) -> dict:
    """MDAgents complexity routing distribution."""
    engine = get_engine()
    since = (datetime.utcnow() - timedelta(days=days)).isoformat()

    with engine.connect() as conn:
        rows = conn.execute(text("""
            SELECT result_json
            FROM audit_log
            WHERE processed_at >= :since
        """), {"since": since}).fetchall()

    tiers = {"LOW": 0, "MODERATE": 0, "HIGH": 0}
    token_cost = {"LOW": 2000, "MODERATE": 8000, "HIGH": 20000}

    for (result_json,) in rows:
        try:
            result = json.loads(result_json) if result_json else {}
            tier = result.get("complexity_tier", "")
            if tier in tiers:
                tiers[tier] += 1
        except Exception:
            pass

    total = sum(tiers.values()) or 1
    tokens_used = sum(tiers[t] * token_cost[t] for t in tiers)
    tokens_if_all_high = total * token_cost["HIGH"]
    tokens_saved = tokens_if_all_high - tokens_used

    return {
        "distribution": tiers,
        "pct_low": round(tiers["LOW"] / total * 100, 1),
        "pct_moderate": round(tiers["MODERATE"] / total * 100, 1),
        "pct_high": round(tiers["HIGH"] / total * 100, 1),
        "estimated_tokens_used": tokens_used,
        "estimated_tokens_saved": tokens_saved,
        "estimated_cost_savings_pct": round(tokens_saved / max(tokens_if_all_high, 1) * 100, 1)
    }


def get_patient_cohort_analysis() -> list:
    """Patient cohort analysis grouped by condition clusters."""
    try:
        with open("data/synthetic_patients.json") as f:
            patients = json.load(f)
    except Exception:
        return []

    cohorts = {
        "Diabetic + CKD": {"conditions": ["Type 2 Diabetes", "CKD"], "patients": []},
        "Cardiovascular Complex": {"conditions": ["Heart Failure", "Atrial Fibrillation", "Coronary Artery Disease"], "patients": []},
        "Oncology": {"conditions": ["Cancer", "cancer"], "patients": []},
        "Metabolic Syndrome": {"conditions": ["Obesity", "Hypertension", "Hyperlipidemia"], "patients": []},
        "Autoimmune": {"conditions": ["Rheumatoid Arthritis", "Lupus", "Crohn"], "patients": []},
        "Other": {"conditions": [], "patients": []},
    }

    for p in patients:
        dx = " ".join(p.get("diagnoses", []))
        assigned = False
        for cohort_name, cohort_data in cohorts.items():
            if cohort_name == "Other":
                continue
            if any(c.lower() in dx.lower() for c in cohort_data["conditions"]):
                cohort_data["patients"].append(p["patient_id"])
                assigned = True
                break
        if not assigned:
            cohorts["Other"]["patients"].append(p["patient_id"])

    result = []
    for cohort_name, cohort_data in cohorts.items():
        count = len(cohort_data["patients"])
        if count > 0:
            result.append({
                "cohort": cohort_name,
                "patient_count": count,
                "patient_ids": cohort_data["patients"],
                "pct_of_population": round(count / len(patients) * 100, 1),
            })

    return sorted(result, key=lambda x: x["patient_count"], reverse=True)


def get_review_queue_sla_metrics() -> dict:
    """SLA compliance for the HITL review queue."""
    engine = get_engine()

    with engine.connect() as conn:
        rows = conn.execute(text("""
            SELECT id, patient_id, created_at, resolved_at, status, resolved_by, resolution
            FROM review_queue
            ORDER BY created_at DESC
            LIMIT 100
        """)).fetchall()

    total = len(rows)
    resolved = sum(1 for r in rows if r[4] == "RESOLVED")
    pending = sum(1 for r in rows if r[4] == "PENDING")

    resolution_counts = {}
    review_times = []

    for row in rows:
        if row[4] == "RESOLVED" and row[2] and row[3]:
            try:
                created = datetime.fromisoformat(row[2])
                resolved_dt = datetime.fromisoformat(row[3])
                hours = (resolved_dt - created).total_seconds() / 3600
                review_times.append(hours)
            except Exception:
                pass
        res = row[6]
        if res:
            resolution_counts[res] = resolution_counts.get(res, 0) + 1

    return {
        "total_cases": total,
        "resolved": resolved,
        "pending": pending,
        "resolution_rate": round(resolved / max(total, 1) * 100, 1),
        "avg_review_time_hours": round(sum(review_times) / max(len(review_times), 1), 1),
        "resolution_breakdown": resolution_counts,
        "sla_target_hours": 4,
        "sla_met_pct": round(
            sum(1 for t in review_times if t <= 4) / max(len(review_times), 1) * 100, 1
        )
    }


if __name__ == "__main__":
    print("=== Agent Performance ===")
    print(json.dumps(get_agent_performance_summary(), indent=2))
    print("\n=== Prior Auth Metrics ===")
    print(json.dumps(get_prior_auth_metrics(), indent=2))
    print("\n=== Complexity Distribution ===")
    print(json.dumps(get_complexity_distribution(), indent=2))
    print("\n=== Care Gap Metrics ===")
    print(json.dumps(get_care_gap_metrics(), indent=2))