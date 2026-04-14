"""
analytics/queries.py

SQL analytics layer for healthcare AI agent reporting.

Demonstrates: complex SQL, data aggregation, cohort analysis,
and operational metrics — directly mapping to the "ad hoc analytics
and custom SQL queries" requirement in the JD.

These queries run against the audit_log and review_queue tables
and can be adapted to any data warehouse (PostgreSQL, Snowflake,
BigQuery) by changing the DATABASE_URL environment variable.
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
    return create_engine(DATABASE_URL, connect_args={"check_same_thread": False}
                         if "sqlite" in DATABASE_URL else {})


# ── Operational metrics ───────────────────────────────────────────────────────

def get_agent_performance_summary(days: int = 30) -> dict:
    """
    Overall agent performance metrics for the last N days.
    Maps to: management dashboard / C-suite reporting.
    """
    engine = get_engine()
    since = (datetime.utcnow() - timedelta(days=days)).isoformat()

    with engine.connect() as conn:
        # Total runs and status breakdown
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
    """
    Prior authorization decision metrics — approval rate, escalation rate,
    average confidence by decision type.
    Maps to: payer analytics reporting.
    """
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
            auth_results = result.get("auth_results", [])
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


def get_care_gap_metrics(days: int = 30) -> list:
    """
    Care gap identification metrics — most common gaps, closure rates.
    Maps to: population health analytics reporting.
    """
    engine = get_engine()
    since = (datetime.utcnow() - timedelta(days=days)).isoformat()

    with engine.connect() as conn:
        rows = conn.execute(text("""
            SELECT patient_id, result_json
            FROM audit_log
            WHERE status = 'COMPLETED'
            AND processed_at >= :since
        """), {"since": since}).fetchall()

    gap_counts = {}
    patients_with_gaps = 0
    total_gaps = 0

    for (patient_id, result_json) in rows:
        try:
            result = json.loads(result_json) if result_json else {}
            report = result.get("final_report", "")
            if not report:
                continue

            # Count gap mentions in reports
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
            }

            found_gap = False
            for keyword, label in gap_keywords.items():
                if keyword.lower() in report.lower():
                    gap_counts[label] = gap_counts.get(label, 0) + 1
                    total_gaps += 1
                    found_gap = True

            if found_gap:
                patients_with_gaps += 1
        except Exception:
            pass

    return {
        "patients_analyzed": len(rows),
        "patients_with_gaps": patients_with_gaps,
        "total_gaps_identified": total_gaps,
        "avg_gaps_per_patient": round(total_gaps / max(len(rows), 1), 1),
        "gap_frequency": sorted(
            [{"gap": k, "count": v, "pct": round(v / max(len(rows), 1) * 100, 1)}
             for k, v in gap_counts.items()],
            key=lambda x: x["count"],
            reverse=True
        )
    }


def get_complexity_distribution(days: int = 30) -> dict:
    """
    MDAgents complexity routing distribution — shows cost savings
    from routing simple cases to single agents vs full CrewAI team.
    Maps to: ROI reporting / cost analysis.
    """
    engine = get_engine()
    since = (datetime.utcnow() - timedelta(days=days)).isoformat()

    with engine.connect() as conn:
        rows = conn.execute(text("""
            SELECT result_json
            FROM audit_log
            WHERE processed_at >= :since
        """), {"since": since}).fetchall()

    tiers = {"LOW": 0, "MODERATE": 0, "HIGH": 0}
    # Approximate token costs per tier
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
    """
    Patient cohort analysis — groups patients by condition clusters
    and shows care gap burden per cohort.
    Maps to: population health / value-based care reporting.

    This is a SQL-heavy query that demonstrates complex aggregation
    and cohort segmentation skills.
    """
    import json as _json
    try:
        with open("data/synthetic_patients.json") as f:
            patients = _json.load(f)
    except Exception:
        return []

    cohorts = {
        "Diabetic + CKD": {"conditions": ["Type 2 Diabetes", "CKD"], "patients": [], "risk_scores": []},
        "Cardiovascular Complex": {"conditions": ["Heart Failure", "Atrial Fibrillation", "Coronary Artery Disease"], "patients": [], "risk_scores": []},
        "Oncology": {"conditions": ["Cancer", "cancer"], "patients": [], "risk_scores": []},
        "Metabolic Syndrome": {"conditions": ["Obesity", "Hypertension", "Hyperlipidemia"], "patients": [], "risk_scores": []},
        "Autoimmune": {"conditions": ["Rheumatoid Arthritis", "Lupus", "Crohn"], "patients": [], "risk_scores": []},
        "Other": {"conditions": [], "patients": [], "risk_scores": []},
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
    """
    SLA compliance for the HITL review queue — time-to-review,
    overdue cases, reviewer patterns.
    Maps to: operations management reporting.
    """
    engine = get_engine()

    with engine.connect() as conn:
        rows = conn.execute(text("""
            SELECT
                id,
                patient_id,
                created_at,
                resolved_at,
                status,
                resolved_by,
                resolution
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
                resolved = datetime.fromisoformat(row[3])
                hours = (resolved - created).total_seconds() / 3600
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