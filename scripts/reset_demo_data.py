"""
scripts/reset_demo_data.py

Seed the SQLite audit_log and review_queue tables with realistic demo data.

Why this exists:
  - Locally with a persisted DB, the dashboard shows historical runs.
  - On Railway/Render, SQLite is wiped on every redeploy. Without seed data,
    first-time visitors see empty analytics, empty audit log, and no pending
    reviews — making the dashboard look broken.

Invoked by:
  - Dockerfile (RUN python scripts/reset_demo_data.py) during image build
  - api/main.py on_startup if audit_log is empty (Railway re-seeds each boot)
  - Manually:  python scripts/reset_demo_data.py

Idempotent: clears existing seed rows before inserting.

The result dicts match the exact shape that live agents produce, so
render_crew_output() and render_care_gap_report() parse them identically.
"""
import json
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from sqlalchemy import create_engine, text
from dotenv import load_dotenv

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./healthcare_agents.db")


def _make_engine():
    return create_engine(
        DATABASE_URL,
        connect_args={"check_same_thread": False} if "sqlite" in DATABASE_URL else {}
    )


# ── Demo runs covering LOW / MODERATE / HIGH complexity + all three modes ────

def _build_demo_runs() -> list[dict]:
    """Return 10 demo audit_log entries spanning different pathways."""
    return [
        # ── MODERATE: P001 Eleanor Vance, T2DM + CKD Stage 3 ──────────────────
        {
            "patient_id": "P001",
            "hours_ago": 1,
            "status": "COMPLETED",
            "result": {
                "mode": "full",
                "patient_id": "P001",
                "complexity_tier": "MODERATE",
                "complexity_score": 6,
                "complexity_rationale": "Multi-factor case — MDT collaboration",
                "workflow_completed": True,
                "escalation_triggered": False,
                "status": "COMPLETED",
                "crew_output": (
                    "MODERATE pathway completed for patient P001.\n\n"
                    "KNOWLEDGE GRAPH:\n"
                    "**Clinical Knowledge Graph Analysis:**\n\n"
                    "🔴 **High Priority Clinical Connections:**\n"
                    "  - [HIGH] HbA1c > 9.0% → Insulin pump therapy (direct)\n"
                    "    Evidence: ADA 2025: CSII indicated for HbA1c > 9% despite optimal MDI therapy\n"
                    "  - [HIGH] CKD Stage 3 → Nephrology referral (direct)\n"
                    "    Evidence: KDIGO 2024: Nephrology co-management recommended for eGFR < 60\n\n"
                    "CARE GAPS:\n"
                    "Care Gap Report for Patient P001: Eleanor Vance\n\n"
                    "1. Care Gaps Found:\n\n"
                    "* Mammogram — HIGH priority [USPSTF: Breast Cancer Screening] — Biennial mammography for women 40-74. Action: Schedule mammogram within 30 days.\n\n"
                    "* Colonoscopy — HIGH priority [USPSTF: Colorectal Cancer Screening] — Adults 45-75 should be screened. Action: Schedule colonoscopy or FIT test within 60 days.\n\n"
                    "* Nephrology Referral — HIGH priority [KDIGO 2024: CKD Management] — eGFR 45 with CKD Stage 3. Action: Initiate nephrology referral within 14 days.\n\n"
                    "* Annual Flu Vaccine — LOW priority [CDC: Adult Immunization 2025] — Annual influenza vaccination. Action: Administer at next visit.\n\n"
                    "2. Clinical Review Items:\n\n"
                    "* Review HbA1c 10.2% — insulin pump consideration per ADA 2025.\n\n"
                    "Summary: Immediate priorities are mammogram, colonoscopy, and nephrology referral. All HIGH gaps should be addressed within 30 days.\n\n"
                    "PRIOR AUTH:\n"
                    "[{\"patient_id\": \"P001\", \"request_id\": \"REQ001\", \"item\": \"insulin_pump\", "
                    "\"decision\": \"APPROVE\", \"confidence\": 0.92, "
                    "\"justification\": \"Patient meets BlueCross PPO coverage criteria: HbA1c 10.2% > 9.0% threshold, "
                    "documented failure of MDI therapy.\", "
                    "\"critic_reviewed\": true, \"was_revised\": false}]"
                )
            }
        },

        # ── HIGH: P004 James Whitfield (escalated) ────────────────────────────
        {
            "patient_id": "P004",
            "hours_ago": 2,
            "status": "PENDING_REVIEW",
            "result": {
                "mode": "full",
                "patient_id": "P004",
                "complexity_tier": "HIGH",
                "complexity_score": 12,
                "complexity_rationale": "Complex multi-system case — ICT with full specialist team",
                "workflow_completed": True,
                "escalation_triggered": True,
                "status": "PENDING_REVIEW",
                "workflow_results": {
                    "pathway": "HIGH",
                    "drug_safety": {
                        "safety_tier": "CRITICAL",
                        "fda_findings_count": 4,
                        "kg_findings_count": 6
                    }
                },
                "crew_output": (
                    "HIGH pathway completed for patient P004.\n\n"
                    "KNOWLEDGE GRAPH:\n"
                    "**Clinical Knowledge Graph Analysis:**\n\n"
                    "⚠️ **Drug Interactions Detected:**\n"
                    "  - Warfarin → INTERACTS WITH → NSAIDs\n"
                    "    Evidence: FDA: Warfarin + NSAIDs — significantly increased bleeding risk\n\n"
                    "🔴 **High Priority Clinical Connections:**\n"
                    "  - [URGENT] CKD Stage 4 → Metformin (direct)\n"
                    "    Evidence: FDA: Metformin contraindicated in eGFR < 30 (lactic acidosis risk)\n"
                    "  - [URGENT] CKD Stage 4 → Nephrology referral (direct)\n"
                    "    Evidence: KDIGO 2024: Urgent nephrology referral — prepare for RRT\n\n"
                    "CARE GAPS:\n"
                    "Care Gap Report for Patient P004: James Whitfield\n\n"
                    "1. Care Gaps Found:\n\n"
                    "* Nephrology Referral — HIGH priority [KDIGO 2024: CKD Management] — eGFR 22, CKD Stage 4. Action: Urgent nephrology referral within 7 days.\n\n"
                    "* Metformin Discontinuation — HIGH priority [FDA: Metformin Safety] — Contraindicated at eGFR < 30. Action: Discontinue immediately.\n\n"
                    "* Warfarin/NSAID Review — HIGH priority [FDA: Warfarin Interactions] — Major bleeding risk. Action: Review all medications.\n\n"
                    "* Cardiac Rehabilitation — MEDIUM priority [ACC/AHA 2022] — Indicated for HF. Action: Schedule evaluation.\n\n"
                    "Summary: Case ESCALATED to clinician review due to critical drug safety and multi-system complexity.\n\n"
                    "PRIOR AUTH:\n"
                    "[{\"patient_id\": \"P004\", \"request_id\": \"REQ004\", \"item\": \"home_dialysis_evaluation\", "
                    "\"decision\": \"ESCALATE\", \"confidence\": 0.55, "
                    "\"justification\": \"Complex case requires multidisciplinary review — CKD Stage 4 progression with anticoagulation considerations.\", "
                    "\"critic_reviewed\": true, \"was_revised\": false}]\n\n"
                    "SYNTHESIS:\n"
                    "URGENT findings:\n"
                    "1. CKD Stage 4 progression — urgent nephrology referral within 7 days for RRT preparation\n"
                    "2. Metformin contraindicated — discontinue immediately, switch to DPP-4 inhibitor\n"
                    "3. Warfarin + NSAID interaction — major bleeding risk, medication review required\n\n"
                    "Medication safety alerts:\n"
                    "- Furosemide: monitor electrolytes weekly\n"
                    "- Warfarin: weekly INR given interactions\n\n"
                    "Prior auth: Home dialysis evaluation ESCALATED — multidisciplinary review needed\n\n"
                    "Recommended referrals: Nephrology (urgent), Cardiology, Clinical Pharmacy"
                )
            }
        },

        # ── LOW: P019 Natalie Russo — routine case ────────────────────────────
        {
            "patient_id": "P019",
            "hours_ago": 4,
            "status": "COMPLETED",
            "result": {
                "mode": "full",
                "patient_id": "P019",
                "complexity_tier": "LOW",
                "complexity_score": 2,
                "complexity_rationale": "Routine case — single agent pathway",
                "workflow_completed": True,
                "escalation_triggered": False,
                "status": "COMPLETED",
                "crew_output": (
                    "LOW pathway completed for patient P019.\n\n"
                    "CARE GAPS:\n"
                    "Care Gap Report for Patient P019: Natalie Russo\n\n"
                    "1. Care Gaps Found:\n\n"
                    "* Annual Flu Vaccine — LOW priority [CDC: Adult Immunization 2025] — Annual influenza vaccine due. Action: Administer at next visit.\n\n"
                    "* Depression Screening — MEDIUM priority [USPSTF: Depression Screening] — Annual PHQ-9 screening for patients with anxiety disorders. Action: Complete PHQ-9 at next appointment.\n\n"
                    "2. Clinical Review Items:\n\n"
                    "* Migraine management appears well-controlled on current regimen.\n\n"
                    "Summary: Standard preventive care updates needed. No urgent gaps.\n\n"
                    "PRIOR AUTH:\nNo pending authorization requests for this patient."
                )
            }
        },

        # ── P007 Linda Okafor — breast cancer (care gap only) ─────────────────
        {
            "patient_id": "P007",
            "hours_ago": 5,
            "status": "COMPLETED",
            "result": {
                "mode": "care_gap_only",
                "steps_executed": 9,
                "final_report": (
                    "Care Gap Report for Patient P007: Linda Okafor\n\n"
                    "1. Care Gaps Found:\n\n"
                    "* Bone Density Screening — HIGH priority [NCCN: Breast Cancer Survivorship] — "
                    "Tamoxifen and aromatase inhibitor use require baseline DEXA and annual monitoring. "
                    "Action: Schedule DEXA scan within 30 days.\n\n"
                    "* Contralateral Mammogram — HIGH priority [NCI: Breast Cancer Follow-up] — "
                    "Annual mammography of remaining breast tissue post-treatment. "
                    "Action: Schedule within 60 days.\n\n"
                    "* Cardiovascular Risk Assessment — MEDIUM priority [ACC/AHA] — "
                    "Chemotherapy survivors have elevated CVD risk. Action: Annual lipid panel and BP monitoring.\n\n"
                    "* Annual Flu Vaccine — LOW priority [CDC] — Cancer survivors should receive annual flu vaccine. "
                    "Action: Administer at next visit.\n\n"
                    "2. Clinical Review Items:\n\n"
                    "* Stage II breast cancer on Tamoxifen — monitor for endometrial symptoms.\n"
                    "* Coordinate oncology survivorship plan with primary care.\n\n"
                    "Summary: Immediate priorities are bone density and contralateral mammogram given Tamoxifen use. "
                    "All HIGH gaps should be addressed within 30-60 days."
                )
            }
        },

        # ── P014 Harold Washington — stroke + HTN + T2DM ──────────────────────
        {
            "patient_id": "P014",
            "hours_ago": 7,
            "status": "PENDING_REVIEW",
            "result": {
                "mode": "full",
                "patient_id": "P014",
                "complexity_tier": "HIGH",
                "complexity_score": 10,
                "complexity_rationale": "Complex multi-system case — ICT with full specialist team",
                "workflow_completed": True,
                "escalation_triggered": True,
                "status": "PENDING_REVIEW",
                "workflow_results": {
                    "pathway": "HIGH",
                    "drug_safety": {
                        "safety_tier": "WARNING",
                        "fda_findings_count": 3,
                        "kg_findings_count": 4
                    }
                },
                "crew_output": (
                    "HIGH pathway completed for patient P014.\n\n"
                    "KNOWLEDGE GRAPH:\n"
                    "**Clinical Knowledge Graph Analysis:**\n\n"
                    "🔴 **High Priority Clinical Connections:**\n"
                    "  - [HIGH] Stroke → Cardiology referral (direct)\n"
                    "    Evidence: AHA/ASA: Cardiology evaluation for secondary stroke prevention\n"
                    "  - [HIGH] Stroke → Cardiac monitoring (direct)\n"
                    "    Evidence: AHA/ASA: 30-day cardiac monitoring to detect AFib\n\n"
                    "CARE GAPS:\n"
                    "Care Gap Report for Patient P014: Harold Washington\n\n"
                    "1. Care Gaps Found:\n\n"
                    "* Cardiology Referral — HIGH priority [AHA/ASA 2021: Stroke Secondary Prevention] — Post-stroke cardiology evaluation required. Action: Refer within 14 days.\n\n"
                    "* Extended Cardiac Monitoring — HIGH priority [AHA/ASA] — 30-day monitor to detect occult AFib. Action: Order Holter monitor.\n\n"
                    "* Colonoscopy — MEDIUM priority [USPSTF] — Age 65, overdue. Action: Schedule within 90 days.\n\n"
                    "Summary: Post-stroke care coordination required. Case escalated for specialist review.\n\n"
                    "PRIOR AUTH:\n"
                    "[{\"patient_id\": \"P014\", \"request_id\": \"REQ014\", \"item\": \"extended_cardiac_monitor\", "
                    "\"decision\": \"ESCALATE\", \"confidence\": 0.60, "
                    "\"justification\": \"Post-stroke cardiac monitoring indicated but payer criteria require prior Holter attempt — escalating for care coordinator review.\", "
                    "\"critic_reviewed\": true, \"was_revised\": false}]"
                )
            }
        },

        # ── P020 Charles Obi — HTN + T2DM + Hyperlipidemia (MODERATE) ─────────
        {
            "patient_id": "P020",
            "hours_ago": 8,
            "status": "COMPLETED",
            "result": {
                "mode": "full",
                "patient_id": "P020",
                "complexity_tier": "MODERATE",
                "complexity_score": 5,
                "complexity_rationale": "Multi-factor case — MDT collaboration",
                "workflow_completed": True,
                "escalation_triggered": False,
                "status": "COMPLETED",
                "crew_output": (
                    "MODERATE pathway completed for patient P020.\n\n"
                    "KNOWLEDGE GRAPH:\n"
                    "**Clinical Knowledge Graph Analysis:**\n\n"
                    "🔴 **High Priority Clinical Connections:**\n"
                    "  - [HIGH] HbA1c > 9.0% → Endocrinology referral (direct)\n"
                    "    Evidence: ADA 2025: Endocrinology referral when HbA1c > 9%\n"
                    "  - [HIGH] Hypertension → Cardiovascular event (direct)\n"
                    "    Evidence: ACC/AHA 2017: HTN is #1 modifiable CVD risk factor\n\n"
                    "CARE GAPS:\n"
                    "Care Gap Report for Patient P020: Charles Obi\n\n"
                    "1. Care Gaps Found:\n\n"
                    "* Endocrinology Referral — HIGH priority [ADA 2025] — HbA1c 9.5% on maximal therapy. Action: Refer within 30 days.\n\n"
                    "* Colonoscopy — HIGH priority [USPSTF] — Age 62, not screened. Action: Schedule within 60 days.\n\n"
                    "* Lipid Panel — MEDIUM priority [ACC/AHA 2018] — Annual monitoring on statin therapy. Action: Order at next visit.\n\n"
                    "Summary: Diabetes management and preventive screening are priorities.\n\n"
                    "PRIOR AUTH:\n"
                    "[{\"patient_id\": \"P020\", \"request_id\": \"REQ020\", \"item\": \"glp1_agonist\", "
                    "\"decision\": \"APPROVE\", \"confidence\": 0.88, "
                    "\"justification\": \"Patient meets coverage criteria: HbA1c 9.5% > 9.0%, BMI 32, documented metformin trial.\", "
                    "\"critic_reviewed\": true, \"was_revised\": false}]"
                )
            }
        },

        # ── P005 Priya Nair — RA + osteoporosis (care gap only) ───────────────
        {
            "patient_id": "P005",
            "hours_ago": 12,
            "status": "COMPLETED",
            "result": {
                "mode": "care_gap_only",
                "steps_executed": 8,
                "final_report": (
                    "Care Gap Report for Patient P005: Priya Nair\n\n"
                    "1. Care Gaps Found:\n\n"
                    "* DEXA Bone Density Scan — HIGH priority [ACR 2021: RA Guidelines] — "
                    "Methotrexate and glucocorticoid use increase fracture risk. Action: Schedule DEXA within 30 days.\n\n"
                    "* Annual Flu Vaccine — HIGH priority [ACR: RA Comorbidities] — "
                    "Strongly recommended for all RA patients on biologic DMARDs. Action: Administer this visit.\n\n"
                    "* Cardiovascular Risk Assessment — MEDIUM priority [ACR 2021] — "
                    "RA increases CVD risk 1.5-2x. Action: Annual lipid panel and BP check.\n\n"
                    "2. Clinical Review Items:\n\n"
                    "* Monitor CBC every 2-3 months on methotrexate therapy.\n"
                    "* Verify folic acid supplementation ongoing.\n\n"
                    "Summary: Bone density screening and flu vaccine are immediate priorities."
                )
            }
        },

        # ── P012 Kevin O'Brien — auth only ────────────────────────────────────
        {
            "patient_id": "P012",
            "hours_ago": 18,
            "status": "COMPLETED",
            "result": {
                "mode": "auth_only",
                "auth_results": [
                    {
                        "patient_id": "P012",
                        "request_id": "REQ012",
                        "item": "cpap_machine",
                        "decision": "APPROVE",
                        "confidence": 0.94,
                        "justification": "Patient meets all payer criteria for CPAP: AHI 28 events/hour on polysomnography, BMI 34, documented daytime sleepiness (Epworth 14).",
                        "critic_reviewed": True,
                        "was_revised": False
                    }
                ]
            }
        },

        # ── P009 Susan Bergman — T2DM + OA + Obesity ──────────────────────────
        {
            "patient_id": "P009",
            "hours_ago": 24,
            "status": "COMPLETED",
            "result": {
                "mode": "care_gap_only",
                "steps_executed": 10,
                "final_report": (
                    "Care Gap Report for Patient P009: Susan Bergman\n\n"
                    "1. Care Gaps Found:\n\n"
                    "* Mammogram — HIGH priority [USPSTF: Breast Cancer Screening] — Biennial mammography for women 40-74. Action: Schedule within 60 days.\n\n"
                    "* Colonoscopy — HIGH priority [USPSTF: Colorectal Cancer Screening] — Adults 45-75 should be screened. Action: Schedule or offer FIT test.\n\n"
                    "* Weight Management — MEDIUM priority [USPSTF 2018: Obesity Management] — BMI 33 with weight-related comorbidities. Action: Refer to intensive behavioral intervention.\n\n"
                    "* Diabetic Eye Exam — MEDIUM priority [ADA 2025] — Annual retinal exam required. Action: Refer to ophthalmology.\n\n"
                    "Summary: Multiple preventive screenings overdue."
                )
            }
        },

        # ── P016 Frank Deluca — Parkinson's + depression + HTN ────────────────
        {
            "patient_id": "P016",
            "hours_ago": 30,
            "status": "COMPLETED",
            "result": {
                "mode": "full",
                "patient_id": "P016",
                "complexity_tier": "MODERATE",
                "complexity_score": 6,
                "complexity_rationale": "Multi-factor case — MDT collaboration",
                "workflow_completed": True,
                "escalation_triggered": False,
                "status": "COMPLETED",
                "crew_output": (
                    "MODERATE pathway completed for patient P016.\n\n"
                    "KNOWLEDGE GRAPH:\n"
                    "**Clinical Knowledge Graph Analysis:**\n\n"
                    "🔴 **High Priority Clinical Connections:**\n"
                    "  - [HIGH] Depression → Cardiovascular event (direct)\n"
                    "    Evidence: AHA: Depression is independent CVD risk factor\n"
                    "  - [HIGH] Age 65+ → Fall risk assessment (direct)\n"
                    "    Evidence: USPSTF 2018: Fall risk assessment for adults 65+\n\n"
                    "CARE GAPS:\n"
                    "Care Gap Report for Patient P016: Frank Deluca\n\n"
                    "1. Care Gaps Found:\n\n"
                    "* Fall Risk Assessment — HIGH priority [USPSTF 2018] — Parkinson's + age increase fall risk. Action: Complete Timed Up and Go test.\n\n"
                    "* Colonoscopy — HIGH priority [USPSTF] — Age 58, overdue. Action: Schedule within 60 days.\n\n"
                    "* PHQ-9 Re-screening — MEDIUM priority [USPSTF] — On sertraline for depression; reassess response. Action: Complete PHQ-9 at next visit.\n\n"
                    "Summary: Fall prevention and depression monitoring are priorities.\n\n"
                    "PRIOR AUTH:\nNo pending authorization requests for this patient."
                )
            }
        },
    ]


# ── Seed orchestration ────────────────────────────────────────────────────────

def seed_audit_log(engine) -> int:
    """Insert demo runs into audit_log. Returns row count inserted."""
    runs = _build_demo_runs()
    inserted = 0

    with engine.connect() as conn:
        # Clear old seed rows — keep any live runs that happened after deploy
        conn.execute(text("DELETE FROM audit_log"))
        conn.commit()

        for run in runs:
            ts = datetime.utcnow() - timedelta(hours=run["hours_ago"])
            conn.execute(text("""
                INSERT INTO audit_log (patient_id, processed_at, status, result_json)
                VALUES (:pid, :ts, :status, :result)
            """), {
                "pid": run["patient_id"],
                "ts": ts.isoformat(),
                "status": run["status"],
                "result": json.dumps(run["result"])
            })
            inserted += 1

        conn.commit()

    return inserted


def seed_review_queue(engine) -> int:
    """Insert pending clinical reviews for escalated cases."""
    reviews = [
        {
            "patient_id": "P004",
            "hours_ago": 2,
            "agent_output": (
                "HIGH complexity patient with multi-system involvement. "
                "Drug safety analysis detected CRITICAL interactions (Warfarin + NSAIDs, "
                "Metformin contraindicated at eGFR < 30). "
                "Multidisciplinary review needed for home dialysis evaluation."
            ),
            "reason": "Agent/Critic review flagged drug safety concerns and multi-specialist coordination need"
        },
        {
            "patient_id": "P014",
            "hours_ago": 7,
            "agent_output": (
                "Post-stroke patient requiring cardiac monitoring. "
                "Payer criteria for extended monitoring require prior Holter attempt on file — "
                "not confirmed in available records. Care coordinator review needed."
            ),
            "reason": "Payer criteria verification required for extended cardiac monitor authorization"
        },
    ]

    inserted = 0
    with engine.connect() as conn:
        conn.execute(text("DELETE FROM review_queue WHERE status = 'PENDING'"))
        conn.commit()

        for r in reviews:
            ts = datetime.utcnow() - timedelta(hours=r["hours_ago"])
            conn.execute(text("""
                INSERT INTO review_queue
                (patient_id, created_at, status, agent_output, auth_results, reason)
                VALUES (:pid, :ts, 'PENDING', :output, '[]', :reason)
            """), {
                "pid": r["patient_id"],
                "ts": ts.isoformat(),
                "output": r["agent_output"],
                "reason": r["reason"]
            })
            inserted += 1

        conn.commit()

    return inserted


def reset_demo_data() -> dict:
    """Main entry: create tables if needed, then seed demo rows."""
    engine = _make_engine()

    # Ensure tables exist (safe to run repeatedly)
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

    audit_count = seed_audit_log(engine)
    review_count = seed_review_queue(engine)

    result = {
        "audit_log_rows": audit_count,
        "review_queue_rows": review_count,
        "database_url": DATABASE_URL,
    }
    print(f"[SeedData] ✅ Seeded {audit_count} audit rows, {review_count} review rows")
    print(f"[SeedData] DB: {DATABASE_URL}")
    return result


if __name__ == "__main__":
    reset_demo_data()