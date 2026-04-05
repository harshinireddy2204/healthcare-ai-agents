"""
frontend/app.py

Streamlit Human-in-the-Loop Dashboard.

Clinicians use this dashboard to:
  - See all cases escalated by the AI agents
  - Read the agent's reasoning and decision
  - Approve, reject, or modify the recommendation
  - View the audit trail

Run:
    streamlit run frontend/app.py
"""
import json
import os
from datetime import datetime

import requests
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

API_BASE = os.getenv("API_BASE_URL", "http://localhost:8000")

# ── Page config ───────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Healthcare AI — Clinical Review Dashboard",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Sidebar navigation ────────────────────────────────────────────────────────

st.sidebar.title("🏥 Clinical AI Dashboard")
st.sidebar.markdown("---")
page = st.sidebar.radio(
    "Navigate",
    ["📋 Pending Reviews", "⚡ Process Patient", "📊 Audit Log", "🔍 System Status"]
)

# ── Helper functions ───────────────────────────────────────────────────────────

def api_get(path: str) -> dict:
    try:
        r = requests.get(f"{API_BASE}{path}", timeout=10)
        r.raise_for_status()
        return r.json()
    except requests.exceptions.ConnectionError:
        st.error("Cannot connect to the API. Is `uvicorn api.main:app --reload` running?")
        return {}
    except Exception as e:
        st.error(f"API error: {e}")
        return {}


def api_post(path: str, data: dict) -> dict:
    try:
        r = requests.post(f"{API_BASE}{path}", json=data, timeout=30)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        st.error(f"API error: {e}")
        return {}


def format_confidence(confidence: float) -> str:
    if confidence >= 0.85:
        return f"🟢 {confidence:.0%}"
    elif confidence >= 0.70:
        return f"🟡 {confidence:.0%}"
    else:
        return f"🔴 {confidence:.0%}"


def decision_badge(decision: str) -> str:
    badges = {
        "APPROVE": "✅ APPROVE",
        "DENY": "❌ DENY",
        "ESCALATE": "⚠️ ESCALATE",
        "PENDING_REVIEW": "🔄 PENDING"
    }
    return badges.get(decision, decision)


# ── Page: Pending Reviews ─────────────────────────────────────────────────────

if page == "📋 Pending Reviews":
    st.title("📋 Cases Pending Clinical Review")
    st.markdown(
        "These cases were flagged by the AI agent for human review — either because "
        "the agent's confidence was below threshold or the decision required clinical judgment."
    )
    st.markdown("---")

    data = api_get("/pending-reviews")
    reviews = data.get("reviews", [])
    count = data.get("pending_count", 0)

    # Summary metric
    col1, col2, col3 = st.columns(3)
    col1.metric("Pending Cases", count)
    col2.metric("Today's Date", datetime.now().strftime("%Y-%m-%d"))
    col3.metric("Review Target", "< 4 hours")

    st.markdown("---")

    if not reviews:
        st.success("✅ No cases pending review. All agent decisions are complete.")
    else:
        for review in reviews:
            with st.expander(
                f"🔶 Patient {review['patient_id']} — "
                f"Escalated at {review['created_at'][:16]}",
                expanded=True
            ):
                col_left, col_right = st.columns([2, 1])

                with col_left:
                    st.markdown("**Escalation reason:**")
                    st.info(review.get("reason", "No reason provided"))

                    if review.get("agent_output_preview"):
                        st.markdown("**Agent reasoning (preview):**")
                        st.text_area(
                            "Agent output",
                            value=review["agent_output_preview"],
                            height=150,
                            key=f"output_{review['review_id']}",
                            disabled=True
                        )

                with col_right:
                    st.markdown("**Take action:**")
                    reviewer_name = st.text_input(
                        "Your name",
                        placeholder="Dr. Jane Smith",
                        key=f"reviewer_{review['review_id']}"
                    )
                    notes = st.text_area(
                        "Clinical notes (optional)",
                        placeholder="Add any relevant clinical context...",
                        height=80,
                        key=f"notes_{review['review_id']}"
                    )

                    col_a, col_r, col_m = st.columns(3)
                    if col_a.button("✅ Approve", key=f"approve_{review['review_id']}"):
                        if not reviewer_name:
                            st.warning("Please enter your name before resolving.")
                        else:
                            result = api_post(
                                f"/resolve-review/{review['review_id']}",
                                {"resolution": "APPROVED", "resolved_by": reviewer_name, "notes": notes}
                            )
                            if result:
                                st.success(f"Case approved by {reviewer_name}")
                                st.rerun()

                    if col_r.button("❌ Reject", key=f"reject_{review['review_id']}"):
                        if not reviewer_name:
                            st.warning("Please enter your name before resolving.")
                        else:
                            result = api_post(
                                f"/resolve-review/{review['review_id']}",
                                {"resolution": "REJECTED", "resolved_by": reviewer_name, "notes": notes}
                            )
                            if result:
                                st.success(f"Case rejected by {reviewer_name}")
                                st.rerun()

                    if col_m.button("✏️ Modify", key=f"modify_{review['review_id']}"):
                        if not reviewer_name:
                            st.warning("Please enter your name before resolving.")
                        else:
                            result = api_post(
                                f"/resolve-review/{review['review_id']}",
                                {"resolution": "MODIFIED", "resolved_by": reviewer_name, "notes": notes}
                            )
                            if result:
                                st.success(f"Case modified by {reviewer_name}")
                                st.rerun()


# ── Page: Process Patient ─────────────────────────────────────────────────────

elif page == "⚡ Process Patient":
    st.title("⚡ Process a Patient")
    st.markdown("Manually trigger the AI agent workflow for a specific patient.")
    st.markdown("---")

    with st.form("process_form"):
        patient_id = st.selectbox(
            "Select patient",
            ["P001", "P002", "P003"],
            help="Patient IDs from the synthetic dataset"
        )
        mode = st.radio(
            "Workflow mode",
            ["full", "auth_only", "care_gap_only"],
            format_func=lambda m: {
                "full": "Full triage (prior auth + care gaps + risk)",
                "auth_only": "Prior authorization only",
                "care_gap_only": "Care gap review only"
            }[m]
        )
        submitted = st.form_submit_button("🚀 Run Agent Workflow")

    if submitted:
        with st.spinner(f"Triggering {mode} workflow for {patient_id}..."):
            result = api_post("/process-patient", {"patient_id": patient_id, "mode": mode})
        if result:
            st.success(f"✅ {result.get('message')}")
            st.info(f"Workflow: {result.get('workflow_triggered')}")
            st.markdown("Check the **Audit Log** tab for results in 30–60 seconds.")


# ── Page: Audit Log ───────────────────────────────────────────────────────────

elif page == "📊 Audit Log":
    st.title("📊 Decision Audit Log")
    st.markdown("Complete history of all agent decisions.")
    st.markdown("---")

    filter_patient = st.text_input("Filter by patient ID (optional)", placeholder="P001")

    data = api_get(f"/audit-log{'?patient_id=' + filter_patient if filter_patient else ''}")
    entries = data.get("entries", [])

    if not entries:
        st.info("No audit log entries yet. Process a patient to see results here.")
    else:
        st.metric("Total records", data.get("total", 0))
        st.markdown("---")

        for entry in entries:
            status_icon = {"COMPLETED": "✅", "PENDING_REVIEW": "⚠️", "FAILED": "❌"}.get(
                entry["status"], "🔄"
            )
            with st.expander(
                f"{status_icon} Patient {entry['patient_id']} — "
                f"{entry['status']} — {entry['processed_at'][:16]}"
            ):
                st.json(entry.get("result", {}))


# ── Page: System Status ───────────────────────────────────────────────────────

elif page == "🔍 System Status":
    st.title("🔍 System Status")
    st.markdown("---")

    data = api_get("/health")
    if data:
        st.success(f"✅ API is healthy — version {data.get('version')}")
        col1, col2 = st.columns(2)
        col1.metric("API Status", "Online")
        col2.metric("Last checked", data.get("timestamp", "")[:19])
    else:
        st.error("❌ API is offline. Start it with: `uvicorn api.main:app --reload`")

    st.markdown("---")
    st.markdown("### Quick start commands")
    st.code("""
# 1. Start the API
uvicorn api.main:app --reload --port 8000

# 2. Run the Prefect workflow (processes all patients)
python orchestration/prefect_flow.py

# 3. Test the prior auth agent directly
python agents/prior_auth_agent.py

# 4. View agent traces
# Open https://smith.langchain.com → your project
    """, language="bash")

# ── Footer ─────────────────────────────────────────────────────────────────────

st.sidebar.markdown("---")
st.sidebar.markdown("**Healthcare AI Agents**")
st.sidebar.markdown("LangGraph · CrewAI · Prefect · LangSmith")
st.sidebar.caption("Synthetic data only — no real patient information")
