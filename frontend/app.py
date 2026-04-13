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
                        with st.container(border=True):
                            st.markdown(review["agent_output_preview"])

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
            ["P001","P002","P003","P004","P005","P006","P007","P008","P009","P010","P011","P012","P013","P014","P015","P016","P017","P018","P019","P020"],
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
            result = entry.get("result", {})
            mode = result.get("mode", "full")

            with st.expander(
                f"{status_icon} Patient {entry['patient_id']} — "
                f"{entry['status']} — {entry['processed_at'][:16]}",
                expanded=False
            ):
                # ── FAILED ────────────────────────────────────────────────────
                if entry["status"] == "FAILED":
                    st.error(f"**Error:** {result.get('error', 'Unknown error')}")

                # ── CARE GAP REPORT ───────────────────────────────────────────
                elif mode == "care_gap_only":
                    col1, col2 = st.columns(2)
                    col1.metric("Mode", "Care Gap Review")
                    col2.metric("Steps Executed", result.get("steps_executed", "—"))
                    st.markdown("---")
                    st.markdown("#### 📋 Care Gap Report")
                    report = result.get("final_report", "")
                    if report:
                        st.markdown(report)
                    else:
                        st.info("No report generated.")

                # ── PRIOR AUTH RESULTS ────────────────────────────────────────
                elif mode == "auth_only":
                    st.markdown("#### 🔐 Prior Authorization Results")
                    auth_results = result.get("auth_results", [])
                    if not auth_results:
                        st.info("No authorization requests found for this patient.")
                    for r in auth_results:
                        decision = r.get("decision", "UNKNOWN")
                        color = {"APPROVE": "✅", "DENY": "❌", "ESCALATE": "⚠️"}.get(decision, "🔄")
                        confidence = r.get("confidence", 0)
                        st.markdown(f"""
**{color} Request:** `{r.get('request_id')}` — **{r.get('item', '').replace('_', ' ').title()}**

| Field | Value |
|---|---|
| Decision | **{decision}** |
| Confidence | {confidence:.0%} |
| Justification | {r.get('justification', '—')} |
""")
                        st.markdown("---")

                # ── FULL TRIAGE (CrewAI) ──────────────────────────────────────
                elif mode == "full" or "crew_output" in result:
                    escalated = result.get("escalation_triggered", False)
                    st.markdown(f"**Escalation triggered:** {'⚠️ Yes' if escalated else '✅ No'}")
                    st.markdown("---")
                    st.markdown("#### 🤖 Supervisor Report")
                    crew_output = result.get("crew_output", "")
                    if crew_output:
                        st.markdown(crew_output)
                    else:
                        st.json(result)

                # ── FALLBACK ──────────────────────────────────────────────────
                else:
                    st.json(result)


# ── Page: System Status ───────────────────────────────────────────────────────

elif page == "🔍 System Status":
    st.title("🔍 System Status")
    st.markdown("---")

    # API health
    data = api_get("/health")
    if data:
        st.success(f"✅ API is healthy — version {data.get('version')}")
        col1, col2 = st.columns(2)
        col1.metric("API Status", "Online")
        col2.metric("Last checked", data.get("timestamp", "")[:19])
    else:
        st.error("❌ API is offline. Start it with: `uvicorn api.main:app --reload`")

    st.markdown("---")

    # Guidelines RAG status
    st.markdown("### 📚 Clinical Guidelines Knowledge Base")
    g_data = api_get("/guidelines-status")
    if g_data:
        collection = g_data.get("collection", {})
        col1, col2, col3 = st.columns(3)
        col1.metric("Total Chunks", collection.get("total_chunks", 0))
        col2.metric("Sources Loaded", collection.get("total_sources", 0))
        status = g_data.get("status", "unknown")
        col3.metric("Status", "✅ Ready" if collection.get("total_chunks", 0) > 0 else "⚠️ Empty")

        if collection.get("total_chunks", 0) == 0:
            st.warning("Guidelines not loaded yet. Run a refresh below.")

        # Sources table
        sources = collection.get("sources", [])
        if sources:
            st.markdown("**Loaded sources:**")
            for s in sources:
                last = s.get("last_updated", "")[:10]
                st.markdown(f"- `{s['source_id']}` — last updated {last}")

        # Recent refreshes
        recent = g_data.get("recent_refreshes", [])
        if recent:
            st.markdown("**Recent refreshes:**")
            for r in recent[:3]:
                ts = r.get("timestamp", "")[:16]
                mode = r.get("mode", "")
                embedded = r.get("sources_embedded", 0)
                valid = r.get("validation", "")
                st.markdown(f"- `{ts}` — {mode} — {embedded} sources embedded — {valid}")

    st.markdown("---")

    # Manual refresh controls
    st.markdown("### 🔄 Trigger Guidelines Refresh")
    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown("**Weekly sources only** (USPSTF, CDC)")
        st.caption("Use for routine weekly updates")
        if st.button("▶ Run Weekly Refresh", use_container_width=True):
            from rag.guideline_sources import WEEKLY_SOURCES
            ids = [s["id"] for s in WEEKLY_SOURCES]
            result = api_post("/refresh-guidelines", {"source_ids": ids, "force": False, "triggered_by": "dashboard"})
            if result:
                st.success(result.get("message", "Refresh started"))

    with col_b:
        st.markdown("**Full refresh — ALL sources**")
        st.caption("Use when ADA / ACC guidelines are updated")
        force = st.checkbox("Force re-embed (ignore change detection)")
        if st.button("⚡ Run Full Refresh", use_container_width=True):
            result = api_post("/refresh-guidelines", {"source_ids": None, "force": force, "triggered_by": "dashboard_manual"})
            if result:
                st.success(result.get("message", "Full refresh started"))

    st.markdown("---")

    # Test search
    st.markdown("### 🔍 Test Guideline Search")
    test_q = st.text_input("Search query", placeholder="mammogram screening 67 year old woman")
    if test_q:
        results = api_get(f"/guidelines-search?q={test_q}&n=3")
        if results and results.get("results"):
            for r in results["results"]:
                with st.expander(f"[{r['relevance_score']:.0%}] {r['source_name']}"):
                    st.markdown(f"**Source:** [{r['source_name']}]({r['url']})")
                    st.markdown(f"**Retrieved:** {r['scraped_at']}")
                    st.markdown(f"> {r['text'][:400]}...")
        else:
            st.info("No results — guidelines may not be loaded yet.")

    st.markdown("---")
    st.markdown("### Quick start commands")
    st.code("""
# 1. Install new RAG dependencies
pip install chromadb sentence-transformers

# 2. Run initial guidelines refresh (first time)
python rag/refresh_flow.py

# 3. Start the API
uvicorn api.main:app --reload --port 8000

# 4. Run Prefect weekly scheduler
python orchestration/prefect_flow.py
    """, language="bash")

# ── Footer ─────────────────────────────────────────────────────────────────────

st.sidebar.markdown("---")
st.sidebar.markdown("**Healthcare AI Agents**")
st.sidebar.markdown("LangGraph · CrewAI · Prefect · LangSmith")
st.sidebar.caption("Synthetic data only — no real patient information")