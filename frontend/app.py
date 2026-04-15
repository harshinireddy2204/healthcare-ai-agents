"""
frontend/app.py

Healthcare AI Multi-Agent System — Clinical Operations Dashboard

Production-grade Streamlit UI showing the multi-agent system handling
patients in real time. Features:
  - Live agent activity feed (auto-refreshing)
  - Complexity routing visualization (LOW/MODERATE/HIGH)
  - Drug safety alerts with FDA citations
  - Knowledge graph findings
  - Prior auth with Agent/Critic review status
  - Care gap reports with guideline citations
  - HITL review queue with approve/reject/modify
  - System health and RAG knowledge base status
"""
import json
import os
import time
from datetime import datetime, timedelta

import requests
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

# ── API URL: reads from Streamlit secrets (cloud) or .env (local) ─────────────
# On Streamlit Community Cloud: App Settings → Secrets → add API_BASE_URL
# Locally: set API_BASE_URL in your .env file or leave as default
def _get_api_base() -> str:
    # Streamlit Cloud secrets take priority
    try:
        url = st.secrets.get("API_BASE_URL", "")
        if url:
            return url.rstrip("/")
    except Exception:
        pass
    # Fall back to environment variable or localhost
    return os.getenv("API_BASE_URL", "http://localhost:8000")

API_BASE = _get_api_base()

# ── Page config ───────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Healthcare AI — Multi-Agent Clinical Operations",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Custom CSS ─────────────────────────────────────────────────────────────────

st.markdown("""
<style>
.agent-card {
    background: #1A1D27;
    border-radius: 10px;
    padding: 14px 16px;
    border: 1px solid #2D3142;
    margin-bottom: 10px;
}
.agent-header {
    display: flex;
    align-items: center;
    gap: 10px;
    margin-bottom: 8px;
}
.complexity-low    { color: #1D9E75; font-weight: 600; }
.complexity-mod    { color: #BA7517; font-weight: 600; }
.complexity-high   { color: #E24B4A; font-weight: 600; }
.tier-badge-safe     { background: #0F6E56; color: #E1F5EE; padding: 2px 8px; border-radius: 10px; font-size: 12px; }
.tier-badge-warn     { background: #854F0B; color: #FAEEDA; padding: 2px 8px; border-radius: 10px; font-size: 12px; }
.tier-badge-critical { background: #A32D2D; color: #FCEBEB; padding: 2px 8px; border-radius: 10px; font-size: 12px; }
.live-dot { width: 8px; height: 8px; border-radius: 50%; background: #1D9E75; display: inline-block; margin-right: 6px; animation: pulse 1.5s infinite; }
@keyframes pulse { 0%,100%{opacity:1} 50%{opacity:0.4} }
.stat-num { font-size: 28px; font-weight: 600; }
</style>
""", unsafe_allow_html=True)

# ── API helpers ───────────────────────────────────────────────────────────────

def api_get(path: str, timeout: int = 10) -> dict:
    try:
        r = requests.get(f"{API_BASE}{path}", timeout=timeout)
        r.raise_for_status()
        return r.json()
    except requests.exceptions.ConnectionError:
        return {"_error": "API offline"}
    except Exception as e:
        return {"_error": str(e)}


def api_post(path: str, data: dict, timeout: int = 30) -> dict:
    try:
        r = requests.post(f"{API_BASE}{path}", json=data, timeout=timeout)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        return {"_error": str(e)}

# ── Sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("## 🏥 Clinical AI Dashboard")
    st.markdown("---")

    health = api_get("/health")
    if "_error" not in health:
        st.markdown('<span class="live-dot"></span> **API Online**', unsafe_allow_html=True)
    else:
        st.error("⚫ API Offline")

    st.markdown("---")
    page = st.radio("Navigate", [
        "🏠 Live Overview",
        "⚡ Run Agent Workflow",
        "📋 Pending Reviews",
        "📊 Audit Log",
        "📈 Analytics & Reporting",
        "💊 Drug Safety",
        "🧠 Knowledge Graph",
        "📚 Guidelines KB",
        "🔧 System Status"
    ])

    st.markdown("---")
    st.markdown("**Agent Stack**")
    st.markdown("🔵 LangGraph ReAct")
    st.markdown("🟣 CrewAI MDT/ICT")
    st.markdown("🟡 OpenFDA Live")
    st.markdown("🟢 Knowledge Graph")
    st.markdown("📖 RAG Guidelines")
    st.markdown("👤 HITL Escalation")

    st.markdown("---")
    st.markdown("**💬 Share feedback**")
    st.markdown("[→ Leave feedback on GitHub](https://github.com/harshinireddy2204/healthcare-ai-agents/issues/new?title=Feedback&labels=feedback)")
    st.markdown("[→ Connect on LinkedIn](https://linkedin.com/in/harshini-reddy22/)")
    st.markdown("---")
    st.caption("Synthetic data only — HIPAA-safe")
    st.caption("Research: MDAgents (NeurIPS'24), TxAgent, MALADE")


# ── Page: Live Overview ────────────────────────────────────────────────────────

if page == "🏠 Live Overview":
    st.title("🏠 Multi-Agent Clinical Operations — Live")
    # Welcome banner for LinkedIn visitors
    st.info(
        "👋 **Welcome!** This is a live multi-agent clinical AI system — "
        "go to **⚡ Run Agent Workflow**, pick a patient, and trigger a workflow. "
        "Results appear here in 60–120 seconds. "
        "Built by [Harshini Reddy](https://linkedin.com/in/harshini-reddy22/) · "
        "[GitHub](https://github.com/harshinireddy2204/healthcare-ai-agents)",
        icon=None
    )

    st.markdown("Real-time view of the AI agent system processing patients.")

    # Auto-refresh toggle
    col_r, col_i = st.columns([3, 1])
    with col_r:
        auto_refresh = st.checkbox("Auto-refresh every 15s", value=False)
    with col_i:
        if st.button("🔄 Refresh"):
            st.rerun()

    audit = api_get("/audit-log?limit=50")
    entries = audit.get("entries", [])
    reviews = api_get("/pending-reviews").get("reviews", [])

    # ── Summary metrics ────────────────────────────────────────────────────────
    total = len(entries)
    completed = sum(1 for e in entries if e["status"] == "COMPLETED")
    failed    = sum(1 for e in entries if e["status"] == "FAILED")
    pending   = len(reviews)

    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Total Runs", total)
    col2.metric("✅ Completed", completed)
    col3.metric("⚠️ Pending Review", pending)
    col4.metric("❌ Failed", failed)
    success_rate = f"{(completed/total*100):.0f}%" if total > 0 else "—"
    col5.metric("Success Rate", success_rate)

    st.markdown("---")

    # ── Complexity distribution ────────────────────────────────────────────────
    if entries:
        tiers = {"LOW": 0, "MODERATE": 0, "HIGH": 0, "UNKNOWN": 0}
        for e in entries:
            tier = e.get("result", {}).get("complexity_tier", "UNKNOWN")
            tiers[tier] = tiers.get(tier, 0) + 1

        if any(v > 0 for k, v in tiers.items() if k != "UNKNOWN"):
            st.markdown("#### Complexity Routing Distribution (MDAgents pattern)")
            c1, c2, c3 = st.columns(3)
            c1.metric("🟢 LOW (single agent)", tiers["LOW"])
            c2.metric("🟡 MODERATE (MDT)", tiers["MODERATE"])
            c3.metric("🔴 HIGH (ICT + drug safety)", tiers["HIGH"])
            st.markdown("---")

    # ── Live activity feed ─────────────────────────────────────────────────────
    st.markdown("#### 🔴 Live Agent Activity Feed")

    if not entries:
        st.info("No agent runs yet. Use 'Run Agent Workflow' to start processing patients.")
    else:
        for entry in entries[:10]:
            result = entry.get("result", {})
            mode   = result.get("mode", "full")
            tier   = result.get("complexity_tier", "")
            status = entry["status"]

            status_icon = {"COMPLETED": "✅", "PENDING_REVIEW": "⚠️", "FAILED": "❌"}.get(status, "🔄")
            tier_color  = {"LOW": "complexity-low", "MODERATE": "complexity-mod", "HIGH": "complexity-high"}.get(tier, "")
            tier_badge  = f'<span class="{tier_color}">{tier}</span>' if tier else ""

            ts = entry["processed_at"][:16].replace("T", " ")

            with st.expander(
                f"{status_icon} Patient **{entry['patient_id']}** — {status} — {ts}",
                expanded=(status == "PENDING_REVIEW")
            ):
                # Mode + complexity row
                info_cols = st.columns(4)
                info_cols[0].markdown(f"**Mode:** `{mode}`")
                if tier:
                    info_cols[1].markdown(f"**Complexity:** {tier_badge}", unsafe_allow_html=True)
                if result.get("complexity_score") is not None:
                    info_cols[2].metric("Score", result["complexity_score"])
                if result.get("workflow_results", {}).get("pathway"):
                    info_cols[3].markdown(f"**Pathway:** `{result['workflow_results']['pathway']}`")

                st.markdown("---")

                # Drug safety alert
                drug_report = (result.get("workflow_results", {}).get("drug_safety") or
                               result.get("drug_safety"))
                if drug_report:
                    safety_tier = drug_report.get("safety_tier", "CAUTION")
                    color = {"CRITICAL": "🔴", "WARNING": "🟠", "CAUTION": "🟡", "SAFE": "🟢"}.get(safety_tier, "🟡")
                    st.markdown(f"**{color} Drug Safety: {safety_tier}**")
                    if drug_report.get("fda_findings_count", 0) > 0:
                        st.caption(f"OpenFDA: {drug_report['fda_findings_count']} drug label findings")

                # Care gap report
                if mode == "care_gap_only":
                    report = result.get("final_report", "")
                    if report:
                        st.markdown("**📋 Care Gap Report:**")
                        with st.container(border=True):
                            st.markdown(report)

                # Full mode — show care gap from workflow_results
                if mode == "full":
                    wf = result.get("workflow_results", {})
                    care_report = (wf.get("care_gaps", {}).get("final_report", "") or
                                   result.get("care_summary", ""))
                    if care_report:
                        st.markdown("**📋 Care Gap Report:**")
                        with st.container(border=True):
                            st.markdown(care_report)

                # Auth results
                auth_results = (result.get("auth_results") or
                                result.get("workflow_results", {}).get("auth_results", []))
                if auth_results:
                    st.markdown("**🔐 Prior Auth Results:**")
                    for r in auth_results:
                        decision = r.get("decision", "?")
                        icon = {"APPROVE": "✅", "DENY": "❌", "ESCALATE": "⚠️"}.get(decision, "❓")
                        conf = r.get("confidence", 0)
                        revised = "🔄 revised" if r.get("was_revised") else ""
                        critic = "✓ critic reviewed" if r.get("critic_reviewed") else ""
                        st.markdown(
                            f"**{icon} {decision}** — `{r.get('item', '')}` — "
                            f"{conf:.0%} confidence {revised} {critic}"
                        )

                # Full mode crew output
                if mode == "full" or "crew_output" in result:
                    crew = result.get("crew_output", "")
                    if crew:
                        with st.expander("View full agent output"):
                            st.markdown(crew[:1500])

                if status == "FAILED":
                    st.error(f"Error: {result.get('error', 'Unknown')}")

    if auto_refresh:
        time.sleep(15)
        st.rerun()


# ── Page: Run Agent Workflow ──────────────────────────────────────────────────

elif page == "⚡ Run Agent Workflow":
    st.title("⚡ Run Agent Workflow")

    tab_synthetic, tab_fhir = st.tabs([
        "🗂️ Synthetic Patients (P001–P020)",
        "🏥 FHIR R4 — Live Patients (hapi.fhir.org)"
    ])

    # ── Tab 1: Synthetic patients ─────────────────────────────────────────────
    with tab_synthetic:
        st.markdown("20 clinically diverse synthetic patients — guaranteed data, no network dependency.")

        patient_profiles = {
            "P001": "Eleanor Vance, 67F | T2DM + HTN + CKD3 | HbA1c 10.2",
            "P002": "Robert Callahan, 54M | CAD + Hyperlipidemia",
            "P003": "Maria Santos, 43F | Asthma + Obesity + Prediabetes",
            "P004": "James Whitfield, 72M | HF + AFib + T2DM + CKD4 ⚠️ HIGH",
            "P005": "Priya Nair, 38F | Rheumatoid Arthritis + Osteoporosis",
            "P006": "Marcus Thompson, 61M | COPD + HTN + Hyperlipidemia",
            "P007": "Linda Okafor, 55F | Breast Cancer Stage II ⚠️ HIGH",
            "P008": "David Hernandez, 47M | T1DM + HTN | HbA1c 8.8",
            "P009": "Susan Bergman, 69F | Osteoarthritis + T2DM + Obesity",
            "P010": "Anthony Rivera, 33M | Crohn's Disease + Anemia",
            "P011": "Dorothy Chang, 78F | HTN + CKD3 + Hyperlipidemia + Osteoporosis",
            "P012": "Kevin O'Brien, 50M | HTN + Sleep Apnea + Obesity",
            "P013": "Gabrielle Martin, 29F | T1DM + Celiac Disease",
            "P014": "Harold Washington, 65M | Stroke + HTN + T2DM ⚠️ HIGH",
            "P015": "Mei-Ling Zhou, 45F | Lupus + HTN",
            "P016": "Frank Deluca, 58M | Parkinson's + Depression + HTN",
            "P017": "Aisha Mohammed, 36F | PCOS + Prediabetes + Obesity",
            "P018": "Raymond Foster, 74M | Prostate Cancer + T2DM + CKD3 ⚠️ HIGH",
            "P019": "Natalie Russo, 41F | Migraine + Anxiety + Hypothyroidism",
            "P020": "Charles Obi, 62M | HTN + T2DM + Hyperlipidemia + Obesity | HbA1c 9.5",
        }

        col1, col2 = st.columns([2, 1])
        with col1:
            patient_id = st.selectbox(
                "Select patient",
                options=list(patient_profiles.keys()),
                format_func=lambda p: f"{p} — {patient_profiles[p]}"
            )
        with col2:
            st.markdown("**Predicted complexity:**")
            if "⚠️ HIGH" in patient_profiles[patient_id]:
                st.markdown("🔴 HIGH — ICT + drug safety")
            elif any(x in patient_profiles[patient_id] for x in ["CKD", "HF", "AFib", "Cancer", "Stroke"]):
                st.markdown("🟡 MODERATE — MDT")
            else:
                st.markdown("🟢 LOW — single agent")

        mode = st.radio(
            "Workflow mode",
            ["full", "auth_only", "care_gap_only"],
            format_func=lambda m: {
                "full": "🤖 Full triage",
                "auth_only": "🔐 Prior auth + Critic",
                "care_gap_only": "📋 Care gaps + RAG"
            }[m],
            horizontal=True
        )
        mode_descriptions = {
            "full": "Complexity router → LOW/MOD/HIGH pathway → all agents",
            "auth_only": "LangGraph ReAct + Agent/Critic review pattern (MALADE)",
            "care_gap_only": "Plan-and-Execute + RAG over 63 clinical guidelines"
        }
        st.caption(f"ℹ️ {mode_descriptions[mode]}")

        if st.button("🚀 Run Workflow", type="primary"):
            with st.spinner(f"Triggering {mode} workflow for {patient_id}..."):
                result = api_post("/process-patient", {"patient_id": patient_id, "mode": mode})
            if "_error" not in result:
                st.success(f"✅ Workflow triggered — {result.get('workflow_triggered')}")
                st.info("Check **Live Overview** or **Audit Log** for results in 60–120 seconds.")
            else:
                st.error(f"Failed: {result['_error']}")

        st.markdown("---")
        st.markdown("#### Quick batch — run all 20 patients")
        st.caption("Runs care_gap_only on all patients to populate the dashboard.")
        if st.button("▶ Run batch (all 20 patients, care_gap_only)"):
            progress = st.progress(0)
            for i, pid in enumerate(list(patient_profiles.keys())):
                api_post("/process-patient", {"patient_id": pid, "mode": "care_gap_only"})
                progress.progress((i + 1) / 20)
                time.sleep(0.3)
            st.success("Batch triggered for all 20 patients.")

    # ── Tab 2: FHIR R4 live patients ──────────────────────────────────────────
    with tab_fhir:
        st.markdown(
            "**Live FHIR R4 data** from `hapi.fhir.org/baseR4` — "
            "This is the healthcare industry standard for EHR data exchange."
        )

        fhir_server = st.text_input(
            "FHIR R4 server",
            value="https://hapi.fhir.org/baseR4",
            help="Any FHIR R4 compliant endpoint"
        )

        st.caption("Resources: Patient · Condition · Observation (LOINC) · MedicationRequest · Immunization · ServiceRequest")
        st.markdown("---")

        st.markdown("#### Step 1 — Search for a FHIR Patient")
        search_col, btn_col = st.columns([3, 1])
        search_name = search_col.text_input(
            "Search by patient name",
            placeholder="Smith, Johnson, Williams...",
            key="fhir_search_name"
        )

        if btn_col.button("🔍 Search FHIR", type="primary"):
            with st.spinner(f"Querying {fhir_server}..."):
                try:
                    import httpx
                    params = {"name": search_name, "_count": "8", "_format": "json"}
                    headers = {"Accept": "application/fhir+json"}
                    with httpx.Client(timeout=10, follow_redirects=True) as client:
                        resp = client.get(f"{fhir_server}/Patient", params=params, headers=headers)
                        resp.raise_for_status()
                        bundle = resp.json()

                    fhir_results = []
                    for entry in bundle.get("entry", []):
                        p = entry.get("resource", {})
                        if p.get("resourceType") == "Patient":
                            name_entry = p.get("name", [{}])[0]
                            given = " ".join(name_entry.get("given", []))
                            family = name_entry.get("family", "")
                            fhir_results.append({
                                "fhir_id": p.get("id", ""),
                                "name": f"{given} {family}".strip() or "Unknown",
                                "gender": p.get("gender", "unknown"),
                                "birth_date": p.get("birthDate", ""),
                            })
                    st.session_state["fhir_results"] = fhir_results
                    if fhir_results:
                        st.success(f"Found {len(fhir_results)} FHIR patients")
                    else:
                        st.warning("No patients found — try Smith, Johnson, or Williams")
                except Exception as e:
                    st.error(f"FHIR server error: {e}")
                    st.caption("The public HAPI server can be slow. Try again in a moment.")

        fhir_results = st.session_state.get("fhir_results", [])
        if fhir_results:
            st.markdown("#### Step 2 — Select and Load Patient Record")
            selected_fhir = st.selectbox(
                "FHIR Patients found",
                options=fhir_results,
                format_func=lambda p: f"{p['name']} | {p['gender']} | DOB: {p['birth_date']} | ID: {p['fhir_id']}"
            )

            if selected_fhir and st.button(f"📋 Load full FHIR record", use_container_width=True):
                fhir_id = selected_fhir["fhir_id"]
                with st.spinner(f"Fetching Patient + Condition + Observation + Medication..."):
                    try:
                        import httpx
                        headers = {"Accept": "application/fhir+json"}

                        def fhir_get(resource, params):
                            with httpx.Client(timeout=10, follow_redirects=True) as c:
                                r = c.get(f"{fhir_server}/{resource}",
                                          params={**params, "_format": "json"}, headers=headers)
                                r.raise_for_status()
                                return r.json()

                        def extract(bundle, rtype):
                            return [e["resource"] for e in bundle.get("entry", [])
                                    if e.get("resource", {}).get("resourceType") == rtype]

                        conditions    = extract(fhir_get("Condition",         {"patient": fhir_id, "_count": "20"}),  "Condition")
                        observations  = extract(fhir_get("Observation",       {"patient": fhir_id, "_count": "15", "_sort": "-date"}), "Observation")
                        medications   = extract(fhir_get("MedicationRequest", {"patient": fhir_id, "_count": "20"}),  "MedicationRequest")
                        immunizations = extract(fhir_get("Immunization",      {"patient": fhir_id, "_count": "15"}),  "Immunization")

                        st.markdown(f"### 🏥 {selected_fhir['name']}")
                        st.caption(f"FHIR ID: `{fhir_id}` | Server: `{fhir_server}`")

                        rc1, rc2, rc3, rc4 = st.columns(4)
                        rc1.metric("Conditions", len(conditions))
                        rc2.metric("Observations", len(observations))
                        rc3.metric("Medications", len(medications))
                        rc4.metric("Immunizations", len(immunizations))

                        if conditions:
                            st.markdown("**Active Conditions:**")
                            for cond in conditions[:8]:
                                code = cond.get("code", {})
                                text = code.get("text") or (code.get("coding", [{}])[0].get("display", "") if code.get("coding") else "")
                                st.markdown(f"- {text or 'Unknown'}")

                        if observations:
                            st.markdown("**Recent Lab Results:**")
                            for obs in observations[:6]:
                                code = obs.get("code", {})
                                label = code.get("text") or (code.get("coding", [{}])[0].get("display", "") if code.get("coding") else "")
                                vq = obs.get("valueQuantity", {})
                                val = vq.get("value", obs.get("valueString", ""))
                                unit = vq.get("unit", "")
                                date = obs.get("effectiveDateTime", "")[:10]
                                if label and val:
                                    st.markdown(f"- **{label}:** {val} {unit} ({date})")

                        if medications:
                            st.markdown("**Active Medications:**")
                            for med in medications[:6]:
                                mc = med.get("medicationCodeableConcept", {})
                                name = mc.get("text") or (mc.get("coding", [{}])[0].get("display", "") if mc.get("coding") else "")
                                st.markdown(f"- {name or 'Unknown'}")

                        st.markdown("---")
                        st.info(
                            f"✅ Live FHIR R4 data from `{fhir_server}`. "
                            f"Set `USE_FHIR=true` and `FHIR_BASE_URL={fhir_server}` in `.env` "
                            f"to use FHIR ID `{fhir_id}` directly in agent workflows."
                        )
                        st.code(f"USE_FHIR=true\nFHIR_BASE_URL={fhir_server}", language="bash")

                    except Exception as e:
                        st.error(f"FHIR fetch error: {e}")

        st.markdown("---")
        st.markdown("#### FHIR R4 Resources used by our agents")
        st.markdown("""
| FHIR Resource | Standard | What agents use it for |
|---|---|---|
| `Patient` | HL7 FHIR R4 | Demographics, age, gender |
| `Condition` | SNOMED CT | Active diagnoses |
| `Observation` | LOINC codes | Labs (HbA1c=4548-4, eGFR=33914-3, LDL=13457-7) |
| `MedicationRequest` | RxNorm | Active medications |
| `Immunization` | CVX codes | Vaccine history |
| `ServiceRequest` | SNOMED | Prior auth requests |
""")
        st.caption("FHIR R4 is the healthcare industry standard adopted by Epic, Cerner, and major health systems.")


# ── Page: Pending Reviews ─────────────────────────────────────────────────────

elif page == "📋 Pending Reviews":
    st.title("📋 Pending Clinical Reviews")
    st.markdown("Cases escalated by AI agents requiring clinician decision.")

    data   = api_get("/pending-reviews")
    reviews = data.get("reviews", [])
    count  = data.get("pending_count", 0)

    col1, col2, col3 = st.columns(3)
    col1.metric("Pending", count)
    col2.metric("Review Target", "< 4 hours")
    col3.metric("Today", datetime.now().strftime("%Y-%m-%d"))

    if st.button("🔄 Refresh"):
        st.rerun()

    st.markdown("---")

    if not reviews:
        st.success("✅ No cases pending — all agent decisions are complete or in progress.")
    else:
        for review in reviews:
            with st.expander(
                f"🔶 Patient {review['patient_id']} — Escalated {review['created_at'][:16]}",
                expanded=True
            ):
                col_left, col_right = st.columns([2, 1])

                with col_left:
                    st.markdown("**Escalation reason:**")
                    st.info(review.get("reason", "Agent/Critic review flagged this case"))
                    if review.get("agent_output_preview"):
                        st.markdown("**Agent reasoning:**")
                        with st.container(border=True):
                            st.markdown(review["agent_output_preview"])

                with col_right:
                    st.markdown("**Clinician Action Required**")
                    reviewer = st.text_input("Your name", placeholder="Dr. Smith",
                                             key=f"rev_{review['review_id']}")
                    notes = st.text_area("Clinical notes", height=80,
                                         key=f"notes_{review['review_id']}")

                    c1, c2, c3 = st.columns(3)
                    for label, resolution, col in [
                        ("✅ Approve", "APPROVED", c1),
                        ("❌ Reject",  "REJECTED",  c2),
                        ("✏️ Modify",  "MODIFIED",  c3)
                    ]:
                        if col.button(label, key=f"{resolution}_{review['review_id']}"):
                            if not reviewer:
                                st.warning("Enter your name first.")
                            else:
                                r = api_post(f"/resolve-review/{review['review_id']}",
                                             {"resolution": resolution, "resolved_by": reviewer, "notes": notes})
                                if "_error" not in r:
                                    st.success(f"Case {resolution.lower()} by {reviewer}")
                                    st.rerun()


# ── Page: Audit Log ───────────────────────────────────────────────────────────

elif page == "📊 Audit Log":
    st.title("📊 Decision Audit Log")

    col_f, col_lim = st.columns([2, 1])
    filter_patient = col_f.text_input("Filter by patient ID", placeholder="P004")
    limit = col_lim.selectbox("Show", [20, 50, 100], index=0)

    url = f"/audit-log?limit={limit}"
    if filter_patient:
        url += f"&patient_id={filter_patient}"
    data = api_get(url)
    entries = data.get("entries", [])

    st.metric("Total records", data.get("total", 0))

    if not entries:
        st.info("No audit log entries yet.")
    else:
        for entry in entries:
            result = entry.get("result", {})
            mode   = result.get("mode", "full")
            tier   = result.get("complexity_tier", "")
            status_icon = {"COMPLETED": "✅", "PENDING_REVIEW": "⚠️", "FAILED": "❌"}.get(entry["status"], "🔄")

            header = f"{status_icon} **{entry['patient_id']}** — {entry['status']}"
            if tier:
                header += f" — {tier}"
            header += f" — {entry['processed_at'][:16]}"

            with st.expander(header, expanded=False):
                if entry["status"] == "FAILED":
                    st.error(result.get("error", "Unknown error"))

                elif mode == "care_gap_only":
                    c1, c2 = st.columns(2)
                    c1.metric("Mode", "Care Gap")
                    c2.metric("Steps", result.get("steps_executed", "—"))
                    report = result.get("final_report", "")
                    if report:
                        st.markdown(report)

                elif mode == "auth_only":
                    st.markdown("#### 🔐 Prior Auth Results")
                    for r in result.get("auth_results", []):
                        decision = r.get("decision", "?")
                        icon = {"APPROVE": "✅", "DENY": "❌", "ESCALATE": "⚠️"}.get(decision, "❓")
                        st.markdown(f"**{icon} {decision}** — `{r.get('request_id')}` — `{r.get('item', '')}`")
                        c1, c2, c3 = st.columns(3)
                        c1.metric("Confidence", f"{r.get('confidence', 0):.0%}")
                        c2.metric("Critic reviewed", "✓" if r.get("critic_reviewed") else "—")
                        c3.metric("Revised", "Yes" if r.get("was_revised") else "No")
                        st.caption(r.get("justification", ""))
                        st.markdown("---")

                elif mode == "full" or "crew_output" in result:
                    tier_val = result.get("complexity_tier", "")
                    if tier_val:
                        tier_colors = {"LOW": "normal", "MODERATE": "off", "HIGH": "inverse"}
                        st.markdown(f"**Complexity:** {tier_val} — {result.get('complexity_rationale', '')}")
                    crew = result.get("crew_output", "")
                    if crew:
                        st.markdown(crew)

                else:
                    st.json(result)


# ── Page: Analytics & Reporting ──────────────────────────────────────────────

elif page == "📈 Analytics & Reporting":
    st.title("📈 Analytics & Reporting")
    st.markdown(
        "Operational metrics, cohort analysis, and data quality reporting. "
        "Powered by the SQL analytics layer — compatible with PostgreSQL and Snowflake."
    )

    try:
        import sys
        sys.path.insert(0, ".")
        from analytics.queries import (
            get_agent_performance_summary,
            get_prior_auth_metrics,
            get_care_gap_metrics,
            get_complexity_distribution,
            get_patient_cohort_analysis,
            get_review_queue_sla_metrics
        )
        from analytics.data_quality import run_data_quality_check

        tab_ops, tab_auth, tab_gaps, tab_cohorts, tab_dq = st.tabs([
            "📊 Operations",
            "🔐 Prior Auth",
            "📋 Care Gaps",
            "👥 Patient Cohorts",
            "✅ Data Quality"
        ])

        # ── Operations tab ─────────────────────────────────────────────────────
        with tab_ops:
            st.markdown("#### Agent Performance — Last 30 Days")
            perf = get_agent_performance_summary(30)
            c1, c2, c3, c4, c5 = st.columns(5)
            c1.metric("Total Runs", perf["total_runs"])
            c2.metric("Completed", perf["completed"])
            c3.metric("Escalated", perf["escalated_to_human"])
            c4.metric("Automation Rate", f"{perf['automation_rate']}%")
            c5.metric("Unique Patients", perf["unique_patients"])

            st.markdown("---")
            comp = get_complexity_distribution(30)
            st.markdown("#### MDAgents Complexity Routing Impact")
            col1, col2, col3 = st.columns(3)
            col1.metric("🟢 LOW pathway", f"{comp['pct_low']}%", "single agent")
            col2.metric("🟡 MODERATE pathway", f"{comp['pct_moderate']}%", "MDT team")
            col3.metric("🔴 HIGH pathway", f"{comp['pct_high']}%", "full ICT")

            if comp.get("estimated_cost_savings_pct", 0) > 0:
                st.success(
                    f"🎯 Adaptive routing estimated to save **{comp['estimated_cost_savings_pct']}%** "
                    f"of token costs vs running full ICT on every patient."
                )

            sla = get_review_queue_sla_metrics()
            st.markdown("---")
            st.markdown("#### HITL Review Queue SLA")
            s1, s2, s3, s4 = st.columns(4)
            s1.metric("Total Cases", sla["total_cases"])
            s2.metric("Resolved", sla["resolved"])
            s3.metric("Avg Review Time", f"{sla['avg_review_time_hours']}h")
            s4.metric("SLA Met (< 4h)", f"{sla['sla_met_pct']}%")

        # ── Prior Auth tab ─────────────────────────────────────────────────────
        with tab_auth:
            st.markdown("#### Prior Authorization Metrics")
            auth = get_prior_auth_metrics(30)
            a1, a2, a3, a4 = st.columns(4)
            a1.metric("Total Requests", auth["total_auth_requests"])
            a2.metric("Approval Rate", f"{auth['approval_rate']}%")
            a3.metric("Escalation Rate", f"{auth['escalation_rate']}%")
            a4.metric("Avg Confidence", f"{auth['avg_confidence']:.0%}")

            st.markdown("---")
            decisions = auth["decisions"]
            if any(decisions.values()):
                import pandas as pd
                df = pd.DataFrame([
                    {"Decision": k, "Count": v,
                     "Pct": f"{v/max(sum(decisions.values()),1)*100:.0f}%"}
                    for k, v in decisions.items()
                ])
                st.dataframe(df, use_container_width=True, hide_index=True)

            if auth["critic_reviewed"] > 0:
                st.markdown("#### Agent/Critic Review Stats (MALADE pattern)")
                cr1, cr2 = st.columns(2)
                cr1.metric("Critic Reviewed", auth["critic_reviewed"])
                cr2.metric("Decisions Revised", f"{auth['revision_rate']}%")
                st.caption("Agent/Critic pattern: a critic agent reviews each decision before finalizing, improving accuracy by catching logical gaps.")

        # ── Care Gaps tab ──────────────────────────────────────────────────────
        with tab_gaps:
            st.markdown("#### Care Gap Population Analysis")
            gaps = get_care_gap_metrics(30)
            g1, g2, g3, g4 = st.columns(4)
            g1.metric("Patients Analyzed", gaps["patients_analyzed"])
            g2.metric("With Gaps", gaps["patients_with_gaps"])
            g3.metric("Total Gaps", gaps["total_gaps_identified"])
            g4.metric("Avg Gaps/Patient", gaps["avg_gaps_per_patient"])

            st.markdown("---")
            st.markdown("#### Most Common Care Gaps")
            gap_freq = gaps.get("gap_frequency", [])
            if gap_freq:
                import pandas as pd
                df = pd.DataFrame(gap_freq)
                df.columns = ["Care Gap", "Patients Affected", "Population %"]
                st.dataframe(df, use_container_width=True, hide_index=True)

                # Simple bar chart
                st.bar_chart(
                    {row["gap"]: row["count"] for row in gap_freq[:8]},
                    use_container_width=True
                )
            else:
                st.info("Run care_gap_only workflows to populate this chart.")

        # ── Patient Cohorts tab ────────────────────────────────────────────────
        with tab_cohorts:
            st.markdown("#### Patient Cohort Segmentation")
            st.caption("SQL-powered cohort analysis — compatible with Snowflake / PostgreSQL / BigQuery")
            cohorts = get_patient_cohort_analysis()
            if cohorts:
                import pandas as pd
                df = pd.DataFrame(cohorts)[["cohort", "patient_count", "pct_of_population", "patient_ids"]]
                df["patient_ids"] = df["patient_ids"].apply(lambda x: ", ".join(x))
                df.columns = ["Cohort", "Patients", "% of Population", "Patient IDs"]
                st.dataframe(df, use_container_width=True, hide_index=True)

                st.bar_chart(
                    {r["cohort"]: r["patient_count"] for r in cohorts},
                    use_container_width=True
                )
            else:
                st.info("No cohort data available.")

            st.markdown("---")
            st.markdown("**dbt model:** `dbt/models/marts/patient_care_gaps.sql`")
            st.caption("Production SQL mart transforms agent outputs into HEDIS-reportable metrics.")
            with st.expander("View dbt SQL model"):
                try:
                    sql = open("dbt/models/marts/patient_care_gaps.sql").read()
                    st.code(sql, language="sql")
                except Exception:
                    st.caption("dbt/models/marts/patient_care_gaps.sql")

        # ── Data Quality tab ───────────────────────────────────────────────────
        with tab_dq:
            st.markdown("#### Data Quality Framework")
            st.caption("Validates patient data before agents process it — catches bad data early in the pipeline.")

            if st.button("▶ Run Data Quality Check", type="primary"):
                with st.spinner("Running quality checks..."):
                    report = run_data_quality_check()

                status_icon = "✅" if report["status"] == "PASS" else "❌"
                st.markdown(f"### {status_icon} {report['summary']}")

                q1, q2, q3, q4 = st.columns(4)
                q1.metric("Quality Score", f"{report['overall_quality_score']}%")
                q2.metric("Checks Run", report["total_checks"])
                q3.metric("Errors", report["total_errors"])
                q4.metric("Warnings", report["total_warnings"])

                st.markdown("---")
                failed = [r for r in report["patient_results"] if r["status"] == "FAIL"]
                if failed:
                    st.markdown("**Patients with errors:**")
                    for r in failed:
                        with st.expander(f"❌ {r['patient_id']} — score: {r['quality_score']}"):
                            for err in r["errors"]:
                                st.error(err["message"])
                            for warn in r["warnings"]:
                                st.warning(warn["message"])
                else:
                    st.success("All patients passed quality checks.")

                st.markdown("**Quality rules applied:**")
                rules = ["not_null (patient_id, age)", "age_in_range [0-120]",
                         "valid_gender (M/F/O/U)", "lab_values physiologically plausible",
                         "known_insurance_plan", "no_duplicate_patient_ids"]
                for rule in rules:
                    st.markdown(f"- `{rule}`")

    except ImportError as e:
        st.error(f"Analytics module not available: {e}")
        st.code("pip install pandas", language="bash")


# ── Page: Drug Safety ─────────────────────────────────────────────────────────

elif page == "💊 Drug Safety":
    st.title("💊 Drug Safety Analysis — Powered by OpenFDA")
    st.markdown(
        "Real-time drug interaction and safety analysis using FDA drug label data. "
        "Inspired by **TxAgent** (Harvard, arXiv 2025.3) and **MALADE** (MLHC 2024)."
    )

    patient_meds = {
        "P001": ["Metformin 1000mg", "Lisinopril 10mg", "Atorvastatin 40mg"],
        "P004": ["Furosemide 40mg", "Warfarin 5mg", "Carvedilol 12.5mg", "Insulin glargine"],
        "P005": ["Methotrexate 15mg", "Folic acid 1mg", "Calcium 1200mg"],
        "P007": ["Tamoxifen 20mg", "Lisinopril 5mg"],
        "P016": ["Levodopa/Carbidopa 25/100mg", "Sertraline 50mg", "Metoprolol 50mg"],
    }

    col1, col2 = st.columns([2, 1])
    selected = col1.selectbox("Select patient for drug safety check",
                              options=["P001", "P004", "P005", "P007", "P016"],
                              format_func=lambda p: f"{p} — {', '.join(patient_meds[p][:2])}...")

    if col2.button("🔍 Run Drug Safety Check", type="primary"):
        with st.spinner(f"Querying OpenFDA for {selected}'s medications..."):
            # Trigger via API
            result = api_post("/process-patient", {"patient_id": selected, "mode": "full"})
        if "_error" not in result:
            st.success("Drug safety check triggered — view results in Audit Log.")
        else:
            st.error(result["_error"])

    st.markdown("---")
    st.markdown("#### How the Drug Safety Agent Works")
    col_a, col_b, col_c = st.columns(3)
    with col_a:
        st.markdown("**1. OpenFDA Query**")
        st.caption("Real-time fetch of FDA drug labels for each medication")
        st.code("GET /drug/label.json\n?search=generic_name:warfarin", language="bash")
    with col_b:
        st.markdown("**2. Interaction Analysis**")
        st.caption("Extract drug_interactions, contraindications, boxed_warnings sections")
        st.code('label["drug_interactions"]\nlabel["contraindications"]\nlabel["boxed_warnings"]', language="python")
    with col_c:
        st.markdown("**3. Knowledge Graph Cross-check**")
        st.caption("Graph traversal for drug-condition contraindications")
        st.code("warfarin → NSAIDs\n[FDA: bleeding risk]", language="text")

    st.markdown("---")
    st.markdown("**High-risk patient spotlight — P004 (James Whitfield):**")
    st.markdown("""
| Drug | Risk Flag | Source |
|------|-----------|--------|
| Warfarin + NSAIDs | ⚠️ Major bleeding risk | FDA drug label |
| Metformin + CKD Stage 4 | 🔴 CONTRAINDICATED (lactic acidosis) | FDA + KDIGO |
| Furosemide | ⚠️ Monitor electrolytes | FDA drug label |
| Warfarin + Aspirin | ⚠️ GI bleed risk | ACC/AHA guideline |
""")


# ── Page: Knowledge Graph ─────────────────────────────────────────────────────

elif page == "🧠 Knowledge Graph":
    st.title("🧠 Clinical Knowledge Graph")
    st.markdown(
        "Evidence-based multi-hop clinical reasoning inspired by "
        "**SNOMED CT Knowledge Graphs** (arXiv 2025.10) and "
        "**KG4Diagnosis** (arXiv 2024.12)."
    )

    try:
        import sys
        sys.path.insert(0, ".")
        from knowledge_graph.clinical_graph import (
            get_graph_stats, find_risks_for_patient, format_findings_for_agent
        )

        stats = get_graph_stats()
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Nodes", stats["total_nodes"])
        col2.metric("Total Edges", stats["total_edges"])
        col3.metric("Node Types", len(stats["node_types"]))
        col4.metric("Edge Types", len(stats["edge_types"]))

        st.markdown("---")
        st.markdown("#### Live Patient Analysis")

        patient_demos = {
            "P001": {"diagnoses": ["Type 2 Diabetes", "Hypertension", "CKD Stage 3"],
                     "labs": {"HbA1c": 10.2, "eGFR": 45},
                     "meds": ["Metformin 1000mg", "Lisinopril 10mg"], "age": 67},
            "P004": {"diagnoses": ["Heart Failure", "Atrial Fibrillation", "Type 2 Diabetes", "CKD Stage 4"],
                     "labs": {"HbA1c": 9.1, "eGFR": 22},
                     "meds": ["Furosemide 40mg", "Warfarin 5mg", "Metformin 1000mg"], "age": 72},
            "P007": {"diagnoses": ["Breast Cancer - Stage II", "Hypertension"],
                     "labs": {}, "meds": ["Tamoxifen 20mg", "Lisinopril 5mg"], "age": 55},
            "P014": {"diagnoses": ["Stroke", "Hypertension", "Hyperlipidemia", "Type 2 Diabetes"],
                     "labs": {"HbA1c": 8.3, "eGFR": 55},
                     "meds": ["Aspirin 81mg", "Clopidogrel 75mg", "Atorvastatin 80mg"], "age": 65},
        }

        selected_p = st.selectbox("Select patient for KG analysis",
                                   options=list(patient_demos.keys()),
                                   format_func=lambda p: f"{p} — {', '.join(patient_demos[p]['diagnoses'][:2])}")

        demo = patient_demos[selected_p]
        findings = find_risks_for_patient(
            diagnoses=demo["diagnoses"],
            lab_values=demo["labs"],
            medications=demo["meds"],
            age=demo["age"]
        )

        col_a, col_b, col_c = st.columns(3)
        urgent = [f for f in findings if f.get("urgency") in ("URGENT", "HIGH")]
        interactions = [f for f in findings if f.get("type") == "drug_interaction"]
        col_a.metric("Total Connections", len(findings))
        col_b.metric("⚠️ High Priority", len(urgent))
        col_c.metric("💊 Drug Interactions", len(interactions))

        st.markdown(format_findings_for_agent(findings))

        st.markdown("---")
        st.markdown("**Graph node type breakdown:**")
        for ntype, count in sorted(stats["node_types"].items()):
            st.markdown(f"- `{ntype}`: {count} nodes")

    except ImportError as e:
        st.error(f"Knowledge graph not available: {e}\nRun: `pip install networkx`")


# ── Page: Guidelines KB ───────────────────────────────────────────────────────

elif page == "📚 Guidelines KB":
    st.title("📚 Clinical Guidelines Knowledge Base")
    st.markdown("RAG over 63 clinical guidelines from USPSTF, ADA, AHA, KDIGO, NCI, CDC and more.")

    g_data = api_get("/guidelines-status")
    collection = g_data.get("collection", {})

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Chunks", collection.get("total_chunks", 0))
    col2.metric("Sources Loaded", collection.get("total_sources", 0))
    col3.metric("Status", "✅ Ready" if collection.get("total_chunks", 0) > 0 else "⚠️ Empty")
    col4.metric("Target Sources", 63)

    if collection.get("total_chunks", 0) == 0:
        st.warning("Guidelines not loaded. Run `python rag/refresh_flow.py` first.")

    st.markdown("---")

    # Search test
    st.markdown("#### 🔍 Test Guideline Search")
    test_q = st.text_input("Search query",
                            placeholder="HbA1c target type 2 diabetes insulin pump",
                            key="rag_search")
    if test_q and st.button("Search"):
        with st.spinner("Searching guidelines..."):
            results = api_get(f"/guidelines-search?q={test_q}&n=3")
        if results.get("results"):
            for r in results["results"]:
                with st.expander(
                    f"[{r.get('relevance_score', 0):.0%}] {r['source_name']}"
                ):
                    st.markdown(f"**Source:** [{r['source_name']}]({r['url']})")
                    st.markdown(f"**Retrieved:** {r.get('scraped_at', '')}")
                    st.markdown(f"> {r['text'][:400]}...")
        else:
            st.info("No results — run `python rag/refresh_flow.py` to populate the KB.")

    st.markdown("---")
    col_a, col_b = st.columns(2)
    with col_a:
        if st.button("▶ Weekly Refresh (USPSTF + CDC)", use_container_width=True):
            r = api_post("/refresh-guidelines",
                         {"source_ids": None, "force": False, "triggered_by": "dashboard"})
            st.success(r.get("message", "Refresh started"))
    with col_b:
        if st.button("⚡ Full Refresh — All 63 Sources", use_container_width=True):
            r = api_post("/refresh-guidelines",
                         {"source_ids": None, "force": True, "triggered_by": "dashboard_manual"})
            st.success(r.get("message", "Full refresh started"))

    st.markdown("---")
    st.markdown("**Research basis:**")
    st.markdown("- MedCoAct confidence-aware RAG (arXiv 2025.10)")
    st.markdown("- Path-RAG knowledge-guided retrieval (MLHS 2025)")
    st.markdown("- RAG-KG-IL incremental knowledge graph + RAG (arXiv 2025.3)")


# ── Page: System Status ───────────────────────────────────────────────────────

elif page == "🔧 System Status":
    st.title("🔧 System Status")

    health = api_get("/health")
    if "_error" not in health:
        st.success(f"✅ API online — v{health.get('version')}")
    else:
        st.error("❌ API offline")

    st.markdown("---")
    st.markdown("### Agent Stack Status")

    checks = [
        ("LangGraph (care gap + prior auth)", "langgraph"),
        ("CrewAI (MDT/ICT supervisor)", "crewai"),
        ("OpenFDA (drug safety)", "httpx"),
        ("Knowledge Graph", "networkx"),
        ("ChromaDB (RAG store)", "chromadb"),
        ("Sentence Transformers (embedder)", "sentence_transformers"),
        ("Prefect (orchestration)", "prefect"),
        ("LangSmith (tracing)", "langsmith"),
    ]

    for name, module in checks:
        try:
            __import__(module)
            st.markdown(f"✅ **{name}**")
        except ImportError:
            st.markdown(f"❌ **{name}** — run `pip install {module.replace('_','-')}`")

    st.markdown("---")
    st.markdown("### Quick Commands")
    st.code("""
# Start API
uvicorn api.main:app --reload --port 8000

# Start dashboard
streamlit run frontend/app.py

# Run Prefect batch
python orchestration/prefect_flow.py

# Populate guidelines KB (first time, ~5 min)
python rag/refresh_flow.py

# Test knowledge graph
python knowledge_graph/clinical_graph.py

# Test drug safety agent
python agents/drug_safety_agent.py
""", language="bash")

    st.markdown("---")
    st.markdown("### Architecture & Research References")
    st.markdown("""
| Component | Pattern | Research |
|---|---|---|
| Adaptive complexity routing | Moderator + Recruiter | MDAgents, NeurIPS 2024 Oral |
| Drug safety (OpenFDA) | Tool-augmented reasoning | TxAgent, Harvard 2025 |
| Agent/Critic prior auth | Agent/Critic pattern | MALADE, MLHC 2024 |
| RAG clinical guidelines | Retrieval-augmented | MedCoAct, Path-RAG 2025 |
| Knowledge graph | Multi-hop clinical reasoning | SNOMED KG, KG4Diagnosis 2024 |
| FHIR R4 integration | Real EHR data standard | FHIR-AgentBench, arXiv 2025.9 |
| HITL escalation | Tiered agentic oversight | arXiv 2025.6 |
| Prefect orchestration | Production scheduling | — |
""")

# ── Footer ─────────────────────────────────────────────────────────────────────

st.sidebar.markdown("---")
st.sidebar.caption(f"Last refreshed: {datetime.now().strftime('%H:%M:%S')}")