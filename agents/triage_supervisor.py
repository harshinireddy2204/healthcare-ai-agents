"""
agents/triage_supervisor.py

CrewAI Triage Supervisor with MDAgents-inspired adaptive complexity routing.

Workflow:
  1. Complexity router scores the patient (LOW / MODERATE / HIGH)
  2. Routes to the appropriate agent pathway:
     LOW      → Individual LangGraph agents (fast, cheap)
     MODERATE → Standard CrewAI 3-agent team
     HIGH     → Full CrewAI ICT + drug safety + knowledge graph
  3. Aggregates results, writes to HITL queue if needed

This saves ~60% of tokens on routine LOW cases while ensuring HIGH cases
get full multi-specialist analysis — directly matching MDAgents design.
"""
import os
import json
from datetime import datetime
from typing import Optional

from dotenv import load_dotenv
from crewai import Agent, Task, Crew, Process
from crewai.tools import BaseTool

from agents.prior_auth_agent import run_prior_auth
from agents.care_gap_agent import run_care_gap_review
from tools.ehr_tools import EHR_TOOLS
from tools.risk_tools import calculate_risk_score

load_dotenv()

CONFIDENCE_THRESHOLD = float(os.getenv("CONFIDENCE_THRESHOLD", "0.75"))


# ── Timeout helper ────────────────────────────────────────────────────────────

def _run_with_timeout(fn, args=(), timeout_seconds=90, fallback=None):
    """
    Run fn(*args) with a hard timeout using a thread.
    Returns fallback dict if timeout is exceeded.
    This prevents single LLM calls from hanging the entire workflow.
    """
    import threading
    result_holder = [fallback]
    exception_holder = [None]

    def target():
        try:
            result_holder[0] = fn(*args)
        except Exception as e:
            exception_holder[0] = e

    t = threading.Thread(target=target, daemon=True)
    t.start()
    t.join(timeout=timeout_seconds)

    if t.is_alive():
        print(f"  [Timeout] {fn.__name__} exceeded {timeout_seconds}s — using fallback")
        return fallback

    if exception_holder[0]:
        raise exception_holder[0]

    return result_holder[0]



class PriorAuthTool(BaseTool):
    name: str = "prior_auth_agent"
    description: str = (
        "Run prior authorization evaluation with Agent/Critic review. "
        "Input: JSON with patient_id, request_id, item_name."
    )

    def _run(self, input_str: str) -> str:
        try:
            data = json.loads(input_str)
            result = run_prior_auth(
                patient_id=data["patient_id"],
                request_id=data["request_id"],
                item_name=data["item_name"]
            )
            return json.dumps(result)
        except Exception as e:
            return json.dumps({"error": str(e), "decision": "ESCALATE", "confidence": 0.0})


class CareGapTool(BaseTool):
    name: str = "care_gap_agent"
    description: str = (
        "Run care gap review with RAG-grounded guideline citations. "
        "Input: JSON with patient_id."
    )

    def _run(self, input_str: str) -> str:
        try:
            data = json.loads(input_str)
            result = run_care_gap_review(patient_id=data["patient_id"])
            return json.dumps({
                "patient_id": result["patient_id"],
                "steps_executed": result["steps_executed"],
                "summary": result["final_report"][:1000]
            })
        except Exception as e:
            return json.dumps({"error": str(e)})


class DrugSafetyTool(BaseTool):
    name: str = "drug_safety_agent"
    description: str = (
        "Run real-time OpenFDA drug safety check for a patient. "
        "Input: JSON with patient_id. Optionally include medications (list) and diagnoses (list) "
        "— if omitted, they are fetched automatically from patient records."
    )

    def _run(self, input_str: str) -> str:
        try:
            data = json.loads(input_str)
            patient_id = data["patient_id"]

            # Auto-fetch medications and diagnoses if not provided
            medications = data.get("medications", [])
            diagnoses = data.get("diagnoses", [])

            if not medications or not diagnoses:
                from tools.ehr_tools import get_patient_demographics
                demo = get_patient_demographics.invoke({"patient_id": patient_id})
                if not medications:
                    medications = demo.get("medications", [])
                if not diagnoses:
                    diagnoses = demo.get("diagnoses", [])

            from agents.drug_safety_agent import run_drug_safety_check
            result = run_drug_safety_check(
                patient_id=patient_id,
                medications=medications,
                diagnoses=diagnoses
            )
            return json.dumps({
                "patient_id": result["patient_id"],
                "safety_tier": result["safety_tier"],
                "fda_findings_count": result["fda_findings_count"],
                "report_summary": result["safety_report"][:800]
            })
        except Exception as e:
            return json.dumps({"error": str(e), "safety_tier": "CAUTION"})


class KnowledgeGraphTool(BaseTool):
    name: str = "knowledge_graph_analysis"
    description: str = (
        "Analyze clinical connections using evidence-based knowledge graph. "
        "Input: JSON with patient_id, diagnoses (list), medications (list), "
        "lab_values (dict), age (int)."
    )

    def _run(self, input_str: str) -> str:
        try:
            data = json.loads(input_str)
            from knowledge_graph.clinical_graph import find_risks_for_patient, format_findings_for_agent
            findings = find_risks_for_patient(
                diagnoses=data.get("diagnoses", []),
                lab_values=data.get("lab_values", {}),
                medications=data.get("medications", []),
                age=data.get("age", 0)
            )
            return format_findings_for_agent(findings)
        except Exception as e:
            return json.dumps({"error": str(e)})


class RiskScoreTool(BaseTool):
    name: str = "risk_score_tool"
    description: str = (
        "Calculate patient risk score. "
        "Input: JSON with patient_id, diagnoses (list), lab_values (dict)."
    )

    def _run(self, input_str: str) -> str:
        try:
            data = json.loads(input_str)
            result = calculate_risk_score.invoke({
                "patient_id": data["patient_id"],
                "diagnoses": data["diagnoses"],
                "lab_values": data.get("lab_values", {})
            })
            return json.dumps(result)
        except Exception as e:
            return json.dumps({"error": str(e)})


# ── Routing helpers ───────────────────────────────────────────────────────────

def _get_patient_profile(patient_id: str) -> dict:
    """Fetch patient data needed for complexity routing."""
    from tools.ehr_tools import get_patient_demographics, get_lab_results, get_pending_auth_requests
    try:
        demo = get_patient_demographics.invoke({"patient_id": patient_id})
        pending = get_pending_auth_requests.invoke({"patient_id": patient_id})
        labs = {}
        for lab in ["HbA1c", "eGFR", "LDL"]:
            r = get_lab_results.invoke({"patient_id": patient_id, "lab_name": lab})
            if r.get("value") is not None:
                labs[lab] = r["value"]
        return {
            "diagnoses": demo.get("diagnoses", []),
            "medications": demo.get("medications", []),
            "labs": labs,
            "pending": pending if isinstance(pending, list) else []
        }
    except Exception as e:
        print(f"[Supervisor] Warning: could not fetch patient profile: {e}")
        return {"diagnoses": [], "medications": [], "labs": {}, "pending": []}


def _run_low_complexity(patient_id: str, profile: dict) -> dict:
    """
    LOW complexity pathway: individual LangGraph agents, no CrewAI overhead.
    Fast (~30s), cheap, accurate for routine cases.
    """
    print(f"[Supervisor] LOW pathway: running individual agents for {patient_id}")
    results = {"pathway": "LOW", "workflow": "individual_langgraph"}

    # Care gap review
    try:
        care_result = run_care_gap_review(patient_id)
        results["care_gaps"] = {
            "steps_executed": care_result["steps_executed"],
            "final_report": care_result["final_report"]
        }
    except Exception as e:
        results["care_gaps"] = {"error": str(e)}

    # Auth requests if any
    if profile["pending"]:
        auth_results = []
        for req in profile["pending"][:2]:
            if isinstance(req, dict) and "request_id" in req:
                try:
                    r = run_prior_auth(patient_id, req["request_id"], req["item"])
                    auth_results.append(r)
                except Exception as e:
                    auth_results.append({"error": str(e)})
        results["auth_results"] = auth_results

    # Build formatted crew_output so the frontend renders the same as MODERATE/HIGH
    import json as _json
    care_section = results.get("care_gaps", {})
    care_text = care_section.get("final_report", care_section.get("error", "Not available"))
    auth_text = _json.dumps(results.get("auth_results", []), indent=2) if results.get("auth_results") else "No pending authorization requests for this patient."
    results["crew_output"] = (
        f"LOW pathway completed for patient {patient_id}.\n\n"
        f"CARE GAPS:\n{care_text}\n\n"
        f"PRIOR AUTH:\n{auth_text}"
    )

    return results



def _run_moderate_complexity(patient_id: str) -> dict:
    """
    MODERATE pathway: pre-computed sequential approach.

    Same pattern as HIGH but without drug safety analysis.
    Runs care gap + prior auth as direct LangGraph calls (no CrewAI tool loops),
    then uses KG for risk summary. No concurrent agents = no rate limit saturation.
    """
    import time
    print(f"[Supervisor] MODERATE pathway: sequential MDT for {patient_id}")

    # Step 1: Knowledge graph risk analysis (zero LLM tokens)
    kg_summary = ""
    try:
        from tools.ehr_tools import get_patient_demographics, get_lab_results
        from knowledge_graph.clinical_graph import find_risks_for_patient, format_findings_for_agent
        demo = get_patient_demographics.invoke({"patient_id": patient_id})
        meds = demo.get("medications", [])
        dxs  = demo.get("diagnoses", [])
        age  = demo.get("age", 0)
        findings = find_risks_for_patient(diagnoses=dxs, lab_values={},
                                          medications=meds, age=age)
        kg_summary = format_findings_for_agent(findings, max_findings=6)
        print(f"  [MODERATE] KG: {len(findings)} findings")
    except Exception as e:
        print(f"  [MODERATE] KG error: {e}")

    # Step 2: Care gap via LangGraph
    print(f"  [MODERATE] Running care gap agent...")
    care_summary = ""
    try:
        from agents.care_gap_agent import run_care_gap_review
        time.sleep(2)
        care_result = _run_with_timeout(
            run_care_gap_review, args=(patient_id,), timeout_seconds=180,
            fallback={"final_report": "Care gap timed out.", "steps_executed": 0}
        )
        care_summary = care_result.get("final_report", "")
        print(f"  [MODERATE] Care gap: {care_result.get('steps_executed', 0)} steps")
    except Exception as e:
        print(f"  [MODERATE] Care gap error: {e}")

    # Wait 60s between care gap and prior auth so the TPM window partially refills.
    # Care gap consumes ~190k/200k tokens; prior auth needs ~7-10k per call.
    # Without this pause, prior auth hits 429 on every attempt and exhausts its
    # 300s timeout before the graph can complete.
    print(f"  [MODERATE] Pausing 60s to allow TPM bucket to refill before auth...")
    time.sleep(60)

    # Step 4: Prior auth via LangGraph (with Agent/Critic)
    # Note: rate limits are now handled automatically by llm_invoke retry logic.
    print(f"  [MODERATE] Running prior auth agent...")
    auth_results = []
    try:
        from agents.prior_auth_agent import run_prior_auth
        from tools.ehr_tools import get_pending_auth_requests
        pending = get_pending_auth_requests.invoke({"patient_id": patient_id})
        for req in (pending or [])[:2]:
            if isinstance(req, dict) and "request_id" in req:
                # Retry loop specifically for rate limit errors on auth start
                auth_r = None
                for attempt in range(3):
                    try:
                        auth_r = _run_with_timeout(
                            run_prior_auth,
                            args=(patient_id, req["request_id"], req["item"]),
                            timeout_seconds=480,
                            fallback={
                                "patient_id": patient_id,
                                "request_id": req["request_id"],
                                "item": req["item"],
                                "decision": "ESCALATE",
                                "confidence": 0.5,
                                "justification": "Auth timed out — escalating for manual review",
                                "critic_reviewed": False,
                                "was_revised": False
                            }
                        )
                        break
                    except Exception as e:
                        if "429" in str(e) or "rate_limit" in str(e).lower():
                            wait = 20 * (attempt + 1)
                            print(f"  [MODERATE] Auth 429 — waiting {wait}s (attempt {attempt+1}/3)")
                            time.sleep(wait)
                        else:
                            raise
                if auth_r:
                    auth_results.append(auth_r)
                    print(f"  [MODERATE] Auth: {auth_r.get('decision')} ({auth_r.get('confidence',0):.0%})")
                time.sleep(5)
    except Exception as e:
        print(f"  [MODERATE] Auth error: {e}")

    import json as _json
    auth_text = _json.dumps(auth_results, indent=2)[:2000] if auth_results else "No pending authorization requests for this patient."
    crew_output = (
        f"MODERATE pathway completed for patient {patient_id}.\n\n"
        f"KNOWLEDGE GRAPH:\n{kg_summary[:800]}\n\n"
        f"CARE GAPS:\n{care_summary[:4000]}\n\n"
        f"PRIOR AUTH:\n{auth_text}"
    )

    return {
        "pathway": "MODERATE",
        "workflow": "sequential_mdt",
        "auth_results": auth_results,
        "care_summary": care_summary,
        "crew_output": crew_output
    }


def _run_high_complexity(patient_id: str) -> dict:
    """
    HIGH pathway: pre-compute drug safety + KG outside CrewAI (zero LLM tokens),
    then run care gap + prior auth as sequential LangGraph agents,
    finally synthesize with a single lean CrewAI agent.

    This reduces token usage ~70% vs running 4 concurrent CrewAI agents.
    Drug safety (OpenFDA) and knowledge graph traversal are deterministic —
    they need no LLM calls at all.
    """
    import time
    import json as _json
    print(f"[Supervisor] HIGH pathway: Full ICT for {patient_id}")
    model = os.getenv("MODEL_NAME", "gpt-4o-mini")
    drug_result = {}
    demo = {}
    meds = []
    dxs = []

    # ── Step 1: Pre-compute deterministic work (zero LLM tokens) ─────────────
    print(f"  [HIGH] Step 1/3: Drug safety + knowledge graph (no LLM)...")

    drug_summary = "Drug safety: not available"
    drug_safety_tier = "CAUTION"
    try:
        from agents.drug_safety_agent import run_drug_safety_check
        from tools.ehr_tools import get_patient_demographics
        demo = get_patient_demographics.invoke({"patient_id": patient_id})
        meds = demo.get("medications", [])
        dxs  = demo.get("diagnoses", [])
        drug_result = run_drug_safety_check(patient_id, meds, dxs)
        drug_safety_tier = drug_result["safety_tier"]
        drug_summary = (
            f"Drug Safety Tier: {drug_result['safety_tier']}\n"
            f"FDA findings: {drug_result['fda_findings_count']}\n"
            f"Report: {drug_result['safety_report'][:500]}"
        )
        print(f"  [HIGH] Drug safety: {drug_safety_tier}")
    except Exception as e:
        print(f"  [HIGH] Drug safety error: {e}")

    kg_summary = "Knowledge graph: not available"
    try:
        from knowledge_graph.clinical_graph import find_risks_for_patient, format_findings_for_agent
        findings = find_risks_for_patient(
            diagnoses=dxs, lab_values={}, medications=meds,
            age=demo.get("age", 0) if demo else 0
        )
        kg_summary = format_findings_for_agent(findings, max_findings=8)
        print(f"  [HIGH] KG: {len(findings)} findings")
    except Exception as e:
        print(f"  [HIGH] KG error: {e}")

    # ── Step 2: Care gap via LangGraph ────────────────────────────────────────
    print(f"  [HIGH] Step 2/3: Care gap agent...")
    care_summary = "Care gap analysis: not available"
    try:
        from agents.care_gap_agent import run_care_gap_review
        time.sleep(3)
        care_result = _run_with_timeout(
            run_care_gap_review, args=(patient_id,), timeout_seconds=180,
            fallback={"final_report": "Care gap timed out — check audit log.", "steps_executed": 0}
        )
        care_summary = care_result.get("final_report", "")
        print(f"  [HIGH] Care gap: {care_result.get('steps_executed', 0)} steps")
    except Exception as e:
        print(f"  [HIGH] Care gap error: {e}")

    # ── Step 3: Prior auth via LangGraph (Agent/Critic) ───────────────────────
    print(f"  [HIGH] Step 3/3: Prior auth agent...")
    auth_results = []
    try:
        from agents.prior_auth_agent import run_prior_auth
        from tools.ehr_tools import get_pending_auth_requests
        pending = get_pending_auth_requests.invoke({"patient_id": patient_id})
        # Same TPM refill pause as MODERATE — care gap exhausts the token budget
        print(f"  [HIGH] Pausing 60s to allow TPM bucket to refill before auth...")
        time.sleep(60)
        for req in (pending or [])[:2]:
            if isinstance(req, dict) and "request_id" in req:
                auth_r = _run_with_timeout(
                    run_prior_auth,
                    args=(patient_id, req["request_id"], req["item"]),
                    timeout_seconds=480,
                    fallback={
                        "patient_id": patient_id,
                        "request_id": req["request_id"],
                        "item": req["item"],
                        "decision": "ESCALATE",
                        "confidence": 0.5,
                        "justification": "Auth timed out — escalating for manual review",
                        "critic_reviewed": False,
                        "was_revised": False
                    }
                )
                auth_results.append(auth_r)
                print(f"  [HIGH] Auth: {auth_r.get('decision')} ({auth_r.get('confidence',0):.0%})")
                time.sleep(3)
    except Exception as e:
        print(f"  [HIGH] Auth error: {e}")

    # ── Step 4: Lean CrewAI synthesis ─────────────────────────────────────────
    print(f"  [HIGH] Synthesizing...")
    auth_summary = _json.dumps(auth_results, indent=2)[:2000] if auth_results else "No pending authorization requests for this patient."
    context = (
        f"Pre-computed analysis for patient {patient_id}:\n\n"
        f"DRUG SAFETY:\n{drug_summary}\n\n"
        f"KNOWLEDGE GRAPH:\n{kg_summary[:800]}\n\n"
        f"CARE GAPS:\n{care_summary[:4000]}\n\n"
        f"PRIOR AUTH:\n{auth_summary}"
    )

    synthesizer = Agent(
        role="Senior Clinical Operations Specialist",
        goal="Synthesize all clinical findings into a prioritized action plan.",
        backstory="Senior clinician synthesizing multi-agent findings for the care team.",
        tools=[],
        llm=model, verbose=False
    )
    crew = Crew(
        agents=[synthesizer],
        tasks=[Task(
            description=(
                f"Synthesize this pre-computed analysis for patient {patient_id} into:\n"
                f"1. Top 3 urgent findings\n2. Medication safety alerts\n"
                f"3. Prior auth status\n4. Recommended referrals\n\n{context}"
            ),
            expected_output="Concise clinical action plan with urgent findings and recommendations.",
            agent=synthesizer
        )],
        process=Process.sequential,
        verbose=False
    )
    time.sleep(5)
    crew_result = crew.kickoff()

    # Build crew_output in the same structured format as MODERATE so the
    # frontend render_crew_output() can parse sections consistently.
    high_auth_text = _json.dumps(auth_results, indent=2)[:2000] if auth_results else "No pending authorization requests for this patient."
    crew_output = (
        f"HIGH pathway completed for patient {patient_id}.\n\n"
        f"KNOWLEDGE GRAPH:\n{kg_summary[:800]}\n\n"
        f"CARE GAPS:\n{care_summary[:4000]}\n\n"
        f"PRIOR AUTH:\n{high_auth_text}\n\n"
        f"SYNTHESIS:\n{str(crew_result)}"
    )

    return {
        "pathway": "HIGH",
        "workflow": "crewai_ict_precomputed",
        "drug_safety": {
            "safety_tier": drug_safety_tier,
            "fda_findings_count": drug_result.get("fda_findings_count", 0),
            "kg_findings_count": drug_result.get("kg_findings_count", 0),
        },
        "auth_results": auth_results,
        "care_summary": care_summary,
        "crew_output": crew_output
    }



# ── Public interface ──────────────────────────────────────────────────────────

def run_triage(patient_id: str) -> dict:
    """
    Main triage entry point with adaptive complexity routing.

    Automatically selects LOW/MODERATE/HIGH pathway based on patient profile.
    This mirrors MDAgents' Moderator + Recruiter pattern.
    """
    print(f"\n[Supervisor] Starting adaptive triage for {patient_id}")

    # Step 1: Get patient profile for routing
    profile = _get_patient_profile(patient_id)

    # Step 2: Score complexity
    from agents.complexity_router import route_patient, COMPLEXITY_LOW, COMPLEXITY_HIGH

    routing = route_patient(
        patient_id=patient_id,
        diagnoses=profile["diagnoses"],
        lab_values=profile["labs"],
        medications=profile["medications"],
        pending_requests=profile["pending"]
    )

    tier = routing["complexity_tier"]
    print(f"[Supervisor] Complexity: {tier} (score={routing['complexity_score']})")

    # Step 3: Route to appropriate pathway
    try:
        if tier == COMPLEXITY_LOW:
            results = _run_low_complexity(patient_id, profile)
        elif tier == COMPLEXITY_HIGH:
            results = _run_high_complexity(patient_id)
        else:
            results = _run_moderate_complexity(patient_id)
    except Exception as e:
        print(f"[Supervisor] ERROR in {tier} pathway: {e}")
        # Fallback to simple pathway
        results = _run_low_complexity(patient_id, profile)
        results["fallback"] = True

    # Step 4: Check for escalation
    crew_output = results.get("crew_output", json.dumps(results))
    escalate = ("ESCALATE" in crew_output or
                any(r.get("decision") == "ESCALATE"
                    for r in results.get("auth_results", [])))

    if escalate:
        print(f"[Supervisor] Escalation triggered for {patient_id}")
        try:
            _write_to_review_queue(patient_id, crew_output)
        except Exception as e:
            print(f"[Supervisor] Warning: could not write review queue: {e}")

    return {
        "patient_id": patient_id,
        "complexity_tier": tier,
        "complexity_score": routing["complexity_score"],
        "complexity_rationale": routing["rationale"],
        "workflow_completed": True,
        "escalation_triggered": escalate,
        "crew_output": crew_output,
        "workflow_results": results,
        "status": "PENDING_REVIEW" if escalate else "COMPLETED"
    }


def _write_to_review_queue(patient_id: str, crew_output: str):
    from sqlalchemy import create_engine, text
    db_url = os.getenv("DATABASE_URL", "sqlite:///./healthcare_agents.db")
    engine = create_engine(db_url)
    with engine.connect() as conn:
        conn.execute(text("""
            INSERT INTO review_queue
            (patient_id, created_at, status, agent_output, auth_results, reason)
            VALUES (:pid, :ts, 'PENDING', :output, '[]', :reason)
        """), {
            "pid": patient_id,
            "ts": datetime.utcnow().isoformat(),
            "output": crew_output[:6000],
            "reason": "Agent/Critic review flagged escalation"
        })
        conn.commit()


if __name__ == "__main__":
    result = run_triage("P001")
    print(f"\n=== Triage Result ===")
    print(f"Complexity: {result['complexity_tier']} (score={result['complexity_score']})")
    print(f"Status: {result['status']}")
    print(f"Rationale: {result['complexity_rationale']}")