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


# ── CrewAI tool wrappers ──────────────────────────────────────────────────────

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
        "Input: JSON with patient_id, medications (list), diagnoses (list)."
    )

    def _run(self, input_str: str) -> str:
        try:
            data = json.loads(input_str)
            from agents.drug_safety_agent import run_drug_safety_check
            result = run_drug_safety_check(
                patient_id=data["patient_id"],
                medications=data.get("medications", []),
                diagnoses=data.get("diagnoses", [])
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

    return results


def _run_moderate_complexity(patient_id: str) -> dict:
    """MODERATE pathway: standard CrewAI 3-agent team."""
    print(f"[Supervisor] MODERATE pathway: CrewAI MDT for {patient_id}")
    model = os.getenv("MODEL_NAME", "gpt-4o-mini")

    risk_assessor = Agent(
        role="Clinical Risk Assessor",
        goal="Evaluate patient risk level and identify priority workflows.",
        backstory="Clinical informaticist specializing in risk stratification.",
        tools=[RiskScoreTool(), KnowledgeGraphTool()],
        llm=model, verbose=False
    )
    auth_coordinator = Agent(
        role="Prior Authorization Coordinator",
        goal="Process all pending prior authorization requests.",
        backstory="Prior auth specialist with deep payer policy knowledge.",
        tools=[PriorAuthTool()],
        llm=model, verbose=False
    )
    care_coordinator = Agent(
        role="Care Gap Coordinator",
        goal="Identify preventive care gaps with guideline citations.",
        backstory="Care coordinator translating guidelines into action items.",
        tools=[CareGapTool()],
        llm=model, verbose=False
    )

    crew = Crew(
        agents=[risk_assessor, auth_coordinator, care_coordinator],
        tasks=[
            Task(
                description=f"Score risk and identify priority workflows for patient {patient_id}.",
                expected_output="Risk tier, contributing factors, workflow recommendation.",
                agent=risk_assessor
            ),
            Task(
                description=f"Process prior auth requests for patient {patient_id}.",
                expected_output="Auth decisions with confidence and justification.",
                agent=auth_coordinator
            ),
            Task(
                description=f"Care gap review for patient {patient_id}.",
                expected_output="Prioritized care gaps with guideline citations.",
                agent=care_coordinator
            )
        ],
        process=Process.sequential,
        verbose=False
    )
    result = crew.kickoff()
    return {"pathway": "MODERATE", "workflow": "crewai_mdt", "crew_output": str(result)}


def _run_high_complexity(patient_id: str) -> dict:
    """HIGH pathway: full ICT with drug safety + knowledge graph specialists."""
    print(f"[Supervisor] HIGH pathway: Full ICT for {patient_id}")
    model = os.getenv("MODEL_NAME", "gpt-4o-mini")

    risk_assessor = Agent(
        role="Clinical Risk Assessor",
        goal="Deep risk analysis using knowledge graph and lab values.",
        backstory="Senior clinical informaticist with complex case expertise.",
        tools=[RiskScoreTool(), KnowledgeGraphTool()],
        llm=model, verbose=False
    )
    drug_safety_specialist = Agent(
        role="Clinical Pharmacist",
        goal="Identify all drug interactions, contraindications, and medication safety issues.",
        backstory="Clinical pharmacist with FDA drug safety expertise.",
        tools=[DrugSafetyTool()],
        llm=model, verbose=False
    )
    auth_coordinator = Agent(
        role="Senior Prior Authorization Specialist",
        goal="Process all auth requests with full clinical evidence review.",
        backstory="Senior auth specialist handling complex multi-system cases.",
        tools=[PriorAuthTool()],
        llm=model, verbose=False
    )
    care_coordinator = Agent(
        role="Complex Care Coordinator",
        goal="Identify care gaps and specialist referrals for complex patients.",
        backstory="Care coordinator specializing in multi-morbidity management.",
        tools=[CareGapTool(), KnowledgeGraphTool()],
        llm=model, verbose=False
    )

    crew = Crew(
        agents=[risk_assessor, drug_safety_specialist, auth_coordinator, care_coordinator],
        tasks=[
            Task(
                description=f"Deep risk and knowledge graph analysis for patient {patient_id}.",
                expected_output="Risk tier, multi-hop clinical connections, urgent flags.",
                agent=risk_assessor
            ),
            Task(
                description=f"Full drug safety review for patient {patient_id} using OpenFDA.",
                expected_output="Safety tier, drug interactions, contraindications with FDA citations.",
                agent=drug_safety_specialist
            ),
            Task(
                description=f"Process all prior auth requests for patient {patient_id}.",
                expected_output="Auth decisions with Agent/Critic review results.",
                agent=auth_coordinator
            ),
            Task(
                description=f"Comprehensive care gap and specialist referral review for patient {patient_id}.",
                expected_output="Care gaps with guideline citations and specialist referrals.",
                agent=care_coordinator
            )
        ],
        process=Process.sequential,
        verbose=False
    )
    result = crew.kickoff()
    return {"pathway": "HIGH", "workflow": "crewai_ict_full", "crew_output": str(result)}


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
            "output": crew_output[:2000],
            "reason": "Agent/Critic review flagged escalation"
        })
        conn.commit()


if __name__ == "__main__":
    result = run_triage("P001")
    print(f"\n=== Triage Result ===")
    print(f"Complexity: {result['complexity_tier']} (score={result['complexity_score']})")
    print(f"Status: {result['status']}")
    print(f"Rationale: {result['complexity_rationale']}")