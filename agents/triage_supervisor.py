"""
agents/triage_supervisor.py

CrewAI Triage Supervisor — multi-agent orchestration layer.

The supervisor:
  1. Receives a patient ID
  2. Runs risk scoring to determine what workflows are needed
  3. Delegates to the appropriate LangGraph sub-agents
  4. Aggregates results and writes to the HITL queue if confidence is low
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
from tools.risk_tools import calculate_risk_score

load_dotenv()

CONFIDENCE_THRESHOLD = float(os.getenv("CONFIDENCE_THRESHOLD", "0.75"))


# ── CrewAI Tool wrappers for LangGraph agents ─────────────────────────────────

class PriorAuthTool(BaseTool):
    name: str = "prior_auth_agent"
    description: str = (
        "Run prior authorization evaluation for a patient. "
        "Input must be a JSON string with keys: patient_id, request_id, item_name."
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
        "Run a full care gap review for a patient. "
        "Input must be a JSON string with key: patient_id."
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


class RiskScoreTool(BaseTool):
    name: str = "risk_score_tool"
    description: str = (
        "Calculate patient risk score. "
        "Input must be a JSON string with keys: patient_id, diagnoses (list), lab_values (dict)."
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


# ── CrewAI crew builder ───────────────────────────────────────────────────────

def build_supervisor_crew(patient_id: str) -> Crew:
    """Assemble a CrewAI crew to handle a patient triage workflow."""

    # Use model name string directly — CrewAI handles LLM creation internally
    model = os.getenv("MODEL_NAME", "gpt-4o")

    risk_assessor = Agent(
        role="Clinical Risk Assessor",
        goal=(
            "Evaluate patient risk level and determine which clinical workflows "
            "are needed (prior auth, care gaps, urgent outreach)."
        ),
        backstory=(
            "You are a clinical informaticist with expertise in risk stratification "
            "and care management. You use data-driven tools to make fast, accurate "
            "triage decisions."
        ),
        tools=[RiskScoreTool()],
        llm=model,
        verbose=True,
        allow_delegation=True,
    )

    auth_coordinator = Agent(
        role="Prior Authorization Coordinator",
        goal=(
            "Process all pending prior authorization requests for the patient "
            "and return structured approval/denial/escalation decisions."
        ),
        backstory=(
            "You are a prior auth specialist who knows payer policies inside out. "
            "You process auth requests quickly and accurately, escalating only "
            "when clinical judgment is truly required."
        ),
        tools=[PriorAuthTool()],
        llm=model,
        verbose=True,
    )

    care_coordinator = Agent(
        role="Care Gap Coordinator",
        goal=(
            "Identify all preventive care gaps for the patient and produce "
            "an actionable outreach plan."
        ),
        backstory=(
            "You are a care coordinator who ensures patients receive all recommended "
            "preventive screenings and follow-ups. You translate clinical guidelines "
            "into clear action items."
        ),
        tools=[CareGapTool()],
        llm=model,
        verbose=True,
    )

    risk_task = Task(
        description=(
            f"Assess the clinical risk level for patient {patient_id}. "
            f"Use the risk_score_tool to calculate their risk score. "
            f"First get their demographics and diagnoses, then score them. "
            f"Determine which additional workflows to run (prior auth, care gaps, or both)."
        ),
        expected_output=(
            "Risk tier (LOW/MEDIUM/HIGH/CRITICAL), risk score, contributing factors, "
            "and a recommendation for which workflows to trigger."
        ),
        agent=risk_assessor,
    )

    auth_task = Task(
        description=(
            f"Process all pending prior authorization requests for patient {patient_id}. "
            f"For each pending request, use the prior_auth_agent tool to evaluate it. "
            f"Return a decision (APPROVE/DENY/ESCALATE) with confidence and justification."
        ),
        expected_output=(
            "A list of prior auth decisions, each with request_id, decision, "
            "confidence score, and one-sentence justification."
        ),
        agent=auth_coordinator,
        context=[risk_task],
    )

    care_task = Task(
        description=(
            f"Run a full care gap review for patient {patient_id} using the care_gap_agent tool. "
            f"Summarize the gaps found and the recommended actions."
        ),
        expected_output=(
            "A prioritized list of care gaps with recommended interventions "
            "and an overall outreach recommendation."
        ),
        agent=care_coordinator,
        context=[risk_task],
    )

    crew = Crew(
        agents=[risk_assessor, auth_coordinator, care_coordinator],
        tasks=[risk_task, auth_task, care_task],
        process=Process.sequential,
        verbose=True,
    )

    return crew


# ── HITL queue helper ─────────────────────────────────────────────────────────

def write_to_review_queue(patient_id: str, crew_output: str, auth_results: list):
    """Write a case to the SQLite HITL review queue."""
    from sqlalchemy import create_engine, text

    db_url = os.getenv("DATABASE_URL", "sqlite:///./healthcare_agents.db")
    engine = create_engine(db_url)

    with engine.connect() as conn:
        conn.execute(text("""
            INSERT INTO review_queue
            (patient_id, created_at, status, agent_output, auth_results, reason)
            VALUES (:patient_id, :created_at, :status, :agent_output, :auth_results, :reason)
        """), {
            "patient_id": patient_id,
            "created_at": datetime.utcnow().isoformat(),
            "status": "PENDING",
            "agent_output": crew_output,
            "auth_results": json.dumps(auth_results),
            "reason": "Low confidence or ESCALATE decision from sub-agent"
        })
        conn.commit()


# ── Public interface ──────────────────────────────────────────────────────────

def run_triage(patient_id: str) -> dict:
    """
    Run the full multi-agent triage workflow for a patient.
    Coordinates risk assessment, prior auth, and care gap sub-agents.
    Writes to HITL queue if escalation is needed.
    """
    print(f"\n[Supervisor] Starting triage for patient {patient_id}")

    crew = build_supervisor_crew(patient_id)
    result = crew.kickoff()

    crew_output = str(result)
    escalate = "ESCALATE" in crew_output or "escalate" in crew_output.lower()

    if escalate:
        print(f"[Supervisor] Escalation needed for {patient_id} — writing to review queue")
        try:
            write_to_review_queue(patient_id, crew_output, [])
        except Exception as e:
            print(f"[Supervisor] Warning: could not write to review queue: {e}")

    return {
        "patient_id": patient_id,
        "workflow_completed": True,
        "escalation_triggered": escalate,
        "crew_output": crew_output,
        "status": "PENDING_REVIEW" if escalate else "COMPLETED"
    }


# ── CLI test ──────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    result = run_triage("P001")
    print("\n=== Triage Result ===")
    print(f"Status: {result['status']}")
    print(f"Escalation triggered: {result['escalation_triggered']}")
    print(f"\nCrew output (truncated):\n{result['crew_output'][:500]}...")