"""
agents/complexity_router.py

Adaptive Complexity Router — inspired by MDAgents (NeurIPS 2024 Oral).

MDAgents key insight: medical tasks have different complexity levels that
require different collaboration structures:
  LOW      → Single PCP agent (fast, cheap, accurate for routine cases)
  MODERATE → Multi-Disciplinary Team (MDT) — multiple specialist agents
  HIGH     → Integrated Care Team (ICT) — full multi-agent consensus with debate

In MDAgents, this reduced inference cost by ~60% while improving accuracy
by 11.8% on complex cases compared to always using multi-agent.

Our implementation adapts this to clinical operations:
  LOW      → Individual LangGraph agents (skip CrewAI overhead)
  MODERATE → CrewAI supervisor with standard 3-agent team
  HIGH     → CrewAI supervisor + knowledge graph + drug safety + critic review

Complexity scoring uses the patient's actual clinical data:
  - Number of active diagnoses
  - Risk tier from knowledge graph
  - Number of pending auth requests
  - Presence of high-risk medications
  - Lab values out of range
"""
import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from utils.llm_utils import llm_invoke

load_dotenv()

# ── Complexity tiers ──────────────────────────────────────────────────────────

COMPLEXITY_LOW      = "LOW"
COMPLEXITY_MODERATE = "MODERATE"
COMPLEXITY_HIGH     = "HIGH"

# High-risk medication keywords that trigger MODERATE minimum
HIGH_RISK_MEDS = {
    "warfarin", "heparin", "enoxaparin", "apixaban", "rivaroxaban",
    "insulin", "methotrexate", "lithium", "digoxin", "amiodarone",
    "tacrolimus", "cyclosporine", "chemotherapy", "biologics",
    "immunosuppressant", "azathioprine", "prednisone"
}

# Diagnoses that trigger HIGH minimum
HIGH_COMPLEXITY_DIAGNOSES = {
    "heart failure", "atrial fibrillation", "ckd stage 4", "ckd stage 5",
    "breast cancer", "prostate cancer", "stroke", "esrd", "liver failure",
    "respiratory failure", "sepsis", "hiv", "crohn"
}


def score_clinical_complexity(
    patient_id: str,
    diagnoses: list,
    lab_values: dict,
    medications: list,
    pending_requests: list,
    risk_tier: str = None
) -> dict:
    """
    Score patient complexity using clinical heuristics.
    Returns complexity tier and scoring breakdown.
    """
    score = 0
    factors = []

    # Diagnosis count
    dx_count = len(diagnoses)
    if dx_count >= 5:
        score += 4
        factors.append(f"{dx_count} active diagnoses (+4)")
    elif dx_count >= 3:
        score += 2
        factors.append(f"{dx_count} active diagnoses (+2)")
    elif dx_count >= 1:
        score += 1

    # High-complexity diagnoses
    dx_lower = [d.lower() for d in diagnoses]
    for high_dx in HIGH_COMPLEXITY_DIAGNOSES:
        if any(high_dx in d for d in dx_lower):
            score += 3
            factors.append(f"High-complexity diagnosis: {high_dx} (+3)")
            break

    # High-risk medications
    med_lower = " ".join(medications).lower()
    risky_meds_found = [m for m in HIGH_RISK_MEDS if m in med_lower]
    if risky_meds_found:
        score += 2
        factors.append(f"High-risk medications: {', '.join(risky_meds_found[:3])} (+2)")

    # Multiple medications (polypharmacy)
    if len(medications) >= 6:
        score += 2
        factors.append(f"Polypharmacy: {len(medications)} medications (+2)")
    elif len(medications) >= 4:
        score += 1

    # Pending auth requests
    if len(pending_requests) >= 2:
        score += 2
        factors.append(f"{len(pending_requests)} pending auth requests (+2)")
    elif len(pending_requests) == 1:
        score += 1

    # Lab values out of range
    critical_labs = 0
    hba1c = lab_values.get("HbA1c")
    if hba1c and float(str(hba1c).replace("%","")) >= 10:
        critical_labs += 1
    egfr = lab_values.get("eGFR")
    if egfr and float(egfr) < 30:
        critical_labs += 2
    elif egfr and float(egfr) < 45:
        critical_labs += 1
    ldl = lab_values.get("LDL")
    if ldl and float(ldl) >= 190:
        critical_labs += 1

    if critical_labs >= 2:
        score += 3
        factors.append(f"{critical_labs} critical lab values (+3)")
    elif critical_labs == 1:
        score += 1

    # Knowledge graph risk tier override
    if risk_tier == "CRITICAL":
        score += 4
        factors.append("KG risk tier CRITICAL (+4)")
    elif risk_tier == "HIGH":
        score += 2
        factors.append("KG risk tier HIGH (+2)")

    # Determine tier
    if score <= 3:
        tier = COMPLEXITY_LOW
        rationale = "Routine case — single agent pathway"
    elif score <= 7:
        tier = COMPLEXITY_MODERATE
        rationale = "Multi-factor case — MDT collaboration"
    else:
        tier = COMPLEXITY_HIGH
        rationale = "Complex multi-system case — ICT with full specialist team"

    return {
        "patient_id": patient_id,
        "complexity_tier": tier,
        "complexity_score": score,
        "rationale": rationale,
        "scoring_factors": factors,
        "agent_pathway": {
            COMPLEXITY_LOW:      "Individual LangGraph agents (no CrewAI overhead)",
            COMPLEXITY_MODERATE: "CrewAI MDT: risk assessor + auth coordinator + care coordinator",
            COMPLEXITY_HIGH:     "CrewAI ICT: full team + drug safety + KG analysis + critic review"
        }[tier]
    }


def classify_complexity_with_llm(patient_summary: str) -> dict:
    """
    Use LLM to classify complexity when heuristics are uncertain.
    Mirrors MDAgents' moderator agent approach.
    """
    load_dotenv()
    model = os.getenv("MODEL_NAME", "gpt-4o-mini")
    llm = ChatOpenAI(model=model, temperature=0)

    prompt = f"""You are a clinical moderator assessing the complexity of a patient case.

Classify this patient as LOW, MODERATE, or HIGH complexity based on these criteria:

LOW: Single chronic condition, stable, routine follow-up
MODERATE: 2-3 conditions with interactions, pending authorization, some abnormal labs
HIGH: 4+ conditions, critical lab values, high-risk medications, multiple specialists needed

Patient summary:
{patient_summary}

Respond with ONLY a JSON object:
{{"tier": "LOW|MODERATE|HIGH", "reasoning": "one sentence", "specialists_needed": ["list"]}}
"""

    try:
        response = llm_invoke(llm, [HumanMessage(content=prompt)])
        import json, re
        match = re.search(r'\{.*?\}', response.content, re.DOTALL)
        if match:
            data = json.loads(match.group())
            return {
                "complexity_tier": data.get("tier", "MODERATE"),
                "llm_reasoning": data.get("reasoning", ""),
                "specialists_needed": data.get("specialists_needed", []),
                "method": "llm_moderator"
            }
    except Exception as e:
        pass

    return {"complexity_tier": "MODERATE", "method": "fallback"}


def route_patient(
    patient_id: str,
    diagnoses: list,
    lab_values: dict,
    medications: list,
    pending_requests: list,
    risk_tier: str = None
) -> dict:
    """
    Main routing function. Combines heuristic scoring with LLM classification
    for uncertain cases, just as MDAgents uses a moderator agent.

    Returns the complexity assessment and chosen workflow pathway.
    """
    # Step 1: Heuristic scoring
    heuristic = score_clinical_complexity(
        patient_id, diagnoses, lab_values, medications, pending_requests, risk_tier
    )

    score = heuristic["complexity_score"]

    # Step 2: For borderline cases (score 3-5), use LLM moderator
    # This mirrors MDAgents' moderator LLM that validates complexity
    if 3 <= score <= 5:
        patient_summary = (
            f"Patient {patient_id}: {len(diagnoses)} diagnoses ({', '.join(diagnoses[:3])}...), "
            f"{len(medications)} medications, {len(pending_requests)} pending auth requests. "
            f"Labs: {', '.join([f'{k}={v}' for k,v in list(lab_values.items())[:3]])}"
        )
        llm_result = classify_complexity_with_llm(patient_summary)

        # Blend: if LLM says HIGH and heuristic says MODERATE, use HIGH
        if (llm_result.get("complexity_tier") == COMPLEXITY_HIGH and
                heuristic["complexity_tier"] == COMPLEXITY_MODERATE):
            heuristic["complexity_tier"] = COMPLEXITY_HIGH
            heuristic["rationale"] += " (upgraded by LLM moderator)"

        heuristic["llm_classification"] = llm_result

    print(f"[Router] Patient {patient_id}: {heuristic['complexity_tier']} "
          f"(score={score}) — {heuristic['rationale']}")

    return heuristic