"""
agents/drug_safety_agent.py

Drug Safety Agent — inspired by TxAgent (Harvard, arXiv 2025.3) and
MALADE (MLHC 2024).

TxAgent key insight: drug reasoning requires real-time biomedical knowledge
from authoritative sources (FDA), not LLM memory. LLMs exhibit high variance
on drug name variants and miss contraindications.

MALADE key insight: Agent/Critic pattern improves pharmacovigilance accuracy.
An FDAHandler agent fetches real drug label data, and a DrugOutcomeInfoAgent
synthesizes the interaction report.

Our implementation:
  1. Fetches real FDA drug label data via OpenFDA API (no key required)
  2. Checks drug-drug interactions from FDA label "drug_interactions" section
  3. Checks contraindications for patient's conditions
  4. Applies knowledge graph interaction rules as a second layer
  5. Generates a structured safety report with FDA citations

OpenFDA endpoints used:
  - /drug/label.json  → drug prescribing information, interactions, contraindications
  - /drug/event.json  → FAERS adverse event reports (for signal detection)
"""
import os
import json
import re
from typing import Optional
from dotenv import load_dotenv

import httpx
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import tool
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from typing import Annotated, TypedDict

load_dotenv()

OPENFDA_BASE = "https://api.fda.gov"
FDA_TIMEOUT  = 10


# ── OpenFDA API client ─────────────────────────────────────────────────────────

class OpenFDAClient:
    """
    Lightweight OpenFDA REST client.
    No API key required for basic usage (240 requests/minute limit).
    """

    def get(self, endpoint: str, params: dict) -> Optional[dict]:
        url = f"{OPENFDA_BASE}{endpoint}"
        try:
            with httpx.Client(timeout=FDA_TIMEOUT, follow_redirects=True) as client:
                r = client.get(url, params=params)
                r.raise_for_status()
                return r.json()
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 404:
                return {"results": []}
            print(f"  [OpenFDA] HTTP error: {e}")
            return None
        except Exception as e:
            print(f"  [OpenFDA] Error: {e}")
            return None

    def get_drug_label(self, drug_name: str) -> Optional[dict]:
        """Fetch FDA drug prescribing label by generic or brand name."""
        # Try generic name first, then brand
        for field in ["openfda.generic_name", "openfda.brand_name"]:
            result = self.get("/drug/label.json", {
                "search": f'{field}:"{drug_name}"',
                "limit": 1
            })
            if result and result.get("results"):
                return result["results"][0]
        return None

    def get_adverse_events(self, drug_name: str, limit: int = 5) -> list:
        """Get FAERS adverse event reports for a drug."""
        result = self.get("/drug/event.json", {
            "search": f'patient.drug.openfda.generic_name:"{drug_name}"',
            "limit": limit,
            "sort": "receivedate:desc"
        })
        if result and result.get("results"):
            return result["results"]
        return []

    def get_drug_interactions_section(self, drug_name: str) -> str:
        """Extract drug interactions section from FDA label."""
        label = self.get_drug_label(drug_name)
        if not label:
            return ""
        interactions = label.get("drug_interactions", [])
        if interactions:
            return " ".join(interactions[:2])[:2000]
        return ""

    def get_contraindications(self, drug_name: str) -> str:
        """Extract contraindications section from FDA label."""
        label = self.get_drug_label(drug_name)
        if not label:
            return ""
        contra = label.get("contraindications", [])
        if contra:
            return " ".join(contra[:1])[:1000]
        return ""

    def get_warnings(self, drug_name: str) -> str:
        """Extract boxed warnings and warnings from FDA label."""
        label = self.get_drug_label(drug_name)
        if not label:
            return ""
        boxed = label.get("boxed_warnings", [])
        warnings = label.get("warnings", [])
        all_warnings = boxed + warnings
        return " ".join(all_warnings[:2])[:1500] if all_warnings else ""


# ── Drug name normalizer ──────────────────────────────────────────────────────

# Maps branded/common names to FDA-searchable generic names
DRUG_NORMALIZATION = {
    "metformin":       "metformin",
    "lisinopril":      "lisinopril",
    "atorvastatin":    "atorvastatin",
    "warfarin":        "warfarin",
    "aspirin":         "aspirin",
    "clopidogrel":     "clopidogrel",
    "furosemide":      "furosemide",
    "carvedilol":      "carvedilol",
    "amlodipine":      "amlodipine",
    "metoprolol":      "metoprolol",
    "insulin":         "insulin",
    "insulin glargine":"insulin glargine",
    "insulin lispro":  "insulin lispro",
    "rosuvastatin":    "rosuvastatin",
    "simvastatin":     "simvastatin",
    "hydrochlorothiazide": "hydrochlorothiazide",
    "spironolactone":  "spironolactone",
    "tamoxifen":       "tamoxifen",
    "methotrexate":    "methotrexate",
    "azathioprine":    "azathioprine",
    "prednisone":      "prednisone",
    "hydroxychloroquine": "hydroxychloroquine",
    "levodopa":        "levodopa",
    "sertraline":      "sertraline",
    "levothyroxine":   "levothyroxine",
    "albuterol":       "albuterol",
    "fluticasone":     "fluticasone",
    "tiotropium":      "tiotropium",
    "bicalutamide":    "bicalutamide",
    "sumatriptan":     "sumatriptan",
    "meloxicam":       "meloxicam",
    "pantoprazole":    "pantoprazole",
    "folic acid":      "folic acid",
}


def normalize_drug_name(med_string: str) -> Optional[str]:
    """Extract and normalize a drug name from a medication string."""
    med_lower = med_string.lower()
    for key, normalized in DRUG_NORMALIZATION.items():
        if key in med_lower:
            return normalized
    # Extract first word (likely the drug name)
    first_word = med_string.split()[0].lower() if med_string else ""
    return first_word if len(first_word) > 3 else None


# ── LangGraph state and agent ─────────────────────────────────────────────────

class DrugSafetyState(TypedDict):
    messages: Annotated[list, add_messages]
    patient_id: str
    medications: list
    diagnoses: list
    fda_findings: list
    kg_findings: list
    final_report: str
    safety_tier: str   # SAFE / CAUTION / WARNING / CRITICAL


def get_llm():
    load_dotenv()
    return ChatOpenAI(model=os.getenv("MODEL_NAME", "gpt-4o-mini"), temperature=0)


SYSTEM_PROMPT = """You are a clinical pharmacist AI agent specializing in drug safety.
Your role is to analyze a patient's medication list for:
1. Drug-drug interactions (DDIs)
2. Drug-disease contraindications
3. Dosing concerns based on lab values (e.g. renal dosing)
4. High-risk medication patterns (polypharmacy, narrow therapeutic index drugs)

You have access to:
- Real FDA drug label data (interactions, contraindications, warnings)
- Knowledge graph clinical relationships
- Patient lab values and diagnoses

For each finding, cite the FDA source or clinical guideline.
Classify overall safety as: SAFE / CAUTION / WARNING / CRITICAL

Output a structured safety report ending with:
SAFETY_TIER: [SAFE|CAUTION|WARNING|CRITICAL]
"""


def fda_lookup_node(state: DrugSafetyState) -> dict:
    """Node: fetch real FDA drug label data for all medications."""
    client = OpenFDAClient()
    fda_findings = []
    medications = state["medications"]

    print(f"  [DrugSafety] Querying OpenFDA for {len(medications)} medications...")

    for med in medications[:8]:  # limit to avoid rate limiting
        drug_name = normalize_drug_name(med)
        if not drug_name:
            continue

        interactions = client.get_drug_interactions_section(drug_name)
        contraindications = client.get_contraindications(drug_name)
        warnings = client.get_warnings(drug_name)

        if any([interactions, contraindications, warnings]):
            finding = {
                "drug": drug_name,
                "original_med": med,
                "interactions_section": interactions[:500] if interactions else "",
                "contraindications": contraindications[:300] if contraindications else "",
                "warnings": warnings[:400] if warnings else "",
                "source": "OpenFDA"
            }
            fda_findings.append(finding)
            print(f"    [OpenFDA] {drug_name}: {'interactions found' if interactions else 'no interactions'}")

    return {"fda_findings": fda_findings}


def kg_analysis_node(state: DrugSafetyState) -> dict:
    """Node: run knowledge graph drug interaction analysis."""
    kg_findings = []
    try:
        from knowledge_graph.clinical_graph import find_risks_for_patient, get_drug_interactions

        # Get lab values placeholder
        findings = find_risks_for_patient(
            diagnoses=state["diagnoses"],
            lab_values={},
            medications=state["medications"],
            age=0
        )
        interactions = get_drug_interactions(findings)
        all_findings = [f for f in findings if f.get("urgency") in ("URGENT", "HIGH")]
        kg_findings = all_findings[:10]
        print(f"  [DrugSafety] KG analysis: {len(kg_findings)} high-priority findings, {len(interactions)} interactions")
    except Exception as e:
        print(f"  [DrugSafety] KG not available: {e}")

    return {"kg_findings": kg_findings}


def synthesize_report_node(state: DrugSafetyState) -> dict:
    """Node: LLM synthesizes FDA + KG data into a safety report."""
    llm = get_llm()

    fda_summary = "\n".join([
        f"Drug: {f['drug']}\n"
        f"  Interactions: {f['interactions_section'][:200]}\n"
        f"  Contraindications: {f['contraindications'][:150]}\n"
        f"  Warnings: {f['warnings'][:200]}"
        for f in state["fda_findings"]
    ]) if state["fda_findings"] else "No FDA data retrieved (API may be unavailable)"

    kg_summary = "\n".join([
        f"[{f.get('urgency')}] {' '.join(str(p) for p in f.get('path', []))} — {f.get('evidence', '')[:120]}"
        for f in state["kg_findings"]
    ]) if state["kg_findings"] else "No KG findings"

    response = llm.invoke([
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=f"""
Patient: {state['patient_id']}
Diagnoses: {', '.join(state['diagnoses'])}
Medications: {', '.join(state['medications'])}

FDA Drug Label Data:
{fda_summary}

Knowledge Graph Clinical Relationships:
{kg_summary}

Generate a comprehensive drug safety report with specific findings,
evidence citations, and recommended actions.
End with: SAFETY_TIER: [SAFE|CAUTION|WARNING|CRITICAL]
""")
    ])

    content = response.content
    safety_tier = "CAUTION"  # default

    for line in content.split("\n"):
        if line.startswith("SAFETY_TIER:"):
            val = line.replace("SAFETY_TIER:", "").strip()
            if val in ("SAFE", "CAUTION", "WARNING", "CRITICAL"):
                safety_tier = val

    return {
        "final_report": content,
        "safety_tier": safety_tier,
        "messages": [response]
    }


def build_drug_safety_graph():
    graph = StateGraph(DrugSafetyState)
    graph.add_node("fda_lookup", fda_lookup_node)
    graph.add_node("kg_analysis", kg_analysis_node)
    graph.add_node("synthesize", synthesize_report_node)

    graph.add_edge(START, "fda_lookup")
    graph.add_edge("fda_lookup", "kg_analysis")
    graph.add_edge("kg_analysis", "synthesize")
    graph.add_edge("synthesize", END)

    return graph.compile()


drug_safety_graph = build_drug_safety_graph()


def run_drug_safety_check(patient_id: str, medications: list,
                           diagnoses: list) -> dict:
    """
    Run the drug safety agent for a patient.
    Queries OpenFDA in real-time for drug label data.
    """
    print(f"\n[DrugSafety] Starting analysis for {patient_id}")

    result = drug_safety_graph.invoke({
        "messages": [],
        "patient_id": patient_id,
        "medications": medications,
        "diagnoses": diagnoses,
        "fda_findings": [],
        "kg_findings": [],
        "final_report": "",
        "safety_tier": "CAUTION"
    })

    return {
        "patient_id": patient_id,
        "medications_checked": len(medications),
        "fda_findings_count": len(result["fda_findings"]),
        "kg_findings_count": len(result["kg_findings"]),
        "safety_tier": result["safety_tier"],
        "fda_findings": result["fda_findings"],
        "safety_report": result["final_report"]
    }


if __name__ == "__main__":
    # Test with P004 — Warfarin + Furosemide + Carvedilol + Insulin + Metformin
    result = run_drug_safety_check(
        patient_id="P004",
        medications=["Furosemide 40mg", "Warfarin 5mg", "Carvedilol 12.5mg",
                     "Insulin glargine", "Metformin 1000mg"],
        diagnoses=["Heart Failure", "Atrial Fibrillation", "Type 2 Diabetes", "CKD Stage 4"]
    )
    print(f"\n=== Drug Safety Report ===")
    print(f"Safety tier: {result['safety_tier']}")
    print(f"FDA findings: {result['fda_findings_count']}")
    print(f"\n{result['safety_report'][:1000]}")