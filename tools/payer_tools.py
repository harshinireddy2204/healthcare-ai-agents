"""
tools/payer_tools.py

Payer policy and coverage rule tools. In production these connect to payer
APIs or a coverage database. Here they read from the synthetic policy file.
"""
import json
from pathlib import Path
from langchain_core.tools import tool

POLICY_PATH = Path(__file__).parent.parent / "data" / "payer_policies.json"

def _load_policies() -> dict:
    with open(POLICY_PATH) as f:
        return json.load(f)


@tool
def get_payer_policy(item_name: str, insurance_plan: str) -> dict:
    """
    Look up coverage policy for a specific item (treatment, device, service)
    under a given insurance plan.
    Returns coverage status, prior auth requirement, and all coverage criteria.
    item_name examples: insulin_pump, cardiac_rehab_extended, pulmonary_rehab
    insurance_plan examples: BlueCross PPO, Aetna HMO, United Health PPO
    """
    policies = _load_policies()
    if insurance_plan not in policies:
        return {"covered": False, "message": f"No policy data for {insurance_plan}"}
    plan_policies = policies[insurance_plan]
    if item_name not in plan_policies:
        return {
            "covered": "unknown",
            "message": f"{item_name} not found in {insurance_plan} policy database. "
                       "Manual verification required."
        }
    policy = plan_policies[item_name]
    return {
        "insurance_plan": insurance_plan,
        "item": item_name,
        **policy
    }


@tool
def check_criteria_met(patient_id: str, criteria_list: list, lab_values: dict) -> dict:
    """
    Evaluate whether a patient meets a list of coverage criteria based on
    provided lab values and context.
    criteria_list: list of criteria strings from get_payer_policy
    lab_values: dict of lab name → value pairs from get_lab_results
    Returns a per-criterion pass/fail assessment and overall determination.
    """
    results = []
    all_met = True

    # Normalize lab_values keys to lowercase for case-insensitive lookup
    lab_lower = {k.lower(): v for k, v in lab_values.items()}

    for criterion in criteria_list:
        criterion_lower = criterion.lower()

        # HbA1c criteria
        if "hba1c" in criterion_lower and "hba1c" in lab_lower:
            threshold_str = [t for t in criterion_lower.split() if "." in t or t.isdigit()]
            if threshold_str:
                threshold = float(threshold_str[0].replace("%", ""))
                actual = float(lab_lower["hba1c"])
                met = actual > threshold
                results.append({
                    "criterion": criterion,
                    "met": met,
                    "evidence": f"Patient HbA1c = {actual}% (threshold: {threshold}%)"
                })
                if not met:
                    all_met = False
                continue

        # eGFR criteria
        if "egfr" in criterion_lower and "egfr" in lab_lower:
            threshold_str = [t for t in criterion_lower.split() if t.isdigit()]
            if threshold_str:
                threshold = float(threshold_str[0])
                actual = float(lab_lower["egfr"])
                met = actual < threshold
                results.append({
                    "criterion": criterion,
                    "met": met,
                    "evidence": f"Patient eGFR = {actual} (threshold < {threshold})"
                })
                if not met:
                    all_met = False
                continue

        # LDL criteria
        if "ldl" in criterion_lower and "ldl" in lab_lower:
            threshold_str = [t for t in criterion_lower.split() if t.isdigit()]
            if threshold_str:
                threshold = float(threshold_str[0])
                actual = float(lab_lower["ldl"])
                met = actual > threshold
                results.append({
                    "criterion": criterion,
                    "met": met,
                    "evidence": f"Patient LDL = {actual} mg/dL (threshold > {threshold})"
                })
                if not met:
                    all_met = False
                continue

        # Default: flag for manual review if we can't auto-evaluate
        results.append({
            "criterion": criterion,
            "met": "manual_review_required",
            "evidence": "Cannot auto-evaluate — clinician verification needed"
        })
        all_met = False

    return {
        "patient_id": patient_id,
        "criteria_evaluation": results,
        "all_criteria_met": all_met,
        "recommendation": "APPROVE" if all_met else "REVIEW_REQUIRED"
    }


PAYER_TOOLS = [get_payer_policy, check_criteria_met]
