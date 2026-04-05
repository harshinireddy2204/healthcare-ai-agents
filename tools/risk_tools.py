"""
tools/risk_tools.py

Risk stratification tools used by the triage agent to score patients and
route them to the appropriate care pathway.
"""
from langchain_core.tools import tool


# Clinical risk factor weights (simplified Charlson-inspired scoring)
COMORBIDITY_WEIGHTS = {
    "Type 2 Diabetes": 1,
    "Type 1 Diabetes": 2,
    "Hypertension": 1,
    "CKD Stage 3": 2,
    "CKD Stage 4": 3,
    "CKD Stage 5": 6,
    "Coronary Artery Disease": 2,
    "Heart Failure": 2,
    "COPD": 1,
    "Asthma": 1,
    "Obesity": 1,
    "Prediabetes": 1,
    "Hyperlipidemia": 1,
    "Atrial Fibrillation": 2,
    "Cancer": 6,
    "Stroke": 2,
}

LAB_RISK_MODIFIERS = {
    "HbA1c": lambda v: 2 if float(v) >= 10 else (1 if float(v) >= 8 else 0),
    "eGFR": lambda v: 3 if float(v) < 30 else (2 if float(v) < 45 else (1 if float(v) < 60 else 0)),
    "LDL": lambda v: 1 if float(v) > 130 else 0,
    "BMI": lambda v: 1 if float(v) >= 35 else 0,
}


@tool
def calculate_risk_score(patient_id: str, diagnoses: list, lab_values: dict) -> dict:
    """
    Calculate a composite risk score for a patient based on their diagnoses
    and most recent lab values. Higher scores indicate higher clinical risk.

    patient_id: the patient's ID
    diagnoses: list of diagnosis strings from get_patient_demographics
    lab_values: dict of {lab_name: value} from get_lab_results calls

    Returns risk_score (0-20+), risk_tier (LOW/MEDIUM/HIGH/CRITICAL),
    and contributing_factors list.
    """
    score = 0
    factors = []

    # Comorbidity scoring
    for dx in diagnoses:
        weight = COMORBIDITY_WEIGHTS.get(dx, 0)
        if weight > 0:
            score += weight
            factors.append(f"{dx} (+{weight})")

    # Lab value modifiers
    for lab, value in lab_values.items():
        modifier_fn = LAB_RISK_MODIFIERS.get(lab)
        if modifier_fn:
            try:
                modifier = modifier_fn(value)
                if modifier > 0:
                    score += modifier
                    factors.append(f"{lab}={value} (+{modifier})")
            except (ValueError, TypeError):
                pass

    # Determine tier
    if score <= 2:
        tier = "LOW"
    elif score <= 5:
        tier = "MEDIUM"
    elif score <= 8:
        tier = "HIGH"
    else:
        tier = "CRITICAL"

    return {
        "patient_id": patient_id,
        "risk_score": score,
        "risk_tier": tier,
        "contributing_factors": factors,
        "recommended_action": {
            "LOW": "Routine annual follow-up",
            "MEDIUM": "6-month follow-up, close monitoring",
            "HIGH": "Proactive outreach within 30 days, care coordination",
            "CRITICAL": "Urgent care manager assignment, immediate outreach"
        }[tier]
    }


@tool
def get_care_gaps(patient_id: str, age: int, gender: str,
                  diagnoses: list, screening_history: dict) -> dict:
    """
    Identify preventive care gaps based on patient profile and screening history.
    Uses standard clinical guidelines (USPSTF, CMS).

    Returns a list of identified gaps and recommended actions.
    """
    from datetime import datetime, date
    import re

    today = datetime.today()
    gaps = []

    def months_since(date_str: str) -> float:
        try:
            d = datetime.strptime(date_str, "%Y-%m-%d")
            return (today - d).days / 30.4
        except Exception:
            return 9999

    # Mammogram: women >= 40, every 12 months
    if gender == "F" and age >= 40:
        last = screening_history.get("mammogram", {}).get("last_date")
        if not last or months_since(last) > 12:
            gaps.append({
                "gap": "Mammogram overdue",
                "guideline": "USPSTF: Annual mammogram for women 40+",
                "last_done": last or "Never",
                "priority": "HIGH" if not last else "MEDIUM",
                "action": "Schedule mammogram"
            })

    # Colonoscopy: all patients >= 45, every 10 years
    if age >= 45:
        last = screening_history.get("colonoscopy", {}).get("last_date")
        if not last or months_since(last) > 120:
            gaps.append({
                "gap": "Colorectal cancer screening overdue",
                "guideline": "USPSTF: Colorectal screening starting at age 45",
                "last_done": last or "Never",
                "priority": "HIGH",
                "action": "Schedule colonoscopy or FIT test"
            })

    # Annual flu vaccine
    last_flu = screening_history.get("flu_vaccine", {}).get("last_date")
    if not last_flu or months_since(last_flu) > 12:
        gaps.append({
            "gap": "Annual flu vaccine due",
            "guideline": "CDC: Annual influenza vaccination for all patients",
            "last_done": last_flu or "Never",
            "priority": "LOW",
            "action": "Administer flu vaccine at next visit"
        })

    # Nephrology referral: CKD patients with eGFR < 60 (check diagnosis)
    ckd_diagnoses = [d for d in diagnoses if "CKD" in d or "Chronic Kidney" in d]
    if ckd_diagnoses:
        gaps.append({
            "gap": "Nephrology referral recommended for CKD patient",
            "guideline": "KDIGO: Nephrology co-management for CKD Stage 3+",
            "priority": "HIGH",
            "action": "Initiate nephrology referral"
        })

    return {
        "patient_id": patient_id,
        "care_gaps_found": len(gaps),
        "gaps": gaps,
        "summary": f"{len(gaps)} care gap(s) identified" if gaps else "No care gaps found"
    }


RISK_TOOLS = [calculate_risk_score, get_care_gaps]
