"""
analytics/data_quality.py

Data Quality Framework for healthcare AI pipelines.

Demonstrates: data validation, quality scoring, and quality
reporting — maps to "data quality frameworks" in the JD.

Runs before agents process any patient to catch bad data early.
Inspired by Great Expectations / dbt test patterns.
"""
import json
from datetime import datetime
from pathlib import Path
from typing import Optional

DATA_PATH = Path("data/synthetic_patients.json")


# ── Validation rules ──────────────────────────────────────────────────────────

class DataQualityRule:
    def __init__(self, name: str, description: str, severity: str = "WARNING"):
        self.name = name
        self.description = description
        self.severity = severity  # ERROR / WARNING / INFO

    def check(self, patient: dict) -> Optional[str]:
        """Return error message if rule fails, None if passes."""
        raise NotImplementedError


class NotNullRule(DataQualityRule):
    def __init__(self, field: str, severity="ERROR"):
        super().__init__(f"not_null_{field}", f"{field} must not be null", severity)
        self.field = field

    def check(self, patient: dict) -> Optional[str]:
        val = patient.get(self.field)
        if val is None or val == "" or val == []:
            return f"Field '{self.field}' is null or empty"
        return None


class AgeRangeRule(DataQualityRule):
    def __init__(self, min_age=0, max_age=120):
        super().__init__("age_in_range", f"Age must be {min_age}-{max_age}", "ERROR")
        self.min_age = min_age
        self.max_age = max_age

    def check(self, patient: dict) -> Optional[str]:
        age = patient.get("age")
        if age is None:
            return None  # caught by NotNullRule
        if not (self.min_age <= int(age) <= self.max_age):
            return f"Age {age} is outside valid range [{self.min_age}, {self.max_age}]"
        return None


class GenderRule(DataQualityRule):
    VALID = {"M", "F", "O", "U"}

    def __init__(self):
        super().__init__("valid_gender", "Gender must be M/F/O/U", "WARNING")

    def check(self, patient: dict) -> Optional[str]:
        gender = patient.get("gender", "")
        if gender not in self.VALID:
            return f"Invalid gender value: '{gender}'"
        return None


class LabValueRule(DataQualityRule):
    LAB_RANGES = {
        "HbA1c":  (3.0, 20.0, "%"),
        "eGFR":   (1.0, 150.0, "mL/min/1.73m2"),
        "LDL":    (20.0, 400.0, "mg/dL"),
        "HDL":    (10.0, 200.0, "mg/dL"),
        "BMI":    (10.0, 80.0, "kg/m2"),
    }

    def __init__(self):
        super().__init__("lab_values_in_range", "Lab values must be physiologically plausible", "WARNING")

    def check(self, patient: dict) -> Optional[str]:
        labs = patient.get("labs", {})
        errors = []
        for lab_name, lab_data in labs.items():
            if lab_name in self.LAB_RANGES:
                lo, hi, unit = self.LAB_RANGES[lab_name]
                try:
                    val = float(lab_data.get("value", 0))
                    if not (lo <= val <= hi):
                        errors.append(f"{lab_name}={val} outside [{lo},{hi}]")
                except (ValueError, TypeError):
                    errors.append(f"{lab_name} value is non-numeric")
        return "; ".join(errors) if errors else None


class InsurancePlanRule(DataQualityRule):
    KNOWN_PLANS = {"BlueCross PPO", "Aetna HMO", "United Health PPO",
                   "Cigna PPO", "Humana HMO", "Medicare", "Medicaid"}

    def __init__(self):
        super().__init__("known_insurance_plan", "Insurance plan must be recognized", "INFO")

    def check(self, patient: dict) -> Optional[str]:
        plan = patient.get("insurance", "")
        if plan and plan not in self.KNOWN_PLANS:
            return f"Unrecognized insurance plan: '{plan}' — payer policies may not be available"
        return None


class DuplicatePatientRule(DataQualityRule):
    def __init__(self, all_ids: list):
        super().__init__("no_duplicate_ids", "Patient IDs must be unique", "ERROR")
        self.seen = set()
        self.duplicates = set(id for id in all_ids if all_ids.count(id) > 1)

    def check(self, patient: dict) -> Optional[str]:
        pid = patient.get("patient_id", "")
        if pid in self.duplicates:
            return f"Duplicate patient_id: '{pid}'"
        return None


# ── Quality runner ────────────────────────────────────────────────────────────

def run_data_quality_check(patients: list = None) -> dict:
    """
    Run all quality rules against patient data.
    Returns a quality report with pass/fail per patient and overall score.
    """
    if patients is None:
        with open(DATA_PATH) as f:
            patients = json.load(f)

    all_ids = [p.get("patient_id", "") for p in patients]

    rules = [
        NotNullRule("patient_id", "ERROR"),
        NotNullRule("name", "WARNING"),
        NotNullRule("age", "ERROR"),
        NotNullRule("diagnoses", "WARNING"),
        NotNullRule("insurance", "WARNING"),
        AgeRangeRule(),
        GenderRule(),
        LabValueRule(),
        InsurancePlanRule(),
        DuplicatePatientRule(all_ids),
    ]

    patient_results = []
    total_checks = 0
    total_passed = 0
    errors = 0
    warnings = 0

    for patient in patients:
        pid = patient.get("patient_id", "UNKNOWN")
        patient_errors = []
        patient_warnings = []
        patient_info = []

        for rule in rules:
            total_checks += 1
            issue = rule.check(patient)
            if issue:
                entry = {"rule": rule.name, "message": issue, "severity": rule.severity}
                if rule.severity == "ERROR":
                    patient_errors.append(entry)
                    errors += 1
                elif rule.severity == "WARNING":
                    patient_warnings.append(entry)
                    warnings += 1
                else:
                    patient_info.append(entry)
            else:
                total_passed += 1

        quality_score = 100
        quality_score -= len(patient_errors) * 20
        quality_score -= len(patient_warnings) * 5
        quality_score = max(0, quality_score)

        patient_results.append({
            "patient_id": pid,
            "quality_score": quality_score,
            "status": "PASS" if not patient_errors else "FAIL",
            "errors": patient_errors,
            "warnings": patient_warnings,
            "info": patient_info
        })

    overall_score = round(total_passed / max(total_checks, 1) * 100, 1)
    failed_patients = [r for r in patient_results if r["status"] == "FAIL"]

    return {
        "run_timestamp": datetime.utcnow().isoformat(),
        "total_patients": len(patients),
        "total_checks": total_checks,
        "checks_passed": total_passed,
        "total_errors": errors,
        "total_warnings": warnings,
        "overall_quality_score": overall_score,
        "failed_patients": len(failed_patients),
        "status": "PASS" if not failed_patients else "FAIL",
        "patient_results": patient_results,
        "summary": f"{overall_score}% quality score — {len(failed_patients)} patients with errors"
    }


def validate_patient_before_processing(patient_id: str) -> dict:
    """
    Validate a single patient before running agents.
    Returns validation result — agents should check this before proceeding.
    """
    try:
        with open(DATA_PATH) as f:
            patients = json.load(f)
        patient_map = {p["patient_id"]: p for p in patients}

        if patient_id not in patient_map:
            return {"valid": False, "errors": [f"Patient {patient_id} not found"]}

        report = run_data_quality_check([patient_map[patient_id]])
        result = report["patient_results"][0]

        return {
            "patient_id": patient_id,
            "valid": result["status"] == "PASS",
            "quality_score": result["quality_score"],
            "errors": [e["message"] for e in result["errors"]],
            "warnings": [w["message"] for w in result["warnings"]]
        }
    except Exception as e:
        return {"valid": False, "errors": [str(e)]}


if __name__ == "__main__":
    report = run_data_quality_check()
    print(f"=== Data Quality Report ===")
    print(f"Status: {report['status']}")
    print(f"Score: {report['overall_quality_score']}%")
    print(f"Patients: {report['total_patients']} | Checks: {report['total_checks']}")
    print(f"Errors: {report['total_errors']} | Warnings: {report['total_warnings']}")
    print(f"\n{report['summary']}")