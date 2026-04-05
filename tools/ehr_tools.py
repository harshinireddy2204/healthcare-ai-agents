"""
tools/ehr_tools.py

MCP-style EHR data tools. In production these would call a real FHIR R4 API
or an internal data platform (e.g. Innovaccer Gravity). For demo purposes
they read from the synthetic JSON data file.
"""
import json
from pathlib import Path
from langchain_core.tools import tool

DATA_PATH = Path(__file__).parent.parent / "data" / "synthetic_patients.json"

def _load_patients() -> dict:
    with open(DATA_PATH) as f:
        patients = json.load(f)
    return {p["patient_id"]: p for p in patients}


@tool
def get_patient_demographics(patient_id: str) -> dict:
    """
    Retrieve basic demographics and active diagnoses for a patient.
    Returns name, age, gender, insurance plan, and current diagnosis list.
    """
    patients = _load_patients()
    if patient_id not in patients:
        return {"error": f"Patient {patient_id} not found"}
    p = patients[patient_id]
    return {
        "patient_id": patient_id,
        "name": p["name"],
        "age": p["age"],
        "gender": p["gender"],
        "insurance": p["insurance"],
        "diagnoses": p["diagnoses"],
        "medications": p["medications"]
    }


@tool
def get_lab_results(patient_id: str, lab_name: str) -> dict:
    """
    Retrieve the most recent lab result for a specific test.
    Common lab_name values: HbA1c, eGFR, LDL, HDL, BMI, blood_pressure.
    Returns value, unit, and date of the result.
    """
    patients = _load_patients()
    if patient_id not in patients:
        return {"error": f"Patient {patient_id} not found"}
    labs = patients[patient_id].get("labs", {})
    if lab_name not in labs:
        return {"result": "not_found", "message": f"No {lab_name} result on file"}
    return {"patient_id": patient_id, "lab": lab_name, **labs[lab_name]}


@tool
def get_screening_history(patient_id: str, screening_type: str) -> dict:
    """
    Retrieve the date of the patient's last preventive screening.
    Common screening_type values: mammogram, colonoscopy, flu_vaccine,
    pneumonia_vaccine, cardiac_stress_test, pulmonary_function_test.
    """
    patients = _load_patients()
    if patient_id not in patients:
        return {"error": f"Patient {patient_id} not found"}
    screenings = patients[patient_id].get("screenings", {})
    if screening_type not in screenings:
        return {"result": "never_done", "message": f"No {screening_type} on record"}
    return {
        "patient_id": patient_id,
        "screening": screening_type,
        **screenings[screening_type]
    }


@tool
def get_pending_auth_requests(patient_id: str) -> list:
    """
    Retrieve all open prior authorization requests for a patient.
    Returns a list of request objects with request_id, type, item, and requesting provider.
    """
    patients = _load_patients()
    if patient_id not in patients:
        return [{"error": f"Patient {patient_id} not found"}]
    return patients[patient_id].get("pending_requests", [])


# Expose all tools as a list for easy agent binding
EHR_TOOLS = [
    get_patient_demographics,
    get_lab_results,
    get_screening_history,
    get_pending_auth_requests,
]
