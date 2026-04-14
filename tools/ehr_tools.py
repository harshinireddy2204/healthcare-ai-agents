"""
tools/ehr_tools.py

MCP-style EHR data tools — FHIR R4 + synthetic fallback.

Architecture:
  1. Try the configured FHIR R4 server (HAPI public test server by default)
  2. If unavailable or patient not found, fall back to synthetic JSON
  3. All responses are normalized to the same dict schema regardless of source

FHIR R4 Integration:
  - Server: https://hapi.fhir.org/baseR4 (public, no auth required)
  - Resources used: Patient, Condition, Observation, MedicationRequest,
                    AllergyIntolerance, Immunization, ServiceRequest
  - FHIR R4 is the industry-standard healthcare data interchange format

In production, swap FHIR_BASE_URL for the customer's FHIR endpoint and add
OAuth2 SMART-on-FHIR authentication via the FHIR_CLIENT_ID / FHIR_CLIENT_SECRET
environment variables.

References:
  - FHIR-AgentBench (arXiv 2025.9) — benchmarks LLM agents on real FHIR R4 queries
  - Infherno (arXiv 2025.7) — FHIR resource synthesis from clinical notes
"""
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Optional

import httpx
from dotenv import load_dotenv
from langchain_core.tools import tool

load_dotenv()

# ── Configuration ─────────────────────────────────────────────────────────────

FHIR_BASE_URL = os.getenv("FHIR_BASE_URL", "https://hapi.fhir.org/baseR4")
FHIR_TIMEOUT  = int(os.getenv("FHIR_TIMEOUT", "8"))
USE_FHIR      = os.getenv("USE_FHIR", "true").lower() == "true"

DATA_PATH = Path(__file__).parent.parent / "data" / "synthetic_patients.json"

# FHIR LOINC codes for common lab tests
LOINC_CODES = {
    "HbA1c":           "4548-4",
    "eGFR":            "33914-3",
    "LDL":             "13457-7",
    "HDL":             "2085-9",
    "BMI":             "39156-5",
    "blood_pressure":  "55284-4",
    "creatinine":      "2160-0",
    "potassium":       "2823-3",
    "hemoglobin":      "718-7",
}

# SNOMED CT codes for common diagnoses
SNOMED_CODES = {
    "Type 2 Diabetes":        "44054006",
    "Type 1 Diabetes":        "46635009",
    "Hypertension":           "38341003",
    "CKD Stage 3":            "433144002",
    "CKD Stage 4":            "431855005",
    "Heart Failure":          "84114007",
    "Atrial Fibrillation":    "49436004",
    "Coronary Artery Disease": "53741008",
    "COPD":                   "13645005",
    "Asthma":                 "195967001",
    "Obesity":                "414916001",
    "Hyperlipidemia":         "55822004",
}


# ── FHIR client ───────────────────────────────────────────────────────────────

class FHIRClient:
    """
    Lightweight FHIR R4 REST client.
    Used for all real EHR queries when USE_FHIR=true.
    """

    def __init__(self, base_url: str = FHIR_BASE_URL):
        self.base_url = base_url.rstrip("/")
        self.headers = {
            "Accept": "application/fhir+json",
            "Content-Type": "application/fhir+json",
        }

    def get(self, resource_type: str, params: dict = None) -> Optional[dict]:
        """Execute a FHIR search query. Returns Bundle or None on failure."""
        url = f"{self.base_url}/{resource_type}"
        try:
            with httpx.Client(timeout=FHIR_TIMEOUT, follow_redirects=True) as client:
                response = client.get(url, params=params, headers=self.headers)
                response.raise_for_status()
                return response.json()
        except Exception as e:
            print(f"  [FHIR] {resource_type} query failed: {e}")
            return None

    def get_by_id(self, resource_type: str, resource_id: str) -> Optional[dict]:
        """Fetch a single FHIR resource by ID."""
        url = f"{self.base_url}/{resource_type}/{resource_id}"
        try:
            with httpx.Client(timeout=FHIR_TIMEOUT, follow_redirects=True) as client:
                response = client.get(url, headers=self.headers)
                response.raise_for_status()
                return response.json()
        except Exception as e:
            print(f"  [FHIR] {resource_type}/{resource_id} fetch failed: {e}")
            return None

    def search_patients(self, name: str = None, count: int = 5) -> list[dict]:
        """Search for patients by name. Returns list of Patient resources."""
        params = {"_count": count, "_format": "json"}
        if name:
            params["name"] = name
        bundle = self.get("Patient", params)
        if not bundle:
            return []
        return [
            entry["resource"]
            for entry in bundle.get("entry", [])
            if entry.get("resource", {}).get("resourceType") == "Patient"
        ]

    def get_conditions(self, patient_id: str) -> list[dict]:
        """Get all active conditions for a patient."""
        bundle = self.get("Condition", {
            "patient": patient_id,
            "clinical-status": "active",
            "_count": 50,
            "_format": "json"
        })
        if not bundle:
            return []
        return [
            entry["resource"]
            for entry in bundle.get("entry", [])
            if entry.get("resource", {}).get("resourceType") == "Condition"
        ]

    def get_observations(self, patient_id: str, loinc_code: str = None,
                         count: int = 10) -> list[dict]:
        """Get lab results / vital signs for a patient."""
        params = {
            "patient": patient_id,
            "_count": count,
            "_sort": "-date",
            "_format": "json"
        }
        if loinc_code:
            params["code"] = f"http://loinc.org|{loinc_code}"
        bundle = self.get("Observation", params)
        if not bundle:
            return []
        return [
            entry["resource"]
            for entry in bundle.get("entry", [])
            if entry.get("resource", {}).get("resourceType") == "Observation"
        ]

    def get_medications(self, patient_id: str) -> list[dict]:
        """Get active medication requests for a patient."""
        bundle = self.get("MedicationRequest", {
            "patient": patient_id,
            "status": "active",
            "_count": 50,
            "_format": "json"
        })
        if not bundle:
            return []
        return [
            entry["resource"]
            for entry in bundle.get("entry", [])
            if entry.get("resource", {}).get("resourceType") == "MedicationRequest"
        ]

    def get_immunizations(self, patient_id: str) -> list[dict]:
        """Get immunization history for a patient."""
        bundle = self.get("Immunization", {
            "patient": patient_id,
            "_count": 50,
            "_sort": "-date",
            "_format": "json"
        })
        if not bundle:
            return []
        return [
            entry["resource"]
            for entry in bundle.get("entry", [])
            if entry.get("resource", {}).get("resourceType") == "Immunization"
        ]


# ── FHIR → normalized dict converters ────────────────────────────────────────

def _fhir_patient_to_dict(patient: dict) -> dict:
    """Convert a FHIR R4 Patient resource to our normalized schema."""
    name_entry = patient.get("name", [{}])[0]
    given = " ".join(name_entry.get("given", []))
    family = name_entry.get("family", "")
    name = f"{given} {family}".strip() or "Unknown"

    gender_map = {"male": "M", "female": "F", "other": "O", "unknown": "U"}
    gender = gender_map.get(patient.get("gender", "").lower(), "U")

    birth_date = patient.get("birthDate", "")
    age = 0
    if birth_date:
        try:
            dob = datetime.strptime(birth_date, "%Y-%m-%d")
            age = (datetime.today() - dob).days // 365
        except Exception:
            pass

    # Extract insurance from coverage extension or identifier
    insurance = "Unknown"
    for ext in patient.get("extension", []):
        if "coverage" in ext.get("url", "").lower():
            insurance = ext.get("valueString", "Unknown")

    return {
        "fhir_id": patient.get("id", ""),
        "name": name,
        "age": age,
        "gender": gender,
        "birth_date": birth_date,
        "insurance": insurance,
        "source": "FHIR_R4"
    }


def _fhir_conditions_to_list(conditions: list[dict]) -> list[str]:
    """Convert FHIR Condition resources to a list of diagnosis strings."""
    diagnoses = []
    for cond in conditions:
        code_entry = cond.get("code", {})
        # Try SNOMED text first, then coding display, then text
        text = code_entry.get("text", "")
        if not text:
            codings = code_entry.get("coding", [{}])
            text = codings[0].get("display", "") if codings else ""
        if text:
            diagnoses.append(text)
    return diagnoses


def _fhir_observation_to_lab(obs: dict, lab_name: str) -> dict:
    """Convert a FHIR Observation to our lab result schema."""
    value_quantity = obs.get("valueQuantity", {})
    value = value_quantity.get("value")
    unit = value_quantity.get("unit", "")

    # Handle string values (e.g. blood pressure "120/80")
    if not value:
        value_string = obs.get("valueString", "")
        if value_string:
            return {"value": value_string, "unit": unit,
                    "date": obs.get("effectiveDateTime", "")[:10],
                    "source": "FHIR_R4"}

    effective = obs.get("effectiveDateTime", obs.get("effectivePeriod", {}).get("start", ""))
    return {
        "value": value,
        "unit": unit,
        "date": effective[:10] if effective else "",
        "source": "FHIR_R4",
        "fhir_id": obs.get("id", "")
    }


def _fhir_medications_to_list(meds: list[dict]) -> list[str]:
    """Convert FHIR MedicationRequest resources to medication strings."""
    result = []
    for med in meds:
        med_ref = med.get("medicationCodeableConcept", {})
        text = med_ref.get("text", "")
        if not text:
            codings = med_ref.get("coding", [{}])
            text = codings[0].get("display", "") if codings else ""
        dosage = med.get("dosageInstruction", [{}])
        dose_text = dosage[0].get("text", "") if dosage else ""
        if text:
            result.append(f"{text} {dose_text}".strip())
    return result


def _fhir_immunizations_to_dict(immunizations: list[dict]) -> dict:
    """Convert FHIR Immunization resources to our screening_history schema."""
    history = {}
    vaccine_map = {
        "influenza": "flu_vaccine",
        "flu": "flu_vaccine",
        "pneumococcal": "pneumonia_vaccine",
        "pneumonia": "pneumonia_vaccine",
        "zoster": "shingles_vaccine",
        "shingrix": "shingles_vaccine",
        "tdap": "tdap_vaccine",
        "hepatitis": "hepatitis_vaccine",
        "covid": "covid_vaccine",
    }
    for imm in immunizations:
        code_entry = imm.get("vaccineCode", {})
        text = (code_entry.get("text", "") or
                (code_entry.get("coding", [{}])[0].get("display", "")
                 if code_entry.get("coding") else "")).lower()
        occurrence = imm.get("occurrenceDateTime", imm.get("occurrenceString", ""))

        for keyword, normalized_name in vaccine_map.items():
            if keyword in text:
                if normalized_name not in history or (
                    occurrence and occurrence > history[normalized_name].get("last_date", "")
                ):
                    history[normalized_name] = {"last_date": occurrence[:10] if occurrence else ""}
                break
    return history


# ── Synthetic data fallback ───────────────────────────────────────────────────

def _load_synthetic_patients() -> dict:
    with open(DATA_PATH) as f:
        patients = json.load(f)
    return {p["patient_id"]: p for p in patients}


def _get_patient_data(patient_id: str) -> Optional[dict]:
    """
    Try FHIR first, fall back to synthetic data.
    FHIR patient IDs starting with 'P' (P001..P020) map to synthetic data.
    Numeric or UUID IDs are looked up directly on the FHIR server.
    """
    # Synthetic patient IDs (P001-P020) — always use local data
    if patient_id.startswith("P") and patient_id[1:].isdigit():
        patients = _load_synthetic_patients()
        return patients.get(patient_id)

    # Real FHIR patient ID — query the server
    if USE_FHIR:
        client = FHIRClient()
        patient = client.get_by_id("Patient", patient_id)
        if patient:
            return {
                "_fhir_resource": patient,
                "_fhir_patient_id": patient_id,
                "_source": "fhir"
            }

    return None


# ── LangChain tools (MCP-style) ───────────────────────────────────────────────

@tool
def get_patient_demographics(patient_id: str) -> dict:
    """
    Retrieve basic demographics and active diagnoses for a patient.
    Queries FHIR R4 Patient + Condition resources if available,
    otherwise uses synthetic data.

    patient_id: either a synthetic ID (P001-P020) or a real FHIR patient ID
    Returns name, age, gender, insurance plan, diagnoses, medications, and data source.
    """
    data = _get_patient_data(patient_id)
    if not data:
        return {"error": f"Patient {patient_id} not found"}

    # Real FHIR patient
    if data.get("_source") == "fhir":
        client = FHIRClient()
        fhir_id = data["_fhir_patient_id"]

        patient_dict = _fhir_patient_to_dict(data["_fhir_resource"])

        # Fetch conditions
        conditions = client.get_conditions(fhir_id)
        diagnoses = _fhir_conditions_to_list(conditions)

        # Fetch medications
        meds = client.get_medications(fhir_id)
        medications = _fhir_medications_to_list(meds)

        return {
            "patient_id": patient_id,
            "fhir_id": fhir_id,
            "name": patient_dict["name"],
            "age": patient_dict["age"],
            "gender": patient_dict["gender"],
            "insurance": patient_dict["insurance"],
            "diagnoses": diagnoses,
            "medications": medications,
            "data_source": "FHIR_R4",
            "fhir_server": FHIR_BASE_URL
        }

    # Synthetic patient
    p = data
    return {
        "patient_id": patient_id,
        "name": p["name"],
        "age": p["age"],
        "gender": p["gender"],
        "insurance": p["insurance"],
        "diagnoses": p["diagnoses"],
        "medications": p["medications"],
        "data_source": "synthetic_json"
    }


@tool
def get_lab_results(patient_id: str, lab_name: str) -> dict:
    """
    Retrieve the most recent lab result for a specific test.
    Queries FHIR R4 Observation resource using LOINC codes if available.

    patient_id: synthetic ID (P001-P020) or real FHIR patient ID
    lab_name: HbA1c, eGFR, LDL, HDL, BMI, blood_pressure, creatinine, potassium
    Returns value, unit, date, and data source.
    """
    data = _get_patient_data(patient_id)
    if not data:
        return {"error": f"Patient {patient_id} not found"}

    # Real FHIR patient — query Observation by LOINC code
    if data.get("_source") == "fhir":
        client = FHIRClient()
        fhir_id = data["_fhir_patient_id"]
        loinc_code = LOINC_CODES.get(lab_name)

        if loinc_code:
            observations = client.get_observations(fhir_id, loinc_code=loinc_code, count=1)
            if observations:
                lab_dict = _fhir_observation_to_lab(observations[0], lab_name)
                lab_dict["lab"] = lab_name
                lab_dict["patient_id"] = patient_id
                return lab_dict

        return {
            "result": "not_found",
            "message": f"No FHIR Observation found for {lab_name} (LOINC: {loinc_code})",
            "data_source": "FHIR_R4"
        }

    # Synthetic patient
    labs = data.get("labs", {})
    if lab_name not in labs:
        return {"result": "not_found", "message": f"No {lab_name} result on file",
                "data_source": "synthetic_json"}
    return {"patient_id": patient_id, "lab": lab_name, **labs[lab_name],
            "data_source": "synthetic_json"}


@tool
def get_screening_history(patient_id: str, screening_type: str) -> dict:
    """
    Retrieve the date of the patient's last preventive screening or immunization.
    Queries FHIR R4 Immunization resource for vaccines, Procedure for screenings.

    patient_id: synthetic ID (P001-P020) or real FHIR patient ID
    screening_type: mammogram, colonoscopy, flu_vaccine, pneumonia_vaccine,
                    cardiac_stress_test, pulmonary_function_test, bone_density
    Returns last_date and data source.
    """
    data = _get_patient_data(patient_id)
    if not data:
        return {"error": f"Patient {patient_id} not found"}

    # Real FHIR patient
    if data.get("_source") == "fhir":
        client = FHIRClient()
        fhir_id = data["_fhir_patient_id"]

        # Check immunizations for vaccine-type screenings
        vaccine_screenings = {"flu_vaccine", "pneumonia_vaccine", "shingles_vaccine",
                               "tdap_vaccine", "covid_vaccine"}
        if screening_type in vaccine_screenings:
            immunizations = client.get_immunizations(fhir_id)
            history = _fhir_immunizations_to_dict(immunizations)
            if screening_type in history:
                return {
                    "patient_id": patient_id,
                    "screening": screening_type,
                    **history[screening_type],
                    "data_source": "FHIR_R4"
                }

        return {
            "result": "not_found",
            "message": f"No FHIR record for {screening_type}",
            "data_source": "FHIR_R4"
        }

    # Synthetic patient
    screenings = data.get("screenings", {})
    if screening_type not in screenings:
        return {"result": "never_done", "message": f"No {screening_type} on record",
                "data_source": "synthetic_json"}
    entry = screenings[screening_type]
    if entry.get("last_date") is None:
        return {"result": "never_done", "message": f"No {screening_type} on record",
                "data_source": "synthetic_json"}
    return {"patient_id": patient_id, "screening": screening_type,
            **entry, "data_source": "synthetic_json"}


@tool
def get_pending_auth_requests(patient_id: str) -> list:
    """
    Retrieve all open prior authorization requests for a patient.
    In FHIR R4 these map to ServiceRequest resources with status=active.

    Returns list of request objects with request_id, type, item, and requesting provider.
    """
    data = _get_patient_data(patient_id)
    if not data:
        return [{"error": f"Patient {patient_id} not found"}]

    # Real FHIR patient — query ServiceRequest
    if data.get("_source") == "fhir":
        client = FHIRClient()
        fhir_id = data["_fhir_patient_id"]
        bundle = client.get("ServiceRequest", {
            "patient": fhir_id,
            "status": "active",
            "_count": 20,
            "_format": "json"
        })
        if not bundle:
            return []
        requests = []
        for entry in bundle.get("entry", []):
            req = entry.get("resource", {})
            if req.get("resourceType") == "ServiceRequest":
                code = req.get("code", {})
                item = code.get("text", "") or (
                    code.get("coding", [{}])[0].get("display", "") if code.get("coding") else ""
                )
                requester = req.get("requester", {}).get("display", "Unknown")
                requests.append({
                    "request_id": req.get("id", ""),
                    "type": "prior_auth",
                    "item": item.lower().replace(" ", "_"),
                    "item_display": item,
                    "requested_by": requester,
                    "fhir_id": req.get("id", ""),
                    "data_source": "FHIR_R4"
                })
        return requests

    # Synthetic patient
    return data.get("pending_requests", [])


@tool
def search_fhir_patients(name: str, count: int = 5) -> list:
    """
    Search the FHIR server for patients by name.
    Returns a list of matching patients with their FHIR IDs.

    Use this to find real FHIR patient IDs to pass to other tools.
    name: full or partial patient name
    count: max number of results (default 5)
    """
    if not USE_FHIR:
        return [{"message": "FHIR disabled — set USE_FHIR=true in .env"}]

    client = FHIRClient()
    patients = client.search_patients(name=name, count=count)
    if not patients:
        return [{"message": f"No FHIR patients found matching '{name}'"}]

    results = []
    for p in patients:
        pdict = _fhir_patient_to_dict(p)
        results.append({
            "fhir_id": pdict["fhir_id"],
            "name": pdict["name"],
            "age": pdict["age"],
            "gender": pdict["gender"],
            "birth_date": pdict["birth_date"],
            "data_source": "FHIR_R4",
            "server": FHIR_BASE_URL
        })
    return results


@tool
def get_fhir_summary(patient_id: str) -> dict:
    """
    Get a comprehensive FHIR R4 patient summary — all resources in one call.
    Fetches Patient + Condition + Observation + MedicationRequest + Immunization.

    Use for real FHIR patient IDs (numeric or UUID).
    For synthetic patients (P001-P020), use get_patient_demographics instead.
    """
    if patient_id.startswith("P") and patient_id[1:].isdigit():
        return {"message": "Use get_patient_demographics for synthetic patients P001-P020"}

    if not USE_FHIR:
        return {"error": "FHIR disabled — set USE_FHIR=true in .env"}

    client = FHIRClient()
    patient = client.get_by_id("Patient", patient_id)
    if not patient:
        return {"error": f"FHIR patient {patient_id} not found on {FHIR_BASE_URL}"}

    patient_dict = _fhir_patient_to_dict(patient)

    conditions   = client.get_conditions(patient_id)
    observations = client.get_observations(patient_id, count=20)
    medications  = client.get_medications(patient_id)
    immunizations = client.get_immunizations(patient_id)

    # Process labs into named dict
    labs = {}
    for obs in observations:
        code_entry = obs.get("code", {})
        codings = code_entry.get("coding", [{}])
        loinc = codings[0].get("code", "") if codings else ""
        display = codings[0].get("display", code_entry.get("text", "")) if codings else ""
        vq = obs.get("valueQuantity", {})
        if vq.get("value") is not None:
            labs[display or loinc] = {
                "value": vq["value"],
                "unit": vq.get("unit", ""),
                "date": obs.get("effectiveDateTime", "")[:10]
            }

    return {
        "patient_id": patient_id,
        "fhir_server": FHIR_BASE_URL,
        "data_source": "FHIR_R4",
        **patient_dict,
        "diagnoses": _fhir_conditions_to_list(conditions),
        "medications": _fhir_medications_to_list(medications),
        "labs": labs,
        "immunization_history": _fhir_immunizations_to_dict(immunizations),
        "resource_counts": {
            "conditions": len(conditions),
            "observations": len(observations),
            "medications": len(medications),
            "immunizations": len(immunizations)
        }
    }


# ── Tool registry ─────────────────────────────────────────────────────────────

EHR_TOOLS = [
    get_patient_demographics,
    get_lab_results,
    get_screening_history,
    get_pending_auth_requests,
    search_fhir_patients,
    get_fhir_summary,
]


# ── CLI test ──────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=== EHR Tools Test ===\n")

    # Test synthetic path
    result = get_patient_demographics.invoke({"patient_id": "P001"})
    print(f"P001 demographics (source: {result.get('data_source')}):")
    print(f"  {result['name']}, {result['age']}yo {result['gender']}")
    print(f"  Diagnoses: {result['diagnoses'][:3]}")

    result = get_lab_results.invoke({"patient_id": "P001", "lab_name": "HbA1c"})
    print(f"\nP001 HbA1c: {result.get('value')} {result.get('unit')} ({result.get('date')})")

    # Test FHIR path (if accessible)
    if USE_FHIR:
        print(f"\nTesting FHIR server: {FHIR_BASE_URL}")
        results = search_fhir_patients.invoke({"name": "Smith", "count": 2})
        if results and "error" not in results[0] and "message" not in results[0]:
            print(f"FHIR search: found {len(results)} patient(s)")
            for r in results[:1]:
                print(f"  FHIR ID: {r['fhir_id']} — {r['name']}, {r['age']}yo")
                # Full summary
                summary = get_fhir_summary.invoke({"patient_id": r["fhir_id"]})
                print(f"  Conditions: {len(summary.get('diagnoses', []))}")
                print(f"  Medications: {len(summary.get('medications', []))}")
        else:
            print(f"  FHIR server not reachable: {results}")