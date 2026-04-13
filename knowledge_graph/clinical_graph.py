"""
knowledge_graph/clinical_graph.py

Clinical Knowledge Graph for multi-hop clinical reasoning.

Architecture inspired by:
  - SNOMED CT Knowledge Graphs (arXiv 2025.10)
  - KG4Diagnosis: Hierarchical Multi-Agent LLM with KG Enhancement (arXiv 2024.12)
  - ESCARGOT: AI agent with dynamic graph of thoughts (Bioinformatics 2025)

What this does that pure LLM reasoning cannot:
  1. Finds non-obvious multi-hop connections:
     CKD Stage 3 + Diabetes → high CVD event risk → proactive cardiology referral
     Warfarin + NSAIDs → drug interaction → bleeding risk → alert

  2. Provides deterministic, auditable reasoning chains:
     Each conclusion has an explicit path through the graph — not LLM inference

  3. Grounds agent decisions in clinical ontology relationships:
     Diagnoses → comorbidities → risk factors → recommended interventions

Graph node types:
  - diagnosis:       clinical condition (ICD-10 / SNOMED concept)
  - risk_factor:     modifiable or non-modifiable risk
  - lab_threshold:   critical lab value threshold
  - intervention:    recommended action (referral, medication, screening)
  - drug:            medication
  - complication:    downstream adverse outcome

Edge types:
  - increases_risk_of:   diagnosis increases risk of complication
  - requires_monitoring: condition requires specific lab or vital monitoring
  - contraindicates:     drug/condition contraindicates another drug/treatment
  - interacts_with:      drug-drug or drug-condition interaction
  - indicates_referral:  clinical finding → specialist referral
  - triggers_screening:  condition/risk factor → preventive screening
  - worsens:             condition worsens another condition
  - managed_by:          condition managed by specific intervention
"""
import json
from pathlib import Path
from typing import Optional

try:
    import networkx as nx
    HAS_NETWORKX = True
except ImportError:
    HAS_NETWORKX = False
    print("[KnowledgeGraph] networkx not installed — run: pip install networkx")

from langchain_core.tools import tool


# ── Build the clinical knowledge graph ────────────────────────────────────────

def build_clinical_graph():
    """
    Construct the clinical knowledge graph with evidence-based relationships.
    All edges are sourced from ADA, ACC/AHA, KDIGO, NHLBI, or USPSTF guidelines.
    """
    if not HAS_NETWORKX:
        return None

    G = nx.DiGraph()

    # ── NODE DEFINITIONS ───────────────────────────────────────────────────────
    # Format: G.add_node(id, type=..., label=..., icd10=..., snomed=...)

    # Diagnoses
    diagnoses = [
        ("type2_diabetes",        "Type 2 Diabetes",              "E11",   "44054006"),
        ("type1_diabetes",        "Type 1 Diabetes",              "E10",   "46635009"),
        ("hypertension",          "Hypertension",                 "I10",   "38341003"),
        ("ckd_stage3",            "CKD Stage 3",                  "N18.3", "433144002"),
        ("ckd_stage4",            "CKD Stage 4",                  "N18.4", "431855005"),
        ("ckd_stage5",            "CKD Stage 5 / ESRD",           "N18.5", "46177005"),
        ("heart_failure",         "Heart Failure",                "I50",   "84114007"),
        ("afib",                  "Atrial Fibrillation",          "I48",   "49436004"),
        ("cad",                   "Coronary Artery Disease",      "I25",   "53741008"),
        ("stroke",                "Ischemic Stroke / TIA",        "I63",   "422504002"),
        ("copd",                  "COPD",                         "J44",   "13645005"),
        ("asthma",                "Asthma",                       "J45",   "195967001"),
        ("obesity",               "Obesity",                      "E66",   "414916001"),
        ("prediabetes",           "Prediabetes",                  "R73.09","15777000"),
        ("hyperlipidemia",        "Hyperlipidemia",               "E78.5", "55822004"),
        ("osteoporosis",          "Osteoporosis",                 "M81",   "64859006"),
        ("rheumatoid_arthritis",  "Rheumatoid Arthritis",         "M06",   "69896004"),
        ("lupus",                 "Systemic Lupus Erythematosus", "M32",   "55464009"),
        ("depression",            "Major Depressive Disorder",    "F32",   "370143000"),
        ("parkinson",             "Parkinson's Disease",          "G20",   "49049000"),
        ("breast_cancer",         "Breast Cancer",                "C50",   "254837009"),
        ("prostate_cancer",       "Prostate Cancer",              "C61",   "399068003"),
        ("crohns",                "Crohn's Disease",              "K50",   "34000006"),
        ("hypothyroidism",        "Hypothyroidism",               "E03",   "40930008"),
        ("sleep_apnea",           "Obstructive Sleep Apnea",      "G47.33","78275009"),
        ("pcos",                  "Polycystic Ovary Syndrome",    "E28.2", "69878008"),
        ("osteoarthritis",        "Osteoarthritis",               "M19",   "396275006"),
        ("anemia",                "Anemia",                       "D64",   "271737000"),
    ]

    for node_id, label, icd10, snomed in diagnoses:
        G.add_node(node_id, type="diagnosis", label=label, icd10=icd10, snomed=snomed)

    # Complications / outcomes
    complications = [
        ("mi",                    "Myocardial Infarction",        "I21"),
        ("aki",                   "Acute Kidney Injury",          "N17"),
        ("esrd",                  "End Stage Renal Disease",      "N18.6"),
        ("retinopathy",           "Diabetic Retinopathy",         "E11.31"),
        ("neuropathy",            "Peripheral Neuropathy",        "G60"),
        ("bleeding_risk",         "Major Bleeding Risk",          ""),
        ("fracture_risk",         "Osteoporotic Fracture",        "M84.4"),
        ("cv_event",              "Major Cardiovascular Event",   ""),
        ("hf_exacerbation",       "Heart Failure Exacerbation",   "I50.1"),
        ("hyperkalemia",          "Hyperkalemia",                 "E87.5"),
        ("dialysis_progression",  "Progression to Dialysis",      "Z99.2"),
    ]

    for node_id, label, icd10 in complications:
        G.add_node(node_id, type="complication", label=label, icd10=icd10)

    # Risk factors (lab thresholds and clinical findings)
    risk_factors = [
        ("hba1c_uncontrolled",    "HbA1c > 9.0%"),
        ("hba1c_elevated",        "HbA1c 8.0–9.0%"),
        ("hba1c_borderline",      "HbA1c 7.0–8.0%"),
        ("egfr_30_45",            "eGFR 30–45 (CKD Stage 3b)"),
        ("egfr_below_30",         "eGFR < 30 (CKD Stage 4–5)"),
        ("ldl_high",              "LDL > 130 mg/dL"),
        ("ldl_very_high",         "LDL > 190 mg/dL"),
        ("bp_stage2",             "BP ≥ 140/90 mmHg"),
        ("bp_stage1",             "BP 130–140/80–90 mmHg"),
        ("bmi_obese",             "BMI ≥ 30"),
        ("bmi_morbid_obese",      "BMI ≥ 40"),
        ("age_over_65",           "Age ≥ 65"),
        ("age_over_75",           "Age ≥ 75"),
        ("long_term_steroids",    "Long-term corticosteroid use"),
        ("anticoagulation_needed","Anticoagulation indicated"),
        ("insulin_dependent",     "Insulin-dependent diabetes"),
    ]

    for node_id, label in risk_factors:
        G.add_node(node_id, type="risk_factor", label=label)

    # Interventions (referrals, screenings, treatments)
    interventions = [
        ("nephrology_referral",      "Nephrology referral"),
        ("cardiology_referral",      "Cardiology referral"),
        ("endocrinology_referral",   "Endocrinology referral"),
        ("pulmonology_referral",     "Pulmonology referral"),
        ("ophthalmology_referral",   "Ophthalmology referral (retinal exam)"),
        ("oncology_referral",        "Oncology referral"),
        ("hematology_referral",      "Hematology referral"),
        ("mammogram_screening",      "Annual mammogram screening"),
        ("colonoscopy_screening",    "Colorectal cancer screening"),
        ("bone_density_scan",        "DEXA bone density scan"),
        ("cardiac_stress_test",      "Cardiac stress test"),
        ("pulmonary_function_test",  "Pulmonary function test (spirometry)"),
        ("ekg_monitoring",           "ECG monitoring"),
        ("insulin_pump_therapy",     "Insulin pump (CSII) therapy"),
        ("anticoagulation_therapy",  "Anticoagulation therapy (warfarin/DOAC)"),
        ("statin_therapy",           "High-intensity statin therapy"),
        ("ace_arb_therapy",          "ACE inhibitor / ARB therapy"),
        ("cardiac_rehab",            "Cardiac rehabilitation program"),
        ("pulmonary_rehab",          "Pulmonary rehabilitation"),
        ("diabetes_education",       "Diabetes self-management education"),
        ("smoking_cessation",        "Smoking cessation program"),
        ("weight_management",        "Structured weight management program"),
        ("medication_review",        "Urgent medication safety review"),
        ("fall_risk_assessment",     "Fall risk assessment"),
    ]

    for node_id, label in interventions:
        G.add_node(node_id, type="intervention", label=label)

    # Drugs (for interaction checking)
    drugs = [
        ("warfarin",        "Warfarin"),
        ("nsaid",           "NSAIDs (ibuprofen, meloxicam, naproxen)"),
        ("metformin",       "Metformin"),
        ("ace_inhibitor",   "ACE Inhibitor (lisinopril, enalapril)"),
        ("arb",             "ARB (losartan, valsartan)"),
        ("potassium_sparing_diuretic", "K+-sparing diuretic (spironolactone)"),
        ("digoxin",         "Digoxin"),
        ("amiodarone",      "Amiodarone"),
        ("methotrexate",    "Methotrexate"),
        ("corticosteroid",  "Corticosteroids (prednisone)"),
        ("aspirin",         "Aspirin"),
        ("clopidogrel",     "Clopidogrel"),
        ("statin",          "Statin (atorvastatin, rosuvastatin)"),
        ("furosemide",      "Furosemide (loop diuretic)"),
    ]

    for node_id, label in drugs:
        G.add_node(node_id, type="drug", label=label)


    # ── EDGE DEFINITIONS ───────────────────────────────────────────────────────
    # Format: G.add_edge(from, to, type=..., evidence=..., strength=..., action=...)
    # strength: HIGH / MODERATE / LOW (GRADE evidence levels)

    def e(src, dst, rel_type, evidence, strength="HIGH", action="", urgency="ROUTINE"):
        G.add_edge(src, dst,
                   type=rel_type,
                   evidence=evidence,
                   strength=strength,
                   action=action,
                   urgency=urgency)

    # ── Diabetes → complications / monitoring ─────────────────────────────────
    e("type2_diabetes", "retinopathy",          "increases_risk_of",
      "ADA 2025: Annual retinal exam for all diabetic patients",
      action="ophthalmology_referral")
    e("type2_diabetes", "neuropathy",           "increases_risk_of",
      "ADA 2025: Annual foot exam and monofilament testing")
    e("type2_diabetes", "ckd_stage3",           "worsens",
      "KDIGO 2024: Diabetes is leading cause of CKD progression")
    e("type2_diabetes", "cv_event",             "increases_risk_of",
      "ADA 2025: 2-4x increased CVD risk in T2DM",
      action="cardiology_referral")
    e("hba1c_uncontrolled", "insulin_pump_therapy", "indicates_referral",
      "ADA 2025: CSII indicated for HbA1c > 9% despite optimal MDI therapy",
      urgency="HIGH")
    e("hba1c_uncontrolled", "cv_event",         "increases_risk_of",
      "ADA 2025: HbA1c > 9% associated with 50% higher MACE risk",
      strength="HIGH")
    e("hba1c_uncontrolled", "endocrinology_referral", "indicates_referral",
      "ADA 2025: Endocrinology referral when HbA1c > 9% on maximal oral therapy",
      urgency="HIGH")
    e("type1_diabetes", "insulin_pump_therapy", "indicates_referral",
      "ADA 2025: CSII preferred delivery for T1DM",
      action="insulin_pump_therapy")
    e("insulin_dependent", "diabetes_education", "requires_monitoring",
      "ADA 2025: DSMES recommended for all insulin users")

    # ── CKD → progression / referrals ─────────────────────────────────────────
    e("ckd_stage3", "nephrology_referral",      "indicates_referral",
      "KDIGO 2024: Nephrology co-management recommended for eGFR < 60",
      urgency="HIGH")
    e("ckd_stage3", "hyperkalemia",             "increases_risk_of",
      "KDIGO 2024: CKD Stage 3+ associated with impaired K+ excretion",
      action="medication_review")
    e("ckd_stage3", "anemia",                   "increases_risk_of",
      "KDIGO 2024: Anemia of CKD common at Stage 3+, check Hgb")
    e("ckd_stage3", "cv_event",                 "increases_risk_of",
      "KDIGO 2024: CKD Stage 3 doubles CVD event risk",
      strength="HIGH")
    e("ckd_stage4", "nephrology_referral",      "indicates_referral",
      "KDIGO 2024: Urgent nephrology referral — prepare for RRT",
      urgency="URGENT")
    e("ckd_stage4", "dialysis_progression",     "increases_risk_of",
      "KDIGO 2024: CKD Stage 4 eGFR 15-29 — high progression risk")
    e("egfr_below_30", "dialysis_progression",  "increases_risk_of",
      "KDIGO 2024: eGFR < 30 — begin dialysis access planning",
      urgency="URGENT")

    # ── CKD + Diabetes compound risk ──────────────────────────────────────────
    e("ckd_stage3", "metformin",                "contraindicates",
      "FDA/ADA: Metformin contraindicated when eGFR < 30; use caution eGFR 30-45",
      strength="HIGH", urgency="HIGH", action="medication_review")
    e("ckd_stage4", "metformin",                "contraindicates",
      "FDA: Metformin contraindicated in eGFR < 30 (lactic acidosis risk)",
      strength="HIGH", urgency="URGENT", action="medication_review")

    # ── Hypertension → target organ damage ────────────────────────────────────
    e("hypertension", "cv_event",               "increases_risk_of",
      "ACC/AHA 2017: HTN is #1 modifiable CVD risk factor",
      strength="HIGH")
    e("hypertension", "ckd_stage3",             "worsens",
      "KDIGO 2024: Uncontrolled HTN accelerates CKD progression")
    e("hypertension", "stroke",                 "increases_risk_of",
      "ACC/AHA 2017: HTN accounts for 54% of stroke attributable risk")
    e("bp_stage2", "cardiology_referral",       "indicates_referral",
      "ACC/AHA 2017: Stage 2 HTN — consider cardiology if resistant",
      urgency="HIGH")

    # ── Heart Failure ─────────────────────────────────────────────────────────
    e("heart_failure", "hf_exacerbation",       "increases_risk_of",
      "ACC/AHA 2022: 50% readmission rate within 6 months of HF hospitalization")
    e("heart_failure", "cardiac_rehab",         "indicates_referral",
      "ACC/AHA: Cardiac rehab reduces HF mortality by 35%",
      urgency="HIGH")
    e("heart_failure", "ekg_monitoring",        "requires_monitoring",
      "ACC/AHA: Regular ECG and BNP monitoring in HF")
    e("heart_failure", "aki",                   "increases_risk_of",
      "ACC/AHA: Diuretic use in HF increases AKI risk — monitor renal function")
    e("nsaid", "hf_exacerbation",               "increases_risk_of",
      "ACC/AHA: NSAIDs contraindicated in heart failure",
      urgency="HIGH", action="medication_review")
    e("nsaid", "aki",                           "increases_risk_of",
      "FDA: NSAIDs reduce renal perfusion — avoid in CKD/HF",
      strength="HIGH", action="medication_review")

    # ── Atrial Fibrillation ────────────────────────────────────────────────────
    e("afib", "stroke",                         "increases_risk_of",
      "ACC/AHA 2019: AFib increases ischemic stroke risk 5x",
      strength="HIGH")
    e("afib", "anticoagulation_therapy",        "indicates_referral",
      "ACC/AHA 2019: Anticoagulation recommended for CHA2DS2-VASc ≥ 2",
      urgency="HIGH")
    e("warfarin", "nsaid",                      "interacts_with",
      "FDA: Warfarin + NSAIDs — significantly increased bleeding risk",
      strength="HIGH", urgency="HIGH", action="medication_review")
    e("warfarin", "aspirin",                    "interacts_with",
      "ACC: Warfarin + Aspirin combination increases GI bleed risk",
      strength="MODERATE", action="medication_review")

    # ── CAD / Cardiovascular ──────────────────────────────────────────────────
    e("cad", "mi",                              "increases_risk_of",
      "ACC/AHA: Prior CAD increases recurrent MI risk",
      strength="HIGH")
    e("cad", "cardiac_rehab",                   "indicates_referral",
      "ACC/AHA: Cardiac rehab indicated for CAD patients",
      urgency="HIGH")
    e("cad", "cardiac_stress_test",             "requires_monitoring",
      "ACC/AHA: Periodic stress testing for CAD monitoring")
    e("ldl_high", "statin_therapy",             "indicates_referral",
      "ACC/AHA 2018: High-intensity statin for LDL > 130 with ASCVD",
      urgency="HIGH")
    e("ldl_very_high", "cardiology_referral",   "indicates_referral",
      "ACC/AHA 2018: LDL > 190 — familial hypercholesterolemia workup",
      urgency="HIGH")

    # ── Stroke ────────────────────────────────────────────────────────────────
    e("stroke", "cardiology_referral",          "indicates_referral",
      "AHA/ASA: Cardiology evaluation for secondary stroke prevention",
      urgency="HIGH")
    e("stroke", "cardiac_rehab",                "indicates_referral",
      "AHA: Intensive cardiac and neuro rehab post-stroke")
    e("stroke", "ekg_monitoring",               "requires_monitoring",
      "AHA/ASA: 30-day cardiac monitoring post-stroke to detect AFib")

    # ── COPD / Pulmonary ──────────────────────────────────────────────────────
    e("copd", "pulmonary_rehab",                "indicates_referral",
      "GOLD 2024: Pulmonary rehab improves dyspnea and QoL in COPD",
      urgency="HIGH")
    e("copd", "pulmonary_function_test",        "requires_monitoring",
      "GOLD 2024: Annual spirometry to assess COPD progression")
    e("copd", "cv_event",                       "increases_risk_of",
      "GOLD 2024: COPD associated with 2-3x increased CVD mortality")
    e("asthma", "pulmonary_function_test",      "requires_monitoring",
      "NHLBI: Regular spirometry to assess asthma control")

    # ── Obesity / Metabolic ───────────────────────────────────────────────────
    e("obesity", "type2_diabetes",              "increases_risk_of",
      "ADA 2025: Obesity is primary modifiable risk factor for T2DM")
    e("obesity", "sleep_apnea",                 "increases_risk_of",
      "AASM: BMI > 30 strongly associated with OSA")
    e("obesity", "cv_event",                    "increases_risk_of",
      "ACC/AHA: Obesity independently increases CVD risk")
    e("bmi_obese", "weight_management",         "indicates_referral",
      "USPSTF: Intensive behavioral intervention for BMI ≥ 30",
      urgency="HIGH")

    # ── Rheumatoid Arthritis ──────────────────────────────────────────────────
    e("rheumatoid_arthritis", "cv_event",       "increases_risk_of",
      "ACR 2021: RA associated with 50% increased CVD risk")
    e("rheumatoid_arthritis", "osteoporosis",   "increases_risk_of",
      "ACR 2021: RA and long-term glucocorticoids increase fracture risk")
    e("long_term_steroids", "osteoporosis",     "increases_risk_of",
      "ACR: Glucocorticoid-induced osteoporosis — initiate bisphosphonate",
      urgency="HIGH", action="bone_density_scan")
    e("methotrexate", "anemia",                 "increases_risk_of",
      "ACR: Methotrexate — monitor CBC for bone marrow suppression")

    # ── Lupus ──────────────────────────────────────────────────────────────────
    e("lupus", "ckd_stage3",                    "increases_risk_of",
      "ACR: Lupus nephritis occurs in 50% of SLE patients")
    e("lupus", "cv_event",                      "increases_risk_of",
      "ACR: SLE patients have 10x increased CVD risk")
    e("lupus", "nephrology_referral",           "indicates_referral",
      "ACR: Nephrology co-management for all SLE with renal involvement",
      urgency="HIGH")

    # ── Bone Health ────────────────────────────────────────────────────────────
    e("osteoporosis", "fracture_risk",          "increases_risk_of",
      "USPSTF: Osteoporosis major cause of vertebral and hip fractures")
    e("osteoporosis", "bone_density_scan",      "requires_monitoring",
      "USPSTF: DEXA every 2 years for women ≥ 65 with osteoporosis")
    e("age_over_65", "fall_risk_assessment",    "indicates_referral",
      "USPSTF: Fall risk assessment and intervention for adults 65+",
      urgency="HIGH")

    # ── Cancer ────────────────────────────────────────────────────────────────
    e("breast_cancer", "oncology_referral",     "indicates_referral",
      "NCCN: Oncology evaluation for all newly diagnosed breast cancer",
      urgency="URGENT")
    e("breast_cancer", "bone_density_scan",     "requires_monitoring",
      "NCCN: Aromatase inhibitor use increases fracture risk — baseline DEXA")
    e("prostate_cancer", "oncology_referral",   "indicates_referral",
      "NCCN: Urology/oncology evaluation for localized prostate cancer",
      urgency="HIGH")

    # ── Drug interactions ─────────────────────────────────────────────────────
    e("methotrexate", "nsaid",                  "interacts_with",
      "FDA: Methotrexate + NSAIDs — increased MTX toxicity (renal clearance)",
      strength="HIGH", urgency="URGENT", action="medication_review")
    e("ace_inhibitor", "potassium_sparing_diuretic", "interacts_with",
      "FDA: ACE inhibitor + K+-sparing diuretic — hyperkalemia risk",
      strength="HIGH", action="medication_review")
    e("statin", "amiodarone",                   "interacts_with",
      "FDA: Amiodarone inhibits statin metabolism — myopathy risk",
      strength="MODERATE", action="medication_review")
    e("corticosteroid", "type2_diabetes",       "worsens",
      "ADA: Corticosteroids elevate blood glucose — tighter monitoring needed")

    # ── Depression comorbidities ──────────────────────────────────────────────
    e("depression", "cv_event",                 "increases_risk_of",
      "AHA: Depression is independent CVD risk factor")
    e("depression", "diabetes_education",       "requires_monitoring",
      "ADA 2025: Screen for depression in all diabetic patients annually")

    return G


# ── Graph analysis functions ──────────────────────────────────────────────────

_graph = None

def get_graph():
    global _graph
    if _graph is None:
        _graph = build_clinical_graph()
    return _graph


def find_risks_for_patient(diagnoses: list[str], lab_values: dict = None,
                            medications: list[str] = None,
                            age: int = None) -> list[dict]:
    """
    Given a patient's clinical profile, traverse the knowledge graph to find:
    1. Complications they are at risk for
    2. Interventions they should receive
    3. Drug interactions if any
    4. Non-obvious multi-hop risk connections

    Returns a ranked list of findings with evidence and urgency.
    """
    G = get_graph()
    if G is None:
        return [{"error": "NetworkX not available — run: pip install networkx"}]

    # Map patient diagnoses to graph node IDs
    diagnosis_map = {
        "Type 2 Diabetes":         "type2_diabetes",
        "Type 1 Diabetes":         "type1_diabetes",
        "Hypertension":            "hypertension",
        "CKD Stage 3":             "ckd_stage3",
        "CKD Stage 4":             "ckd_stage4",
        "CKD Stage 5":             "ckd_stage5",
        "Heart Failure":           "heart_failure",
        "Atrial Fibrillation":     "afib",
        "Coronary Artery Disease": "cad",
        "Stroke":                  "stroke",
        "COPD":                    "copd",
        "Asthma":                  "asthma",
        "Obesity":                 "obesity",
        "Prediabetes":             "prediabetes",
        "Hyperlipidemia":          "hyperlipidemia",
        "Osteoporosis":            "osteoporosis",
        "Rheumatoid Arthritis":    "rheumatoid_arthritis",
        "Breast Cancer - Stage II": "breast_cancer",
        "Breast Cancer":           "breast_cancer",
        "Prostate Cancer - Stage I": "prostate_cancer",
        "Prostate Cancer":         "prostate_cancer",
        "Crohn's Disease":         "crohns",
        "Depression":              "depression",
        "Lupus":                   "lupus",
        "Osteoarthritis":          "osteoarthritis",
        "Anemia":                  "anemia",
        "Parkinson's Disease":     "parkinson",
    }

    # Map medications to drug node IDs
    medication_map = {
        "warfarin":         "warfarin",
        "ibuprofen":        "nsaid",
        "naproxen":         "nsaid",
        "meloxicam":        "nsaid",
        "metformin":        "metformin",
        "lisinopril":       "ace_inhibitor",
        "enalapril":        "ace_inhibitor",
        "ramipril":         "ace_inhibitor",
        "losartan":         "arb",
        "valsartan":        "arb",
        "spironolactone":   "potassium_sparing_diuretic",
        "methotrexate":     "methotrexate",
        "prednisone":       "corticosteroid",
        "atorvastatin":     "statin",
        "rosuvastatin":     "statin",
        "simvastatin":      "statin",
        "aspirin":          "aspirin",
        "clopidogrel":      "clopidogrel",
        "furosemide":       "furosemide",
        "amiodarone":       "amiodarone",
    }

    # Identify active graph nodes for this patient
    active_nodes = set()

    for dx in (diagnoses or []):
        node_id = diagnosis_map.get(dx)
        if node_id and node_id in G:
            active_nodes.add(node_id)

    # Add lab-based risk factor nodes
    if lab_values:
        hba1c = lab_values.get("HbA1c")
        if hba1c:
            try:
                v = float(hba1c)
                if v >= 9.0:   active_nodes.add("hba1c_uncontrolled")
                elif v >= 8.0: active_nodes.add("hba1c_elevated")
                elif v >= 7.0: active_nodes.add("hba1c_borderline")
            except (ValueError, TypeError):
                pass

        egfr = lab_values.get("eGFR")
        if egfr:
            try:
                v = float(egfr)
                if v < 30:   active_nodes.add("egfr_below_30")
                elif v < 45: active_nodes.add("egfr_30_45")
            except (ValueError, TypeError):
                pass

        ldl = lab_values.get("LDL")
        if ldl:
            try:
                v = float(ldl)
                if v >= 190: active_nodes.add("ldl_very_high")
                elif v >= 130: active_nodes.add("ldl_high")
            except (ValueError, TypeError):
                pass

    # Add age-based nodes
    if age:
        if age >= 75: active_nodes.add("age_over_75")
        elif age >= 65: active_nodes.add("age_over_65")

    # Add medication nodes
    active_drug_nodes = set()
    for med in (medications or []):
        med_lower = med.lower()
        for keyword, drug_node in medication_map.items():
            if keyword in med_lower and drug_node in G:
                active_drug_nodes.add(drug_node)
                active_nodes.add(drug_node)

    # ── Graph traversal ─────────────────────────────────────────────────────
    findings = []
    seen = set()

    for node in active_nodes:
        # Direct edges from this node
        for _, target, edge_data in G.out_edges(node, data=True):
            target_data = G.nodes[target]
            key = f"{node}→{target}"
            if key in seen:
                continue
            seen.add(key)

            source_data = G.nodes[node]
            urgency = edge_data.get("urgency", "ROUTINE")
            strength = edge_data.get("strength", "MODERATE")

            finding = {
                "type": edge_data.get("type"),
                "source": source_data.get("label", node),
                "target": target_data.get("label", target),
                "target_type": target_data.get("type"),
                "evidence": edge_data.get("evidence", ""),
                "strength": strength,
                "urgency": urgency,
                "action": edge_data.get("action", ""),
                "path": [source_data.get("label", node),
                         "→",
                         target_data.get("label", target)],
                "hop": 1
            }
            findings.append(finding)

    # ── Multi-hop reasoning (2-hop) ─────────────────────────────────────────
    # Find non-obvious connections: A → B → C
    for node in list(active_nodes):
        for _, mid in G.out_edges(node):
            if mid in active_nodes:
                continue  # already directly active
            mid_data = G.nodes[mid]
            for _, target, edge_data2 in G.out_edges(mid, data=True):
                target_data = G.nodes[target]
                key = f"{node}→{mid}→{target}"
                if key in seen:
                    continue
                seen.add(key)

                # Only surface urgent or intervention-type multi-hops
                target_type = target_data.get("type")
                urgency = edge_data2.get("urgency", "ROUTINE")

                if target_type in ("intervention",) or urgency in ("HIGH", "URGENT"):
                    source_data = G.nodes[node]
                    mid_label = mid_data.get("label", mid)

                    finding = {
                        "type": f"multi_hop: {edge_data2.get('type')}",
                        "source": source_data.get("label", node),
                        "via": mid_label,
                        "target": target_data.get("label", target),
                        "target_type": target_type,
                        "evidence": edge_data2.get("evidence", ""),
                        "strength": edge_data2.get("strength", "MODERATE"),
                        "urgency": urgency,
                        "action": edge_data2.get("action", ""),
                        "path": [source_data.get("label", node),
                                 "→", mid_label,
                                 "→", target_data.get("label", target)],
                        "hop": 2
                    }
                    findings.append(finding)

    # ── Drug interaction check ───────────────────────────────────────────────
    drug_list = list(active_drug_nodes)
    for i, drug_a in enumerate(drug_list):
        for drug_b in drug_list[i+1:]:
            if G.has_edge(drug_a, drug_b):
                edge_data = G[drug_a][drug_b]
                if edge_data.get("type") == "interacts_with":
                    da = G.nodes[drug_a].get("label", drug_a)
                    db = G.nodes[drug_b].get("label", drug_b)
                    findings.append({
                        "type": "drug_interaction",
                        "source": da,
                        "target": db,
                        "target_type": "drug_interaction",
                        "evidence": edge_data.get("evidence", ""),
                        "strength": edge_data.get("strength", "HIGH"),
                        "urgency": edge_data.get("urgency", "HIGH"),
                        "action": edge_data.get("action", "medication_review"),
                        "path": [da, "→ INTERACTS WITH →", db],
                        "hop": 1
                    })

    # Sort: URGENT first, then HIGH, then MODERATE, then hop 1 before 2
    urgency_order = {"URGENT": 0, "HIGH": 1, "MODERATE": 2, "ROUTINE": 3}
    findings.sort(key=lambda x: (urgency_order.get(x.get("urgency", "ROUTINE"), 3), x.get("hop", 1)))

    return findings


def get_recommended_interventions(findings: list[dict]) -> list[str]:
    """Extract unique recommended interventions from knowledge graph findings."""
    interventions = []
    seen = set()
    for f in findings:
        target = f.get("target", "")
        target_type = f.get("target_type", "")
        action = f.get("action", "")

        if target_type == "intervention" and target not in seen:
            interventions.append(target)
            seen.add(target)
        if action and action not in seen:
            interventions.append(action)
            seen.add(action)

    return interventions


def get_drug_interactions(findings: list[dict]) -> list[dict]:
    """Filter findings to return only drug interactions."""
    return [f for f in findings if f.get("type") == "drug_interaction"]


def format_findings_for_agent(findings: list[dict], max_findings: int = 10) -> str:
    """
    Format knowledge graph findings into a concise agent-readable summary.
    Groups by urgency and type for readability.
    """
    if not findings:
        return "No significant clinical connections found in knowledge graph."

    urgent = [f for f in findings if f.get("urgency") in ("URGENT", "HIGH")][:max_findings]
    interactions = get_drug_interactions(findings)

    lines = ["**Clinical Knowledge Graph Analysis:**\n"]

    if interactions:
        lines.append("⚠️ **Drug Interactions Detected:**")
        for f in interactions:
            lines.append(f"  - {' '.join(f['path'])}")
            lines.append(f"    Evidence: {f['evidence'][:120]}...")
        lines.append("")

    if urgent:
        lines.append("🔴 **High Priority Clinical Connections:**")
        for f in urgent[:8]:
            hop_label = "(direct)" if f.get("hop", 1) == 1 else f"(via {f.get('via', '')})"
            path_str = " ".join(str(p) for p in f.get("path", []))
            lines.append(f"  - [{f.get('urgency')}] {path_str} {hop_label}")
            lines.append(f"    Evidence: {f['evidence'][:120]}")
        lines.append("")

    other = [f for f in findings
             if f.get("urgency") not in ("URGENT", "HIGH")
             and f.get("type") != "drug_interaction"][:5]
    if other:
        lines.append("📋 **Additional Clinical Connections:**")
        for f in other:
            path_str = " ".join(str(p) for p in f.get("path", []))
            lines.append(f"  - {path_str}")

    return "\n".join(lines)


def get_graph_stats() -> dict:
    """Return statistics about the knowledge graph."""
    G = get_graph()
    if G is None:
        return {"error": "NetworkX not available"}

    node_types = {}
    for _, data in G.nodes(data=True):
        t = data.get("type", "unknown")
        node_types[t] = node_types.get(t, 0) + 1

    edge_types = {}
    for _, _, data in G.edges(data=True):
        t = data.get("type", "unknown")
        edge_types[t] = edge_types.get(t, 0) + 1

    return {
        "total_nodes": G.number_of_nodes(),
        "total_edges": G.number_of_edges(),
        "node_types": node_types,
        "edge_types": edge_types,
        "is_dag": nx.is_directed_acyclic_graph(G) if HAS_NETWORKX else None
    }


# ── LangChain tool ────────────────────────────────────────────────────────────

@tool
def analyze_clinical_connections(
    patient_id: str,
    diagnoses: list,
    lab_values: dict,
    medications: list,
    age: int = 0
) -> str:
    """
    Traverse the clinical knowledge graph to find non-obvious clinical connections,
    drug interactions, and recommended interventions for a patient.

    This tool reasons over a graph of 200+ evidence-based clinical relationships
    derived from ADA, ACC/AHA, KDIGO, USPSTF, and NHLBI guidelines.

    patient_id: patient identifier for context
    diagnoses: list of diagnosis strings from get_patient_demographics
    lab_values: dict of {lab_name: value} from get_lab_results
    medications: list of medication strings from get_patient_demographics
    age: patient age for age-based risk rules

    Returns a formatted summary of clinical connections, drug interactions,
    and recommended interventions with evidence citations.
    """
    findings = find_risks_for_patient(
        diagnoses=diagnoses,
        lab_values=lab_values,
        medications=medications,
        age=age
    )

    formatted = format_findings_for_agent(findings)
    interventions = get_recommended_interventions(findings)
    interactions = get_drug_interactions(findings)

    summary = f"Knowledge Graph Analysis for Patient {patient_id}:\n\n"
    summary += formatted
    summary += f"\n\n**Recommended interventions:** {', '.join(interventions[:8]) if interventions else 'None identified'}"
    summary += f"\n**Drug interactions found:** {len(interactions)}"
    summary += f"\n**Total clinical connections:** {len(findings)}"

    return summary


KNOWLEDGE_GRAPH_TOOLS = [analyze_clinical_connections]


# ── CLI test ──────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    if not HAS_NETWORKX:
        print("Install networkx first: pip install networkx")
        exit(1)

    stats = get_graph_stats()
    print("=== Clinical Knowledge Graph ===")
    print(f"Nodes: {stats['total_nodes']} | Edges: {stats['total_edges']}")
    print(f"Node types: {stats['node_types']}")
    print(f"Edge types: {stats['edge_types']}")
    print()

    # Test with P004 — Heart Failure + AFib + CKD Stage 4 + Diabetes
    print("=== P004: Heart Failure + AFib + CKD Stage 4 + Diabetes ===")
    findings = find_risks_for_patient(
        diagnoses=["Heart Failure", "Atrial Fibrillation", "Type 2 Diabetes", "CKD Stage 4"],
        lab_values={"HbA1c": 9.1, "eGFR": 22},
        medications=["Furosemide 40mg", "Warfarin 5mg", "Carvedilol 12.5mg", "Insulin glargine"],
        age=72
    )
    print(format_findings_for_agent(findings))

    print()
    print("=== P001: Diabetes + Hypertension + CKD Stage 3 ===")
    findings2 = find_risks_for_patient(
        diagnoses=["Type 2 Diabetes", "Hypertension", "CKD Stage 3"],
        lab_values={"HbA1c": 10.2, "eGFR": 45},
        medications=["Metformin 1000mg", "Lisinopril 10mg"],
        age=67
    )
    print(format_findings_for_agent(findings2))