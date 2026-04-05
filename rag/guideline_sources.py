"""
rag/guideline_sources.py

Registry of all clinical guideline sources.
Add new sources here — the scraper and embedder pick them up automatically.

Each source has:
  - id:         unique slug used in ChromaDB metadata
  - name:       human-readable name shown in citations
  - url:        URL to fetch the guideline content from
  - category:   clinical domain for filtering
  - refresh:    "weekly" or "manual" — manual means only refresh when explicitly triggered
  - priority:   HIGH sources block agent decisions until refreshed if stale > 90 days
"""

GUIDELINE_SOURCES = [

    # ── USPSTF Preventive Care ────────────────────────────────────────────────
    {
        "id": "uspstf_breast_cancer",
        "name": "USPSTF: Breast Cancer Screening",
        "url": "https://www.uspreventiveservicestaskforce.org/uspstf/recommendation/breast-cancer-screening",
        "category": "preventive_screening",
        "refresh": "weekly",
        "priority": "HIGH",
        "keywords": ["mammogram", "breast cancer", "screening", "women", "40", "50"]
    },
    {
        "id": "uspstf_colorectal",
        "name": "USPSTF: Colorectal Cancer Screening",
        "url": "https://www.uspreventiveservicestaskforce.org/uspstf/recommendation/colorectal-cancer-screening",
        "category": "preventive_screening",
        "refresh": "weekly",
        "priority": "HIGH",
        "keywords": ["colonoscopy", "colorectal", "colon cancer", "45", "FIT", "stool test"]
    },
    {
        "id": "uspstf_hypertension",
        "name": "USPSTF: Hypertension Screening",
        "url": "https://www.uspreventiveservicestaskforce.org/uspstf/recommendation/hypertension-in-adults-screening",
        "category": "cardiovascular",
        "refresh": "weekly",
        "priority": "HIGH",
        "keywords": ["hypertension", "blood pressure", "screening", "adults"]
    },
    {
        "id": "uspstf_diabetes",
        "name": "USPSTF: Prediabetes and Type 2 Diabetes Screening",
        "url": "https://www.uspreventiveservicestaskforce.org/uspstf/recommendation/screening-for-prediabetes-and-type-2-diabetes",
        "category": "diabetes",
        "refresh": "weekly",
        "priority": "HIGH",
        "keywords": ["diabetes", "prediabetes", "HbA1c", "fasting glucose", "screening"]
    },
    {
        "id": "uspstf_lung_cancer",
        "name": "USPSTF: Lung Cancer Screening",
        "url": "https://www.uspreventiveservicestaskforce.org/uspstf/recommendation/lung-cancer-screening",
        "category": "preventive_screening",
        "refresh": "weekly",
        "priority": "MEDIUM",
        "keywords": ["lung cancer", "LDCT", "smoking", "low-dose CT"]
    },
    {
        "id": "uspstf_depression",
        "name": "USPSTF: Depression Screening in Adults",
        "url": "https://www.uspreventiveservicestaskforce.org/uspstf/recommendation/depression-in-adults-screening",
        "category": "mental_health",
        "refresh": "weekly",
        "priority": "MEDIUM",
        "keywords": ["depression", "PHQ-9", "mental health", "screening", "adults"]
    },

    # ── ADA Standards of Medical Care in Diabetes ─────────────────────────────
    {
        "id": "ada_standards_2025",
        "name": "ADA Standards of Medical Care in Diabetes 2025",
        "url": "https://diabetesjournals.org/care/issue/48/Supplement_1",
        "category": "diabetes",
        "refresh": "manual",
        "priority": "HIGH",
        "keywords": ["HbA1c", "diabetes management", "insulin", "metformin",
                     "blood glucose", "A1c target", "GLP-1", "SGLT2"]
    },
    {
        "id": "ada_ckd_diabetes",
        "name": "ADA: Chronic Kidney Disease and Risk Management",
        "url": "https://diabetesjournals.org/care/article/48/Supplement_1/S327/157563",
        "category": "diabetes_ckd",
        "refresh": "manual",
        "priority": "HIGH",
        "keywords": ["CKD", "diabetic nephropathy", "eGFR", "UACR", "kidney", "nephrology"]
    },

    # ── ACC/AHA Cardiovascular Guidelines ─────────────────────────────────────
    {
        "id": "aha_hypertension_2017",
        "name": "ACC/AHA Hypertension Guidelines",
        "url": "https://www.ahajournals.org/doi/10.1161/HYP.0000000000000065",
        "category": "cardiovascular",
        "refresh": "manual",
        "priority": "HIGH",
        "keywords": ["hypertension", "blood pressure", "130/80", "140/90",
                     "antihypertensive", "lifestyle", "CVD risk"]
    },
    {
        "id": "aha_cholesterol_2018",
        "name": "ACC/AHA Cholesterol Guidelines",
        "url": "https://www.ahajournals.org/doi/10.1161/CIR.0000000000000625",
        "category": "cardiovascular",
        "refresh": "manual",
        "priority": "HIGH",
        "keywords": ["LDL", "cholesterol", "statin", "ASCVD", "cardiovascular risk",
                     "hyperlipidemia", "HDL"]
    },

    # ── KDIGO CKD Guidelines ──────────────────────────────────────────────────
    {
        "id": "kdigo_ckd_2024",
        "name": "KDIGO CKD Management Guidelines 2024",
        "url": "https://kdigo.org/guidelines/ckd-evaluation-and-management/",
        "category": "ckd",
        "refresh": "manual",
        "priority": "HIGH",
        "keywords": ["CKD", "eGFR", "nephrology", "kidney disease", "UACR",
                     "proteinuria", "dialysis", "Stage 3", "Stage 4"]
    },

    # ── CDC Immunization Guidelines ───────────────────────────────────────────
    {
        "id": "cdc_adult_immunization_2025",
        "name": "CDC Adult Immunization Schedule 2025",
        "url": "https://www.cdc.gov/vaccines/schedules/hcp/imz/adult.html",
        "category": "immunization",
        "refresh": "weekly",
        "priority": "HIGH",
        "keywords": ["flu vaccine", "influenza", "pneumonia vaccine", "shingles",
                     "Tdap", "COVID", "RSV", "immunization", "annual"]
    },

    # ── CMS HEDIS Quality Measures ────────────────────────────────────────────
    {
        "id": "hedis_diabetes_measures",
        "name": "NCQA HEDIS: Comprehensive Diabetes Care",
        "url": "https://www.ncqa.org/hedis/measures/comprehensive-diabetes-care/",
        "category": "diabetes",
        "refresh": "manual",
        "priority": "HIGH",
        "keywords": ["HbA1c testing", "eye exam", "kidney care", "blood pressure control",
                     "statin therapy", "HEDIS", "diabetes quality measure"]
    },
    {
        "id": "hedis_colorectal_screening",
        "name": "NCQA HEDIS: Colorectal Cancer Screening",
        "url": "https://www.ncqa.org/hedis/measures/colorectal-cancer-screening/",
        "category": "preventive_screening",
        "refresh": "manual",
        "priority": "MEDIUM",
        "keywords": ["colorectal screening", "HEDIS", "colonoscopy", "stool test",
                     "FIT", "quality measure", "45 to 75"]
    },

    # ── JNC Hypertension (fallback static content) ────────────────────────────
    {
        "id": "jnc8_hypertension",
        "name": "JNC 8 Hypertension Treatment Guidelines",
        "url": "https://jamanetwork.com/journals/jama/fullarticle/1791497",
        "category": "cardiovascular",
        "refresh": "manual",
        "priority": "MEDIUM",
        "keywords": ["hypertension treatment", "blood pressure target", "ACE inhibitor",
                     "ARB", "diuretic", "calcium channel blocker", "JNC"]
    },
]

# Quick lookup by ID
SOURCES_BY_ID = {s["id"]: s for s in GUIDELINE_SOURCES}

# Categories for filtering
CATEGORIES = list(set(s["category"] for s in GUIDELINE_SOURCES))

# Sources that refresh automatically every week
WEEKLY_SOURCES = [s for s in GUIDELINE_SOURCES if s["refresh"] == "weekly"]

# Sources that only refresh on manual trigger
MANUAL_SOURCES = [s for s in GUIDELINE_SOURCES if s["refresh"] == "manual"]
