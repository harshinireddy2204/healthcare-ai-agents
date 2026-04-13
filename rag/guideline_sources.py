"""
rag/guideline_sources.py

Comprehensive registry of clinical guideline sources across all major
medical specialties.

Source selection criteria:
  1. Must be freely accessible (no paywall, no login required)
  2. Must be authoritative (government, major professional society, NIH)
  3. Must have substantive text content (not just a landing page)
  4. Mirrors used where primary sites block scrapers (journals → NIH PMC / NHLBI / NIDDK)

Coverage:
  - Cardiovascular (AHA/ACC, NHLBI)
  - Diabetes (ADA via PMC, NIDDK)
  - Chronic Kidney Disease (KDIGO, NIDDK)
  - Oncology (NCI, NCCN summaries)
  - Preventive Care (USPSTF)
  - Immunization (CDC)
  - Mental Health (NICE, SAMHSA)
  - Infectious Disease (CDC, NIH)
  - Pulmonary / COPD (NHLBI, GOLD)
  - Geriatrics (NIA, AGS)
  - Obesity / Metabolic (CDC, NHLBI)
  - Rheumatology (NIH, NIAMS)
  - Neurology / Stroke (NHLBI, NIH)
  - Women's Health (ACOG via NIH, USPSTF)
  - HEDIS Quality Measures (NCQA)
"""

GUIDELINE_SOURCES = [

    # ════════════════════════════════════════════════════════════════════════
    # PREVENTIVE CARE — USPSTF
    # ════════════════════════════════════════════════════════════════════════
    {
        "id": "uspstf_breast_cancer",
        "name": "USPSTF: Breast Cancer Screening",
        "url": "https://www.uspreventiveservicestaskforce.org/uspstf/recommendation/breast-cancer-screening",
        "category": "preventive_screening",
        "refresh": "weekly",
        "priority": "HIGH",
        "keywords": ["mammogram", "breast cancer", "screening", "women", "40", "50", "biennial"]
    },
    {
        "id": "uspstf_colorectal",
        "name": "USPSTF: Colorectal Cancer Screening",
        "url": "https://www.uspreventiveservicestaskforce.org/uspstf/recommendation/colorectal-cancer-screening",
        "category": "preventive_screening",
        "refresh": "weekly",
        "priority": "HIGH",
        "keywords": ["colonoscopy", "colorectal", "colon cancer", "45", "FIT", "stool test", "cologuard"]
    },
    {
        "id": "uspstf_hypertension",
        "name": "USPSTF: Hypertension Screening in Adults",
        "url": "https://www.uspreventiveservicestaskforce.org/uspstf/recommendation/hypertension-in-adults-screening",
        "category": "cardiovascular",
        "refresh": "weekly",
        "priority": "HIGH",
        "keywords": ["hypertension", "blood pressure", "screening", "adults", "130", "140"]
    },
    {
        "id": "uspstf_diabetes",
        "name": "USPSTF: Prediabetes and Type 2 Diabetes Screening",
        "url": "https://www.uspreventiveservicestaskforce.org/uspstf/recommendation/screening-for-prediabetes-and-type-2-diabetes",
        "category": "diabetes",
        "refresh": "weekly",
        "priority": "HIGH",
        "keywords": ["diabetes", "prediabetes", "HbA1c", "fasting glucose", "screening", "overweight"]
    },
    {
        "id": "uspstf_lung_cancer",
        "name": "USPSTF: Lung Cancer Screening",
        "url": "https://www.uspreventiveservicestaskforce.org/uspstf/recommendation/lung-cancer-screening",
        "category": "oncology",
        "refresh": "weekly",
        "priority": "MEDIUM",
        "keywords": ["lung cancer", "LDCT", "smoking", "low-dose CT", "50 years", "pack-year"]
    },
    {
        "id": "uspstf_depression",
        "name": "USPSTF: Depression Screening in Adults",
        "url": "https://www.uspreventiveservicestaskforce.org/uspstf/recommendation/depression-in-adults-screening",
        "category": "mental_health",
        "refresh": "weekly",
        "priority": "HIGH",
        "keywords": ["depression", "PHQ-9", "mental health", "screening", "adults", "PHQ-2"]
    },
    {
        "id": "uspstf_anxiety",
        "name": "USPSTF: Anxiety Disorders Screening in Adults",
        "url": "https://www.uspreventiveservicestaskforce.org/uspstf/recommendation/anxiety-adults-screening",
        "category": "mental_health",
        "refresh": "weekly",
        "priority": "MEDIUM",
        "keywords": ["anxiety", "GAD-7", "mental health", "generalized anxiety", "screening"]
    },
    {
        "id": "uspstf_osteoporosis",
        "name": "USPSTF: Osteoporosis Screening in Women",
        "url": "https://www.uspreventiveservicestaskforce.org/uspstf/recommendation/osteoporosis-screening",
        "category": "bone_health",
        "refresh": "weekly",
        "priority": "MEDIUM",
        "keywords": ["osteoporosis", "bone density", "DEXA", "fracture risk", "women", "65"]
    },
    {
        "id": "uspstf_obesity",
        "name": "USPSTF: Weight Loss to Prevent Obesity-Related Morbidity",
        "url": "https://www.uspreventiveservicestaskforce.org/uspstf/recommendation/obesity-in-adults-interventions",
        "category": "obesity_metabolic",
        "refresh": "weekly",
        "priority": "HIGH",
        "keywords": ["obesity", "BMI", "weight loss", "bariatric", "intensive behavioral counseling"]
    },
    {
        "id": "uspstf_cervical_cancer",
        "name": "USPSTF: Cervical Cancer Screening",
        "url": "https://www.uspreventiveservicestaskforce.org/uspstf/recommendation/cervical-cancer-screening",
        "category": "womens_health",
        "refresh": "weekly",
        "priority": "HIGH",
        "keywords": ["cervical cancer", "Pap smear", "HPV test", "21 years", "65 years", "colposcopy"]
    },
    {
        "id": "uspstf_prostate_cancer",
        "name": "USPSTF: Prostate Cancer Screening",
        "url": "https://www.uspreventiveservicestaskforce.org/uspstf/recommendation/prostate-cancer-screening",
        "category": "oncology",
        "refresh": "weekly",
        "priority": "MEDIUM",
        "keywords": ["prostate cancer", "PSA", "55 years", "70 years", "shared decision making"]
    },
    {
        "id": "uspstf_alcohol",
        "name": "USPSTF: Unhealthy Alcohol Use in Adolescents and Adults",
        "url": "https://www.uspreventiveservicestaskforce.org/uspstf/recommendation/unhealthy-alcohol-use-in-adolescents-and-adults-screening-and-behavioral-counseling-interventions",
        "category": "behavioral_health",
        "refresh": "weekly",
        "priority": "MEDIUM",
        "keywords": ["alcohol", "AUDIT", "AUDIT-C", "screening", "brief intervention", "counseling"]
    },
    {
        "id": "uspstf_tobacco",
        "name": "USPSTF: Tobacco Smoking Cessation in Adults",
        "url": "https://www.uspreventiveservicestaskforce.org/uspstf/recommendation/tobacco-use-in-adults-and-pregnant-women-counseling-and-interventions",
        "category": "behavioral_health",
        "refresh": "weekly",
        "priority": "HIGH",
        "keywords": ["smoking", "tobacco", "cessation", "nicotine", "varenicline", "NRT", "counseling"]
    },
    {
        "id": "uspstf_falls_prevention",
        "name": "USPSTF: Falls Prevention in Community-Dwelling Older Adults",
        "url": "https://www.uspreventiveservicestaskforce.org/uspstf/recommendation/falls-prevention-in-older-adults-interventions",
        "category": "geriatrics",
        "refresh": "weekly",
        "priority": "MEDIUM",
        "keywords": ["falls", "older adults", "exercise", "vitamin D", "65 years", "fall risk"]
    },
    {
        "id": "uspstf_statin",
        "name": "USPSTF: Statin Use for Prevention of CVD in Adults",
        "url": "https://www.uspreventiveservicestaskforce.org/uspstf/recommendation/statin-use-in-adults-preventive-medication",
        "category": "cardiovascular",
        "refresh": "weekly",
        "priority": "HIGH",
        "keywords": ["statin", "CVD prevention", "ASCVD risk", "40 years", "75 years", "cardiovascular"]
    },

    # ════════════════════════════════════════════════════════════════════════
    # IMMUNIZATION — CDC
    # ════════════════════════════════════════════════════════════════════════
    {
        "id": "cdc_adult_immunization_schedule",
        "name": "CDC Adult Immunization Schedule 2025",
        "url": "https://www.cdc.gov/vaccines/hcp/imz-schedules/adult-age.html",
        "category": "immunization",
        "refresh": "weekly",
        "priority": "HIGH",
        "keywords": ["flu vaccine", "influenza", "pneumonia", "shingles", "Tdap", "COVID",
                     "RSV", "immunization", "annual", "adult vaccines", "Shingrix"]
    },
    {
        "id": "cdc_immunization_notes",
        "name": "CDC Adult Immunization Schedule Notes 2025",
        "url": "https://www.cdc.gov/vaccines/hcp/imz-schedules/adult-notes.html",
        "category": "immunization",
        "refresh": "weekly",
        "priority": "HIGH",
        "keywords": ["PCV15", "PCV20", "pneumococcal", "PPSV23", "Prevnar", "Shingrix",
                     "hepatitis B", "meningococcal", "special populations", "immunocompromised"]
    },
    {
        "id": "cdc_flu_guidance",
        "name": "CDC: Influenza Antiviral Medications",
        "url": "https://www.cdc.gov/flu/treatment/index.htm",
        "category": "immunization",
        "refresh": "weekly",
        "priority": "MEDIUM",
        "keywords": ["influenza treatment", "Tamiflu", "oseltamivir", "antiviral", "high risk",
                     "hospitalized", "pneumonia", "flu complications"]
    },

    # ════════════════════════════════════════════════════════════════════════
    # DIABETES — ADA / NIDDK / PMC
    # ════════════════════════════════════════════════════════════════════════
    {
        "id": "ada_standards_2025",
        "name": "ADA Standards of Medical Care in Diabetes 2025",
        "url": "https://pmc.ncbi.nlm.nih.gov/articles/PMC12018997/",
        "category": "diabetes",
        "refresh": "manual",
        "priority": "HIGH",
        "keywords": ["HbA1c", "diabetes management", "insulin", "metformin", "A1c target",
                     "GLP-1", "SGLT2", "glycemic goals", "type 2 diabetes", "CGM",
                     "insulin pump", "7%", "8%", "9%"]
    },
    {
        "id": "niddk_diabetes_overview",
        "name": "NIDDK: Diabetes — Diagnosis and Treatment",
        "url": "https://www.niddk.nih.gov/health-information/diabetes",
        "category": "diabetes",
        "refresh": "manual",
        "priority": "HIGH",
        "keywords": ["type 2 diabetes", "insulin resistance", "A1C", "fasting glucose",
                     "diabetes symptoms", "oral medications", "blood sugar", "hyperglycemia"]
    },
    {
        "id": "niddk_diabetes_medicines",
        "name": "NIDDK: Insulin, Medicines, and Other Diabetes Treatments",
        "url": "https://www.niddk.nih.gov/health-information/diabetes/overview/insulin-medicines-treatments",
        "category": "diabetes",
        "refresh": "manual",
        "priority": "HIGH",
        "keywords": ["insulin pump", "CSII", "metformin", "GLP-1 agonist", "SGLT2 inhibitor",
                     "insulin lispro", "insulin glargine", "basal bolus", "continuous glucose"]
    },
    {
        "id": "ada_ckd_diabetes",
        "name": "NIDDK: Diabetic Kidney Disease",
        "url": "https://www.niddk.nih.gov/health-information/diabetes/overview/preventing-problems/diabetic-kidney-disease",
        "category": "diabetes_ckd",
        "refresh": "manual",
        "priority": "HIGH",
        "keywords": ["CKD", "diabetic nephropathy", "eGFR", "UACR", "kidney",
                     "albuminuria", "kidney disease diabetes", "ACE inhibitor"]
    },
    {
        "id": "hedis_diabetes_measures",
        "name": "NCQA HEDIS: Comprehensive Diabetes Care",
        "url": "https://www.ncqa.org/hedis/measures/comprehensive-diabetes-care/",
        "category": "diabetes",
        "refresh": "manual",
        "priority": "HIGH",
        "keywords": ["HbA1c testing", "eye exam", "kidney care", "blood pressure control",
                     "statin therapy", "HEDIS", "diabetes quality measure", "retinal exam"]
    },

    # ════════════════════════════════════════════════════════════════════════
    # CARDIOVASCULAR — NHLBI / AHA / ACC
    # ════════════════════════════════════════════════════════════════════════
    {
        "id": "nhlbi_hypertension",
        "name": "NHLBI: High Blood Pressure — Causes, Diagnosis, Treatment",
        "url": "https://www.nhlbi.nih.gov/health/high-blood-pressure",
        "category": "cardiovascular",
        "refresh": "manual",
        "priority": "HIGH",
        "keywords": ["hypertension", "blood pressure", "130/80", "140/90",
                     "antihypertensive", "CVD risk", "systolic", "diastolic",
                     "blood pressure target", "stage 1", "stage 2"]
    },
    {
        "id": "nhlbi_hypertension_treatment",
        "name": "NHLBI: High Blood Pressure Treatment",
        "url": "https://www.nhlbi.nih.gov/health/high-blood-pressure/treatment",
        "category": "cardiovascular",
        "refresh": "manual",
        "priority": "HIGH",
        "keywords": ["ACE inhibitor", "ARB", "diuretic", "calcium channel blocker",
                     "beta blocker", "lisinopril", "amlodipine", "DASH diet",
                     "lifestyle modification", "antihypertensive drugs"]
    },
    {
        "id": "nhlbi_cholesterol",
        "name": "NHLBI: Blood Cholesterol — Diagnosis and Management",
        "url": "https://www.nhlbi.nih.gov/health/blood-cholesterol",
        "category": "cardiovascular",
        "refresh": "manual",
        "priority": "HIGH",
        "keywords": ["LDL", "cholesterol", "statin", "ASCVD", "cardiovascular risk",
                     "hyperlipidemia", "HDL", "triglycerides", "lipid panel",
                     "atorvastatin", "rosuvastatin", "10-year risk"]
    },
    {
        "id": "nhlbi_heart_failure",
        "name": "NHLBI: Heart Failure — Causes, Symptoms, Treatment",
        "url": "https://www.nhlbi.nih.gov/health/heart-failure",
        "category": "cardiovascular",
        "refresh": "manual",
        "priority": "HIGH",
        "keywords": ["heart failure", "ejection fraction", "HFrEF", "HFpEF",
                     "BNP", "edema", "ACE inhibitor", "beta blocker", "diuretic",
                     "cardiac rehab", "carvedilol", "furosemide", "reduced EF"]
    },
    {
        "id": "nhlbi_coronary_artery",
        "name": "NHLBI: Coronary Heart Disease",
        "url": "https://www.nhlbi.nih.gov/health/coronary-heart-disease",
        "category": "cardiovascular",
        "refresh": "manual",
        "priority": "HIGH",
        "keywords": ["coronary artery disease", "CAD", "angina", "chest pain",
                     "cardiac stress test", "angiogram", "stent", "CABG",
                     "aspirin", "clopidogrel", "antiplatelet"]
    },
    {
        "id": "nhlbi_afib",
        "name": "NHLBI: Atrial Fibrillation",
        "url": "https://www.nhlbi.nih.gov/health/atrial-fibrillation",
        "category": "cardiovascular",
        "refresh": "manual",
        "priority": "HIGH",
        "keywords": ["atrial fibrillation", "AFib", "warfarin", "anticoagulant",
                     "stroke prevention", "CHA2DS2-VASc", "rate control",
                     "rhythm control", "cardioversion", "ablation"]
    },
    {
        "id": "aha_cardiac_rehab",
        "name": "AHA: Cardiac Rehabilitation",
        "url": "https://www.heart.org/en/health-topics/cardiac-rehab",
        "category": "cardiovascular",
        "refresh": "manual",
        "priority": "MEDIUM",
        "keywords": ["cardiac rehab", "cardiac rehabilitation", "exercise training",
                     "post-MI", "post-CABG", "heart failure rehab", "36 sessions"]
    },

    # ════════════════════════════════════════════════════════════════════════
    # CHRONIC KIDNEY DISEASE — KDIGO / NIDDK
    # ════════════════════════════════════════════════════════════════════════
    {
        "id": "kdigo_ckd_2024",
        "name": "KDIGO CKD Management Guidelines 2024",
        "url": "https://kdigo.org/guidelines/ckd-evaluation-and-management/",
        "category": "ckd",
        "refresh": "manual",
        "priority": "HIGH",
        "keywords": ["CKD", "eGFR", "nephrology", "kidney disease", "UACR",
                     "proteinuria", "dialysis", "Stage 3", "Stage 4", "Stage 5",
                     "nephrology referral", "progression"]
    },
    {
        "id": "niddk_ckd_overview",
        "name": "NIDDK: Chronic Kidney Disease — Stages and Management",
        "url": "https://www.niddk.nih.gov/health-information/kidney-disease/chronic-kidney-disease-ckd",
        "category": "ckd",
        "refresh": "manual",
        "priority": "HIGH",
        "keywords": ["CKD stage 3", "CKD stage 4", "eGFR 30", "eGFR 45", "eGFR 60",
                     "creatinine", "kidney function", "nephrology referral",
                     "renal diet", "phosphorus", "potassium", "anemia of CKD"]
    },
    {
        "id": "niddk_kidney_failure",
        "name": "NIDDK: Kidney Failure — ESRD Treatment Options",
        "url": "https://www.niddk.nih.gov/health-information/kidney-disease/kidney-failure",
        "category": "ckd",
        "refresh": "manual",
        "priority": "MEDIUM",
        "keywords": ["ESRD", "end stage renal disease", "dialysis", "hemodialysis",
                     "peritoneal dialysis", "kidney transplant", "eGFR 15"]
    },

    # ════════════════════════════════════════════════════════════════════════
    # ONCOLOGY — NCI / NIH
    # ════════════════════════════════════════════════════════════════════════
    {
        "id": "nci_breast_cancer_treatment",
        "name": "NCI: Breast Cancer Treatment (PDQ)",
        "url": "https://www.cancer.gov/types/breast/patient/breast-treatment-pdq",
        "category": "oncology",
        "refresh": "manual",
        "priority": "HIGH",
        "keywords": ["breast cancer", "tamoxifen", "chemotherapy", "hormone therapy",
                     "HER2", "radiation", "mastectomy", "lumpectomy", "stage II",
                     "aromatase inhibitor", "endocrine therapy"]
    },
    {
        "id": "nci_prostate_cancer_treatment",
        "name": "NCI: Prostate Cancer Treatment (PDQ)",
        "url": "https://www.cancer.gov/types/prostate/patient/prostate-treatment-pdq",
        "category": "oncology",
        "refresh": "manual",
        "priority": "HIGH",
        "keywords": ["prostate cancer", "bicalutamide", "ADT", "radiation therapy",
                     "active surveillance", "radical prostatectomy", "gleason",
                     "PSA", "stage I", "localized"]
    },
    {
        "id": "nci_colorectal_treatment",
        "name": "NCI: Colon Cancer Treatment (PDQ)",
        "url": "https://www.cancer.gov/types/colorectal/patient/colon-treatment-pdq",
        "category": "oncology",
        "refresh": "manual",
        "priority": "MEDIUM",
        "keywords": ["colon cancer", "colorectal cancer", "FOLFOX", "FOLFIRI",
                     "resection", "colostomy", "stage III", "adjuvant chemotherapy"]
    },
    {
        "id": "nci_cancer_screening",
        "name": "NCI: Cancer Screening Overview (PDQ)",
        "url": "https://www.cancer.gov/about-cancer/screening/patient-screening-overview-pdq",
        "category": "oncology",
        "refresh": "manual",
        "priority": "MEDIUM",
        "keywords": ["cancer screening", "early detection", "sensitivity", "specificity",
                     "false positive", "overdiagnosis", "PSA", "mammography"]
    },

    # ════════════════════════════════════════════════════════════════════════
    # PULMONARY / COPD — NHLBI / NIH
    # ════════════════════════════════════════════════════════════════════════
    {
        "id": "nhlbi_copd",
        "name": "NHLBI: COPD — Diagnosis and Management",
        "url": "https://www.nhlbi.nih.gov/health/copd",
        "category": "pulmonary",
        "refresh": "manual",
        "priority": "HIGH",
        "keywords": ["COPD", "chronic obstructive pulmonary disease", "FEV1", "FVC",
                     "spirometry", "bronchodilator", "tiotropium", "albuterol",
                     "pulmonary rehab", "oxygen therapy", "GOLD criteria"]
    },
    {
        "id": "nhlbi_asthma",
        "name": "NHLBI: Asthma — Diagnosis and Management",
        "url": "https://www.nhlbi.nih.gov/health/asthma",
        "category": "pulmonary",
        "refresh": "manual",
        "priority": "HIGH",
        "keywords": ["asthma", "inhaled corticosteroid", "bronchospasm", "FEV1",
                     "peak flow", "albuterol", "fluticasone", "step therapy",
                     "asthma action plan", "controller medication"]
    },
    {
        "id": "nhlbi_sleep_apnea",
        "name": "NHLBI: Sleep Apnea — Diagnosis and Treatment",
        "url": "https://www.nhlbi.nih.gov/health/sleep-apnea",
        "category": "pulmonary",
        "refresh": "manual",
        "priority": "MEDIUM",
        "keywords": ["sleep apnea", "OSA", "CPAP", "AHI", "polysomnography",
                     "snoring", "oxygen desaturation", "BMI obesity"]
    },

    # ════════════════════════════════════════════════════════════════════════
    # MENTAL HEALTH — SAMHSA / NIH / NAMI
    # ════════════════════════════════════════════════════════════════════════
    {
        "id": "nimh_depression",
        "name": "NIMH: Depression — Treatment and Management",
        "url": "https://www.nimh.nih.gov/health/topics/depression",
        "category": "mental_health",
        "refresh": "manual",
        "priority": "HIGH",
        "keywords": ["depression", "MDD", "antidepressant", "SSRI", "sertraline",
                     "fluoxetine", "CBT", "psychotherapy", "PHQ-9", "major depressive"]
    },
    {
        "id": "nimh_anxiety",
        "name": "NIMH: Anxiety Disorders — Treatment and Management",
        "url": "https://www.nimh.nih.gov/health/topics/anxiety-disorders",
        "category": "mental_health",
        "refresh": "manual",
        "priority": "HIGH",
        "keywords": ["anxiety", "GAD", "generalized anxiety", "panic disorder",
                     "SSRI", "SNRI", "buspirone", "CBT", "benzodiazepine",
                     "GAD-7", "PHQ anxiety"]
    },
    {
        "id": "samhsa_substance_use",
        "name": "SAMHSA: Alcohol and Drug Use Disorders Treatment",
        "url": "https://www.samhsa.gov/find-help/disorders",
        "category": "behavioral_health",
        "refresh": "manual",
        "priority": "MEDIUM",
        "keywords": ["substance use", "alcohol use disorder", "AUD", "naltrexone",
                     "buprenorphine", "opioid", "AUDIT", "CAGE", "detox",
                     "MAT", "medication assisted treatment"]
    },

    # ════════════════════════════════════════════════════════════════════════
    # OBESITY / METABOLIC — CDC / NHLBI
    # ════════════════════════════════════════════════════════════════════════
    {
        "id": "cdc_obesity_adult",
        "name": "CDC: Adult Obesity — Causes, Consequences, Interventions",
        "url": "https://www.cdc.gov/obesity/adult/index.html",
        "category": "obesity_metabolic",
        "refresh": "manual",
        "priority": "HIGH",
        "keywords": ["obesity", "BMI 30", "BMI 35", "overweight", "weight loss",
                     "caloric deficit", "physical activity", "bariatric surgery",
                     "GLP-1", "semaglutide", "lifestyle intervention"]
    },
    {
        "id": "nhlbi_obesity_guidelines",
        "name": "NHLBI: Obesity Management — Clinical Guidelines",
        "url": "https://www.nhlbi.nih.gov/health/overweight-and-obesity",
        "category": "obesity_metabolic",
        "refresh": "manual",
        "priority": "HIGH",
        "keywords": ["obesity treatment", "weight management", "BMI threshold",
                     "intensive lifestyle intervention", "pharmacotherapy",
                     "orlistat", "5% weight loss", "10% weight loss"]
    },

    # ════════════════════════════════════════════════════════════════════════
    # GERIATRICS — NIA / AGS
    # ════════════════════════════════════════════════════════════════════════
    {
        "id": "nia_older_adult_health",
        "name": "NIA: Health and Aging — Clinical Guidance",
        "url": "https://www.nia.nih.gov/health/doctors-and-health-care-professionals",
        "category": "geriatrics",
        "refresh": "manual",
        "priority": "MEDIUM",
        "keywords": ["older adults", "geriatric", "aging", "frailty", "cognitive",
                     "polypharmacy", "functional decline", "65 years", "75 years"]
    },
    {
        "id": "nia_dementia",
        "name": "NIA: Alzheimer's Disease and Dementia",
        "url": "https://www.nia.nih.gov/health/alzheimers",
        "category": "geriatrics",
        "refresh": "manual",
        "priority": "MEDIUM",
        "keywords": ["dementia", "Alzheimer", "cognitive impairment", "MCI",
                     "MMSE", "donepezil", "memantine", "caregiver", "memory"]
    },

    # ════════════════════════════════════════════════════════════════════════
    # RHEUMATOLOGY / AUTOIMMUNE — NIAMS / NIH
    # ════════════════════════════════════════════════════════════════════════
    {
        "id": "niams_rheumatoid_arthritis",
        "name": "NIAMS: Rheumatoid Arthritis — Diagnosis and Treatment",
        "url": "https://www.niams.nih.gov/health-topics/rheumatoid-arthritis",
        "category": "rheumatology",
        "refresh": "manual",
        "priority": "HIGH",
        "keywords": ["rheumatoid arthritis", "RA", "methotrexate", "DMARD",
                     "biologics", "TNF inhibitor", "DAS28", "remission",
                     "joint damage", "infliximab", "etanercept", "adalimumab"]
    },
    {
        "id": "niams_lupus",
        "name": "NIAMS: Lupus — Diagnosis and Management",
        "url": "https://www.niams.nih.gov/health-topics/lupus",
        "category": "rheumatology",
        "refresh": "manual",
        "priority": "HIGH",
        "keywords": ["lupus", "SLE", "hydroxychloroquine", "prednisone",
                     "ANA", "anti-dsDNA", "lupus nephritis", "flare",
                     "photosensitivity", "immunosuppressive"]
    },
    {
        "id": "niams_osteoporosis",
        "name": "NIAMS: Osteoporosis — Prevention and Treatment",
        "url": "https://www.niams.nih.gov/health-topics/osteoporosis",
        "category": "bone_health",
        "refresh": "manual",
        "priority": "HIGH",
        "keywords": ["osteoporosis", "bone density", "DEXA scan", "T-score",
                     "bisphosphonate", "alendronate", "calcium", "vitamin D",
                     "fracture risk", "FRAX", "vertebral fracture"]
    },
    {
        "id": "niams_osteoarthritis",
        "name": "NIAMS: Osteoarthritis — Diagnosis and Treatment",
        "url": "https://www.niams.nih.gov/health-topics/osteoarthritis",
        "category": "rheumatology",
        "refresh": "manual",
        "priority": "MEDIUM",
        "keywords": ["osteoarthritis", "OA", "joint pain", "NSAIDs", "meloxicam",
                     "physical therapy", "knee replacement", "hip replacement",
                     "acetaminophen", "weight management"]
    },

    # ════════════════════════════════════════════════════════════════════════
    # NEUROLOGY / STROKE — NHLBI / NIH / NINDS
    # ════════════════════════════════════════════════════════════════════════
    {
        "id": "nhlbi_stroke",
        "name": "NHLBI: Stroke — Risk Factors, Treatment, Recovery",
        "url": "https://www.nhlbi.nih.gov/health/stroke",
        "category": "neurology",
        "refresh": "manual",
        "priority": "HIGH",
        "keywords": ["stroke", "TIA", "ischemic stroke", "anticoagulation",
                     "aspirin", "clopidogrel", "warfarin", "DOAC", "risk factors",
                     "blood pressure control", "post-stroke rehabilitation"]
    },
    {
        "id": "ninds_parkinson",
        "name": "NINDS: Parkinson's Disease",
        "url": "https://www.ninds.nih.gov/health-information/disorders/parkinsons-disease",
        "category": "neurology",
        "refresh": "manual",
        "priority": "MEDIUM",
        "keywords": ["Parkinson", "levodopa", "carbidopa", "dopamine",
                     "tremor", "bradykinesia", "deep brain stimulation",
                     "motor symptoms", "non-motor symptoms"]
    },
    {
        "id": "nhlbi_migraine",
        "name": "NIH: Migraine — Diagnosis and Treatment",
        "url": "https://www.ninds.nih.gov/health-information/disorders/migraine",
        "category": "neurology",
        "refresh": "manual",
        "priority": "MEDIUM",
        "keywords": ["migraine", "headache", "triptan", "sumatriptan",
                     "preventive treatment", "aura", "CGRP", "topiramate",
                     "propranolol", "acute treatment"]
    },

    # ════════════════════════════════════════════════════════════════════════
    # WOMEN'S HEALTH — NICHD / ACOG via NIH
    # ════════════════════════════════════════════════════════════════════════
    {
        "id": "nichd_pcos",
        "name": "NICHD: Polycystic Ovary Syndrome (PCOS)",
        "url": "https://www.nichd.nih.gov/health/topics/pcos",
        "category": "womens_health",
        "refresh": "manual",
        "priority": "HIGH",
        "keywords": ["PCOS", "polycystic ovary", "metformin", "spironolactone",
                     "irregular periods", "androgen", "insulin resistance",
                     "infertility", "hirsutism", "weight management"]
    },
    {
        "id": "nichd_thyroid",
        "name": "NIDDK: Thyroid Disease — Hypothyroidism and Hyperthyroidism",
        "url": "https://www.niddk.nih.gov/health-information/endocrine-diseases/thyroid-disease",
        "category": "endocrine",
        "refresh": "manual",
        "priority": "HIGH",
        "keywords": ["hypothyroidism", "levothyroxine", "TSH", "thyroid",
                     "hyperthyroidism", "Hashimoto", "thyroid nodule",
                     "free T4", "thyroid replacement"]
    },

    # ════════════════════════════════════════════════════════════════════════
    # INFECTIOUS DISEASE — CDC / NIH
    # ════════════════════════════════════════════════════════════════════════
    {
        "id": "cdc_antibiotic_stewardship",
        "name": "CDC: Antibiotic Use and Stewardship",
        "url": "https://www.cdc.gov/antibiotic-use/index.html",
        "category": "infectious_disease",
        "refresh": "manual",
        "priority": "MEDIUM",
        "keywords": ["antibiotic", "stewardship", "resistance", "appropriate use",
                     "duration", "de-escalation", "culture-guided therapy"]
    },
    {
        "id": "cdc_hiv_prevention",
        "name": "CDC: HIV — Prevention, Testing, Treatment",
        "url": "https://www.cdc.gov/hiv/clinicians/index.html",
        "category": "infectious_disease",
        "refresh": "manual",
        "priority": "MEDIUM",
        "keywords": ["HIV", "PrEP", "antiretroviral", "CD4", "viral load",
                     "AIDS", "treatment", "prevention", "testing"]
    },

    # ════════════════════════════════════════════════════════════════════════
    # GI / INFLAMMATORY BOWEL — NIH / NIDDK
    # ════════════════════════════════════════════════════════════════════════
    {
        "id": "niddk_crohns",
        "name": "NIDDK: Crohn's Disease — Diagnosis and Treatment",
        "url": "https://www.niddk.nih.gov/health-information/digestive-diseases/crohns-disease",
        "category": "gastroenterology",
        "refresh": "manual",
        "priority": "HIGH",
        "keywords": ["Crohn's disease", "IBD", "azathioprine", "biologic",
                     "infliximab", "steroid", "aminosalicylate", "colonoscopy",
                     "remission", "flare", "maintenance therapy"]
    },
    {
        "id": "niddk_ibs",
        "name": "NIDDK: Irritable Bowel Syndrome",
        "url": "https://www.niddk.nih.gov/health-information/digestive-diseases/irritable-bowel-syndrome",
        "category": "gastroenterology",
        "refresh": "manual",
        "priority": "LOW",
        "keywords": ["IBS", "irritable bowel", "fiber", "antispasmodic",
                     "low FODMAP", "gut-brain", "diarrhea", "constipation"]
    },

    # ════════════════════════════════════════════════════════════════════════
    # HEDIS QUALITY MEASURES — NCQA
    # ════════════════════════════════════════════════════════════════════════
    {
        "id": "hedis_colorectal_screening",
        "name": "NCQA HEDIS: Colorectal Cancer Screening",
        "url": "https://www.ncqa.org/hedis/measures/colorectal-cancer-screening/",
        "category": "preventive_screening",
        "refresh": "manual",
        "priority": "MEDIUM",
        "keywords": ["colorectal screening", "HEDIS", "quality measure", "45 to 75",
                     "colonoscopy", "FIT", "stool test", "reporting"]
    },
    {
        "id": "hedis_cardiovascular",
        "name": "NCQA HEDIS: Cardiovascular Care",
        "url": "https://www.ncqa.org/hedis/measures/controlling-high-blood-pressure/",
        "category": "cardiovascular",
        "refresh": "manual",
        "priority": "MEDIUM",
        "keywords": ["blood pressure control", "HEDIS", "cardiovascular quality",
                     "hypertension control", "140/90", "quality measure"]
    },
    {
        "id": "hedis_mental_health",
        "name": "NCQA HEDIS: Mental Health Treatment",
        "url": "https://www.ncqa.org/hedis/measures/antidepressant-medication-management/",
        "category": "mental_health",
        "refresh": "manual",
        "priority": "MEDIUM",
        "keywords": ["antidepressant", "depression treatment", "HEDIS",
                     "medication management", "adherence", "acute phase", "continuation"]
    },
]

# ── Quick lookup helpers ──────────────────────────────────────────────────────

SOURCES_BY_ID = {s["id"]: s for s in GUIDELINE_SOURCES}
CATEGORIES = sorted(set(s["category"] for s in GUIDELINE_SOURCES))
WEEKLY_SOURCES = [s for s in GUIDELINE_SOURCES if s["refresh"] == "weekly"]
MANUAL_SOURCES = [s for s in GUIDELINE_SOURCES if s["refresh"] == "manual"]
HIGH_PRIORITY = [s for s in GUIDELINE_SOURCES if s["priority"] == "HIGH"]

# Category descriptions for agents
CATEGORY_DESCRIPTIONS = {
    "preventive_screening": "Screening recommendations for cancer, diabetes, hypertension",
    "cardiovascular":       "Heart disease, hypertension, cholesterol, stroke",
    "diabetes":             "Type 1/2 diabetes management, HbA1c targets, medications",
    "diabetes_ckd":         "Diabetic kidney disease, nephropathy",
    "ckd":                  "Chronic kidney disease staging, management, referral",
    "oncology":             "Cancer screening, treatment, and staging guidelines",
    "immunization":         "Vaccine schedules and recommendations",
    "mental_health":        "Depression, anxiety, mental health screening",
    "behavioral_health":    "Substance use, tobacco, alcohol counseling",
    "pulmonary":            "COPD, asthma, sleep apnea management",
    "obesity_metabolic":    "Obesity management, weight loss interventions",
    "geriatrics":           "Older adult care, fall prevention, dementia",
    "rheumatology":         "RA, lupus, osteoarthritis, biologics",
    "bone_health":          "Osteoporosis, fracture prevention, DEXA",
    "neurology":            "Stroke, Parkinson's, migraine management",
    "womens_health":        "PCOS, cervical cancer screening, OB/GYN",
    "endocrine":            "Thyroid disease, hormonal disorders",
    "infectious_disease":   "HIV, antibiotics, infection management",
    "gastroenterology":     "Crohn's disease, IBD, IBS",
}

if __name__ == "__main__":
    print(f"Total sources: {len(GUIDELINE_SOURCES)}")
    print(f"Weekly (auto-refresh): {len(WEEKLY_SOURCES)}")
    print(f"Manual: {len(MANUAL_SOURCES)}")
    print(f"High priority: {len(HIGH_PRIORITY)}")
    print(f"\nCategories ({len(CATEGORIES)}):")
    for cat in CATEGORIES:
        count = sum(1 for s in GUIDELINE_SOURCES if s["category"] == cat)
        print(f"  {cat}: {count} source(s)")