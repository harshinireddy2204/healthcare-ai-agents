"""
rag/guideline_seed.py

Hardcoded clinical guideline text for Railway/cloud deployment.

Why this exists:
  Railway (and most PaaS platforms) use ephemeral filesystems. ChromaDB's
  persistent store is wiped on every redeploy/restart. Scraping 63 URLs
  takes 5+ minutes and fails silently when rate-limited.

  This file embeds the 20 most clinically critical guideline excerpts as
  Python constants. They are embedded into ChromaDB at API startup if the
  collection is empty. This guarantees:
    - Agents always have RAG access from the first request
    - No network dependency at startup
    - No 5-minute wait for first-time visitors
    - Full compatibility with the existing retriever pipeline

  The scraping pipeline (rag/refresh_flow.py) still works — it will ADD
  to and REPLACE these seed chunks when it runs successfully.
"""

SEED_GUIDELINES = [
    # ── DIABETES ──────────────────────────────────────────────────────────────
    {
        "source_id": "seed_ada_hba1c",
        "name": "ADA 2025: HbA1c Targets and Glycemic Goals",
        "url": "https://diabetesjournals.org/care/issue/48/Supplement_1",
        "category": "diabetes",
        "chunks": [
            "ADA 2025 Standards of Medical Care: The glycemic target for most nonpregnant adults with diabetes is an HbA1c below 7.0%. A less stringent HbA1c goal such as below 8.0% may be appropriate for patients with limited life expectancy, advanced microvascular or macrovascular complications, extensive comorbid conditions, or longstanding diabetes where the general goal is difficult to attain. More stringent glycemic goals (HbA1c below 6.5%) may be appropriate for selected patients with short disease duration, long life expectancy, no significant cardiovascular disease, and type 2 diabetes treated with lifestyle or metformin alone if achievable without significant hypoglycemia.",
            "ADA 2025: Continuous glucose monitoring (CGM) is recommended for all adults with type 1 diabetes and for adults with type 2 diabetes on intensive insulin regimens. Insulin pump therapy (CSII) is indicated for patients with type 1 diabetes and for adults with type 2 diabetes who have HbA1c above 9.0% despite optimal multiple daily injection therapy. Insulin pump therapy reduces HbA1c by approximately 0.5% compared to MDI in randomized trials. Endocrinology referral is recommended when HbA1c exceeds 9.0% on maximal oral therapy.",
            "ADA 2025: Metformin is the preferred initial pharmacologic agent for type 2 diabetes in the absence of contraindications. GLP-1 receptor agonists (semaglutide, liraglutide) are recommended for patients with established cardiovascular disease, high cardiovascular risk, CKD, or obesity. SGLT2 inhibitors (empagliflozin, dapagliflozin) are recommended for patients with heart failure with reduced ejection fraction or CKD to reduce cardiovascular and renal outcomes. Annual HbA1c testing is recommended; testing every 3 months when treatment changes or when goals are not being met.",
            "ADA 2025: Diabetes self-management education and support (DSMES) is recommended for all patients with type 2 diabetes at diagnosis and as needed thereafter. Annual foot examination including assessment of protective sensation, foot structure, vascular status, and skin integrity is recommended for all patients with diabetes. Annual dilated eye examination by an ophthalmologist or optometrist is recommended starting at diagnosis for type 2 diabetes and 5 years after diagnosis for type 1 diabetes. Urine albumin-to-creatinine ratio (UACR) and estimated glomerular filtration rate (eGFR) should be assessed annually.",
        ]
    },
    {
        "source_id": "seed_ada_ckd_diabetes",
        "name": "ADA 2025 / KDIGO: Diabetic Kidney Disease Management",
        "url": "https://diabetesjournals.org/care/issue/48/Supplement_1",
        "category": "diabetes_ckd",
        "chunks": [
            "ADA 2025 and KDIGO 2024: Metformin is contraindicated when eGFR falls below 30 mL/min/1.73m2 due to risk of lactic acidosis. When eGFR is between 30 and 45, metformin should be used with caution and the dose should not be increased; discontinue if eGFR falls below 30. SGLT2 inhibitors are recommended for patients with type 2 diabetes and CKD (eGFR ≥ 20 mL/min/1.73m2 and UACR ≥ 200 mg/g) to reduce CKD progression and cardiovascular events. ACE inhibitors or ARBs are recommended for hypertensive patients with diabetes and albuminuria (UACR ≥ 30 mg/g) to slow nephropathy progression.",
            "KDIGO 2024 CKD Guidelines: Nephrology referral is recommended for all patients with eGFR below 30 mL/min/1.73m2 (CKD Stage 4-5) for preparation of renal replacement therapy. Co-management with nephrology is recommended for eGFR below 45 mL/min/1.73m2 (CKD Stage 3b) in patients with rapid eGFR decline (>5 mL/min/1.73m2/year), persistent UACR above 300 mg/g, or hypertension difficult to control. Patients with CKD Stage 4 should receive education about dialysis modalities and kidney transplantation options. Estimated time to dialysis should be discussed when eGFR is below 20.",
        ]
    },

    # ── CARDIOVASCULAR ────────────────────────────────────────────────────────
    {
        "source_id": "seed_acc_aha_bp",
        "name": "ACC/AHA 2017: Hypertension Guidelines",
        "url": "https://www.heart.org/en/health-topics/high-blood-pressure",
        "category": "cardiovascular",
        "chunks": [
            "ACC/AHA 2017 Hypertension Guidelines: Blood pressure targets for most adults with hypertension are below 130/80 mmHg. Stage 1 hypertension is defined as systolic 130-139 mmHg or diastolic 80-89 mmHg. Stage 2 hypertension is systolic at or above 140 mmHg or diastolic at or above 90 mmHg. Lifestyle modification including DASH diet, weight reduction, sodium restriction to less than 1500 mg/day, and physical activity are first-line interventions. Pharmacotherapy is recommended for Stage 1 hypertension with 10-year ASCVD risk of 10% or greater, and for all patients with Stage 2 hypertension. First-line agents include thiazide diuretics, ACE inhibitors, ARBs, and calcium channel blockers.",
            "ACC/AHA Heart Failure Guidelines 2022: Guideline-directed medical therapy for HFrEF (EF below 40%) includes: ACE inhibitor/ARB/ARNI + beta blocker + MRA + SGLT2 inhibitor. This four-pillar regimen reduces mortality by approximately 73% compared to placebo. Loop diuretics (furosemide) are used for symptomatic relief of volume overload but do not improve mortality. Cardiac rehabilitation is recommended for clinically stable outpatients with heart failure (Class I recommendation, Level A evidence). NSAIDs are contraindicated in heart failure as they promote sodium retention and worsen renal function.",
        ]
    },
    {
        "source_id": "seed_acc_aha_afib",
        "name": "ACC/AHA 2019: Atrial Fibrillation Management",
        "url": "https://www.ahajournals.org/doi/10.1161/CIR.0000000000000665",
        "category": "cardiovascular",
        "chunks": [
            "ACC/AHA 2019 Atrial Fibrillation Guidelines: Anticoagulation is recommended for all patients with nonvalvular AFib and CHA2DS2-VASc score of 2 or greater in men and 3 or greater in women to prevent stroke. Warfarin (target INR 2.0-3.0) or direct oral anticoagulants (DOACs: apixaban, rivaroxaban, dabigatran) are recommended. DOACs are preferred over warfarin in most patients with nonvalvular AFib. Warfarin combined with aspirin or NSAIDs significantly increases bleeding risk and should be avoided unless a compelling indication exists. Warfarin combined with NSAIDs increases the risk of major GI bleeding by 3-fold.",
            "ACC/AHA Afib: Rate control with beta blockers or calcium channel blockers is recommended as first-line therapy for most patients. Rhythm control (cardioversion, antiarrhythmic drugs, ablation) may be preferred for patients with persistent symptoms despite rate control. Annual cardiology follow-up is recommended for all patients with AFib on anticoagulation. CHA2DS2-VASc scoring: Congestive heart failure (+1), Hypertension (+1), Age 75+ (+2), Diabetes (+1), Stroke/TIA (+2), Vascular disease (+1), Age 65-74 (+1), Sex female (+1).",
        ]
    },
    {
        "source_id": "seed_acc_cholesterol",
        "name": "ACC/AHA 2018: Cholesterol Management Guidelines",
        "url": "https://www.ahajournals.org/doi/10.1161/CIR.0000000000000625",
        "category": "cardiovascular",
        "chunks": [
            "ACC/AHA 2018 Cholesterol Guidelines: High-intensity statin therapy (atorvastatin 40-80mg or rosuvastatin 20-40mg) is recommended for patients with clinical ASCVD (prior MI, stroke, or symptomatic PAD), patients with LDL-C of 190 mg/dL or higher, and patients aged 40-75 with diabetes and 10-year ASCVD risk of 7.5% or greater. Moderate-intensity statin therapy is recommended for primary prevention in adults 40-75 with LDL-C 70-189 mg/dL and 10-year ASCVD risk of 7.5-19.9%. LDL-C should be rechecked 4-12 weeks after initiating statin therapy to assess adherence and response. Familial hypercholesterolemia workup is recommended for LDL-C above 190 mg/dL.",
        ]
    },

    # ── CKD ───────────────────────────────────────────────────────────────────
    {
        "source_id": "seed_kdigo_ckd",
        "name": "KDIGO 2024: CKD Evaluation and Management",
        "url": "https://kdigo.org/guidelines/ckd-evaluation-and-management/",
        "category": "ckd",
        "chunks": [
            "KDIGO 2024 CKD Guidelines: CKD is classified by GFR category (G1-G5) and albuminuria category (A1-A3). CKD Stage 3a: eGFR 45-59; Stage 3b: eGFR 30-44; Stage 4: eGFR 15-29; Stage 5: eGFR below 15. Nephrology referral is recommended for: eGFR below 30 (urgent preparation for renal replacement therapy), rapidly declining eGFR (>5 mL/min/1.73m2 per year), eGFR below 45 with complications difficult to manage, or persistent UACR above 300 mg/g. Blood pressure target is below 120 mmHg systolic for most CKD patients. ACE inhibitors or ARBs are preferred antihypertensives for CKD patients with albuminuria.",
            "KDIGO 2024: Anemia is common in CKD Stage 3-5 due to reduced erythropoietin production. Hemoglobin target is 10-11.5 g/dL with erythropoiesis-stimulating agents (ESAs). Iron deficiency should be corrected before starting ESAs. Metabolic acidosis (bicarbonate below 22 mEq/L) should be treated with oral bicarbonate supplementation to slow CKD progression. Dietary protein restriction to 0.6-0.8 g/kg/day is recommended for non-dialysis CKD patients. Phosphate restriction and phosphate binders are indicated when serum phosphate rises above normal in CKD Stage 4-5.",
            "KDIGO 2024 CKD and Diabetes: Combination therapy with SGLT2 inhibitors AND ACE inhibitors/ARBs provides additive kidney protection in diabetic CKD. SGLT2 inhibitors reduce the risk of CKD progression (40% relative risk reduction) and are recommended when eGFR is at or above 20 mL/min/1.73m2. Finerenone (non-steroidal MRA) is recommended for CKD Stage 3-4 with UACR above 300 mg/g and type 2 diabetes to reduce CKD progression and cardiovascular events.",
        ]
    },

    # ── PREVENTIVE SCREENING ──────────────────────────────────────────────────
    {
        "source_id": "seed_uspstf_breast",
        "name": "USPSTF 2024: Breast Cancer Screening Recommendations",
        "url": "https://www.uspreventiveservicestaskforce.org/uspstf/recommendation/breast-cancer-screening",
        "category": "preventive_screening",
        "chunks": [
            "USPSTF 2024 Breast Cancer Screening: The USPSTF recommends biennial screening mammography for women aged 40 to 74 years (Grade B recommendation). This recommendation applies to asymptomatic women who do not have preexisting breast cancer or a previously diagnosed high-risk breast lesion, and who are not at high risk for breast cancer because of a known underlying genetic mutation (such as a BRCA1 or BRCA2 gene mutation) or a history of chest radiation at a young age. Women with a family history of breast cancer or other risk factors should be offered individualized screening discussions. Annual mammography may be considered for high-risk women starting at age 30.",
            "USPSTF Colorectal Cancer Screening 2021: The USPSTF recommends screening for colorectal cancer in all adults aged 45 to 75 years (Grade B). Screening options include annual high-sensitivity guaiac-based fecal occult blood test (gFOBT), annual fecal immunochemical test (FIT), stool DNA-FIT every 1-3 years, CT colonography every 5 years, flexible sigmoidoscopy every 5 years, or colonoscopy every 10 years. Adults aged 76 to 85 should make individualized screening decisions. The USPSTF recommends against screening in adults older than 85.",
        ]
    },
    {
        "source_id": "seed_uspstf_preventive",
        "name": "USPSTF: Preventive Care Recommendations for Adults",
        "url": "https://www.uspreventiveservicestaskforce.org/uspstf/recommendation-topics",
        "category": "preventive_screening",
        "chunks": [
            "USPSTF Depression Screening in Adults 2023: The USPSTF recommends screening for depression in the general adult population (Grade B). Screening should be implemented with adequate systems in place to ensure accurate diagnosis, effective treatment, and appropriate follow-up. The PHQ-2 is used as an initial screener, followed by the PHQ-9 for positive screens. The PHQ-9 score of 10 or higher indicates moderate to severe depression requiring treatment. Antidepressant therapy and psychotherapy (particularly cognitive behavioral therapy) are effective treatments.",
            "USPSTF Hypertension Screening 2021: The USPSTF recommends screening for hypertension in adults 18 years or older (Grade A). The USPSTF recommends obtaining measurements outside of the clinical setting for diagnostic confirmation before starting treatment. Office blood pressure measurement of 140/90 mmHg or higher on at least two separate occasions is diagnostic for hypertension. Ambulatory blood pressure monitoring (ABPM) over 24 hours is the reference standard for confirming hypertension. Lifestyle interventions including weight loss, dietary changes, reduced sodium intake, increased physical activity, and reduced alcohol consumption should be recommended for all patients with hypertension.",
            "USPSTF Statin Use for Prevention 2022: The USPSTF recommends prescribing a statin for the primary prevention of CVD events and mortality for adults aged 40 to 75 who have one or more CVD risk factors (dyslipidemia, diabetes, hypertension, or smoking) and an estimated 10-year CVD event risk of 10% or greater (Grade B). Adults with an estimated 10-year CVD event risk of 7.5-9.9% may also benefit from a statin (Grade C). The 10-year ASCVD risk calculator (pooled cohort equations) should be used to estimate risk.",
        ]
    },

    # ── IMMUNIZATION ──────────────────────────────────────────────────────────
    {
        "source_id": "seed_cdc_immunization",
        "name": "CDC Adult Immunization Schedule 2025",
        "url": "https://www.cdc.gov/vaccines/hcp/imz-schedules/adult-age.html",
        "category": "immunization",
        "chunks": [
            "CDC Adult Immunization Schedule 2025: Influenza vaccine: 1 dose annually for all adults. COVID-19 vaccine: 1 updated dose annually for all adults 65 years and older; 1 updated dose for adults 18-64 years with shared clinical decision-making. Tdap or Td: 1 dose Tdap then Td or Tdap booster every 10 years. RSV vaccine: 1 dose for adults 60 years and older (shared clinical decision-making). Shingrix (recombinant zoster vaccine): 2-dose series (2-6 months apart) for all immunocompetent adults 50 years and older and for immunocompromised adults 19 years and older.",
            "CDC 2025 Immunization: Pneumococcal vaccines: Adults 65 years and older who have not previously received PCV15 or PCV20 should receive 1 dose of PCV20 alone, or 1 dose of PCV15 followed by PPSV23 (at least 1 year later). Hepatitis B vaccine: 3-dose series (0, 1-2, 4-6 months) for all adults through age 59; adults 60 years and older may receive the vaccine based on shared decision-making. Hepatitis A vaccine: 2 or 3-dose series for adults at increased risk (travel, chronic liver disease, HIV). HPV vaccine: 2 or 3-dose series through age 26; shared decision-making for ages 27-45.",
        ]
    },

    # ── ONCOLOGY ─────────────────────────────────────────────────────────────
    {
        "source_id": "seed_nci_breast_cancer",
        "name": "NCI/NCCN: Breast Cancer Treatment Standards",
        "url": "https://www.cancer.gov/types/breast/patient/breast-treatment-pdq",
        "category": "oncology",
        "chunks": [
            "NCCN Breast Cancer Guidelines: Hormone receptor-positive (ER+/PR+) breast cancer is treated with endocrine therapy as the primary systemic approach. Tamoxifen (20 mg daily for 5-10 years) is standard for premenopausal women and for postmenopausal women who cannot tolerate aromatase inhibitors. Aromatase inhibitors (anastrozole, letrozole, exemestane) are preferred for postmenopausal women with HR+ breast cancer and are associated with superior outcomes compared to tamoxifen. HER2-positive breast cancer requires targeted therapy with trastuzumab (Herceptin). Bone density monitoring is recommended for patients on aromatase inhibitors due to accelerated bone loss — baseline DEXA scan and annual monitoring.",
            "NCI Breast Cancer Survivorship: Annual mammography of the contralateral breast is recommended after breast cancer treatment. Surveillance for local recurrence includes clinical breast examination every 3-6 months for the first 3 years, then annually. Tamoxifen increases the risk of endometrial cancer; report any abnormal uterine bleeding promptly. Aromatase inhibitors are associated with arthralgia, bone loss, and cardiovascular risk. Oncology follow-up with survivorship care plan should be coordinated with primary care.",
        ]
    },
    {
        "source_id": "seed_nci_prostate_cancer",
        "name": "NCI/NCCN: Prostate Cancer Treatment Standards",
        "url": "https://www.cancer.gov/types/prostate/patient/prostate-treatment-pdq",
        "category": "oncology",
        "chunks": [
            "NCCN Prostate Cancer Guidelines: Active surveillance is the preferred management for very low-risk and low-risk localized prostate cancer (Gleason score 6 or below, PSA below 10 ng/mL, clinical stage T1c-T2a). Active surveillance includes PSA every 3-6 months, digital rectal exam annually, and repeat biopsy every 2-3 years. Radical prostatectomy or radiation therapy are appropriate for intermediate- and high-risk localized disease. Androgen deprivation therapy (ADT) with bicalutamide, leuprolide, or enzalutamide is used for locally advanced or metastatic prostate cancer. Bone density monitoring is essential for patients on ADT due to significant bone loss risk.",
        ]
    },

    # ── MENTAL HEALTH ─────────────────────────────────────────────────────────
    {
        "source_id": "seed_nimh_depression",
        "name": "NIMH / APA: Depression Diagnosis and Treatment",
        "url": "https://www.nimh.nih.gov/health/topics/depression",
        "category": "mental_health",
        "chunks": [
            "APA/NIMH Depression Treatment Guidelines: Antidepressant medications and psychotherapy are both effective for major depressive disorder (MDD). First-line pharmacotherapy includes SSRIs (sertraline, fluoxetine, escitalopram) and SNRIs (venlafaxine, duloxetine). Adequate antidepressant trial is 4-8 weeks at therapeutic dose. PHQ-9 score of 10 or higher indicates moderate-severe depression. Treatment response is defined as 50% or greater reduction in PHQ-9 score; remission is PHQ-9 below 5. Patients with inadequate response to two adequate antidepressant trials should be referred to psychiatry. Cognitive behavioral therapy (CBT) is as effective as antidepressants for mild-moderate depression and is preferred for patients who cannot tolerate medications.",
        ]
    },

    # ── RHEUMATOLOGY ─────────────────────────────────────────────────────────
    {
        "source_id": "seed_acr_ra",
        "name": "ACR 2021: Rheumatoid Arthritis Treatment Guidelines",
        "url": "https://www.rheumatology.org/Practice-Quality/Clinical-Support/Clinical-Practice-Guidelines/Rheumatoid-Arthritis",
        "category": "rheumatology",
        "chunks": [
            "ACR 2021 RA Guidelines: Methotrexate is the preferred initial DMARD for most patients with RA. Target dose is 15-25 mg weekly (oral or subcutaneous). Folic acid 1 mg daily (or 5 mg once weekly) should be prescribed with methotrexate to reduce side effects including nausea, mucositis, and hepatotoxicity. CBC and hepatic function should be monitored every 2-3 months. If inadequate response to methotrexate after 3-6 months, biologic DMARD (TNF inhibitor: etanercept, adalimumab, infliximab) or JAK inhibitor (tofacitinib, baricitinib) should be added. Treat-to-target (T2T) strategy with low disease activity or remission as the goal is recommended.",
            "ACR 2021 RA Comorbidities: RA patients have 1.5-2x increased cardiovascular risk. Annual cardiovascular risk assessment and aggressive management of CVD risk factors is recommended. Glucocorticoids (prednisone) accelerate bone loss — baseline DEXA scan and prophylactic bisphosphonate therapy should be initiated for patients requiring long-term glucocorticoid therapy at doses of 5 mg/day or higher for 3 months or longer. Vaccination should be updated before initiating biologic DMARDs (live vaccines contraindicated once biologics started). Annual influenza and pneumococcal vaccines are recommended.",
        ]
    },

    # ── PULMONARY ─────────────────────────────────────────────────────────────
    {
        "source_id": "seed_gold_copd",
        "name": "GOLD 2024: COPD Diagnosis and Management",
        "url": "https://goldcopd.org/2024-gold-report/",
        "category": "pulmonary",
        "chunks": [
            "GOLD 2024 COPD Guidelines: COPD is diagnosed by post-bronchodilator spirometry showing FEV1/FVC ratio below 0.70. GOLD severity grades: Grade 1 (mild): FEV1 ≥ 80% predicted; Grade 2 (moderate): FEV1 50-79%; Grade 3 (severe): FEV1 30-49%; Grade 4 (very severe): FEV1 below 30%. Initial pharmacotherapy: LAMA (tiotropium) or LABA + ICS for patients with ≥2 exacerbations per year. Short-acting bronchodilators (albuterol, ipratropium) as rescue therapy. Annual influenza vaccination and pneumococcal vaccination are strongly recommended. Pulmonary rehabilitation is recommended for GOLD Grade 2-4 patients and improves exercise tolerance and quality of life.",
        ]
    },

    # ── GERIATRICS / FALLS ────────────────────────────────────────────────────
    {
        "source_id": "seed_uspstf_falls",
        "name": "USPSTF 2018: Falls Prevention in Older Adults",
        "url": "https://www.uspreventiveservicestaskforce.org/uspstf/recommendation/falls-prevention-in-older-adults-interventions",
        "category": "geriatrics",
        "chunks": [
            "USPSTF 2018 Falls Prevention: The USPSTF recommends exercise interventions to prevent falls in community-dwelling adults 65 years or older who are at increased risk for falls (Grade B). Exercise programs should include balance training, gait training, and strength training. Vitamin D supplementation (at least 800 IU daily) reduces fall risk in older adults with vitamin D deficiency. Multifactorial interventions addressing vision, medications (deprescribing fall-risk-increasing drugs), home hazard reduction, and orthostatic hypotension are recommended for high-risk older adults. Fall risk assessment should include: history of falls, gait and balance testing (Timed Up and Go test), medication review, orthostatic blood pressure, and vision assessment.",
        ]
    },

    # ── OBESITY ───────────────────────────────────────────────────────────────
    {
        "source_id": "seed_obesity_management",
        "name": "USPSTF 2018 / AHA-ACC-TOS: Obesity Management in Adults",
        "url": "https://www.uspreventiveservicestaskforce.org/uspstf/recommendation/obesity-in-adults-interventions",
        "category": "obesity_metabolic",
        "chunks": [
            "USPSTF 2018 Obesity: The USPSTF recommends offering or referring adults with a BMI of 30 or higher to intensive, multicomponent behavioral interventions (Grade B). Intensive lifestyle interventions include: 12-26 sessions in the first year, dietary counseling with caloric restriction, supervised physical activity (150-300 minutes/week), and behavioral strategies. A weight loss of 5-10% of initial body weight reduces cardiometabolic risk factors. Pharmacotherapy (orlistat, phentermine-topiramate, naltrexone-bupropion, semaglutide) should be offered to adults with BMI ≥ 30 or BMI ≥ 27 with weight-related comorbidities when lifestyle intervention alone is insufficient. Bariatric surgery should be considered for adults with BMI ≥ 40 or BMI ≥ 35 with serious weight-related comorbidities.",
        ]
    },

    # ── STROKE SECONDARY PREVENTION ──────────────────────────────────────────
    {
        "source_id": "seed_aha_stroke",
        "name": "AHA/ASA 2021: Secondary Stroke Prevention Guidelines",
        "url": "https://www.ahajournals.org/doi/10.1161/STR.0000000000000375",
        "category": "neurology",
        "chunks": [
            "AHA/ASA 2021 Secondary Stroke Prevention: For patients with ischemic stroke or TIA and atrial fibrillation, anticoagulation with warfarin (INR 2.0-3.0) or a DOAC (apixaban, rivaroxaban, dabigatran) is recommended. DOACs are preferred over warfarin in most patients with nonvalvular AFib. For non-cardioembolic ischemic stroke, antiplatelet therapy (aspirin 81-325 mg daily, clopidogrel 75 mg daily, or aspirin/dipyridamole) is recommended rather than anticoagulation. High-intensity statin therapy targeting LDL-C below 70 mg/dL is recommended after ischemic stroke. Blood pressure should be lowered to below 130/80 mmHg. Cardiac monitoring for at least 24 hours is recommended after acute ischemic stroke to detect AFib.",
        ]
    },
]


def get_seed_count() -> int:
    """Return the total number of seed chunks."""
    return sum(len(s["chunks"]) for s in SEED_GUIDELINES)


def get_seed_as_scraped_format() -> list[dict]:
    """
    Convert seed guidelines to the same dict format that scrape_guideline() returns,
    so embed_source() can process them without modification.
    """
    from datetime import datetime
    results = []
    for source in SEED_GUIDELINES:
        results.append({
            "source_id": source["source_id"],
            "name": source["name"],
            "url": source["url"],
            "category": source["category"],
            "chunks": source["chunks"],
            "changed": True,          # always embed seeds
            "hash": f"seed_{source['source_id']}",
            "scraped_at": datetime.utcnow().isoformat(),
            "error": None,
            "keywords": [],
        })
    return results