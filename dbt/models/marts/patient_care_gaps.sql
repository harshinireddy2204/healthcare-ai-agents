-- dbt model: patient_care_gaps
-- Transforms raw agent audit log into a structured care gaps mart
-- for population health analytics and HEDIS measure reporting.
-- Compatible with PostgreSQL and Snowflake.

WITH agent_runs AS (
    SELECT
        patient_id,
        processed_at,
        status,
        result_json,
        ROW_NUMBER() OVER (
            PARTITION BY patient_id
            ORDER BY processed_at DESC
        ) AS run_rank
    FROM {{ source('healthcare_ai', 'audit_log') }}
    WHERE status = 'COMPLETED'
),

latest_runs AS (
    SELECT * FROM agent_runs WHERE run_rank = 1
),

care_gap_flags AS (
    SELECT
        patient_id,
        processed_at AS last_assessed_at,
        CASE WHEN result_json LIKE '%mammogram%'    THEN 1 ELSE 0 END AS gap_mammogram,
        CASE WHEN result_json LIKE '%colonoscopy%'  THEN 1 ELSE 0 END AS gap_colonoscopy,
        CASE WHEN result_json LIKE '%flu vaccine%'  THEN 1 ELSE 0 END AS gap_flu_vaccine,
        CASE WHEN result_json LIKE '%nephrology%'   THEN 1 ELSE 0 END AS gap_nephrology,
        CASE WHEN result_json LIKE '%HbA1c%'        THEN 1 ELSE 0 END AS gap_hba1c,
        CASE WHEN result_json LIKE '%depression%'   THEN 1 ELSE 0 END AS gap_depression,
        CASE WHEN result_json LIKE '%bone density%' THEN 1 ELSE 0 END AS gap_bone_density
    FROM latest_runs
)

SELECT
    patient_id,
    last_assessed_at,
    (gap_mammogram + gap_colonoscopy + gap_flu_vaccine +
     gap_nephrology + gap_hba1c + gap_depression + gap_bone_density) AS total_gaps,
    gap_mammogram, gap_colonoscopy, gap_flu_vaccine,
    gap_nephrology, gap_hba1c, gap_depression, gap_bone_density,
    CASE
        WHEN (gap_mammogram + gap_colonoscopy + gap_nephrology + gap_hba1c) >= 2 THEN 'HIGH'
        WHEN (gap_mammogram + gap_colonoscopy + gap_flu_vaccine) >= 1 THEN 'MEDIUM'
        ELSE 'LOW'
    END AS gap_risk_tier
FROM care_gap_flags
ORDER BY total_gaps DESC, last_assessed_at DESC