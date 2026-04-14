# AI-Powered Clinical Operations — Solution Brief for Mass General Brigham

**Prepared by:** Harshini Reddy, Applied AI & Analytics Consultant
**Platform basis:** Innovaccer Gravity™ + custom multi-agent layer

---

## The Problem MGB Faces Today

Prior authorization delays affect **94% of physicians** (AMA 2024). Care coordinators manually identify gaps across 1.5M+ patients. Both processes are high-volume, rule-based, and consuming clinical staff time.

| Challenge | Scale |
|---|---|
| Physician time on prior auth | 13+ hours per physician per week |
| HEDIS care gap closure rate | ~60–70% at most health systems |
| Prior auth denial rate | 27% — each requiring costly appeals |
| Drug interaction detection | Largely manual, error-prone |

---

## Three Automated Clinical Workflows

### 1. Prior Authorization (Flow Auth equivalent)
Epic FHIR R4 → payer policy → clinical criteria → APPROVE / DENY / ESCALATE.
Agent/Critic pattern (MALADE, MLHC 2024): critic reviews every decision before finalizing.
Target escalation rate: <20%.

### 2. Care Gap Detection (HEDIS Closure)
Systematic review of preventive obligations per patient, grounded in USPSTF, ADA, KDIGO, CDC guidelines.
dbt SQL mart produces HEDIS-reportable metrics for quality team.

### 3. Drug Safety (Pharmacovigilance)
Real-time OpenFDA API for every patient's medication list.
Flags DDIs and contraindications with FDA label citations.
Knowledge graph catches multi-hop risks (CKD Stage 4 → Metformin contraindicated).

---

## Adaptive Complexity Routing (MDAgents, NeurIPS 2024)

| Tier | Criteria | Agent Team | Time |
|---|---|---|---|
| LOW | 1–2 stable conditions | Single LangGraph agent | ~30s |
| MODERATE | 3+ conditions, pending auth | CrewAI 3-agent MDT | ~90s |
| HIGH | Multi-system, high-risk meds | Full ICT + drug safety + KG | ~3min |

~60% of patients routed to LOW — full resources reserved for complex cases.

---

## Epic Integration Path

```
Epic → SMART on FHIR OAuth2 → FHIR R4 API → Agents
                                               ↓
                             Audit Log → Snowflake → dbt → HEDIS
                                               ↓
                             Streamlit HITL Dashboard (clinicians)
```

No rip-and-replace. Connects to existing Epic FHIR endpoints.
Results flow into MGB's Snowflake via standard dbt ETL.

---

## Expected Outcomes (Innovaccer Flow Auth benchmarks)

| Metric | Projected Impact |
|---|---|
| Physician time on prior auth | 50% reduction |
| HEDIS care gap closure | Target 80%+ |
| Prior auth ROI | Up to 5x through denial reduction |
| Drug interaction detection | 100% of medication lists, real-time |

---

## 6-Week Deployment Timeline

| Week | Deliverable |
|---|---|
| 1–2 | Epic FHIR R4 connection, data validation, payer policy mapping |
| 3–4 | Pilot: 50 patients — prior auth + care gap automation |
| 5 | Accuracy validation against clinician ground truth |
| 6 | Production: Snowflake pipeline, HEDIS dashboard, staff training |

*Built on Innovaccer Gravity™ architecture — FHIR R4, HIPAA-structured, Snowflake-compatible.*