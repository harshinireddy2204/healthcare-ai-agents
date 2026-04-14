# Healthcare AI Multi-Agent System
### Built on the same architecture as Innovaccer Gravity™

> A production-grade, multi-agent clinical operations platform that automates prior authorization, care gap detection, and patient risk triage — independently built to demonstrate readiness for customer-facing AI consulting delivery at health systems like Mass General Brigham.

[![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)](https://python.org)
[![LangGraph](https://img.shields.io/badge/LangGraph-0.2-green.svg)](https://langchain-ai.github.io/langgraph/)
[![CrewAI](https://img.shields.io/badge/CrewAI-0.80-purple.svg)](https://crewai.com)
[![Prefect](https://img.shields.io/badge/Prefect-3.x-coral.svg)](https://prefect.io)
[![LangSmith](https://img.shields.io/badge/LangSmith-traced-orange.svg)](https://smith.langchain.com)
[![FHIR R4](https://img.shields.io/badge/FHIR-R4-red.svg)](https://hl7.org/fhir/R4/)
[![Snowflake](https://img.shields.io/badge/Snowflake-compatible-blue.svg)](https://snowflake.com)

---

## What This Demonstrates for a Consulting Role

This project is a working proof-of-concept of what an Applied AI & Analytics Consultant would build and deliver on-site at a health system customer. It covers every dimension of the role:

| Role Requirement | How This Project Demonstrates It |
|---|---|
| End-to-end data pipelines (ETL/ELT) | FHIR R4 ingestion → normalization → audit log → dbt mart |
| Data quality frameworks | `analytics/data_quality.py` — 10 validation rules, quality scoring |
| SQL + data modeling | `analytics/queries.py` — cohort analysis, SLA metrics, care gap rates |
| AI agents and agentic workflows | 5 LangGraph + CrewAI agents running end-to-end |
| Predictive / ML models | Risk scoring (Charlson-inspired), complexity classification |
| BI dashboards | 8-tab Streamlit dashboard with live analytics |
| AI orchestration | Prefect 3.x scheduled flows with retries |
| Customer-facing delivery | HITL clinician dashboard, audit trail, SLA reporting |
| LLM prompt engineering | 8 specialized system prompts across agent types |
| API design + integration | FastAPI with 8 endpoints, FHIR R4 client |

---

## How It Maps to Innovaccer's Product Stack

Innovaccer's Gravity™ platform (launched May 2025) delivers the exact use cases this system automates:

| Innovaccer Product | This System's Equivalent | Key Difference |
|---|---|---|
| **Flow Auth** (prior auth automation) | `agents/prior_auth_agent.py` — LangGraph ReAct + Agent/Critic | Added Agent/Critic review pattern (MALADE, MLHC 2024) for higher accuracy |
| **Gravity Agentic AI** (care management) | `agents/care_gap_agent.py` — Plan-and-Execute with RAG citations | RAG over 63 clinical guidelines with cross-encoder reranking |
| **Galaxy** (HEDIS/quality analytics) | `analytics/queries.py` + `dbt/models/` | dbt mart produces HEDIS-reportable care gap metrics |
| **Gravity Data Fabric** (FHIR ingestion) | `tools/ehr_tools.py` — FHIR R4 dual-mode client | Supports synthetic + live HAPI FHIR server queries |
| **Snowflake integration** | PostgreSQL-compatible, Snowflake-ready SQL | All queries use standard SQL, no SQLite-specific syntax |

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                    Complexity Router (MDAgents)                      │
│         LOW → single agent | MODERATE → MDT | HIGH → ICT           │
└──────────────────────┬──────────────────────────────────────────────┘
                       │
       ┌───────────────┼───────────────────────┐
       │               │                       │
┌──────▼──────┐ ┌──────▼──────┐  ┌─────────────▼───────────┐
│ Prior Auth  │ │  Care Gap   │  │    Drug Safety Agent    │
│  LangGraph  │ │  LangGraph  │  │   OpenFDA Real-time     │
│ ReAct+Critic│ │Plan+Execute │  │   + Knowledge Graph     │
└──────┬──────┘ └──────┬──────┘  └─────────────┬───────────┘
       │               │                       │
┌──────▼───────────────▼───────────────────────▼───────────┐
│                     Tool Layer (MCP-style)                │
│  FHIR R4 (Patient/Condition/Observation/MedicationReq)   │
│  OpenFDA API (drug labels, adverse events, no key needed)│
│  Clinical Knowledge Graph (93 nodes, 67 evidence edges)  │
│  RAG: 63 guideline sources, ChromaDB + cross-encoder     │
└────────────────────────────┬─────────────────────────────┘
                             │
┌────────────────────────────▼─────────────────────────────┐
│              Production Infrastructure                    │
│  Prefect (scheduling, retries) | LangSmith (tracing)     │
│  FastAPI (REST API) | PostgreSQL/SQLite (audit + HITL)   │
│  dbt models | Data Quality Framework | SQL Analytics      │
└────────────────────────────┬─────────────────────────────┘
                             │
┌────────────────────────────▼─────────────────────────────┐
│              Streamlit Clinical Dashboard                 │
│  Live agent feed | Analytics & Reporting | HITL reviews  │
│  Drug safety alerts | Knowledge graph | FHIR R4 search   │
└──────────────────────────────────────────────────────────┘
```

---

## Customer Deployment Scenario: Mass General Brigham

Mass General Brigham (MGB) is a top Innovaccer target account — an academic medical system with Epic EHR, active AI initiatives, and documented prior auth burden affecting thousands of patients.

**How this system would be deployed on-site:**

### Week 1–2: Data Integration
```python
# Connect to MGB's Epic FHIR R4 endpoint
FHIR_BASE_URL = "https://epicfhir.massgeneralbrigham.org/api/FHIR/R4"
USE_FHIR = true  # flip in .env — all agents switch to live data
```
- SMART on FHIR OAuth2 authentication layer (2-week build)
- Map MGB's insurance plans to payer_policies.json
- Load MGB-specific clinical protocols into RAG guideline store

### Week 3–4: Pilot with 50 patients
- Run prior auth automation on backlog (target: 80%+ auto-approval rate)
- Deploy care gap identification across diabetic patient cohort
- Validate against MGB clinician decisions (ground truth)
- Generate accuracy report: precision, recall, escalation rate

### Week 5–6: Production handover
- Connect audit_log to MGB's Snowflake data warehouse via dbt
- Train care coordinators on HITL review dashboard
- Set up Prefect scheduled runs (nightly batch + real-time triggers)
- Deliver HEDIS-reportable care gap metrics to quality team

**Expected outcomes (based on Innovaccer Flow Auth benchmarks):**
- 50% reduction in physician time on prior authorization
- 2x staff productivity on care gap outreach
- Up to 5x ROI through reduced denials and faster approvals

---

## Research Foundation

This system implements patterns from 5 peer-reviewed papers:

| Paper | Venue | Pattern Implemented |
|---|---|---|
| MDAgents | NeurIPS 2024 (Oral) | Adaptive complexity routing: LOW/MODERATE/HIGH → different agent teams |
| TxAgent | arXiv 2025.3, Harvard | OpenFDA real-time drug safety with multi-step tool reasoning |
| MALADE | MLHC 2024 | Agent/Critic pattern for prior auth — critic reviews each decision |
| FHIR-AgentBench | arXiv 2025.9 | FHIR R4 resource queries with LOINC/SNOMED coding |
| KG4Diagnosis | arXiv 2024.12 | Clinical knowledge graph with 93 nodes, 67 evidence-based edges |

---

## Tech Stack

| Layer | Technology | JD Mapping |
|---|---|---|
| Agent frameworks | LangGraph 0.2, CrewAI 0.80 | Agentic AI workflows |
| LLM | GPT-4o-mini (OpenAI) | LLM-based coding + AI solutions |
| Data warehouse | PostgreSQL / Snowflake-compatible | Snowflake, PostgreSQL |
| ETL / data modeling | dbt schema + SQL marts | dbt, data quality frameworks |
| Orchestration | Prefect 3.x | AI orchestration tools |
| Observability | LangSmith | Evaluation, tracing |
| Real-time data | OpenFDA API, FHIR R4 | APIs, data integration patterns |
| RAG | ChromaDB + sentence-transformers | Knowledge retrieval |
| API | FastAPI | REST APIs, microservices |
| Dashboard | Streamlit | BI / visualization tools |
| Language | Python 3.11+ | Python proficiency |

---

## Running the System

```bash
# 1. Install
pip install -r requirements.txt

# 2. Configure
cp .env.example .env  # add OPENAI_API_KEY

# 3. Initialize DB
python -c "from api.main import init_db; init_db()"

# 4. Load clinical guidelines RAG (first time, ~5 min)
python rag/refresh_flow.py

# 5. Start API
uvicorn api.main:app --reload --port 8000

# 6. Start dashboard
streamlit run frontend/app.py

# 7. Run Prefect batch
python orchestration/prefect_flow.py

# 8. Data quality check
python analytics/data_quality.py
```

---

## Project Structure

```
healthcare-ai-agents/
├── agents/
│   ├── complexity_router.py      # MDAgents adaptive routing
│   ├── prior_auth_agent.py       # LangGraph ReAct + Agent/Critic (MALADE)
│   ├── care_gap_agent.py         # LangGraph Plan-and-Execute + RAG
│   ├── drug_safety_agent.py      # OpenFDA real-time + LangGraph (TxAgent)
│   └── triage_supervisor.py      # CrewAI MDT/ICT orchestration
├── tools/
│   ├── ehr_tools.py              # FHIR R4 dual-mode client (6 tools)
│   ├── payer_tools.py            # Payer policy tools (2 tools)
│   └── risk_tools.py             # Risk scoring tools (2 tools)
├── rag/
│   ├── guideline_sources.py      # 63 sources: USPSTF, ADA, AHA, KDIGO, NCI...
│   ├── scraper.py                # Hash-based change detection scraper
│   ├── embedder.py               # ChromaDB + normalized cosine embeddings
│   ├── retriever.py              # Cross-encoder reranker + query expansion
│   └── refresh_flow.py           # Prefect weekly refresh flow
├── knowledge_graph/
│   └── clinical_graph.py         # 93 nodes, 67 edges, SNOMED/LOINC coded
├── analytics/
│   ├── queries.py                # SQL analytics: cohorts, SLA, auth metrics
│   └── data_quality.py           # Data quality framework: 10 validation rules
├── dbt/
│   └── models/
│       ├── schema.yml            # dbt column tests and source definitions
│       └── marts/
│           └── patient_care_gaps.sql  # HEDIS-reportable care gap mart
├── api/
│   └── main.py                   # FastAPI: 8 endpoints + guidelines API
├── frontend/
│   └── app.py                    # Streamlit: 8-tab dashboard
├── orchestration/
│   └── prefect_flow.py           # Prefect batch workflow with retries
├── data/
│   ├── synthetic_patients.json   # 20 clinically diverse patients
│   └── payer_policies.json       # BlueCross, Aetna, United Health policies
└── tests/
    ├── test_prior_auth.py         # 20+ unit tests
    ├── test_care_gap.py
    └── test_triage.py
```

---

## Author

**Harshini Reddy**
Business & Data Analyst | Applied AI & Analytics
[LinkedIn](https://www.linkedin.com/in/harshini-reddy22/) · [GitHub](https://github.com/harshinireddy2204)

> *Available for Applied AI & Analytics Consultant roles. This project demonstrates end-to-end delivery of AI-driven healthcare solutions — from FHIR R4 data integration through multi-agent orchestration to clinical analytics dashboards — the exact scope of work Innovaccer consultants deliver on-site at health system customers.*