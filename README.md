# Healthcare AI Multi-Agent Workflow System

> A production-grade, multi-agent system that automates clinical operations tasks — prior authorization review, care gap detection, and patient risk triage — using LangGraph, CrewAI, MCP tools, Prefect orchestration, and a human-in-the-loop escalation layer.

[![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)](https://python.org)
[![LangGraph](https://img.shields.io/badge/LangGraph-0.2-green.svg)](https://langchain-ai.github.io/langgraph/)
[![CrewAI](https://img.shields.io/badge/CrewAI-0.80-purple.svg)](https://crewai.com)
[![Prefect](https://img.shields.io/badge/Prefect-3.x-coral.svg)](https://prefect.io)
[![LangSmith](https://img.shields.io/badge/LangSmith-traced-orange.svg)](https://smith.langchain.com)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115-teal.svg)](https://fastapi.tiangolo.com)

---

## Problem Statement

Clinical operations teams at healthcare organizations spend significant manual effort on:
- **Prior authorization** — verifying whether a treatment meets payer coverage criteria
- **Care gap detection** — identifying patients missing preventive measures or follow-ups
- **Risk triage** — prioritizing high-risk patients for proactive outreach

This system automates all three workflows using specialized AI agents that reason over patient data, apply payer policy rules, and escalate ambiguous cases to clinicians for review.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                      FastAPI Service                        │
│              POST /process-patient  GET /reviews            │
└────────────────────────────┬────────────────────────────────┘
                             │
┌────────────────────────────▼────────────────────────────────┐
│              CrewAI Triage Supervisor Agent                  │
│    Routes tasks · Scores confidence · Escalates to HITL     │
└──────────┬──────────────────┬─────────────────┬─────────────┘
           │                  │                  │
    ┌──────▼──────┐   ┌───────▼──────┐   ┌──────▼──────┐
    │ Prior Auth  │   │  Care Gap    │   │ Risk Triage │
    │   Agent     │   │   Agent      │   │   Agent     │
    │  LangGraph  │   │  LangGraph   │   │  LangGraph  │
    │  ReAct Loop │   │ Plan+Execute │   │   Scoring   │
    └──────┬──────┘   └───────┬──────┘   └──────┬──────┘
           │                  │                  │
┌──────────▼──────────────────▼──────────────────▼──────────┐
│                    MCP Tool Layer                           │
│   EHR Tools        Payer Policy Tools     Risk Tools       │
│  (patient data,    (coverage rules,      (comorbidity,     │
│   labs, history)    formulary, limits)    risk scores)     │
└────────────────────────────┬──────────────────────────────┘
                             │
┌────────────────────────────▼──────────────────────────────┐
│                  Observability & Storage                    │
│   Prefect (scheduling, retries)  LangSmith (tracing)       │
│   SQLite (HITL queue)            FastAPI (REST API)        │
└────────────────────────────┬──────────────────────────────┘
                             │
┌────────────────────────────▼──────────────────────────────┐
│            Streamlit Human-in-the-Loop Dashboard           │
│     Pending reviews · Approve / Reject / Modify            │
└───────────────────────────────────────────────────────────┘
```

### Agent Design Patterns Used

| Agent | Framework | Pattern | Purpose |
|-------|-----------|---------|---------|
| Prior Auth Agent | LangGraph | ReAct (Reason+Act) | Iterative tool use: fetch patient → check policy → decide |
| Care Gap Agent | LangGraph | Plan-and-Execute | Plan all gap checks upfront, execute sequentially |
| Risk Triage Agent | LangGraph | Scoring loop | Multi-factor risk assessment with threshold routing |
| Triage Supervisor | CrewAI | Multi-agent orchestration | Route, coordinate, and escalate across sub-agents |

---

## Key Features

- **Multi-agent coordination** — CrewAI supervisor delegates to specialized LangGraph agents
- **MCP tool architecture** — standardized tool interfaces for EHR and payer data access
- **ReAct + Plan-and-Execute** — different reasoning patterns per workflow complexity
- **Human-in-the-loop** — automatic escalation when agent confidence < configurable threshold
- **Prefect orchestration** — scheduled batch runs with retry logic, failure alerts, and task dependencies
- **LangSmith observability** — full trace capture: latency per node, token usage, tool call sequences
- **FastAPI service layer** — REST endpoints for triggering workflows and consuming results
- **Streamlit review dashboard** — clinicians approve, reject, or modify escalated decisions

---

## Project Structure

```
healthcare-ai-agents/
├── README.md
├── requirements.txt
├── .env.example
│
├── agents/
│   ├── __init__.py
│   ├── prior_auth_agent.py       # LangGraph ReAct agent for prior auth
│   ├── care_gap_agent.py         # LangGraph Plan-and-Execute for care gaps
│   └── triage_supervisor.py      # CrewAI multi-agent supervisor
│
├── tools/
│   ├── __init__.py
│   ├── ehr_tools.py              # MCP-style EHR data tools
│   ├── payer_tools.py            # Payer policy & coverage tools
│   └── risk_tools.py             # Risk scoring & comorbidity tools
│
├── orchestration/
│   ├── __init__.py
│   └── prefect_flow.py           # Prefect scheduled workflow
│
├── api/
│   ├── __init__.py
│   └── main.py                   # FastAPI endpoints
│
├── frontend/
│   └── app.py                    # Streamlit HITL dashboard
│
├── data/
│   ├── synthetic_patients.json   # Sample patient data (HIPAA-safe synthetic)
│   └── payer_policies.json       # Sample payer coverage rules
│
└── tests/
    ├── test_prior_auth.py
    ├── test_care_gap.py
    └── test_triage.py
```

---

## Setup

### 1. Clone the repository

```bash
git clone https://github.com/harshinireddy2204/healthcare-ai-agents.git
cd healthcare-ai-agents
```

### 2. Create a virtual environment

```bash
python -m venv venv
source venv/bin/activate        # macOS/Linux
# venv\Scripts\activate         # Windows
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure environment variables

```bash
cp .env.example .env
# Edit .env and fill in your API keys
```

Required keys:

| Variable | Where to get it |
|----------|-----------------|
| `OPENAI_API_KEY` | [platform.openai.com](https://platform.openai.com) |
| `LANGCHAIN_API_KEY` | [smith.langchain.com](https://smith.langchain.com) |
| `LANGCHAIN_PROJECT` | Set any name, e.g. `healthcare-ai-agents` |

### 5. Initialize the database

```bash
python -c "from api.main import init_db; init_db()"
```

---

## Running the System

### Start the FastAPI server

```bash
uvicorn api.main:app --reload --port 8000
```

API docs available at: `http://localhost:8000/docs`

### Run a single patient through the pipeline

```bash
curl -X POST http://localhost:8000/process-patient \
  -H "Content-Type: application/json" \
  -d '{"patient_id": "P001"}'
```

### Start the Prefect scheduled workflow

```bash
python orchestration/prefect_flow.py
```

### Launch the HITL review dashboard

```bash
streamlit run frontend/app.py
```

Dashboard available at: `http://localhost:8501`

---

## API Reference

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/process-patient` | Trigger full agent workflow for a patient |
| `GET` | `/pending-reviews` | List all cases awaiting human review |
| `POST` | `/resolve-review/{id}` | Approve, reject, or modify an escalated case |
| `GET` | `/audit-log` | Full decision audit trail |
| `GET` | `/health` | Service health check |

---

## Agent Workflow Detail

### Prior Authorization Agent (ReAct Loop)

```
User request → [Reason] What patient data do I need?
             → [Act]    get_patient_demographics(P001)
             → [Observe] Patient: 67F, Dx: Type 2 Diabetes
             → [Reason] What does the payer policy say?
             → [Act]    get_payer_policy("insulin_pump", "BlueCross")
             → [Observe] Covered if HbA1c > 9.0 and tried ≥2 oral meds
             → [Reason] Does this patient qualify?
             → [Act]    get_lab_results(P001, "HbA1c")
             → [Observe] HbA1c = 10.2 — qualifies
             → [Final]  APPROVE — generate auth justification
```

### Care Gap Agent (Plan-and-Execute)

```
[Plan]  1. Check mammogram (due if female > 40, last > 12mo)
        2. Check HbA1c (due if diabetic, last > 3mo)
        3. Check flu vaccine (due annually)
        4. Check nephrology referral (due if eGFR < 60)

[Execute step 1] get_screening_history(P001, "mammogram") → last: 18mo ago → GAP
[Execute step 2] get_lab_results(P001, "HbA1c") → last: 2mo ago → OK
[Execute step 3] get_immunization_history(P001, "flu") → current → OK
[Execute step 4] get_lab_results(P001, "eGFR") → 45 → GAP → schedule_referral()

[Output] 2 care gaps identified. Outreach scheduled.
```

### Confidence-Based HITL Escalation

```python
if supervisor_confidence < CONFIDENCE_THRESHOLD:  # default 0.75
    write_to_review_queue(case)    # → SQLite
    notify_clinician(case)         # → dashboard alert
    return "PENDING_HUMAN_REVIEW"
```

---

## Observability

This project uses **LangSmith** for full agent tracing. Every agent run captures:
- Token usage per node
- Latency per tool call
- Full reasoning chain
- Tool inputs and outputs
- Final decision with confidence score

Set `LANGCHAIN_TRACING_V2=true` in your `.env` to enable.

View traces at: [smith.langchain.com](https://smith.langchain.com)

---

## Synthetic Data

All patient data in `data/synthetic_patients.json` is fully synthetic and HIPAA-safe. It was generated to reflect realistic clinical distributions without containing any real patient information.

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| LLM backbone | GPT-4o via OpenAI API |
| Agent framework | LangGraph 0.2 + CrewAI 0.80 |
| Tool protocol | MCP-style structured tools |
| Orchestration | Prefect 3.x |
| Observability | LangSmith |
| API | FastAPI + Uvicorn |
| HITL dashboard | Streamlit |
| Storage | SQLite (dev) / PostgreSQL (prod) |
| Language | Python 3.11+ |

---

## Roadmap

- [ ] Add RAG layer over clinical guidelines (USPSTF, CMS)
- [ ] Replace synthetic EHR with FHIR R4 API integration
- [ ] Add AutoGen debate pattern for high-stakes auth decisions
- [ ] Export Prefect run reports to PDF
- [ ] Add evaluation harness with LangSmith datasets

---

## Author

**Harshini Reddy**  
Business & Data Analyst | AI Engineer  
[LinkedIn](https://linkedin.com/in/harshinireddy) · [GitHub](https://github.com/harshinireddy2204)
