"""
agents/prior_auth_agent.py

Prior Authorization Agent — LangGraph ReAct loop.

RAG integration: the agent can now call search_clinical_guidelines()
to retrieve coverage criteria from USPSTF, ADA, KDIGO, and ACC/AHA
guidelines to support or challenge payer coverage decisions.
"""
import os
from typing import Annotated, TypedDict, Literal

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode

from tools.ehr_tools import EHR_TOOLS
from tools.payer_tools import PAYER_TOOLS

load_dotenv()

# ── State ──────────────────────────────────────────────────────────────────────

class AuthState(TypedDict):
    messages: Annotated[list, add_messages]
    patient_id: str
    request_id: str
    decision: str
    confidence: float
    justification: str


# ── Tools (EHR + Payer + RAG Guidelines) ──────────────────────────────────────

def _get_all_tools_kg():
    """Load EHR + Payer + RAG + Knowledge Graph tools."""
    tools = EHR_TOOLS + PAYER_TOOLS
    try:
        from rag.retriever import GUIDELINE_TOOLS
        tools = tools + GUIDELINE_TOOLS
    except Exception:
        pass
    try:
        from knowledge_graph.clinical_graph import KNOWLEDGE_GRAPH_TOOLS
        tools = tools + KNOWLEDGE_GRAPH_TOOLS
        print("[PriorAuthAgent] Knowledge graph tools loaded ✓")
    except Exception as e:
        pass
    return tools
    tools = EHR_TOOLS + PAYER_TOOLS
    try:
        from rag.retriever import GUIDELINE_TOOLS
        tools = tools + GUIDELINE_TOOLS
        print("[PriorAuthAgent] RAG guideline tools loaded ✓")
    except Exception as e:
        print(f"[PriorAuthAgent] RAG tools not available: {e}")
    return tools


# ── Lazy LLM factory ──────────────────────────────────────────────────────────

def get_llm():
    load_dotenv()
    model = os.getenv("MODEL_NAME", "gpt-4o-mini")
    all_tools = _get_all_tools_kg()
    llm = ChatOpenAI(model=model, temperature=0).bind_tools(all_tools)
    tool_node = ToolNode(all_tools)
    return llm, tool_node


# ── Prompt ────────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are a clinical prior authorization specialist AI agent.
Your job is to evaluate whether a prior authorization request should be approved,
denied, or escalated to a human reviewer.

For every request you must:
1. Retrieve the patient's demographics and relevant lab results using EHR tools
2. Look up the payer's coverage policy for the requested item
3. Search clinical guidelines (search_clinical_guidelines) to verify the clinical
   appropriateness of the requested item based on the patient's condition
4. Check whether the patient meets the coverage criteria
5. Make a final decision: APPROVE, DENY, or ESCALATE

ESCALATE when:
- Criteria evaluation requires clinical judgment beyond the data
- A criterion says "manual_review_required"
- You are less than 75% confident in your determination

When guidelines support the auth request but the payer policy is unclear,
cite the guideline in your justification (e.g. "ADA 2025 recommends insulin pump
for HbA1c > 9% — patient qualifies").

Always end your response with:
DECISION: [APPROVE|DENY|ESCALATE]
CONFIDENCE: [0.0-1.0]
JUSTIFICATION: [one sentence including guideline citation if applicable]
"""


# ── Nodes ─────────────────────────────────────────────────────────────────────

def call_agent(state: AuthState) -> dict:
    llm, _ = get_llm()

    messages = state["messages"]
    if not any(isinstance(m, SystemMessage) for m in messages):
        messages = [SystemMessage(content=SYSTEM_PROMPT)] + messages

    response = llm.invoke(messages)
    return {"messages": [response]}


def run_tools(state: AuthState) -> dict:
    _, tool_node = get_llm()
    return tool_node.invoke(state)


def parse_decision(state: AuthState) -> dict:
    last_message = state["messages"][-1]
    content = last_message.content if hasattr(last_message, "content") else ""

    decision = "ESCALATE"
    confidence = 0.5
    justification = "Unable to parse agent output — escalating for safety."

    for line in content.split("\n"):
        line = line.strip()
        if line.startswith("DECISION:"):
            val = line.replace("DECISION:", "").strip()
            if val in ("APPROVE", "DENY", "ESCALATE"):
                decision = val
        elif line.startswith("CONFIDENCE:"):
            try:
                confidence = float(line.replace("CONFIDENCE:", "").strip())
            except ValueError:
                pass
        elif line.startswith("JUSTIFICATION:"):
            justification = line.replace("JUSTIFICATION:", "").strip()

    return {
        "decision": decision,
        "confidence": confidence,
        "justification": justification
    }


def should_continue(state: AuthState) -> Literal["run_tools", "parse_decision"]:
    last_message = state["messages"][-1]
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        return "run_tools"
    return "parse_decision"


# ── Graph assembly ─────────────────────────────────────────────────────────────

def build_prior_auth_graph() -> StateGraph:
    graph = StateGraph(AuthState)

    graph.add_node("agent", call_agent)
    graph.add_node("run_tools", run_tools)
    graph.add_node("parse_decision", parse_decision)

    graph.add_edge(START, "agent")
    graph.add_conditional_edges("agent", should_continue)
    graph.add_edge("run_tools", "agent")
    graph.add_edge("parse_decision", END)

    return graph.compile()


prior_auth_graph = build_prior_auth_graph()


def run_prior_auth(patient_id: str, request_id: str, item_name: str) -> dict:
    initial_message = HumanMessage(
        content=(
            f"Please evaluate prior authorization request {request_id} "
            f"for patient {patient_id}. "
            f"The requested item is: {item_name}. "
            f"Use EHR tools to check patient data, payer tools to check coverage policy, "
            f"and search_clinical_guidelines to find supporting clinical evidence. "
            f"Then make a determination."
        )
    )

    result = prior_auth_graph.invoke({
        "messages": [initial_message],
        "patient_id": patient_id,
        "request_id": request_id,
        "decision": "PENDING",
        "confidence": 0.0,
        "justification": ""
    })

    return {
        "patient_id": patient_id,
        "request_id": request_id,
        "item": item_name,
        "decision": result["decision"],
        "confidence": result["confidence"],
        "justification": result["justification"],
    }


if __name__ == "__main__":
    result = run_prior_auth("P001", "REQ001", "insulin_pump")
    print("\n=== Prior Auth Result ===")
    for k, v in result.items():
        print(f"  {k}: {v}")