"""
agents/prior_auth_agent.py

Prior Authorization Agent — LangGraph ReAct loop.

Reasoning pattern:
  [Reason] What data do I need to evaluate this auth request?
  [Act]    Call EHR / payer tools
  [Observe] Process tool results
  [Repeat] Until a final APPROVE / DENY / ESCALATE decision is reached

The agent traces every step to LangSmith automatically via the
LANGCHAIN_TRACING_V2 environment variable.
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
    """State that flows through the prior auth graph nodes."""
    messages: Annotated[list, add_messages]
    patient_id: str
    request_id: str
    decision: str          # APPROVE | DENY | ESCALATE | PENDING
    confidence: float
    justification: str


# ── Tools ─────────────────────────────────────────────────────────────────────

ALL_TOOLS = EHR_TOOLS + PAYER_TOOLS


# ── Lazy LLM factory ──────────────────────────────────────────────────────────

def get_llm():
    """
    Create LLM instance inside the function so it is only initialized
    when actually called — not at module import time.
    Prevents bind_tools errors when imported in background threads.
    """
    load_dotenv()
    model = os.getenv("MODEL_NAME", "gpt-4o")
    llm = ChatOpenAI(model=model, temperature=0).bind_tools(ALL_TOOLS)
    tool_node = ToolNode(ALL_TOOLS)
    return llm, tool_node


# ── Prompts ───────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are a clinical prior authorization specialist AI agent.
Your job is to evaluate whether a prior authorization request should be approved,
denied, or escalated to a human reviewer.

For every request you must:
1. Retrieve the patient's demographics and relevant lab results using EHR tools
2. Look up the payer's coverage policy for the requested item
3. Check whether the patient meets the coverage criteria
4. Make a final decision: APPROVE, DENY, or ESCALATE

ESCALATE when:
- Criteria evaluation requires clinical judgment beyond the data
- A criterion says "manual_review_required"
- You are less than 75% confident in your determination

Always end your response with a structured decision block:
DECISION: [APPROVE|DENY|ESCALATE]
CONFIDENCE: [0.0-1.0]
JUSTIFICATION: [one sentence summary]
"""


# ── Nodes ─────────────────────────────────────────────────────────────────────

def call_agent(state: AuthState) -> dict:
    """Node: invoke the LLM with current messages."""
    llm, _ = get_llm()

    messages = state["messages"]
    if not any(isinstance(m, SystemMessage) for m in messages):
        messages = [SystemMessage(content=SYSTEM_PROMPT)] + messages

    response = llm.invoke(messages)
    return {"messages": [response]}


def run_tools(state: AuthState) -> dict:
    """Node: execute any tool calls in the last message."""
    _, tool_node = get_llm()
    return tool_node.invoke(state)


def parse_decision(state: AuthState) -> dict:
    """Node: extract the structured decision from the last AI message."""
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
    """Edge: route to tools if there are pending tool calls, else finalize."""
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
    graph.add_edge("run_tools", "agent")   # loop back after tool use
    graph.add_edge("parse_decision", END)

    return graph.compile()


# ── Public interface ──────────────────────────────────────────────────────────

prior_auth_graph = build_prior_auth_graph()


def run_prior_auth(patient_id: str, request_id: str, item_name: str) -> dict:
    """
    Run the prior authorization ReAct agent for a single request.

    Returns:
        dict with patient_id, request_id, decision, confidence, justification
    """
    initial_message = HumanMessage(
        content=(
            f"Please evaluate prior authorization request {request_id} "
            f"for patient {patient_id}. "
            f"The requested item is: {item_name}. "
            f"Use the available tools to check patient data and payer policy, "
            f"then make a determination."
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


# ── CLI test ──────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    result = run_prior_auth("P001", "REQ001", "insulin_pump")
    print("\n=== Prior Auth Result ===")
    for k, v in result.items():
        print(f"  {k}: {v}")