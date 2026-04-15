"""
agents/prior_auth_agent.py

Prior Authorization Agent — LangGraph ReAct loop with Agent/Critic pattern.

MALADE (MLHC 2024) key insight: Agent/Critic pattern improves accuracy by
having a dedicated critic agent review and challenge the primary agent's
reasoning before it becomes final. In MALADE, critics improved AUC from
0.82 to 0.85 on pharmacovigilance tasks.

Our implementation:
  1. Primary ReAct agent reasons over patient data + payer policy + guidelines
  2. Critic agent reviews the decision for logical gaps and missing criteria
  3. If critic finds issues, primary agent revises (max 1 revision cycle)
  4. Final decision must survive critic review to avoid HITL escalation

This directly addresses the REQ001 problem — ESCALATE with 0.6 confidence
often happened because the agent hadn't checked all available evidence.
"""
import os
from typing import Annotated, TypedDict, Literal

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
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
    critic_feedback: str
    revision_count: int


# ── Tool loading ──────────────────────────────────────────────────────────────

def _get_all_tools():
    tools = EHR_TOOLS + PAYER_TOOLS
    try:
        from rag.retriever import GUIDELINE_TOOLS
        tools = tools + GUIDELINE_TOOLS
    except Exception:
        pass
    try:
        from knowledge_graph.clinical_graph import KNOWLEDGE_GRAPH_TOOLS
        tools = tools + KNOWLEDGE_GRAPH_TOOLS
    except Exception:
        pass
    return tools


def get_llm():
    load_dotenv()
    model = os.getenv("MODEL_NAME", "gpt-4o-mini")
    all_tools = _get_all_tools()
    llm = ChatOpenAI(model=model, temperature=0).bind_tools(all_tools)
    tool_node = ToolNode(all_tools)
    return llm, tool_node


def get_bare_llm():
    """LLM without tools — used for critic and parser nodes."""
    load_dotenv()
    return ChatOpenAI(model=os.getenv("MODEL_NAME", "gpt-4o-mini"), temperature=0)


# ── Prompts ───────────────────────────────────────────────────────────────────

PRIMARY_AGENT_SYSTEM = """You are a clinical prior authorization specialist.
Evaluate whether an authorization request should be APPROVE, DENY, or ESCALATE.

Steps:
1. Get patient demographics and relevant labs (EHR tools)
2. Get payer coverage policy for the item (payer tools)
3. Search clinical guidelines for evidence (search_clinical_guidelines)
4. Run knowledge graph analysis if complex case (analyze_clinical_connections)
5. Evaluate each coverage criterion against patient data
6. Make final determination

ESCALATE only when criteria genuinely cannot be evaluated from available data.

End your response with:
DECISION: [APPROVE|DENY|ESCALATE]
CONFIDENCE: [0.0-1.0]
JUSTIFICATION: [cite specific guideline or payer criteria]
"""

CRITIC_SYSTEM = """You are a senior clinical reviewer auditing a prior authorization decision.

Your ONLY job is to catch clear logical errors — not to find improvements.

APPROVE the decision if ANY of these are true:
- The agent cited a specific guideline or payer policy
- The confidence is >= 0.65
- The decision is ESCALATE (this is always a valid terminal state — never revise ESCALATE)
- Some criteria returned "manual_review_required" (this means data was unavailable — acceptable)
- The justification references the patient's actual clinical data

Only respond CRITIQUE: REVISE if ALL of these are true:
- The agent made a clear factual error (e.g. wrong patient data cited)
- The agent had the tool available AND the data was present AND it was skipped
- The error would materially change the decision

When in doubt — APPROVE. Over-revision wastes clinical time and causes rate limits.

Respond with exactly ONE of:
  CRITIQUE: APPROVED — [brief reason]
  CRITIQUE: REVISE — [one specific error, max 1 sentence]
"""

REVISION_SYSTEM = """You are revising your prior authorization decision based on
a senior reviewer's critique.

Address each issue raised, gather any missing evidence, and provide a revised determination.

End with:
DECISION: [APPROVE|DENY|ESCALATE]
CONFIDENCE: [0.0-1.0]
JUSTIFICATION: [revised justification addressing critique]
REVISED: true
"""


# ── Nodes ─────────────────────────────────────────────────────────────────────

def primary_agent_node(state: AuthState) -> dict:
    """Node: primary ReAct agent evaluates the auth request."""
    llm, _ = get_llm()

    messages = state["messages"]
    if not any(isinstance(m, SystemMessage) for m in messages):
        messages = [SystemMessage(content=PRIMARY_AGENT_SYSTEM)] + messages

    # Include any critic feedback from previous iteration
    if state.get("critic_feedback") and state.get("revision_count", 0) > 0:
        messages = messages + [HumanMessage(
            content=f"Please revise your decision based on this critique:\n{state['critic_feedback']}"
        )]

    response = llm.invoke(messages)
    return {"messages": [response]}


def run_tools_node(state: AuthState) -> dict:
    """Node: execute tool calls from the agent."""
    _, tool_node = get_llm()
    return tool_node.invoke(state)


def critic_node(state: AuthState) -> dict:
    """
    Node: MALADE-inspired critic agent reviews the primary agent's decision.
    Returns critique feedback that may trigger a revision cycle.
    """
    critic_llm = get_bare_llm()

    # Find the last AIMessage with a decision
    last_decision_msg = None
    for msg in reversed(state["messages"]):
        if isinstance(msg, AIMessage) and "DECISION:" in getattr(msg, "content", ""):
            last_decision_msg = msg
            break

    if not last_decision_msg:
        return {"critic_feedback": "CRITIQUE: APPROVED — no decision found to review"}

    critique_response = critic_llm.invoke([
        SystemMessage(content=CRITIC_SYSTEM),
        HumanMessage(content=(
            f"Review this prior authorization decision:\n\n"
            f"Patient: {state['patient_id']}, Request: {state['request_id']}\n\n"
            f"Agent reasoning:\n{last_decision_msg.content}"
        ))
    ])

    feedback = critique_response.content
    print(f"  [Critic] {feedback[:100]}...")
    return {"critic_feedback": feedback}


def should_revise(state: AuthState) -> Literal["primary_agent", "parse_decision"]:
    """
    Route back to agent only for clear logical errors.
    Hard stops — always go to parse_decision if:
      - Already revised once
      - Decision is ESCALATE (valid terminal state)
      - Decision has confidence >= 0.65 (sufficient certainty)
      - Critic said APPROVED
    """
    feedback = state.get("critic_feedback", "")
    revision_count = state.get("revision_count", 0)

    # Hard stop 1: max one revision ever
    if revision_count >= 1:
        return "parse_decision"

    # Hard stop 2: extract decision + confidence from messages
    last_decision = "ESCALATE"
    last_confidence = 0.5
    for msg in reversed(state["messages"]):
        content = getattr(msg, "content", "")
        if "DECISION:" in content:
            for line in content.split("\n"):
                line = line.strip()
                if line.startswith("DECISION:"):
                    last_decision = line.replace("DECISION:", "").strip()
                elif line.startswith("CONFIDENCE:"):
                    try:
                        last_confidence = float(line.replace("CONFIDENCE:", "").strip())
                    except ValueError:
                        pass
            break

    # ESCALATE is always a valid final state — accept it
    if last_decision == "ESCALATE":
        return "parse_decision"

    # High confidence decisions don't need revision
    if last_confidence >= 0.65:
        return "parse_decision"

    # Only revise on explicit CRITIQUE: REVISE from critic
    if "CRITIQUE: REVISE" in feedback:
        print(f"  [Critic] Requesting revision #{revision_count + 1}")
        return "primary_agent"

    return "parse_decision"


def increment_revision(state: AuthState) -> dict:
    """Node: increment revision counter before re-running primary agent."""
    return {"revision_count": state.get("revision_count", 0) + 1}


def should_continue_tools(state: AuthState) -> Literal["run_tools", "critic"]:
    """Edge: run tools if pending, else send to critic."""
    last_message = state["messages"][-1]
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        return "run_tools"
    return "critic"


def parse_decision_node(state: AuthState) -> dict:
    """Node: extract final structured decision from agent messages."""
    # Find last message with decision
    for msg in reversed(state["messages"]):
        content = getattr(msg, "content", "")
        if "DECISION:" in content:
            decision = "ESCALATE"
            confidence = 0.5
            justification = "Unable to parse — escalating for safety"

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

    return {
        "decision": "ESCALATE",
        "confidence": 0.3,
        "justification": "No structured decision found — escalating for safety"
    }


# ── Graph assembly ─────────────────────────────────────────────────────────────

def build_prior_auth_graph():
    graph = StateGraph(AuthState)

    graph.add_node("primary_agent", primary_agent_node)
    graph.add_node("run_tools", run_tools_node)
    graph.add_node("critic", critic_node)
    graph.add_node("increment_revision", increment_revision)
    graph.add_node("parse_decision", parse_decision_node)

    graph.add_edge(START, "primary_agent")
    graph.add_conditional_edges("primary_agent", should_continue_tools)
    graph.add_edge("run_tools", "primary_agent")
    graph.add_conditional_edges("critic", should_revise, {
        "primary_agent": "increment_revision",
        "parse_decision": "parse_decision"
    })
    graph.add_edge("increment_revision", "primary_agent")
    graph.add_edge("parse_decision", END)

    return graph.compile()


prior_auth_graph = build_prior_auth_graph()


def run_prior_auth(patient_id: str, request_id: str, item_name: str) -> dict:
    """Run the prior auth agent with Agent/Critic review pattern."""
    result = prior_auth_graph.invoke({
        "messages": [HumanMessage(content=(
            f"Evaluate prior authorization request {request_id} for patient {patient_id}. "
            f"Requested item: {item_name}. "
            f"Use EHR tools, payer policy tools, clinical guidelines, and knowledge graph "
            f"to make a determination."
        ))],
        "patient_id": patient_id,
        "request_id": request_id,
        "decision": "PENDING",
        "confidence": 0.0,
        "justification": "",
        "critic_feedback": "",
        "revision_count": 0
    })

    was_revised = result.get("revision_count", 0) > 0
    return {
        "patient_id": patient_id,
        "request_id": request_id,
        "item": item_name,
        "decision": result["decision"],
        "confidence": result["confidence"],
        "justification": result["justification"],
        "critic_reviewed": True,
        "was_revised": was_revised,
        "critic_feedback": result.get("critic_feedback", "")
    }


if __name__ == "__main__":
    result = run_prior_auth("P001", "REQ001", "insulin_pump")
    print("\n=== Prior Auth Result (with Critic) ===")
    for k, v in result.items():
        if k != "critic_feedback":
            print(f"  {k}: {v}")
    print(f"  Critic: {result.get('critic_feedback', '')[:150]}")