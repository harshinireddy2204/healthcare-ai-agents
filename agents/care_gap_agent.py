"""
agents/care_gap_agent.py

Care Gap Agent — LangGraph Plan-and-Execute pattern.

Unlike ReAct which reasons one step at a time, Plan-and-Execute:
  1. [PLAN]    Generates a full list of checks to perform
  2. [EXECUTE] Runs each check sequentially using tools
  3. [REPORT]  Synthesizes findings into actionable care gaps

This is better for care gap detection because all the checks are known
upfront and can be planned systematically.
"""
import os
from typing import Annotated, TypedDict, Optional

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode

from tools.ehr_tools import EHR_TOOLS
from tools.risk_tools import RISK_TOOLS
from rag.retriever import GUIDELINE_TOOLS

load_dotenv()

# ── State ──────────────────────────────────────────────────────────────────────

class CareGapState(TypedDict):
    messages: Annotated[list, add_messages]
    patient_id: str
    plan: list[str]        # list of check steps to execute
    plan_index: int        # which plan step we're on
    completed_checks: list # results gathered so far
    gaps_found: list       # identified care gaps
    final_report: str


# ── Tools ─────────────────────────────────────────────────────────────────────

ALL_TOOLS = EHR_TOOLS + RISK_TOOLS + GUIDELINE_TOOLS


# ── Lazy LLM factory (called inside each node, not at import time) ─────────────

def get_llms():
    """
    Create LLM instances inside the function so they are only
    initialized when actually called — not at module import time.
    This prevents 'str has no attribute bind_tools' when the module
    is imported in a background thread before .env loads.
    """
    load_dotenv()
    model = os.getenv("MODEL_NAME", "gpt-4o")
    planner = ChatOpenAI(model=model, temperature=0)
    executor = ChatOpenAI(model=model, temperature=0).bind_tools(ALL_TOOLS)
    tools = ToolNode(ALL_TOOLS)
    return planner, executor, tools


# ── Prompts ───────────────────────────────────────────────────────────────────

PLANNER_SYSTEM = """You are a clinical care gap specialist. Your job is to plan
a systematic review of a patient's preventive care needs.

Given a patient ID, generate a numbered list of specific data checks to perform.
Each check should be a single concrete action. Always include a step to search
clinical guidelines to ground your findings in evidence.

Example plan:
  1. Get patient demographics (age, gender, diagnoses)
  2. Check mammogram screening history
  3. Check colonoscopy history
  4. Check flu vaccine history
  5. Check most recent HbA1c (if diabetic)
  6. Search clinical guidelines for diabetes care gaps
  7. Search clinical guidelines for preventive screening recommendations
  8. Calculate risk score

Output ONLY the numbered list, nothing else. Be specific about what to look up.
"""

EXECUTOR_SYSTEM = """You are executing a care gap check for a patient.
Use the available tools to complete the current step.
After calling tools, summarize what you found in 1-2 sentences.
"""

REPORTER_SYSTEM = """You are a care gap report writer. Given the completed checks,
write a clear, actionable summary.

CRITICAL: Every care gap MUST cite the specific clinical guideline that supports it.
Format: "Per [Guideline Name]: [recommendation]"

Structure your report as:
1. Care gaps found (with priority level and guideline citation)
2. Recommended interventions (with specific action and timeline)
3. Clinical guideline references (list all sources used)

Keep it concise but complete. A care coordinator should be able to act on it immediately.
"""


# ── Nodes ─────────────────────────────────────────────────────────────────────

def plan_node(state: CareGapState) -> dict:
    """Node: generate a list of checks to perform for this patient."""
    planner_llm, _, _ = get_llms()

    response = planner_llm.invoke([
        SystemMessage(content=PLANNER_SYSTEM),
        HumanMessage(content=f"Create a care gap review plan for patient {state['patient_id']}.")
    ])

    # Parse the numbered list into individual steps
    plan_steps = []
    for line in response.content.strip().split("\n"):
        line = line.strip()
        if line and line[0].isdigit():
            step = line.split(".", 1)[-1].split(")", 1)[-1].strip()
            plan_steps.append(step)

    return {
        "plan": plan_steps,
        "plan_index": 0,
        "messages": [HumanMessage(content=f"Plan created with {len(plan_steps)} steps.")]
    }


def execute_step_node(state: CareGapState) -> dict:
    """Node: execute the current plan step using tools."""
    _, executor_llm, tool_node = get_llms()

    current_step = state["plan"][state["plan_index"]]

    messages = [
        SystemMessage(content=EXECUTOR_SYSTEM),
        HumanMessage(
            content=(
                f"Patient ID: {state['patient_id']}\n"
                f"Current step: {current_step}\n"
                f"Execute this step using the available tools."
            )
        )
    ]

    response = executor_llm.invoke(messages)
    new_messages = [response]

    # If tools were called, run them
    if hasattr(response, "tool_calls") and response.tool_calls:
        messages_with_tools = messages + [response]
        tool_results = tool_node.invoke({"messages": messages_with_tools})
        new_messages.extend(tool_results["messages"])

        # Get the final summary after tool calls
        all_msgs = messages + new_messages
        final = executor_llm.invoke(all_msgs)
        new_messages.append(final)

    completed = state.get("completed_checks", [])
    completed.append({
        "step": current_step,
        "result": new_messages[-1].content if hasattr(new_messages[-1], "content") else ""
    })

    return {
        "messages": new_messages,
        "completed_checks": completed,
        "plan_index": state["plan_index"] + 1
    }


def should_continue_plan(state: CareGapState) -> str:
    """Edge: continue executing steps or move to reporting."""
    if state["plan_index"] < len(state["plan"]):
        return "execute_step"
    return "report"


def report_node(state: CareGapState) -> dict:
    """Node: synthesize all check results into a care gap report."""
    planner_llm, _, _ = get_llms()

    checks_summary = "\n".join([
        f"Step {i+1}: {c['step']}\nResult: {c['result']}"
        for i, c in enumerate(state["completed_checks"])
    ])

    response = planner_llm.invoke([
        SystemMessage(content=REPORTER_SYSTEM),
        HumanMessage(
            content=(
                f"Patient: {state['patient_id']}\n\n"
                f"Completed checks:\n{checks_summary}\n\n"
                f"Write the care gap report."
            )
        )
    ])

    return {
        "final_report": response.content,
        "messages": [response]
    }


# ── Graph assembly ─────────────────────────────────────────────────────────────

def build_care_gap_graph() -> StateGraph:
    graph = StateGraph(CareGapState)

    graph.add_node("plan", plan_node)
    graph.add_node("execute_step", execute_step_node)
    graph.add_node("report", report_node)

    graph.add_edge(START, "plan")
    graph.add_edge("plan", "execute_step")
    graph.add_conditional_edges(
        "execute_step",
        should_continue_plan,
        {"execute_step": "execute_step", "report": "report"}
    )
    graph.add_edge("report", END)

    return graph.compile()


# ── Public interface ──────────────────────────────────────────────────────────

care_gap_graph = build_care_gap_graph()


def run_care_gap_review(patient_id: str) -> dict:
    """
    Run the care gap Plan-and-Execute agent for a single patient.

    Returns:
        dict with patient_id, plan, completed_checks, final_report
    """
    result = care_gap_graph.invoke({
        "messages": [],
        "patient_id": patient_id,
        "plan": [],
        "plan_index": 0,
        "completed_checks": [],
        "gaps_found": [],
        "final_report": ""
    })

    return {
        "patient_id": patient_id,
        "steps_executed": len(result["completed_checks"]),
        "plan": result["plan"],
        "completed_checks": result["completed_checks"],
        "final_report": result["final_report"]
    }


# ── CLI test ──────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    result = run_care_gap_review("P001")
    print("\n=== Care Gap Review Result ===")
    print(f"Steps executed: {result['steps_executed']}")
    print(f"\nFinal Report:\n{result['final_report']}")
