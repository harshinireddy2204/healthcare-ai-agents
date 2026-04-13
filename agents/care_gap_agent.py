"""
agents/care_gap_agent.py

Care Gap Agent — LangGraph Plan-and-Execute pattern.

Unlike ReAct which reasons one step at a time, Plan-and-Execute:
  1. [PLAN]    Generates a full list of checks to perform
  2. [EXECUTE] Runs each check sequentially using tools
  3. [REPORT]  Synthesizes findings into actionable care gaps with guideline citations

RAG integration: agents now call search_clinical_guidelines() to ground
every care gap finding in a specific USPSTF, ADA, CDC, or KDIGO citation.
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

def _get_all_tools_with_kg():
    """Load all tools including knowledge graph if available."""
    tools = EHR_TOOLS + RISK_TOOLS
    try:
        from rag.retriever import GUIDELINE_TOOLS
        tools = tools + GUIDELINE_TOOLS
    except Exception:
        pass
    try:
        from knowledge_graph.clinical_graph import KNOWLEDGE_GRAPH_TOOLS
        tools = tools + KNOWLEDGE_GRAPH_TOOLS
        print("[CareGapAgent] Knowledge graph tools loaded ✓")
    except Exception as e:
        print(f"[CareGapAgent] KG tools not available: {e}")
    return tools

load_dotenv()

# ── State ──────────────────────────────────────────────────────────────────────

class CareGapState(TypedDict):
    messages: Annotated[list, add_messages]
    patient_id: str
    plan: list[str]
    plan_index: int
    completed_checks: list
    gaps_found: list
    final_report: str


# ── Tools (EHR + Risk + RAG Guidelines) ───────────────────────────────────────

def _get_all_tools_orig():
    """Load all tools including guideline search if available."""
    tools = EHR_TOOLS + RISK_TOOLS
    try:
        from rag.retriever import GUIDELINE_TOOLS
        tools = tools + GUIDELINE_TOOLS
        print("[CareGapAgent] RAG guideline tools loaded ✓")
    except Exception as e:
        print(f"[CareGapAgent] RAG tools not available (run refresh_flow.py first): {e}")
    return tools


# ── Lazy LLM factory ──────────────────────────────────────────────────────────

def get_llms():
    load_dotenv()
    model = os.getenv("MODEL_NAME", "gpt-4o-mini")
    all_tools = _get_all_tools_with_kg()
    planner = ChatOpenAI(model=model, temperature=0)
    executor = ChatOpenAI(model=model, temperature=0).bind_tools(all_tools)
    tools = ToolNode(all_tools)
    return planner, executor, tools


# ── Prompts ───────────────────────────────────────────────────────────────────

PLANNER_SYSTEM = """You are a clinical care gap specialist. Your job is to plan
a systematic review of a patient's preventive care needs.

Given a patient ID, generate a numbered list of specific data checks to perform.
Each check should be a single concrete action. Always include a step to search
clinical guidelines for evidence-based citations.

Example plan:
  1. Get patient demographics (age, gender, diagnoses)
  2. Check mammogram screening history
  3. Search clinical guidelines for mammogram recommendations (if female 40+)
  4. Check colonoscopy history
  5. Check flu vaccine history
  6. Calculate risk score
  7. Search clinical guidelines for any identified care gaps

Output ONLY the numbered list, nothing else.
"""

EXECUTOR_SYSTEM = """You are executing a care gap check for a patient.
Use the available tools to complete the current step.

When you use the search_clinical_guidelines tool, include the specific
guideline text in your summary so it can be cited in the final report.

After calling tools, summarize what you found in 1-2 sentences, including
any guideline citations retrieved.
"""

REPORTER_SYSTEM = """You are a care gap report writer. Given the completed checks,
write a clear, actionable summary that includes:

1. Care gaps found (with priority level)
2. The specific clinical guideline supporting each gap (cite source name and key criteria)
3. Recommended interventions
4. Any items that need clinical review

Format guideline citations as: [Source Name] — key recommendation text.

Example:
- Mammogram overdue — HIGH priority
  [USPSTF: Breast Cancer Screening] — Recommends biennial mammography for women 40-74 years
  Action: Schedule mammogram within 30 days

Keep the report actionable and clinically grounded.
"""


# ── Nodes ─────────────────────────────────────────────────────────────────────

def plan_node(state: CareGapState) -> dict:
    planner_llm, _, _ = get_llms()

    response = planner_llm.invoke([
        SystemMessage(content=PLANNER_SYSTEM),
        HumanMessage(content=f"Create a care gap review plan for patient {state['patient_id']}.")
    ])

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

    if hasattr(response, "tool_calls") and response.tool_calls:
        messages_with_tools = messages + [response]
        tool_results = tool_node.invoke({"messages": messages_with_tools})
        new_messages.extend(tool_results["messages"])
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
    if state["plan_index"] < len(state["plan"]):
        return "execute_step"
    return "report"


def report_node(state: CareGapState) -> dict:
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
                f"Write the care gap report with guideline citations."
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


care_gap_graph = build_care_gap_graph()


def run_care_gap_review(patient_id: str) -> dict:
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


if __name__ == "__main__":
    result = run_care_gap_review("P001")
    print("\n=== Care Gap Review Result ===")
    print(f"Steps executed: {result['steps_executed']}")
    print(f"\nFinal Report:\n{result['final_report']}")