"""
tests/test_care_gap.py

Tests for the care gap Plan-and-Execute agent.
"""
import pytest


class TestCareGapAgent:

    def test_graph_compiles(self):
        from agents.care_gap_agent import build_care_gap_graph
        graph = build_care_gap_graph()
        assert graph is not None

    def test_initial_state_valid(self):
        """Verify the initial state shape is accepted by the graph."""
        from agents.care_gap_agent import CareGapState
        state: CareGapState = {
            "messages": [],
            "patient_id": "P001",
            "plan": [],
            "plan_index": 0,
            "completed_checks": [],
            "gaps_found": [],
            "final_report": ""
        }
        assert state["patient_id"] == "P001"
        assert state["plan_index"] == 0

    def test_should_continue_returns_execute_when_steps_remain(self):
        from agents.care_gap_agent import should_continue_plan
        state = {
            "plan": ["step1", "step2", "step3"],
            "plan_index": 1,
            "completed_checks": [],
            "messages": [],
            "patient_id": "P001",
            "gaps_found": [],
            "final_report": ""
        }
        assert should_continue_plan(state) == "execute_step"

    def test_should_continue_returns_report_when_done(self):
        from agents.care_gap_agent import should_continue_plan
        state = {
            "plan": ["step1", "step2"],
            "plan_index": 2,  # exhausted
            "completed_checks": [],
            "messages": [],
            "patient_id": "P001",
            "gaps_found": [],
            "final_report": ""
        }
        assert should_continue_plan(state) == "report"
