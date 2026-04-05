"""
tests/test_prior_auth.py

Unit tests for the prior authorization agent.
Uses mocked tools so no real OpenAI calls are made in CI.
"""
import json
import pytest
from unittest.mock import patch, MagicMock


# ── Tool-level tests (no LLM needed) ─────────────────────────────────────────

class TestEHRTools:
    def test_get_patient_demographics_found(self):
        from tools.ehr_tools import get_patient_demographics
        result = get_patient_demographics.invoke({"patient_id": "P001"})
        assert result["patient_id"] == "P001"
        assert result["name"] == "Eleanor Vance"
        assert "Type 2 Diabetes" in result["diagnoses"]

    def test_get_patient_demographics_not_found(self):
        from tools.ehr_tools import get_patient_demographics
        result = get_patient_demographics.invoke({"patient_id": "UNKNOWN"})
        assert "error" in result

    def test_get_lab_results_found(self):
        from tools.ehr_tools import get_lab_results
        result = get_lab_results.invoke({"patient_id": "P001", "lab_name": "HbA1c"})
        assert result["value"] == 10.2
        assert result["unit"] == "%"

    def test_get_lab_results_missing_lab(self):
        from tools.ehr_tools import get_lab_results
        result = get_lab_results.invoke({"patient_id": "P001", "lab_name": "nonexistent"})
        assert result["result"] == "not_found"

    def test_get_screening_history_found(self):
        from tools.ehr_tools import get_screening_history
        result = get_screening_history.invoke(
            {"patient_id": "P001", "screening_type": "flu_vaccine"}
        )
        assert "last_date" in result

    def test_get_pending_auth_requests(self):
        from tools.ehr_tools import get_pending_auth_requests
        result = get_pending_auth_requests.invoke({"patient_id": "P001"})
        assert isinstance(result, list)
        assert len(result) > 0
        assert result[0]["request_id"] == "REQ001"


class TestPayerTools:
    def test_get_payer_policy_found(self):
        from tools.payer_tools import get_payer_policy
        result = get_payer_policy.invoke({
            "item_name": "insulin_pump",
            "insurance_plan": "BlueCross PPO"
        })
        assert result["covered"] is True
        assert result["requires_prior_auth"] is True
        assert len(result["criteria"]) > 0

    def test_get_payer_policy_unknown_plan(self):
        from tools.payer_tools import get_payer_policy
        result = get_payer_policy.invoke({
            "item_name": "insulin_pump",
            "insurance_plan": "Unknown Plan"
        })
        assert result["covered"] is False

    def test_get_payer_policy_unknown_item(self):
        from tools.payer_tools import get_payer_policy
        result = get_payer_policy.invoke({
            "item_name": "flying_car",
            "insurance_plan": "BlueCross PPO"
        })
        assert result["covered"] == "unknown"

    def test_check_criteria_met_hba1c_passes(self):
        from tools.payer_tools import check_criteria_met
        result = check_criteria_met.invoke({
            "patient_id": "P001",
            "criteria_list": ["HbA1c > 9.0% documented within 6 months"],
            "lab_values": {"HbA1c": 10.2}
        })
        assert result["all_criteria_met"] is True
        assert result["recommendation"] == "APPROVE"

    def test_check_criteria_met_hba1c_fails(self):
        from tools.payer_tools import check_criteria_met
        result = check_criteria_met.invoke({
            "patient_id": "P001",
            "criteria_list": ["HbA1c > 9.0% documented within 6 months"],
            "lab_values": {"HbA1c": 7.2}
        })
        assert result["all_criteria_met"] is False


class TestRiskTools:
    def test_calculate_risk_score_high(self):
        from tools.risk_tools import calculate_risk_score
        result = calculate_risk_score.invoke({
            "patient_id": "P001",
            "diagnoses": ["Type 2 Diabetes", "Hypertension", "CKD Stage 3"],
            "lab_values": {"HbA1c": 10.2, "eGFR": 45}
        })
        assert result["risk_score"] > 5
        assert result["risk_tier"] in ("HIGH", "CRITICAL")
        assert len(result["contributing_factors"]) > 0

    def test_calculate_risk_score_low(self):
        from tools.risk_tools import calculate_risk_score
        result = calculate_risk_score.invoke({
            "patient_id": "TEST",
            "diagnoses": [],
            "lab_values": {}
        })
        assert result["risk_tier"] == "LOW"
        assert result["risk_score"] == 0

    def test_get_care_gaps_female_over_40(self):
        from tools.risk_tools import get_care_gaps
        # Mammogram overdue (no screening history)
        result = get_care_gaps.invoke({
            "patient_id": "P001",
            "age": 67,
            "gender": "F",
            "diagnoses": ["Type 2 Diabetes", "CKD Stage 3"],
            "screening_history": {}
        })
        gap_names = [g["gap"] for g in result["gaps"]]
        assert any("mammogram" in g.lower() for g in gap_names)

    def test_get_care_gaps_ckd_patient(self):
        from tools.risk_tools import get_care_gaps
        result = get_care_gaps.invoke({
            "patient_id": "P001",
            "age": 67,
            "gender": "F",
            "diagnoses": ["CKD Stage 3"],
            "screening_history": {}
        })
        gap_names = [g["gap"] for g in result["gaps"]]
        assert any("nephrology" in g.lower() for g in gap_names)


# ── Agent-level tests (mock the LLM) ─────────────────────────────────────────

class TestPriorAuthAgent:
    """Test the graph structure and decision parsing without live LLM calls."""

    def test_graph_compiles(self):
        from agents.prior_auth_agent import build_prior_auth_graph
        graph = build_prior_auth_graph()
        assert graph is not None

    def test_parse_decision_approve(self):
        """Test that the decision parser correctly extracts APPROVE."""
        from agents.prior_auth_agent import parse_decision
        from langchain_core.messages import AIMessage

        state = {
            "messages": [
                AIMessage(content=(
                    "After reviewing the patient data...\n"
                    "DECISION: APPROVE\n"
                    "CONFIDENCE: 0.92\n"
                    "JUSTIFICATION: Patient meets all coverage criteria for insulin pump."
                ))
            ],
            "patient_id": "P001",
            "request_id": "REQ001",
            "decision": "PENDING",
            "confidence": 0.0,
            "justification": ""
        }

        result = parse_decision(state)
        assert result["decision"] == "APPROVE"
        assert result["confidence"] == 0.92
        assert "insulin pump" in result["justification"]

    def test_parse_decision_escalate_on_bad_output(self):
        """Test that malformed LLM output safely escalates."""
        from agents.prior_auth_agent import parse_decision
        from langchain_core.messages import AIMessage

        state = {
            "messages": [AIMessage(content="I'm not sure what to do here.")],
            "patient_id": "P001",
            "request_id": "REQ001",
            "decision": "PENDING",
            "confidence": 0.0,
            "justification": ""
        }

        result = parse_decision(state)
        assert result["decision"] == "ESCALATE"
        assert result["confidence"] == 0.5


# ── Integration smoke test ────────────────────────────────────────────────────

class TestIntegration:
    """Smoke tests that verify the full pipeline without LLM calls."""

    def test_synthetic_data_loads(self):
        from tools.ehr_tools import _load_patients
        patients = _load_patients()
        assert "P001" in patients
        assert "P002" in patients
        assert "P003" in patients

    def test_payer_policies_load(self):
        from tools.payer_tools import _load_policies
        policies = _load_policies()
        assert "BlueCross PPO" in policies
        assert "Aetna HMO" in policies

    def test_all_tools_importable(self):
        from tools.ehr_tools import EHR_TOOLS
        from tools.payer_tools import PAYER_TOOLS
        from tools.risk_tools import RISK_TOOLS
        assert len(EHR_TOOLS) == 4
        assert len(PAYER_TOOLS) == 2
        assert len(RISK_TOOLS) == 2
