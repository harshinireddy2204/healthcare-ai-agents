"""
tests/test_triage.py

Tests for the triage supervisor and API endpoints.
"""
import json
import pytest
from unittest.mock import patch, MagicMock


class TestTriageSupervisor:

    def test_needs_escalation_with_escalate_decision(self):
        from agents.triage_supervisor import needs_escalation
        auth_results = [
            {"request_id": "REQ001", "decision": "ESCALATE", "confidence": 0.4}
        ]
        assert needs_escalation(auth_results) is True

    def test_needs_escalation_with_low_confidence(self):
        from agents.triage_supervisor import needs_escalation
        auth_results = [
            {"request_id": "REQ001", "decision": "APPROVE", "confidence": 0.60}
        ]
        assert needs_escalation(auth_results) is True

    def test_no_escalation_for_high_confidence_approve(self):
        from agents.triage_supervisor import needs_escalation
        auth_results = [
            {"request_id": "REQ001", "decision": "APPROVE", "confidence": 0.95}
        ]
        assert needs_escalation(auth_results) is False

    def test_no_escalation_for_empty_results(self):
        from agents.triage_supervisor import needs_escalation
        assert needs_escalation([]) is False


class TestFastAPIEndpoints:
    """Integration tests for FastAPI endpoints using TestClient."""

    @pytest.fixture
    def client(self):
        from fastapi.testclient import TestClient
        from api.main import app, init_db
        init_db()
        return TestClient(app)

    def test_health_check(self, client):
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "version" in data

    def test_pending_reviews_returns_list(self, client):
        response = client.get("/pending-reviews")
        assert response.status_code == 200
        data = response.json()
        assert "reviews" in data
        assert "pending_count" in data

    def test_audit_log_returns_list(self, client):
        response = client.get("/audit-log")
        assert response.status_code == 200
        data = response.json()
        assert "entries" in data
        assert "total" in data

    def test_resolve_review_not_found(self, client):
        response = client.post(
            "/resolve-review/99999",
            json={"resolution": "APPROVED", "resolved_by": "Dr. Test"}
        )
        assert response.status_code == 404

    def test_process_patient_triggers_background(self, client):
        """POST /process-patient should return 200 immediately."""
        response = client.post(
            "/process-patient",
            json={"patient_id": "P001", "mode": "care_gap_only"}
        )
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "PROCESSING"
        assert data["patient_id"] == "P001"
