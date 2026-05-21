"""
Tests for Escalation Handling.

Tests for EscalationHandler, EscalationRequest, and related classes.
"""

from datetime import datetime

import pytest

from sage.human_oversight.approval_flow import RiskLevel
from sage.human_oversight.escalation import (
    EscalationHandler,
    EscalationLevel,
    EscalationPolicy,
    EscalationRequest,
)

# ============================================================================
# EscalationPolicy Tests
# ============================================================================


class TestEscalationPolicy:
    """Tests for EscalationPolicy dataclass."""

    def test_create_policy(self):
        """Test creating an escalation policy."""
        policy = EscalationPolicy(
            trigger_level=EscalationLevel.TIER_1,
            risk_levels=[RiskLevel.HIGH, RiskLevel.CRITICAL],
            timeout_seconds=300.0,
            notify_channels=["email", "slack"],
        )

        assert policy.trigger_level == EscalationLevel.TIER_1
        assert RiskLevel.HIGH in policy.risk_levels
        assert policy.timeout_seconds == 300.0
        assert "email" in policy.notify_channels


# ============================================================================
# EscalationRequest Tests
# ============================================================================


class TestEscalationRequest:
    """Tests for EscalationRequest dataclass."""

    def test_create_request(self):
        """Test creating an escalation request."""
        request = EscalationRequest(
            escalation_id="esc-001",
            original_request_id="req-001",
            level=EscalationLevel.TIER_1,
            reason="Pending too long",
            created_at=datetime.now(),
        )

        assert request.escalation_id == "esc-001"
        assert request.level == EscalationLevel.TIER_1
        assert request.resolved is False

    def test_to_dict(self):
        """Test to_dict method."""
        request = EscalationRequest(
            escalation_id="esc-001",
            original_request_id="req-001",
            level=EscalationLevel.TIER_2,
            reason="Test reason",
            created_at=datetime.now(),
        )

        d = request.to_dict()
        assert d["escalation_id"] == "esc-001"
        assert d["level"] == "tier_2"
        assert d["resolved"] is False


# ============================================================================
# EscalationHandler Tests
# ============================================================================


class TestEscalationHandler:
    """Tests for EscalationHandler class."""

    @pytest.fixture
    def handler(self):
        """Create an EscalationHandler instance."""
        return EscalationHandler()

    def test_initialization(self, handler):
        """Test handler initialization."""
        assert len(handler._policies) == 0
        assert len(handler._escalations) == 0

    def test_add_policy(self, handler):
        """Test adding a policy."""
        policy = EscalationPolicy(
            trigger_level=EscalationLevel.TIER_1,
            risk_levels=[RiskLevel.HIGH],
        )

        handler.add_policy(policy)
        assert len(handler._policies) == 1

    def test_get_policy(self, handler):
        """Test getting applicable policy."""
        policy = EscalationPolicy(
            trigger_level=EscalationLevel.TIER_1,
            risk_levels=[RiskLevel.HIGH],
        )
        handler.add_policy(policy)

        found = handler.get_policy(EscalationLevel.TIER_1, RiskLevel.HIGH)
        assert found == policy

        not_found = handler.get_policy(EscalationLevel.TIER_1, RiskLevel.LOW)
        assert not_found is None

    def test_escalate(self, handler):
        """Test creating an escalation."""
        escalation = handler.escalate(
            request_id="req-001",
            level=EscalationLevel.TIER_1,
            reason="Pending too long",
        )

        assert escalation.original_request_id == "req-001"
        assert escalation.level == EscalationLevel.TIER_1
        assert escalation.escalation_id in handler._escalations

    def test_resolve(self, handler):
        """Test resolving an escalation."""
        escalation = handler.escalate(
            request_id="req-001",
            reason="Test",
        )

        resolved = handler.resolve(
            escalation_id=escalation.escalation_id,
            resolver="admin",
        )

        assert resolved.resolved is True
        assert resolved.resolver == "admin"
        assert resolved.resolved_at is not None

    def test_resolve_nonexistent(self, handler):
        """Test resolving nonexistent escalation raises error."""
        with pytest.raises(KeyError):
            handler.resolve(escalation_id="nonexistent")

    def test_escalate_further(self, handler):
        """Test escalating to next tier."""
        escalation = handler.escalate(
            request_id="req-001",
            level=EscalationLevel.TIER_1,
            reason="Initial escalation",
        )

        next_escalation = handler.escalate_further(
            escalation_id=escalation.escalation_id,
            reason="Still unresolved",
        )

        assert next_escalation is not None
        assert next_escalation.level == EscalationLevel.TIER_2
        assert escalation.resolved is True

    def test_escalate_further_at_max_level(self, handler):
        """Test escalating further when at max level."""
        escalation = handler.escalate(
            request_id="req-001",
            level=EscalationLevel.EMERGENCY,
            reason="Critical",
        )

        result = handler.escalate_further(
            escalation_id=escalation.escalation_id,
        )

        assert result is None  # Cannot escalate further

    def test_get_active_escalations(self, handler):
        """Test getting active escalations."""
        esc1 = handler.escalate(request_id="req-001", reason="Test")
        esc2 = handler.escalate(request_id="req-002", reason="Test")
        handler.resolve(escalation_id=esc1.escalation_id)

        active = handler.get_active_escalations()
        assert len(active) == 1
        assert active[0].escalation_id == esc2.escalation_id

    def test_get_escalations_for_request(self, handler):
        """Test getting escalations for a request."""
        handler.escalate(request_id="req-001", reason="First")
        handler.escalate(request_id="req-001", reason="Second")
        handler.escalate(request_id="req-002", reason="Other")

        escalations = handler.get_escalations_for_request("req-001")
        assert len(escalations) == 2

    def test_get_escalations_by_level(self, handler):
        """Test getting escalations by level."""
        handler.escalate(
            request_id="req-001", level=EscalationLevel.TIER_1, reason="T1"
        )
        handler.escalate(
            request_id="req-002", level=EscalationLevel.TIER_1, reason="T1"
        )
        handler.escalate(
            request_id="req-003", level=EscalationLevel.TIER_2, reason="T2"
        )

        tier1 = handler.get_escalations_by_level(EscalationLevel.TIER_1)
        assert len(tier1) == 2

    def test_get_statistics(self, handler):
        """Test getting escalation statistics."""
        esc1 = handler.escalate(request_id="req-001", reason="Test")
        handler.escalate(request_id="req-002", reason="Test")
        handler.resolve(escalation_id=esc1.escalation_id)

        stats = handler.get_statistics()
        assert stats["total_escalations"] == 2
        assert stats["active_escalations"] == 1
        assert stats["resolved_escalations"] == 1

    def test_get_audit_log(self, handler):
        """Test audit log recording."""
        esc = handler.escalate(request_id="req-001", reason="Test")
        handler.resolve(escalation_id=esc.escalation_id)

        log = handler.get_audit_log()
        assert len(log) == 2

    def test_clear(self, handler):
        """Test clearing handler."""
        handler.escalate(request_id="req-001", reason="Test")
        handler.clear()

        assert len(handler._escalations) == 0
        assert len(handler._audit_log) == 0

    def test_on_escalate_callback(self):
        """Test escalation callback is called."""
        callback_calls = []

        def on_escalate(esc):
            callback_calls.append(esc)

        handler = EscalationHandler(on_escalate=on_escalate)
        handler.escalate(request_id="req-001", reason="Test")

        assert len(callback_calls) == 1


# ============================================================================
# EscalationLevel Tests
# ============================================================================


class TestEscalationLevel:
    """Tests for EscalationLevel enum."""

    def test_all_levels_exist(self):
        """Test all levels exist."""
        assert EscalationLevel.TIER_1.value == "tier_1"
        assert EscalationLevel.TIER_2.value == "tier_2"
        assert EscalationLevel.TIER_3.value == "tier_3"
        assert EscalationLevel.EMERGENCY.value == "emergency"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
