"""
Tests for Human Approval Flow.

Comprehensive tests for ApprovalHandler, ApprovalRequest, and related classes.
"""

import pytest
import asyncio
from datetime import datetime, timedelta
from unittest.mock import Mock, AsyncMock

from sage.human_oversight.approval_flow import (
    ApprovalHandler,
    ApprovalRequest,
    ApprovalResult,
    ApprovalStatus,
    RiskLevel,
    ApprovalTimeoutError,
    BatchApprovalHandler,
)


# ============================================================================
# ApprovalRequest Tests
# ============================================================================

class TestApprovalRequest:
    """Tests for ApprovalRequest dataclass."""

    def test_create_request(self):
        """Test creating an approval request."""
        now = datetime.now()
        request = ApprovalRequest(
            request_id="req-001",
            action="deploy_model",
            context={"model": "gpt-4"},
            risk_level=RiskLevel.HIGH,
            created_at=now,
            timeout=300.0,
            expires_at=now + timedelta(seconds=300),
        )

        assert request.request_id == "req-001"
        assert request.action == "deploy_model"
        assert request.context == {"model": "gpt-4"}
        assert request.risk_level == RiskLevel.HIGH
        assert request.status == ApprovalStatus.PENDING
        assert request.timeout == 300.0

    def test_is_pending(self):
        """Test is_pending method."""
        request = ApprovalRequest(
            request_id="req-001",
            action="test",
            context={},
            risk_level=RiskLevel.LOW,
            created_at=datetime.now(),
            timeout=None,
            expires_at=None,
        )

        assert request.is_pending() is True

        request.status = ApprovalStatus.APPROVED
        assert request.is_pending() is False

    def test_is_expired(self):
        """Test is_expired method."""
        # No expiration
        request = ApprovalRequest(
            request_id="req-001",
            action="test",
            context={},
            risk_level=RiskLevel.LOW,
            created_at=datetime.now(),
            timeout=None,
            expires_at=None,
        )
        assert request.is_expired() is False

        # Future expiration
        request.expires_at = datetime.now() + timedelta(hours=1)
        assert request.is_expired() is False

        # Past expiration
        request.expires_at = datetime.now() - timedelta(seconds=1)
        assert request.is_expired() is True

    def test_time_remaining(self):
        """Test time_remaining method."""
        # No expiration
        request = ApprovalRequest(
            request_id="req-001",
            action="test",
            context={},
            risk_level=RiskLevel.LOW,
            created_at=datetime.now(),
            timeout=None,
            expires_at=None,
        )
        assert request.time_remaining() is None

        # With expiration
        request.expires_at = datetime.now() + timedelta(seconds=100)
        remaining = request.time_remaining()
        assert remaining is not None
        assert 99 <= remaining <= 100

    def test_to_dict(self):
        """Test to_dict method."""
        now = datetime.now()
        request = ApprovalRequest(
            request_id="req-001",
            action="test",
            context={"key": "value"},
            risk_level=RiskLevel.MEDIUM,
            created_at=now,
            timeout=300.0,
            expires_at=now + timedelta(seconds=300),
            requester="agent-1",
        )

        d = request.to_dict()
        assert d["request_id"] == "req-001"
        assert d["action"] == "test"
        assert d["risk_level"] == "medium"
        assert d["requester"] == "agent-1"


# ============================================================================
# ApprovalResult Tests
# ============================================================================

class TestApprovalResult:
    """Tests for ApprovalResult dataclass."""

    def test_create_approved_result(self):
        """Test creating an approved result."""
        result = ApprovalResult(
            request_id="req-001",
            approved=True,
            status=ApprovalStatus.APPROVED,
            approver="admin",
            reason="Looks good",
        )

        assert result.request_id == "req-001"
        assert result.approved is True
        assert result.status == ApprovalStatus.APPROVED
        assert result.approver == "admin"

    def test_create_rejected_result(self):
        """Test creating a rejected result."""
        result = ApprovalResult(
            request_id="req-001",
            approved=False,
            status=ApprovalStatus.REJECTED,
            reason="Too risky",
        )

        assert result.approved is False
        assert result.status == ApprovalStatus.REJECTED


# ============================================================================
# ApprovalHandler Tests
# ============================================================================

class TestApprovalHandler:
    """Tests for ApprovalHandler class."""

    @pytest.fixture
    def handler(self):
        """Create an ApprovalHandler instance."""
        return ApprovalHandler(
            default_timeout=5.0,
            auto_reject_on_timeout=True,
        )

    def test_initialization(self, handler):
        """Test handler initialization."""
        assert handler.default_timeout == 5.0
        assert handler.auto_reject_on_timeout is True
        assert len(handler._requests) == 0

    def test_create_request(self, handler):
        """Test creating a request."""
        request = handler.create_request(
            action="deploy",
            context={"env": "prod"},
            risk_level=RiskLevel.HIGH,
            requester="agent-1",
        )

        assert request.action == "deploy"
        assert request.context == {"env": "prod"}
        assert request.risk_level == RiskLevel.HIGH
        assert request.status == ApprovalStatus.PENDING
        assert request.request_id in handler._requests

    def test_create_request_with_custom_timeout(self, handler):
        """Test creating a request with custom timeout."""
        request = handler.create_request(
            action="test",
            timeout=60.0,
        )

        assert request.timeout == 60.0

    def test_create_request_with_risk_timeout(self):
        """Test risk-level specific timeouts."""
        handler = ApprovalHandler(
            default_timeout=300.0,
            risk_timeouts={
                RiskLevel.CRITICAL: 60.0,
                RiskLevel.HIGH: 120.0,
            }
        )

        critical_req = handler.create_request(
            action="critical_action",
            risk_level=RiskLevel.CRITICAL,
        )
        assert critical_req.timeout == 60.0

        high_req = handler.create_request(
            action="high_action",
            risk_level=RiskLevel.HIGH,
        )
        assert high_req.timeout == 120.0

        medium_req = handler.create_request(
            action="medium_action",
            risk_level=RiskLevel.MEDIUM,
        )
        assert medium_req.timeout == 300.0  # Default

    def test_approve_request(self, handler):
        """Test approving a request."""
        request = handler.create_request(
            action="test",
            risk_level=RiskLevel.LOW,
        )

        result = handler.approve(
            request_id=request.request_id,
            approver="admin",
            reason="Approved after review",
        )

        assert result.approved is True
        assert result.status == ApprovalStatus.APPROVED
        assert result.approver == "admin"

        # Check request was updated
        updated_request = handler.get_request(request.request_id)
        assert updated_request.status == ApprovalStatus.APPROVED

    def test_reject_request(self, handler):
        """Test rejecting a request."""
        request = handler.create_request(
            action="test",
            risk_level=RiskLevel.HIGH,
        )

        result = handler.reject(
            request_id=request.request_id,
            approver="admin",
            reason="Too risky",
        )

        assert result.approved is False
        assert result.status == ApprovalStatus.REJECTED

    def test_cancel_request(self, handler):
        """Test cancelling a request."""
        request = handler.create_request(
            action="test",
        )

        result = handler.cancel(request.request_id)

        assert result.approved is False
        assert result.status == ApprovalStatus.CANCELLED

    def test_approve_nonexistent_request(self, handler):
        """Test approving a nonexistent request raises error."""
        with pytest.raises(KeyError):
            handler.approve(request_id="nonexistent")

    def test_approve_already_decided_request(self, handler):
        """Test approving an already decided request raises error."""
        request = handler.create_request(action="test")
        handler.approve(request_id=request.request_id)

        with pytest.raises(ValueError, match="not pending"):
            handler.approve(request_id=request.request_id)

    def test_get_pending_requests(self, handler):
        """Test getting pending requests."""
        req1 = handler.create_request(action="test1")
        req2 = handler.create_request(action="test2")
        handler.approve(request_id=req1.request_id)

        pending = handler.get_pending_requests()
        assert len(pending) == 1
        assert pending[0].request_id == req2.request_id

    def test_get_requests_by_action(self, handler):
        """Test filtering requests by action."""
        handler.create_request(action="deploy")
        handler.create_request(action="deploy")
        handler.create_request(action="rollback")

        deploy_requests = handler.get_requests_by_action("deploy")
        assert len(deploy_requests) == 2

    def test_get_requests_by_risk_level(self, handler):
        """Test filtering requests by risk level."""
        handler.create_request(action="test1", risk_level=RiskLevel.HIGH)
        handler.create_request(action="test2", risk_level=RiskLevel.HIGH)
        handler.create_request(action="test3", risk_level=RiskLevel.LOW)

        high_risk = handler.get_requests_by_risk_level(RiskLevel.HIGH)
        assert len(high_risk) == 2

    @pytest.mark.asyncio
    async def test_wait_for_approval_immediate(self, handler):
        """Test waiting for approval when already decided."""
        request = handler.create_request(action="test")
        handler.approve(request_id=request.request_id)

        result = await handler.wait_for_approval(request.request_id)
        assert result.approved is True

    @pytest.mark.asyncio
    async def test_wait_for_approval_timeout(self):
        """Test waiting for approval with timeout."""
        handler = ApprovalHandler(
            default_timeout=0.1,
            auto_reject_on_timeout=True,
        )

        request = handler.create_request(action="test")
        result = await handler.wait_for_approval(request.request_id)

        assert result.approved is False
        assert result.status == ApprovalStatus.TIMEOUT

    @pytest.mark.asyncio
    async def test_wait_for_approval_timeout_raises(self):
        """Test timeout raises exception when auto_reject is False."""
        handler = ApprovalHandler(
            default_timeout=0.1,
            auto_reject_on_timeout=False,
        )

        request = handler.create_request(action="test")

        with pytest.raises(ApprovalTimeoutError) as exc_info:
            await handler.wait_for_approval(request.request_id)

        assert exc_info.value.request_id == request.request_id

    @pytest.mark.asyncio
    async def test_wait_for_approval_concurrent(self, handler):
        """Test concurrent approval wait."""
        request = handler.create_request(action="test")

        async def approve_after_delay():
            await asyncio.sleep(0.1)
            handler.approve(request_id=request.request_id, approver="admin")

        # Start approval in background
        asyncio.create_task(approve_after_delay())

        # Wait for approval
        result = await handler.wait_for_approval(request.request_id)
        assert result.approved is True

    def test_get_audit_log(self, handler):
        """Test audit log recording."""
        request = handler.create_request(action="test")
        handler.approve(request_id=request.request_id)

        audit_log = handler.get_audit_log()
        assert len(audit_log) == 2  # created + approved

    def test_export_audit_log_json(self, handler):
        """Test exporting audit log as JSON."""
        handler.create_request(action="test")
        export = handler.export_audit_log(format="json")
        assert "test" in export

    def test_export_audit_log_csv(self, handler):
        """Test exporting audit log as CSV."""
        handler.create_request(action="test")
        export = handler.export_audit_log(format="csv")
        assert "timestamp,event" in export

    def test_export_audit_log_markdown(self, handler):
        """Test exporting audit log as Markdown."""
        handler.create_request(action="test")
        export = handler.export_audit_log(format="markdown")
        assert "# Approval Audit Log" in export

    def test_get_statistics(self, handler):
        """Test getting approval statistics."""
        handler.create_request(action="test1", risk_level=RiskLevel.HIGH)
        req2 = handler.create_request(action="test2", risk_level=RiskLevel.LOW)
        handler.approve(request_id=req2.request_id)

        stats = handler.get_statistics()
        assert stats["total_requests"] == 2
        assert stats["pending_count"] == 1
        assert "by_risk_level" in stats

    def test_clear(self, handler):
        """Test clearing all requests."""
        handler.create_request(action="test")
        handler.clear()

        assert len(handler._requests) == 0
        assert len(handler._audit_log) == 0


# ============================================================================
# BatchApprovalHandler Tests
# ============================================================================

class TestBatchApprovalHandler:
    """Tests for BatchApprovalHandler class."""

    @pytest.fixture
    def handler(self):
        """Create an ApprovalHandler instance."""
        return ApprovalHandler(default_timeout=300.0)

    @pytest.fixture
    def batch_handler(self, handler):
        """Create a BatchApprovalHandler instance."""
        return BatchApprovalHandler(handler)

    def test_approve_all(self, handler, batch_handler):
        """Test batch approval."""
        req1 = handler.create_request(action="test1")
        req2 = handler.create_request(action="test2")
        req3 = handler.create_request(action="test3")

        results = batch_handler.approve_all(
            request_ids=[req1.request_id, req2.request_id, req3.request_id],
            approver="admin",
            reason="Batch approved",
        )

        assert len(results) == 3
        assert all(r.approved for r in results)

    def test_reject_all(self, handler, batch_handler):
        """Test batch rejection."""
        req1 = handler.create_request(action="test1")
        req2 = handler.create_request(action="test2")

        results = batch_handler.reject_all(
            request_ids=[req1.request_id, req2.request_id],
            approver="admin",
            reason="Batch rejected",
        )

        assert len(results) == 2
        assert all(not r.approved for r in results)

    def test_approve_by_action(self, handler, batch_handler):
        """Test approving all requests for an action."""
        handler.create_request(action="deploy")
        handler.create_request(action="deploy")
        handler.create_request(action="rollback")

        results = batch_handler.approve_by_action(
            action="deploy",
            approver="admin",
        )

        assert len(results) == 2

    def test_approve_by_risk_level(self, handler, batch_handler):
        """Test approving all requests with a risk level."""
        handler.create_request(action="test1", risk_level=RiskLevel.LOW)
        handler.create_request(action="test2", risk_level=RiskLevel.LOW)
        handler.create_request(action="test3", risk_level=RiskLevel.HIGH)

        results = batch_handler.approve_by_risk_level(
            risk_level=RiskLevel.LOW,
            approver="admin",
        )

        assert len(results) == 2

    def test_approve_all_handles_errors(self, handler, batch_handler):
        """Test batch approval handles errors gracefully."""
        req = handler.create_request(action="test")
        handler.approve(request_id=req.request_id)  # Already approved

        # This should not raise
        results = batch_handler.approve_all(
            request_ids=[req.request_id, "nonexistent"],
            approver="admin",
        )

        assert len(results) == 0  # Both failed


# ============================================================================
# RiskLevel Tests
# ============================================================================

class TestRiskLevel:
    """Tests for RiskLevel enum."""

    def test_all_levels_exist(self):
        """Test all risk levels exist."""
        assert RiskLevel.LOW.value == "low"
        assert RiskLevel.MEDIUM.value == "medium"
        assert RiskLevel.HIGH.value == "high"
        assert RiskLevel.CRITICAL.value == "critical"


# ============================================================================
# ApprovalStatus Tests
# ============================================================================

class TestApprovalStatus:
    """Tests for ApprovalStatus enum."""

    def test_all_statuses_exist(self):
        """Test all statuses exist."""
        assert ApprovalStatus.PENDING.value == "pending"
        assert ApprovalStatus.APPROVED.value == "approved"
        assert ApprovalStatus.REJECTED.value == "rejected"
        assert ApprovalStatus.TIMEOUT.value == "timeout"
        assert ApprovalStatus.CANCELLED.value == "cancelled"
        assert ApprovalStatus.ESCALATED.value == "escalated"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
