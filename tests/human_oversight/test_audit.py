"""
Tests for Audit Logging.

Tests for AuditLogger, AuditEntry, and related classes.
"""

import pytest
import json
from datetime import datetime, timedelta
from pathlib import Path
import tempfile

from sage.human_oversight.audit import (
    AuditLogger,
    AuditEntry,
    AuditEventType,
)

# ============================================================================
# AuditEntry Tests
# ============================================================================


class TestAuditEntry:
    """Tests for AuditEntry dataclass."""

    def test_create_entry(self):
        """Test creating an audit entry."""
        entry = AuditEntry(
            entry_id="audit-001",
            timestamp=datetime.now(),
            event_type=AuditEventType.APPROVAL_REQUESTED,
            actor="agent-1",
            action="deploy",
        )

        assert entry.entry_id == "audit-001"
        assert entry.event_type == AuditEventType.APPROVAL_REQUESTED
        assert entry.actor == "agent-1"

    def test_to_dict(self):
        """Test to_dict method."""
        entry = AuditEntry(
            entry_id="audit-001",
            timestamp=datetime.now(),
            event_type=AuditEventType.HUMAN_INTERVENTION,
            actor="admin",
            action="stop_agent",
            outcome="success",
        )

        d = entry.to_dict()
        assert d["entry_id"] == "audit-001"
        assert d["event_type"] == "human_intervention"
        assert d["outcome"] == "success"


# ============================================================================
# AuditLogger Tests
# ============================================================================


class TestAuditLogger:
    """Tests for AuditLogger class."""

    @pytest.fixture
    def logger(self):
        """Create an AuditLogger instance."""
        return AuditLogger(session_id="test-session")

    def test_initialization(self, logger):
        """Test logger initialization."""
        assert logger.session_id == "test-session"
        assert logger.retention_days == 90
        assert len(logger._entries) == 0

    def test_log_event(self, logger):
        """Test logging an event."""
        entry = logger.log_event(
            event_type=AuditEventType.APPROVAL_REQUESTED,
            actor="agent-1",
            action="deploy",
        )

        assert entry.entry_id is not None
        assert entry.event_type == AuditEventType.APPROVAL_REQUESTED
        assert entry.session_id == "test-session"
        assert len(logger._entries) == 1

    def test_log_approval_requested(self, logger):
        """Test logging approval requested event."""
        entry = logger.log_approval_requested(
            request_id="req-001",
            action="deploy",
            actor="agent-1",
            risk_level="high",
        )

        assert entry.event_type == AuditEventType.APPROVAL_REQUESTED
        assert entry.request_id == "req-001"

    def test_log_approval_decision(self, logger):
        """Test logging approval decision."""
        approved_entry = logger.log_approval_decision(
            request_id="req-001",
            approved=True,
            approver="admin",
            reason="Looks good",
        )

        assert approved_entry.event_type == AuditEventType.APPROVAL_GRANTED

        rejected_entry = logger.log_approval_decision(
            request_id="req-002",
            approved=False,
            approver="admin",
            reason="Too risky",
        )

        assert rejected_entry.event_type == AuditEventType.APPROVAL_DENIED

    def test_log_escalation(self, logger):
        """Test logging escalation event."""
        entry = logger.log_escalation(
            escalation_id="esc-001",
            request_id="req-001",
            level="tier_2",
            reason="Unresolved",
        )

        assert entry.event_type == AuditEventType.ESCALATION_CREATED

    def test_log_intervention(self, logger):
        """Test logging intervention event."""
        entry = logger.log_intervention(
            actor="admin",
            action="stop_agent",
            reason="Erratic behavior",
        )

        assert entry.event_type == AuditEventType.HUMAN_INTERVENTION

    def test_log_policy_violation(self, logger):
        """Test logging policy violation."""
        entry = logger.log_policy_violation(
            policy_name="rate_limit",
            violation_details="Exceeded 100 requests/minute",
            actor="agent-1",
        )

        assert entry.event_type == AuditEventType.POLICY_VIOLATION

    def test_get_entries(self, logger):
        """Test getting all entries."""
        logger.log_event(event_type=AuditEventType.SESSION_START)
        logger.log_event(event_type=AuditEventType.SESSION_END)

        entries = logger.get_entries()
        assert len(entries) == 2

    def test_get_events_by_type(self, logger):
        """Test filtering by event type."""
        logger.log_event(event_type=AuditEventType.APPROVAL_REQUESTED)
        logger.log_event(event_type=AuditEventType.APPROVAL_GRANTED)
        logger.log_event(event_type=AuditEventType.APPROVAL_REQUESTED)

        requests = logger.get_events_by_type(AuditEventType.APPROVAL_REQUESTED)
        assert len(requests) == 2

    def test_get_events_by_actor(self, logger):
        """Test filtering by actor."""
        logger.log_event(event_type=AuditEventType.SESSION_START, actor="agent-1")
        logger.log_event(event_type=AuditEventType.SESSION_START, actor="agent-2")
        logger.log_event(event_type=AuditEventType.SESSION_START, actor="agent-1")

        agent1_events = logger.get_events_by_actor("agent-1")
        assert len(agent1_events) == 2

    def test_get_events_by_request(self, logger):
        """Test filtering by request ID."""
        logger.log_event(
            event_type=AuditEventType.APPROVAL_REQUESTED, request_id="req-001"
        )
        logger.log_event(
            event_type=AuditEventType.APPROVAL_GRANTED, request_id="req-001"
        )
        logger.log_event(
            event_type=AuditEventType.APPROVAL_REQUESTED, request_id="req-002"
        )

        req1_events = logger.get_events_by_request("req-001")
        assert len(req1_events) == 2

    def test_get_events_in_timerange(self, logger):
        """Test filtering by time range."""
        now = datetime.now()

        # Log event
        entry = logger.log_event(event_type=AuditEventType.SESSION_START)

        # Query with range that includes now
        start = now - timedelta(hours=1)
        end = now + timedelta(hours=1)

        events = logger.get_events_in_timerange(start, end)
        assert len(events) == 1

        # Query with range in the past
        past_start = now - timedelta(days=2)
        past_end = now - timedelta(days=1)

        events = logger.get_events_in_timerange(past_start, past_end)
        assert len(events) == 0

    def test_get_recent_events(self, logger):
        """Test getting recent events."""
        logger.log_event(event_type=AuditEventType.SESSION_START)

        recent = logger.get_recent_events(hours=24)
        assert len(recent) == 1

    def test_apply_retention_policy(self):
        """Test retention policy application."""
        logger = AuditLogger(retention_days=1)

        # Add entry
        entry = logger.log_event(event_type=AuditEventType.SESSION_START)

        # Manually backdate the entry
        entry.timestamp = datetime.now() - timedelta(days=2)

        removed = logger.apply_retention_policy()
        assert removed == 1
        assert len(logger._entries) == 0

    def test_export_json(self, logger):
        """Test JSON export."""
        logger.log_event(event_type=AuditEventType.APPROVAL_REQUESTED, actor="agent-1")

        export = logger.export(format="json")
        parsed = json.loads(export)

        assert len(parsed) == 1
        assert parsed[0]["event_type"] == "approval_requested"

    def test_export_csv(self, logger):
        """Test CSV export."""
        logger.log_event(event_type=AuditEventType.APPROVAL_REQUESTED, actor="agent-1")

        export = logger.export(format="csv")
        assert "entry_id,timestamp,event_type" in export
        assert "approval_requested" in export

    def test_export_markdown(self, logger):
        """Test Markdown export."""
        logger.log_event(
            event_type=AuditEventType.HUMAN_INTERVENTION, actor="admin", action="stop"
        )

        export = logger.export(format="markdown")
        assert "# Human Oversight Audit Log" in export
        assert "human_intervention" in export

    def test_export_invalid_format(self, logger):
        """Test export with invalid format."""
        with pytest.raises(ValueError, match="Unsupported format"):
            logger.export(format="xml")

    def test_save_to_file(self, logger):
        """Test saving to file."""
        logger.log_event(event_type=AuditEventType.SESSION_START)

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            path = Path(f.name)

        try:
            logger.save_to_file(path, format="json")
            assert path.exists()

            content = json.loads(path.read_text())
            assert len(content) == 1
        finally:
            path.unlink()

    def test_get_statistics(self, logger):
        """Test getting statistics."""
        logger.log_event(
            event_type=AuditEventType.APPROVAL_REQUESTED,
            actor="agent-1",
            outcome="pending",
        )
        logger.log_event(
            event_type=AuditEventType.APPROVAL_GRANTED,
            actor="admin",
            outcome="approved",
        )
        logger.log_event(
            event_type=AuditEventType.APPROVAL_DENIED, actor="admin", outcome="denied"
        )

        stats = logger.get_statistics()

        assert stats["total_entries"] == 3
        assert "by_event_type" in stats
        assert stats["by_event_type"]["approval_requested"] == 1
        assert "by_actor" in stats
        assert stats["by_actor"]["admin"] == 2

    def test_clear(self, logger):
        """Test clearing entries."""
        logger.log_event(event_type=AuditEventType.SESSION_START)
        logger.clear()

        assert len(logger._entries) == 0
        assert logger._entry_counter == 0

    def test_on_log_callback(self):
        """Test callback when entry is logged."""
        callback_calls = []

        def on_log(entry):
            callback_calls.append(entry)

        logger = AuditLogger(on_log=on_log)
        logger.log_event(event_type=AuditEventType.SESSION_START)

        assert len(callback_calls) == 1


# ============================================================================
# AuditEventType Tests
# ============================================================================


class TestAuditEventType:
    """Tests for AuditEventType enum."""

    def test_approval_events_exist(self):
        """Test approval events exist."""
        assert AuditEventType.APPROVAL_REQUESTED.value == "approval_requested"
        assert AuditEventType.APPROVAL_GRANTED.value == "approval_granted"
        assert AuditEventType.APPROVAL_DENIED.value == "approval_denied"
        assert AuditEventType.APPROVAL_TIMEOUT.value == "approval_timeout"
        assert AuditEventType.APPROVAL_CANCELLED.value == "approval_cancelled"

    def test_escalation_events_exist(self):
        """Test escalation events exist."""
        assert AuditEventType.ESCALATION_CREATED.value == "escalation_created"
        assert AuditEventType.ESCALATION_RESOLVED.value == "escalation_resolved"
        assert AuditEventType.ESCALATION_UPGRADED.value == "escalation_upgraded"

    def test_intervention_events_exist(self):
        """Test intervention events exist."""
        assert AuditEventType.HUMAN_INTERVENTION.value == "human_intervention"
        assert AuditEventType.AGENT_OVERRIDE.value == "agent_override"
        assert AuditEventType.EMERGENCY_STOP.value == "emergency_stop"

    def test_policy_events_exist(self):
        """Test policy events exist."""
        assert AuditEventType.POLICY_VIOLATION.value == "policy_violation"
        assert AuditEventType.GUARDRAIL_TRIGGERED.value == "guardrail_triggered"
        assert AuditEventType.CONSTRAINT_VIOLATION.value == "constraint_violation"

    def test_session_events_exist(self):
        """Test session events exist."""
        assert AuditEventType.SESSION_START.value == "session_start"
        assert AuditEventType.SESSION_END.value == "session_end"
        assert AuditEventType.CONTEXT_SWITCH.value == "context_switch"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
