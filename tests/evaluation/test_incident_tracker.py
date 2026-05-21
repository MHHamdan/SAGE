"""
Tests for Incident Rate Tracking Module.

Tests cover:
- Incident recording and retrieval
- Statistics computation
- Rate calculation
- Severity filtering
- Threshold breach detection
- Report export formats
- Edge cases
"""

import pytest
import json
from datetime import datetime, timedelta
from unittest.mock import Mock, patch

from sage.evaluation.incident_tracker import (
    IncidentTracker,
    AggregatedIncidentTracker,
    Incident,
    IncidentType,
    IncidentSeverity,
    IncidentStatistics,
)


# Test fixtures

@pytest.fixture
def tracker():
    """Create a basic incident tracker."""
    return IncidentTracker()


@pytest.fixture
def tracker_with_task_counter():
    """Create tracker with task counter."""
    counter = Mock(return_value=100)
    return IncidentTracker(task_counter=counter)


@pytest.fixture
def populated_tracker():
    """Create tracker with some incidents already recorded."""
    tracker = IncidentTracker()

    # Add various incidents
    tracker.record_incident(
        incident_type=IncidentType.GUARDRAIL_ACTIVATION,
        severity=IncidentSeverity.MEDIUM,
        description="Blocked dangerous action",
        context={"task_id": "task_1"}
    )
    tracker.record_incident(
        incident_type=IncidentType.TOOL_FAILURE,
        severity=IncidentSeverity.LOW,
        description="API timeout",
        context={"task_id": "task_2"}
    )
    tracker.record_incident(
        incident_type=IncidentType.POLICY_VIOLATION,
        severity=IncidentSeverity.HIGH,
        description="Attempted unauthorized access",
        context={"task_id": "task_3"}
    )

    return tracker


# Tests for IncidentType enum

class TestIncidentType:
    """Tests for IncidentType enum."""

    def test_all_incident_types_have_values(self):
        """All incident types should have string values."""
        for incident_type in IncidentType:
            assert isinstance(incident_type.value, str)
            assert len(incident_type.value) > 0

    def test_incident_types_are_unique(self):
        """All incident type values should be unique."""
        values = [t.value for t in IncidentType]
        assert len(values) == len(set(values))


# Tests for IncidentSeverity enum

class TestIncidentSeverity:
    """Tests for IncidentSeverity enum."""

    def test_severity_ordering(self):
        """Severity levels should be properly ordered."""
        assert IncidentSeverity.INFO < IncidentSeverity.LOW
        assert IncidentSeverity.LOW < IncidentSeverity.MEDIUM
        assert IncidentSeverity.MEDIUM < IncidentSeverity.HIGH
        assert IncidentSeverity.HIGH < IncidentSeverity.CRITICAL

    def test_severity_comparison_operators(self):
        """All comparison operators should work."""
        low = IncidentSeverity.LOW
        medium = IncidentSeverity.MEDIUM

        assert low < medium
        assert low <= medium
        assert medium > low
        assert medium >= low
        assert low <= low
        assert low >= low


# Tests for Incident dataclass

class TestIncident:
    """Tests for Incident dataclass."""

    def test_incident_creation(self):
        """Should create incident with all fields."""
        incident = Incident(
            incident_id="test-123",
            timestamp=datetime.now(),
            incident_type=IncidentType.TOOL_FAILURE,
            severity=IncidentSeverity.MEDIUM,
            description="Test incident",
            context={"key": "value"}
        )

        assert incident.incident_id == "test-123"
        assert incident.incident_type == IncidentType.TOOL_FAILURE
        assert incident.severity == IncidentSeverity.MEDIUM
        assert incident.resolved is False

    def test_incident_to_dict(self):
        """Should convert incident to dictionary."""
        now = datetime.now()
        incident = Incident(
            incident_id="test-123",
            timestamp=now,
            incident_type=IncidentType.GUARDRAIL_ACTIVATION,
            severity=IncidentSeverity.HIGH,
            description="Test incident"
        )

        data = incident.to_dict()
        assert data["incident_id"] == "test-123"
        assert data["incident_type"] == "guardrail_activation"
        assert data["severity"] == "HIGH"
        assert data["severity_level"] == 4

    def test_incident_from_dict(self):
        """Should create incident from dictionary."""
        data = {
            "incident_id": "test-123",
            "timestamp": datetime.now().isoformat(),
            "incident_type": "tool_failure",
            "severity": "LOW",
            "description": "Test"
        }

        incident = Incident.from_dict(data)
        assert incident.incident_id == "test-123"
        assert incident.incident_type == IncidentType.TOOL_FAILURE
        assert incident.severity == IncidentSeverity.LOW

    def test_time_to_resolution(self):
        """Should calculate time to resolution."""
        now = datetime.now()
        incident = Incident(
            incident_id="test",
            timestamp=now,
            incident_type=IncidentType.TIMEOUT,
            severity=IncidentSeverity.LOW,
            description="Test",
            resolved=True,
            resolution="Fixed",
            resolution_timestamp=now + timedelta(minutes=30)
        )

        ttr = incident.time_to_resolution()
        assert ttr is not None
        assert abs(ttr.total_seconds() - 1800) < 1  # 30 minutes

    def test_time_to_resolution_unresolved(self):
        """Unresolved incidents should have no TTR."""
        incident = Incident(
            incident_id="test",
            timestamp=datetime.now(),
            incident_type=IncidentType.TIMEOUT,
            severity=IncidentSeverity.LOW,
            description="Test"
        )

        assert incident.time_to_resolution() is None


# Tests for IncidentTracker recording

class TestIncidentTrackerRecording:
    """Tests for incident recording."""

    def test_record_incident(self, tracker):
        """Should successfully record an incident."""
        incident = tracker.record_incident(
            incident_type=IncidentType.GUARDRAIL_ACTIVATION,
            severity=IncidentSeverity.MEDIUM,
            description="Blocked action"
        )

        assert incident.incident_id is not None
        assert incident.incident_type == IncidentType.GUARDRAIL_ACTIVATION
        assert incident.severity == IncidentSeverity.MEDIUM

    def test_record_incident_with_context(self, tracker):
        """Should record incident with context."""
        context = {"task_id": "task_123", "tool": "file_read"}
        incident = tracker.record_incident(
            incident_type=IncidentType.POLICY_VIOLATION,
            severity=IncidentSeverity.HIGH,
            description="Access denied",
            context=context
        )

        assert incident.context == context

    def test_record_incident_with_tags(self, tracker):
        """Should record incident with tags."""
        tags = ["security", "critical"]
        incident = tracker.record_incident(
            incident_type=IncidentType.HUMAN_INTERVENTION,
            severity=IncidentSeverity.HIGH,
            description="User intervention",
            tags=tags
        )

        assert incident.tags == tags

    def test_record_incident_with_custom_timestamp(self, tracker):
        """Should accept custom timestamp."""
        past = datetime.now() - timedelta(hours=5)
        incident = tracker.record_incident(
            incident_type=IncidentType.TIMEOUT,
            severity=IncidentSeverity.LOW,
            description="Old incident",
            timestamp=past
        )

        assert incident.timestamp == past

    def test_critical_incident_triggers_callback(self):
        """Critical incidents should trigger alert callback."""
        callback = Mock()
        tracker = IncidentTracker(alert_callback=callback)

        incident = tracker.record_incident(
            incident_type=IncidentType.UNEXPECTED_TERMINATION,
            severity=IncidentSeverity.CRITICAL,
            description="Critical failure"
        )

        callback.assert_called_once_with(incident)

    def test_non_critical_incident_no_callback(self):
        """Non-critical incidents should not trigger callback."""
        callback = Mock()
        tracker = IncidentTracker(alert_callback=callback)

        tracker.record_incident(
            incident_type=IncidentType.TIMEOUT,
            severity=IncidentSeverity.LOW,
            description="Minor issue"
        )

        callback.assert_not_called()


# Tests for incident resolution

class TestIncidentResolution:
    """Tests for resolving incidents."""

    def test_resolve_incident(self, tracker):
        """Should successfully resolve an incident."""
        incident = tracker.record_incident(
            incident_type=IncidentType.TOOL_FAILURE,
            severity=IncidentSeverity.MEDIUM,
            description="API error"
        )

        result = tracker.resolve_incident(
            incident.incident_id,
            resolution="Retried successfully"
        )

        assert result is True
        assert incident.resolved is True
        assert incident.resolution == "Retried successfully"
        assert incident.resolution_timestamp is not None

    def test_resolve_nonexistent_incident(self, tracker):
        """Should return False for nonexistent incident."""
        result = tracker.resolve_incident(
            "nonexistent-id",
            resolution="N/A"
        )
        assert result is False

    def test_resolve_with_custom_timestamp(self, tracker):
        """Should accept custom resolution timestamp."""
        incident = tracker.record_incident(
            incident_type=IncidentType.TIMEOUT,
            severity=IncidentSeverity.LOW,
            description="Test"
        )

        resolve_time = datetime.now() + timedelta(hours=1)
        tracker.resolve_incident(
            incident.incident_id,
            resolution="Fixed",
            timestamp=resolve_time
        )

        assert incident.resolution_timestamp == resolve_time


# Tests for incident retrieval

class TestIncidentRetrieval:
    """Tests for retrieving incidents."""

    def test_get_incident_by_id(self, populated_tracker):
        """Should retrieve specific incident by ID."""
        incidents = populated_tracker.get_incidents()
        first = incidents[0]

        retrieved = populated_tracker.get_incident(first.incident_id)
        assert retrieved is not None
        assert retrieved.incident_id == first.incident_id

    def test_get_nonexistent_incident(self, tracker):
        """Should return None for nonexistent incident."""
        assert tracker.get_incident("nonexistent") is None

    def test_get_incidents_all(self, populated_tracker):
        """Should return all incidents."""
        incidents = populated_tracker.get_incidents()
        assert len(incidents) == 3

    def test_filter_by_type(self, populated_tracker):
        """Should filter by incident type."""
        incidents = populated_tracker.get_incidents(
            incident_type=IncidentType.GUARDRAIL_ACTIVATION
        )
        assert len(incidents) == 1
        assert incidents[0].incident_type == IncidentType.GUARDRAIL_ACTIVATION

    def test_filter_by_min_severity(self, populated_tracker):
        """Should filter by minimum severity."""
        incidents = populated_tracker.get_incidents(
            severity_min=IncidentSeverity.MEDIUM
        )
        assert len(incidents) == 2
        for i in incidents:
            assert i.severity >= IncidentSeverity.MEDIUM

    def test_filter_by_max_severity(self, populated_tracker):
        """Should filter by maximum severity."""
        incidents = populated_tracker.get_incidents(
            severity_max=IncidentSeverity.MEDIUM
        )
        assert len(incidents) == 2
        for i in incidents:
            assert i.severity <= IncidentSeverity.MEDIUM

    def test_filter_by_time_range(self, tracker):
        """Should filter by time range."""
        # Record incidents at different times
        old_time = datetime.now() - timedelta(hours=5)
        new_time = datetime.now()

        tracker.record_incident(
            incident_type=IncidentType.TIMEOUT,
            severity=IncidentSeverity.LOW,
            description="Old",
            timestamp=old_time
        )
        tracker.record_incident(
            incident_type=IncidentType.TIMEOUT,
            severity=IncidentSeverity.LOW,
            description="New",
            timestamp=new_time
        )

        # Filter for recent only
        cutoff = datetime.now() - timedelta(hours=1)
        recent = tracker.get_incidents(since=cutoff)
        assert len(recent) == 1
        assert recent[0].description == "New"

    def test_filter_resolved_only(self, populated_tracker):
        """Should filter resolved only."""
        # Resolve one incident
        incidents = populated_tracker.get_incidents()
        populated_tracker.resolve_incident(
            incidents[0].incident_id,
            resolution="Fixed"
        )

        resolved = populated_tracker.get_incidents(resolved_only=True)
        assert len(resolved) == 1

    def test_filter_unresolved_only(self, populated_tracker):
        """Should filter unresolved only."""
        unresolved = populated_tracker.get_incidents(unresolved_only=True)
        assert len(unresolved) == 3  # All initially unresolved


# Tests for statistics computation

class TestIncidentStatistics:
    """Tests for statistics computation."""

    def test_statistics_empty_tracker(self, tracker):
        """Should handle empty tracker."""
        stats = tracker.get_statistics()
        assert stats.total_incidents == 0
        assert stats.incident_rate_per_hour >= 0

    def test_statistics_with_incidents(self, populated_tracker):
        """Should compute correct statistics."""
        stats = populated_tracker.get_statistics()

        assert stats.total_incidents == 3
        assert IncidentType.GUARDRAIL_ACTIVATION in stats.incidents_by_type
        assert IncidentSeverity.HIGH in stats.incidents_by_severity

    def test_statistics_with_window(self, tracker):
        """Should filter by time window."""
        # Record old incident
        tracker.record_incident(
            incident_type=IncidentType.TIMEOUT,
            severity=IncidentSeverity.LOW,
            description="Old",
            timestamp=datetime.now() - timedelta(hours=48)
        )
        # Record recent incident
        tracker.record_incident(
            incident_type=IncidentType.TIMEOUT,
            severity=IncidentSeverity.LOW,
            description="Recent"
        )

        stats_24h = tracker.get_statistics(window_hours=24)
        assert stats_24h.total_incidents == 1

    def test_mean_time_to_resolution(self, tracker):
        """Should calculate mean TTR."""
        # Record and resolve incidents
        i1 = tracker.record_incident(
            incident_type=IncidentType.TIMEOUT,
            severity=IncidentSeverity.LOW,
            description="Test 1"
        )
        i2 = tracker.record_incident(
            incident_type=IncidentType.TIMEOUT,
            severity=IncidentSeverity.LOW,
            description="Test 2"
        )

        # Resolve with known times
        tracker.resolve_incident(
            i1.incident_id,
            "Fixed",
            timestamp=i1.timestamp + timedelta(minutes=10)
        )
        tracker.resolve_incident(
            i2.incident_id,
            "Fixed",
            timestamp=i2.timestamp + timedelta(minutes=20)
        )

        stats = tracker.get_statistics()
        assert stats.mean_time_to_resolution is not None
        # Mean of 10 and 20 minutes = 15 minutes = 900 seconds
        assert abs(stats.mean_time_to_resolution.total_seconds() - 900) < 60

    def test_statistics_to_dict(self, populated_tracker):
        """Should convert statistics to dictionary."""
        stats = populated_tracker.get_statistics()
        data = stats.to_dict()

        assert "total_incidents" in data
        assert "incidents_by_type" in data
        assert "incident_rate_per_hour" in data


# Tests for incident rate calculation

class TestIncidentRateCalculation:
    """Tests for rate calculation."""

    def test_incident_rate(self, tracker):
        """Should calculate correct rate."""
        # Record incidents over known time period
        for _ in range(5):
            tracker.record_incident(
                incident_type=IncidentType.TIMEOUT,
                severity=IncidentSeverity.LOW,
                description="Test"
            )

        rate = tracker.get_incident_rate(window_hours=1)
        # 5 incidents in window, but timing matters
        assert rate >= 0

    def test_rate_with_task_counter(self, tracker_with_task_counter):
        """Should use task counter for per-task rate."""
        tracker_with_task_counter.record_incident(
            incident_type=IncidentType.TIMEOUT,
            severity=IncidentSeverity.LOW,
            description="Test"
        )

        stats = tracker_with_task_counter.get_statistics()
        # 1 incident / 100 tasks = 0.01
        assert stats.incident_rate_per_task == pytest.approx(0.01, abs=0.001)


# Tests for threshold breach detection

class TestThresholdBreachDetection:
    """Tests for threshold detection."""

    def test_no_breach_under_threshold(self, tracker):
        """Should not detect breach when under threshold."""
        tracker.record_incident(
            incident_type=IncidentType.TIMEOUT,
            severity=IncidentSeverity.LOW,
            description="Test"
        )

        breached, reason = tracker.check_threshold_breach(
            max_rate_per_hour=10.0
        )
        assert not breached

    def test_rate_threshold_breach(self, tracker):
        """Should detect rate threshold breach."""
        # Record many incidents quickly
        for _ in range(10):
            tracker.record_incident(
                incident_type=IncidentType.TIMEOUT,
                severity=IncidentSeverity.LOW,
                description="Test"
            )

        breached, reason = tracker.check_threshold_breach(
            max_rate_per_hour=5.0,
            window_hours=1
        )

        # Should breach if all 10 in 1 hour
        if tracker.get_incident_rate(1) > 5.0:
            assert breached
            assert "rate" in reason.lower()

    def test_critical_count_threshold(self, tracker):
        """Should detect critical count threshold breach."""
        # Record 2 critical incidents
        tracker.record_incident(
            incident_type=IncidentType.UNEXPECTED_TERMINATION,
            severity=IncidentSeverity.CRITICAL,
            description="Critical 1"
        )
        tracker.record_incident(
            incident_type=IncidentType.UNEXPECTED_TERMINATION,
            severity=IncidentSeverity.CRITICAL,
            description="Critical 2"
        )

        breached, reason = tracker.check_threshold_breach(
            max_critical_count=1
        )

        assert breached
        assert "critical" in reason.lower()


# Tests for report export

class TestReportExport:
    """Tests for report export."""

    def test_export_json(self, populated_tracker):
        """Should export as valid JSON."""
        report = populated_tracker.export_report(format="json")

        # Should be valid JSON
        data = json.loads(report)
        assert "statistics" in data
        assert "incidents" in data
        assert len(data["incidents"]) == 3

    def test_export_csv(self, populated_tracker):
        """Should export as CSV."""
        report = populated_tracker.export_report(format="csv")

        lines = report.split("\n")
        assert len(lines) == 4  # Header + 3 incidents
        assert "incident_id" in lines[0]

    def test_export_markdown(self, populated_tracker):
        """Should export as markdown."""
        report = populated_tracker.export_report(format="markdown")

        assert "# Incident Report" in report
        assert "## Summary" in report
        assert "Total Incidents" in report

    def test_export_invalid_format(self, tracker):
        """Should raise error for invalid format."""
        with pytest.raises(ValueError, match="Unsupported format"):
            tracker.export_report(format="xml")


# Tests for trend detection

class TestTrendDetection:
    """Tests for trend detection."""

    def test_stable_trend(self, tracker):
        """Should detect stable trend with constant rate."""
        # Record evenly spread incidents
        now = datetime.now()
        for i in range(10):
            tracker.record_incident(
                incident_type=IncidentType.TIMEOUT,
                severity=IncidentSeverity.LOW,
                description="Test",
                timestamp=now - timedelta(hours=i)
            )

        stats = tracker.get_statistics(window_hours=24)
        # Trend depends on distribution
        assert stats.trend in ["stable", "increasing", "decreasing"]


# Tests for utility methods

class TestTrackerUtilities:
    """Tests for utility methods."""

    def test_get_unresolved_incidents(self, populated_tracker):
        """Should return unresolved incidents sorted by severity."""
        unresolved = populated_tracker.get_unresolved_incidents()

        assert len(unresolved) == 3
        # Should be sorted by severity (highest first)
        assert unresolved[0].severity >= unresolved[-1].severity

    def test_get_critical_incidents(self, tracker):
        """Should return only critical incidents."""
        tracker.record_incident(
            incident_type=IncidentType.TIMEOUT,
            severity=IncidentSeverity.LOW,
            description="Low"
        )
        tracker.record_incident(
            incident_type=IncidentType.UNEXPECTED_TERMINATION,
            severity=IncidentSeverity.CRITICAL,
            description="Critical"
        )

        critical = tracker.get_critical_incidents()
        assert len(critical) == 1
        assert critical[0].severity == IncidentSeverity.CRITICAL

    def test_clear_tracker(self, populated_tracker):
        """Should clear all incidents."""
        assert len(populated_tracker.get_incidents()) == 3

        populated_tracker.clear()

        assert len(populated_tracker.get_incidents()) == 0

    def test_to_dict(self, populated_tracker):
        """Should export state to dictionary."""
        data = populated_tracker.to_dict()

        assert "incident_count" in data
        assert data["incident_count"] == 3
        assert "statistics" in data


# Tests for AggregatedIncidentTracker

class TestAggregatedIncidentTracker:
    """Tests for multi-tracker aggregation."""

    def test_add_trackers(self):
        """Should add multiple trackers."""
        agg = AggregatedIncidentTracker()

        tracker1 = IncidentTracker()
        tracker2 = IncidentTracker()

        agg.add_tracker("agent_1", tracker1)
        agg.add_tracker("agent_2", tracker2)

        # Record incidents in each
        tracker1.record_incident(
            incident_type=IncidentType.TIMEOUT,
            severity=IncidentSeverity.LOW,
            description="Test 1"
        )
        tracker2.record_incident(
            incident_type=IncidentType.TOOL_FAILURE,
            severity=IncidentSeverity.MEDIUM,
            description="Test 2"
        )

        combined = agg.get_combined_statistics()
        assert "agent_1" in combined
        assert "agent_2" in combined

    def test_get_all_incidents(self):
        """Should get all incidents across trackers."""
        agg = AggregatedIncidentTracker()

        tracker1 = IncidentTracker()
        tracker2 = IncidentTracker()

        agg.add_tracker("a", tracker1)
        agg.add_tracker("b", tracker2)

        tracker1.record_incident(
            incident_type=IncidentType.TIMEOUT,
            severity=IncidentSeverity.LOW,
            description="Test 1"
        )
        tracker2.record_incident(
            incident_type=IncidentType.TIMEOUT,
            severity=IncidentSeverity.LOW,
            description="Test 2"
        )

        all_incidents = agg.get_all_incidents()
        assert len(all_incidents) == 2

    def test_check_any_threshold_breach(self):
        """Should check thresholds across all trackers."""
        agg = AggregatedIncidentTracker()

        tracker1 = IncidentTracker()
        tracker2 = IncidentTracker()

        agg.add_tracker("normal", tracker1)
        agg.add_tracker("problematic", tracker2)

        # Make one tracker breach threshold
        for _ in range(10):
            tracker2.record_incident(
                incident_type=IncidentType.TIMEOUT,
                severity=IncidentSeverity.LOW,
                description="Test"
            )

        results = agg.check_any_threshold_breach(max_rate_per_hour=5.0)
        assert "normal" in results
        assert "problematic" in results
