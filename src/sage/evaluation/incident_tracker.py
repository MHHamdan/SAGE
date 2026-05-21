"""
Incident Rate Tracking Module

Tracks safety-relevant events during agent operation for evaluation
and compliance reporting. Essential for long-horizon evaluation where
safety incidents can compound over time.

Incident types tracked:
- Human intervention requests
- Guardrail/safety filter activations
- Policy and constraint violations
- Unexpected terminations
- Tool failures
- Timeouts
- Resource exhaustion

Example:
    >>> from sage.evaluation.incident_tracker import (
    ...     IncidentTracker, IncidentType, IncidentSeverity
    ... )
    >>>
    >>> tracker = IncidentTracker()
    >>>
    >>> # Record incidents as they occur
    >>> tracker.record_incident(
    ...     incident_type=IncidentType.GUARDRAIL_ACTIVATION,
    ...     severity=IncidentSeverity.MEDIUM,
    ...     description="Blocked attempt to access /etc/passwd",
    ...     context={"task_id": "task_123", "tool": "file_read"}
    ... )
    >>>
    >>> # Get statistics
    >>> stats = tracker.get_statistics(window_hours=24)
    >>> print(f"Incident rate: {stats.incident_rate_per_hour:.2f}/hour")
"""

import json
import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Union

logger = logging.getLogger(__name__)


class IncidentType(Enum):
    """Types of safety-relevant incidents.

    Each type represents a different category of safety event:

    - HUMAN_INTERVENTION: Human took over or intervened in agent operation
    - GUARDRAIL_ACTIVATION: Safety guardrail blocked an action
    - POLICY_VIOLATION: Agent attempted to violate a policy
    - CONSTRAINT_VIOLATION: Agent violated a constraint
    - UNEXPECTED_TERMINATION: Agent terminated unexpectedly
    - TOOL_FAILURE: Tool execution failed
    - TIMEOUT: Operation exceeded time limit
    - RESOURCE_EXHAUSTION: Resource limits exceeded (memory, tokens, etc.)
    """

    HUMAN_INTERVENTION = "human_intervention"
    GUARDRAIL_ACTIVATION = "guardrail_activation"
    POLICY_VIOLATION = "policy_violation"
    CONSTRAINT_VIOLATION = "constraint_violation"
    UNEXPECTED_TERMINATION = "unexpected_termination"
    TOOL_FAILURE = "tool_failure"
    TIMEOUT = "timeout"
    RESOURCE_EXHAUSTION = "resource_exhaustion"


class IncidentSeverity(Enum):
    """Severity levels for incidents.

    Severity determines how incidents are prioritized and reported:

    - INFO (1): Informational, no action needed
    - LOW (2): Minor issue, review at convenience
    - MEDIUM (3): Significant issue, investigate soon
    - HIGH (4): Serious issue, requires prompt attention
    - CRITICAL (5): Critical issue, requires immediate action
    """

    INFO = 1
    LOW = 2
    MEDIUM = 3
    HIGH = 4
    CRITICAL = 5

    def __lt__(self, other):
        if isinstance(other, IncidentSeverity):
            return self.value < other.value
        return NotImplemented

    def __le__(self, other):
        if isinstance(other, IncidentSeverity):
            return self.value <= other.value
        return NotImplemented

    def __gt__(self, other):
        if isinstance(other, IncidentSeverity):
            return self.value > other.value
        return NotImplemented

    def __ge__(self, other):
        if isinstance(other, IncidentSeverity):
            return self.value >= other.value
        return NotImplemented


@dataclass
class Incident:
    """Record of a safety-relevant incident.

    Attributes:
        incident_id: Unique identifier for this incident
        timestamp: When the incident occurred
        incident_type: Category of incident
        severity: Severity level
        description: Human-readable description of what happened
        context: Additional context (task_id, action, tool, etc.)
        resolved: Whether the incident has been resolved
        resolution: Description of how it was resolved
        resolution_timestamp: When it was resolved
        tags: Custom tags for categorization
    """

    incident_id: str
    timestamp: datetime
    incident_type: IncidentType
    severity: IncidentSeverity
    description: str
    context: Dict[str, Any] = field(default_factory=dict)
    resolved: bool = False
    resolution: Optional[str] = None
    resolution_timestamp: Optional[datetime] = None
    tags: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert incident to dictionary for serialization."""
        return {
            "incident_id": self.incident_id,
            "timestamp": self.timestamp.isoformat(),
            "incident_type": self.incident_type.value,
            "severity": self.severity.name,
            "severity_level": self.severity.value,
            "description": self.description,
            "context": self.context,
            "resolved": self.resolved,
            "resolution": self.resolution,
            "resolution_timestamp": (
                self.resolution_timestamp.isoformat()
                if self.resolution_timestamp
                else None
            ),
            "tags": self.tags,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Incident":
        """Create incident from dictionary."""
        return cls(
            incident_id=data["incident_id"],
            timestamp=datetime.fromisoformat(data["timestamp"]),
            incident_type=IncidentType(data["incident_type"]),
            severity=IncidentSeverity[data["severity"]],
            description=data["description"],
            context=data.get("context", {}),
            resolved=data.get("resolved", False),
            resolution=data.get("resolution"),
            resolution_timestamp=(
                datetime.fromisoformat(data["resolution_timestamp"])
                if data.get("resolution_timestamp")
                else None
            ),
            tags=data.get("tags", []),
        )

    def time_to_resolution(self) -> Optional[timedelta]:
        """Get time from incident to resolution."""
        if self.resolution_timestamp:
            return self.resolution_timestamp - self.timestamp
        return None


@dataclass
class IncidentStatistics:
    """Aggregated incident statistics.

    Attributes:
        total_incidents: Total number of incidents
        incidents_by_type: Count per incident type
        incidents_by_severity: Count per severity level
        incident_rate_per_task: Incidents per task (if task count available)
        incident_rate_per_hour: Incidents per hour
        mean_time_to_resolution: Average resolution time
        unresolved_count: Number of unresolved incidents
        trend: Whether incidents are stable, increasing, or decreasing
        window_start: Start of analysis window
        window_end: End of analysis window
    """

    total_incidents: int
    incidents_by_type: Dict[IncidentType, int]
    incidents_by_severity: Dict[IncidentSeverity, int]
    incident_rate_per_task: float
    incident_rate_per_hour: float
    mean_time_to_resolution: Optional[timedelta]
    unresolved_count: int
    trend: str
    window_start: Optional[datetime] = None
    window_end: Optional[datetime] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert statistics to dictionary."""
        return {
            "total_incidents": self.total_incidents,
            "incidents_by_type": {
                k.value: v for k, v in self.incidents_by_type.items()
            },
            "incidents_by_severity": {
                k.name: v for k, v in self.incidents_by_severity.items()
            },
            "incident_rate_per_task": self.incident_rate_per_task,
            "incident_rate_per_hour": self.incident_rate_per_hour,
            "mean_time_to_resolution_seconds": (
                self.mean_time_to_resolution.total_seconds()
                if self.mean_time_to_resolution
                else None
            ),
            "unresolved_count": self.unresolved_count,
            "trend": self.trend,
            "window_start": (
                self.window_start.isoformat() if self.window_start else None
            ),
            "window_end": self.window_end.isoformat() if self.window_end else None,
        }


class IncidentTracker:
    """
    Tracks and analyzes safety incidents during agent operation.

    Provides comprehensive incident tracking with:
    - Recording incidents with type, severity, and context
    - Resolving incidents with resolution descriptions
    - Filtering incidents by various criteria
    - Computing statistics and rates
    - Detecting threshold breaches
    - Exporting reports for compliance

    Example:
        >>> tracker = IncidentTracker()
        >>>
        >>> # Record incidents as they occur
        >>> incident = tracker.record_incident(
        ...     incident_type=IncidentType.GUARDRAIL_ACTIVATION,
        ...     severity=IncidentSeverity.MEDIUM,
        ...     description="Blocked attempt to access /etc/passwd",
        ...     context={"task_id": "task_123", "tool": "file_read"}
        ... )
        >>>
        >>> # Later, mark as resolved
        >>> tracker.resolve_incident(
        ...     incident.incident_id,
        ...     resolution="Action blocked, task continued with alternative"
        ... )
        >>>
        >>> # Get statistics
        >>> stats = tracker.get_statistics(window_hours=24)
        >>> print(f"Total incidents: {stats.total_incidents}")
        >>> print(f"Rate: {stats.incident_rate_per_hour:.2f}/hour")

    Attributes:
        task_counter: Optional callable returning current task count
    """

    def __init__(
        self,
        task_counter: Optional[Callable[[], int]] = None,
        alert_callback: Optional[Callable[[Incident], None]] = None,
    ):
        """Initialize the incident tracker.

        Args:
            task_counter: Optional function returning current task count.
                         Used to calculate incidents per task.
            alert_callback: Optional callback for real-time alerting.
                           Called immediately when critical incidents occur.
        """
        self.task_counter = task_counter
        self.alert_callback = alert_callback

        self._incidents: List[Incident] = []
        self._start_time: datetime = datetime.now()
        self._incident_counts_by_hour: Dict[str, int] = {}  # hour_key -> count

    def record_incident(
        self,
        incident_type: IncidentType,
        severity: IncidentSeverity,
        description: str,
        context: Optional[Dict[str, Any]] = None,
        tags: Optional[List[str]] = None,
        timestamp: Optional[datetime] = None,
    ) -> Incident:
        """Record a new incident.

        Args:
            incident_type: Type of incident
            severity: Severity level
            description: Human-readable description
            context: Additional context (task_id, action, tool, etc.)
            tags: Custom tags for categorization
            timestamp: When incident occurred (default: now)

        Returns:
            The created Incident object
        """
        now = timestamp or datetime.now()

        incident = Incident(
            incident_id=str(uuid.uuid4()),
            timestamp=now,
            incident_type=incident_type,
            severity=severity,
            description=description,
            context=context or {},
            tags=tags or [],
        )

        self._incidents.append(incident)

        # Update hourly count
        hour_key = now.strftime("%Y-%m-%d-%H")
        self._incident_counts_by_hour[hour_key] = (
            self._incident_counts_by_hour.get(hour_key, 0) + 1
        )

        # Log based on severity
        log_msg = (
            f"Incident recorded: [{severity.name}] {incident_type.value} - "
            f"{description[:100]}"
        )
        if severity >= IncidentSeverity.HIGH:
            logger.warning(log_msg)
        else:
            logger.info(log_msg)

        # Alert for critical incidents
        if severity == IncidentSeverity.CRITICAL and self.alert_callback:
            self.alert_callback(incident)

        return incident

    def resolve_incident(
        self, incident_id: str, resolution: str, timestamp: Optional[datetime] = None
    ) -> bool:
        """Mark an incident as resolved.

        Args:
            incident_id: ID of incident to resolve
            resolution: Description of how it was resolved
            timestamp: When resolved (default: now)

        Returns:
            True if incident was found and resolved, False otherwise
        """
        for incident in self._incidents:
            if incident.incident_id == incident_id:
                incident.resolved = True
                incident.resolution = resolution
                incident.resolution_timestamp = timestamp or datetime.now()

                logger.info(
                    f"Incident {incident_id[:8]}... resolved: {resolution[:100]}"
                )
                return True

        logger.warning(f"Incident {incident_id} not found for resolution")
        return False

    def get_incident(self, incident_id: str) -> Optional[Incident]:
        """Get a specific incident by ID.

        Args:
            incident_id: ID of incident to retrieve

        Returns:
            Incident if found, None otherwise
        """
        for incident in self._incidents:
            if incident.incident_id == incident_id:
                return incident
        return None

    def get_incidents(
        self,
        incident_type: Optional[IncidentType] = None,
        severity_min: Optional[IncidentSeverity] = None,
        severity_max: Optional[IncidentSeverity] = None,
        since: Optional[datetime] = None,
        until: Optional[datetime] = None,
        resolved_only: bool = False,
        unresolved_only: bool = False,
        tags: Optional[List[str]] = None,
    ) -> List[Incident]:
        """Get filtered list of incidents.

        Args:
            incident_type: Filter by specific type
            severity_min: Filter by minimum severity (inclusive)
            severity_max: Filter by maximum severity (inclusive)
            since: Filter incidents after this time
            until: Filter incidents before this time
            resolved_only: Only return resolved incidents
            unresolved_only: Only return unresolved incidents
            tags: Filter by tags (any match)

        Returns:
            List of matching incidents
        """
        result = self._incidents.copy()

        if incident_type is not None:
            result = [i for i in result if i.incident_type == incident_type]

        if severity_min is not None:
            result = [i for i in result if i.severity >= severity_min]

        if severity_max is not None:
            result = [i for i in result if i.severity <= severity_max]

        if since is not None:
            result = [i for i in result if i.timestamp >= since]

        if until is not None:
            result = [i for i in result if i.timestamp <= until]

        if resolved_only:
            result = [i for i in result if i.resolved]

        if unresolved_only:
            result = [i for i in result if not i.resolved]

        if tags:
            result = [i for i in result if any(tag in i.tags for tag in tags)]

        return result

    def get_statistics(self, window_hours: Optional[int] = None) -> IncidentStatistics:
        """Get aggregated incident statistics.

        Args:
            window_hours: Only consider incidents within this many hours.
                         If None, considers all incidents.

        Returns:
            IncidentStatistics with aggregated data
        """
        now = datetime.now()

        # Filter by time window
        if window_hours is not None:
            cutoff = now - timedelta(hours=window_hours)
            incidents = [i for i in self._incidents if i.timestamp >= cutoff]
            window_start = cutoff
        else:
            incidents = self._incidents.copy()
            window_start = self._start_time

        # Count by type
        by_type: Dict[IncidentType, int] = {}
        for t in IncidentType:
            count = sum(1 for i in incidents if i.incident_type == t)
            if count > 0:
                by_type[t] = count

        # Count by severity
        by_severity: Dict[IncidentSeverity, int] = {}
        for s in IncidentSeverity:
            count = sum(1 for i in incidents if i.severity == s)
            if count > 0:
                by_severity[s] = count

        # Calculate rates
        total = len(incidents)
        hours_elapsed = max(
            (now - window_start).total_seconds() / 3600, 0.001  # Avoid division by zero
        )
        rate_per_hour = total / hours_elapsed

        # Task rate if counter available
        task_count = self.task_counter() if self.task_counter else 0
        rate_per_task = total / max(task_count, 1)

        # Mean time to resolution
        resolution_times = [
            i.time_to_resolution()
            for i in incidents
            if i.time_to_resolution() is not None
        ]
        mean_ttr = None
        if resolution_times:
            total_seconds = sum(t.total_seconds() for t in resolution_times)
            mean_ttr = timedelta(seconds=total_seconds / len(resolution_times))

        # Unresolved count
        unresolved = sum(1 for i in incidents if not i.resolved)

        # Trend analysis
        trend = self._calculate_trend(window_hours or 24)

        return IncidentStatistics(
            total_incidents=total,
            incidents_by_type=by_type,
            incidents_by_severity=by_severity,
            incident_rate_per_task=rate_per_task,
            incident_rate_per_hour=rate_per_hour,
            mean_time_to_resolution=mean_ttr,
            unresolved_count=unresolved,
            trend=trend,
            window_start=window_start,
            window_end=now,
        )

    def get_incident_rate(self, window_hours: int = 1) -> float:
        """Get incidents per hour over specified window.

        Args:
            window_hours: Number of hours to consider

        Returns:
            Incident rate per hour
        """
        cutoff = datetime.now() - timedelta(hours=window_hours)
        incidents = [i for i in self._incidents if i.timestamp >= cutoff]
        return len(incidents) / max(window_hours, 0.001)

    def check_threshold_breach(
        self,
        max_rate_per_hour: float = 5.0,
        max_critical_count: int = 1,
        window_hours: int = 1,
    ) -> tuple[bool, Optional[str]]:
        """Check if incident thresholds are breached.

        Args:
            max_rate_per_hour: Maximum acceptable incident rate
            max_critical_count: Maximum acceptable critical incidents
            window_hours: Time window to check

        Returns:
            Tuple of (breached: bool, reason: Optional[str])
        """
        cutoff = datetime.now() - timedelta(hours=window_hours)
        recent_incidents = [i for i in self._incidents if i.timestamp >= cutoff]

        # Check rate
        rate = len(recent_incidents) / max(window_hours, 0.001)
        if rate > max_rate_per_hour:
            return (
                True,
                f"Incident rate {rate:.2f}/hour exceeds threshold {max_rate_per_hour}",
            )

        # Check critical count
        critical_count = sum(
            1 for i in recent_incidents if i.severity == IncidentSeverity.CRITICAL
        )
        if critical_count > max_critical_count:
            return (
                True,
                f"Critical incidents ({critical_count}) exceed threshold ({max_critical_count})",
            )

        return False, None

    def export_report(
        self, format: str = "json", window_hours: Optional[int] = None
    ) -> str:
        """Export incident report for compliance.

        Args:
            format: Output format ("json", "csv", "markdown")
            window_hours: Time window to include

        Returns:
            Formatted report string

        Raises:
            ValueError: If format is not supported
        """
        stats = self.get_statistics(window_hours)
        incidents = self.get_incidents(
            since=(
                (datetime.now() - timedelta(hours=window_hours))
                if window_hours
                else None
            )
        )

        if format == "json":
            return json.dumps(
                {
                    "report_generated": datetime.now().isoformat(),
                    "statistics": stats.to_dict(),
                    "incidents": [i.to_dict() for i in incidents],
                },
                indent=2,
            )

        elif format == "csv":
            lines = ["incident_id,timestamp,type,severity,description,resolved"]
            for i in incidents:
                desc = i.description.replace('"', '""')[:100]
                lines.append(
                    f'"{i.incident_id}","{i.timestamp.isoformat()}",'
                    f'"{i.incident_type.value}","{i.severity.name}",'
                    f'"{desc}",{i.resolved}'
                )
            return "\n".join(lines)

        elif format == "markdown":
            md_lines = [
                "# Incident Report",
                f"\nGenerated: {datetime.now().isoformat()}",
                f"\n## Summary",
                f"\n- **Total Incidents**: {stats.total_incidents}",
                f"- **Incident Rate**: {stats.incident_rate_per_hour:.2f}/hour",
                f"- **Unresolved**: {stats.unresolved_count}",
                f"- **Trend**: {stats.trend}",
                f"\n## Incidents by Type\n",
            ]
            for t, count in stats.incidents_by_type.items():
                md_lines.append(f"- {t.value}: {count}")

            md_lines.append(f"\n## Incidents by Severity\n")
            for s, count in stats.incidents_by_severity.items():
                md_lines.append(f"- {s.name}: {count}")

            md_lines.append(f"\n## Incident Details\n")
            for i in incidents[:20]:  # Limit to 20 most recent
                md_lines.append(f"\n### [{i.severity.name}] {i.incident_type.value}")
                md_lines.append(f"\n- **Time**: {i.timestamp.isoformat()}")
                md_lines.append(f"- **Description**: {i.description}")
                md_lines.append(f"- **Resolved**: {i.resolved}")

            return "\n".join(md_lines)

        else:
            raise ValueError(f"Unsupported format: {format}")

    def _calculate_trend(self, window_hours: int = 24) -> str:
        """Calculate incident trend direction.

        Compares first half of window to second half.

        Args:
            window_hours: Window to analyze

        Returns:
            "stable", "increasing", or "decreasing"
        """
        now = datetime.now()
        window_start = now - timedelta(hours=window_hours)
        midpoint = now - timedelta(hours=window_hours / 2)

        first_half = sum(
            1 for i in self._incidents if window_start <= i.timestamp < midpoint
        )
        second_half = sum(1 for i in self._incidents if midpoint <= i.timestamp <= now)

        # Normalize by time (in case window doesn't have equal halves)
        first_rate = first_half / (window_hours / 2) if first_half > 0 else 0
        second_rate = second_half / (window_hours / 2) if second_half > 0 else 0

        # Use 20% threshold for trend detection
        if second_rate > first_rate * 1.2:
            return "increasing"
        elif second_rate < first_rate * 0.8:
            return "decreasing"
        else:
            return "stable"

    def get_unresolved_incidents(self) -> List[Incident]:
        """Get all unresolved incidents.

        Returns:
            List of unresolved incidents sorted by severity (highest first)
        """
        unresolved = [i for i in self._incidents if not i.resolved]
        return sorted(unresolved, key=lambda i: -i.severity.value)

    def get_critical_incidents(
        self, window_hours: Optional[int] = None
    ) -> List[Incident]:
        """Get critical severity incidents.

        Args:
            window_hours: Optional time window

        Returns:
            List of critical incidents
        """
        return self.get_incidents(
            severity_min=IncidentSeverity.CRITICAL,
            since=(
                (datetime.now() - timedelta(hours=window_hours))
                if window_hours
                else None
            ),
        )

    def clear(self) -> None:
        """Clear all incident records."""
        count = len(self._incidents)
        self._incidents = []
        self._incident_counts_by_hour = {}
        self._start_time = datetime.now()
        logger.info(f"Cleared {count} incidents from tracker")

    def to_dict(self) -> Dict[str, Any]:
        """Export tracker state to dictionary."""
        return {
            "incident_count": len(self._incidents),
            "start_time": self._start_time.isoformat(),
            "statistics": self.get_statistics().to_dict(),
        }


class AggregatedIncidentTracker:
    """Aggregates incidents across multiple trackers.

    Useful for tracking incidents across multiple agents or components.

    Example:
        >>> agg = AggregatedIncidentTracker()
        >>> agg.add_tracker("agent_1", tracker1)
        >>> agg.add_tracker("agent_2", tracker2)
        >>> combined_stats = agg.get_combined_statistics()
    """

    def __init__(self):
        """Initialize aggregated tracker."""
        self._trackers: Dict[str, IncidentTracker] = {}

    def add_tracker(self, name: str, tracker: IncidentTracker) -> None:
        """Add a tracker to aggregate.

        Args:
            name: Identifier for this tracker
            tracker: IncidentTracker instance
        """
        self._trackers[name] = tracker

    def get_combined_statistics(
        self, window_hours: Optional[int] = None
    ) -> Dict[str, IncidentStatistics]:
        """Get statistics from all trackers.

        Args:
            window_hours: Time window to analyze

        Returns:
            Dict mapping tracker name to statistics
        """
        return {
            name: tracker.get_statistics(window_hours)
            for name, tracker in self._trackers.items()
        }

    def get_all_incidents(
        self, severity_min: Optional[IncidentSeverity] = None
    ) -> List[tuple[str, Incident]]:
        """Get all incidents from all trackers.

        Args:
            severity_min: Minimum severity filter

        Returns:
            List of (tracker_name, incident) tuples
        """
        all_incidents = []
        for name, tracker in self._trackers.items():
            incidents = tracker.get_incidents(severity_min=severity_min)
            all_incidents.extend((name, i) for i in incidents)

        # Sort by timestamp
        all_incidents.sort(key=lambda x: x[1].timestamp, reverse=True)
        return all_incidents

    def check_any_threshold_breach(
        self, max_rate_per_hour: float = 5.0, max_critical_count: int = 1
    ) -> Dict[str, tuple[bool, Optional[str]]]:
        """Check thresholds across all trackers.

        Args:
            max_rate_per_hour: Maximum acceptable rate
            max_critical_count: Maximum critical incidents

        Returns:
            Dict mapping tracker name to (breached, reason) tuples
        """
        return {
            name: tracker.check_threshold_breach(max_rate_per_hour, max_critical_count)
            for name, tracker in self._trackers.items()
        }
