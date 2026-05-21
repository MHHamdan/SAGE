"""
Audit Logging for Human Oversight

Provides comprehensive audit logging for all human oversight interactions,
including approval requests, escalations, and interventions.

Supports multiple output formats and storage backends for compliance
and forensic analysis.

Example:
    >>> logger = AuditLogger()
    >>> logger.log_event(
    ...     event_type=AuditEventType.APPROVAL_REQUESTED,
    ...     request_id="req-123",
    ...     action="deploy_model",
    ...     risk_level="high"
    ... )
    >>>
    >>> # Export for compliance
    >>> report = logger.export(format="json")
"""

from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List, Callable, Union
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
import logging
import json
import csv
import io

logger = logging.getLogger(__name__)


class AuditEventType(Enum):
    """Types of auditable events in human oversight."""
    # Approval events
    APPROVAL_REQUESTED = "approval_requested"
    APPROVAL_GRANTED = "approval_granted"
    APPROVAL_DENIED = "approval_denied"
    APPROVAL_TIMEOUT = "approval_timeout"
    APPROVAL_CANCELLED = "approval_cancelled"

    # Escalation events
    ESCALATION_CREATED = "escalation_created"
    ESCALATION_RESOLVED = "escalation_resolved"
    ESCALATION_UPGRADED = "escalation_upgraded"

    # Intervention events
    HUMAN_INTERVENTION = "human_intervention"
    AGENT_OVERRIDE = "agent_override"
    EMERGENCY_STOP = "emergency_stop"

    # Policy events
    POLICY_VIOLATION = "policy_violation"
    GUARDRAIL_TRIGGERED = "guardrail_triggered"
    CONSTRAINT_VIOLATION = "constraint_violation"

    # Session events
    SESSION_START = "session_start"
    SESSION_END = "session_end"
    CONTEXT_SWITCH = "context_switch"


@dataclass
class AuditEntry:
    """Single entry in the audit log.

    Attributes:
        entry_id: Unique identifier for this entry
        timestamp: When the event occurred
        event_type: Type of event
        actor: Who/what triggered the event
        action: Specific action taken
        target: Target of the action
        context: Additional context
        outcome: Result of the action
        metadata: Additional metadata
        session_id: Session identifier (if applicable)
        request_id: Request identifier (if applicable)
    """
    entry_id: str
    timestamp: datetime
    event_type: AuditEventType
    actor: Optional[str] = None
    action: Optional[str] = None
    target: Optional[str] = None
    context: Dict[str, Any] = field(default_factory=dict)
    outcome: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    session_id: Optional[str] = None
    request_id: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "entry_id": self.entry_id,
            "timestamp": self.timestamp.isoformat(),
            "event_type": self.event_type.value,
            "actor": self.actor,
            "action": self.action,
            "target": self.target,
            "context": self.context,
            "outcome": self.outcome,
            "metadata": self.metadata,
            "session_id": self.session_id,
            "request_id": self.request_id,
        }


class AuditLogger:
    """
    Comprehensive audit logging for human oversight.

    Records all human oversight interactions for compliance,
    forensic analysis, and continuous improvement.

    Features:
        - Structured event logging
        - Multiple export formats (JSON, CSV, Markdown)
        - Filtering and querying
        - Retention policies
        - Storage backend support

    Example:
        >>> audit = AuditLogger(session_id="session-001")
        >>>
        >>> audit.log_event(
        ...     event_type=AuditEventType.APPROVAL_REQUESTED,
        ...     actor="agent-alpha",
        ...     action="execute_trade",
        ...     context={"symbol": "AAPL", "quantity": 100}
        ... )
        >>>
        >>> # Query events
        >>> approvals = audit.get_events_by_type(AuditEventType.APPROVAL_REQUESTED)
        >>>
        >>> # Export report
        >>> report = audit.export(format="markdown")

    Attributes:
        session_id: Current session identifier
        retention_days: How long to retain entries
    """

    def __init__(
        self,
        session_id: Optional[str] = None,
        retention_days: int = 90,
        on_log: Optional[Callable[[AuditEntry], None]] = None,
    ):
        """Initialize audit logger.

        Args:
            session_id: Session identifier for grouping entries
            retention_days: Number of days to retain entries
            on_log: Callback when entry is logged
        """
        self.session_id = session_id
        self.retention_days = retention_days
        self._on_log = on_log
        self._entries: List[AuditEntry] = []
        self._entry_counter = 0

    def log_event(
        self,
        event_type: AuditEventType,
        actor: Optional[str] = None,
        action: Optional[str] = None,
        target: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
        outcome: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        request_id: Optional[str] = None,
    ) -> AuditEntry:
        """Log an audit event.

        Args:
            event_type: Type of event
            actor: Who/what triggered the event
            action: Specific action taken
            target: Target of the action
            context: Additional context
            outcome: Result of the action
            metadata: Additional metadata
            request_id: Associated request ID

        Returns:
            Created AuditEntry
        """
        self._entry_counter += 1
        entry_id = f"audit-{self._entry_counter:08d}"

        entry = AuditEntry(
            entry_id=entry_id,
            timestamp=datetime.now(),
            event_type=event_type,
            actor=actor,
            action=action,
            target=target,
            context=context or {},
            outcome=outcome,
            metadata=metadata or {},
            session_id=self.session_id,
            request_id=request_id,
        )

        self._entries.append(entry)

        logger.debug(
            f"Audit: {event_type.value} by {actor} - {action} "
            f"({outcome or 'pending'})"
        )

        if self._on_log:
            self._on_log(entry)

        return entry

    def log_approval_requested(
        self,
        request_id: str,
        action: str,
        actor: str,
        risk_level: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> AuditEntry:
        """Log an approval request event.

        Args:
            request_id: Approval request ID
            action: Action requiring approval
            actor: Who requested approval
            risk_level: Risk level of the action
            context: Additional context

        Returns:
            AuditEntry
        """
        return self.log_event(
            event_type=AuditEventType.APPROVAL_REQUESTED,
            actor=actor,
            action=action,
            request_id=request_id,
            context=context or {},
            metadata={"risk_level": risk_level},
        )

    def log_approval_decision(
        self,
        request_id: str,
        approved: bool,
        approver: str,
        reason: Optional[str] = None,
    ) -> AuditEntry:
        """Log an approval decision.

        Args:
            request_id: Approval request ID
            approved: Whether approved or denied
            approver: Who made the decision
            reason: Reason for decision

        Returns:
            AuditEntry
        """
        event_type = (
            AuditEventType.APPROVAL_GRANTED
            if approved
            else AuditEventType.APPROVAL_DENIED
        )

        return self.log_event(
            event_type=event_type,
            actor=approver,
            request_id=request_id,
            outcome="approved" if approved else "denied",
            metadata={"reason": reason} if reason else {},
        )

    def log_escalation(
        self,
        escalation_id: str,
        request_id: str,
        level: str,
        reason: str,
    ) -> AuditEntry:
        """Log an escalation event.

        Args:
            escalation_id: Escalation ID
            request_id: Original request ID
            level: Escalation level
            reason: Reason for escalation

        Returns:
            AuditEntry
        """
        return self.log_event(
            event_type=AuditEventType.ESCALATION_CREATED,
            request_id=request_id,
            metadata={
                "escalation_id": escalation_id,
                "level": level,
                "reason": reason,
            },
        )

    def log_intervention(
        self,
        actor: str,
        action: str,
        reason: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> AuditEntry:
        """Log a human intervention.

        Args:
            actor: Who intervened
            action: What intervention was taken
            reason: Why intervention was needed
            context: Additional context

        Returns:
            AuditEntry
        """
        return self.log_event(
            event_type=AuditEventType.HUMAN_INTERVENTION,
            actor=actor,
            action=action,
            context=context or {},
            metadata={"reason": reason},
        )

    def log_policy_violation(
        self,
        policy_name: str,
        violation_details: str,
        actor: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> AuditEntry:
        """Log a policy violation.

        Args:
            policy_name: Name of the violated policy
            violation_details: Details of the violation
            actor: Who/what caused the violation
            context: Additional context

        Returns:
            AuditEntry
        """
        return self.log_event(
            event_type=AuditEventType.POLICY_VIOLATION,
            actor=actor,
            action=policy_name,
            context=context or {},
            metadata={"violation_details": violation_details},
        )

    def get_entries(self) -> List[AuditEntry]:
        """Get all audit entries."""
        return self._entries.copy()

    def get_events_by_type(
        self,
        event_type: AuditEventType
    ) -> List[AuditEntry]:
        """Get entries filtered by event type.

        Args:
            event_type: Event type to filter by

        Returns:
            Filtered list of entries
        """
        return [e for e in self._entries if e.event_type == event_type]

    def get_events_by_actor(self, actor: str) -> List[AuditEntry]:
        """Get entries filtered by actor.

        Args:
            actor: Actor to filter by

        Returns:
            Filtered list of entries
        """
        return [e for e in self._entries if e.actor == actor]

    def get_events_by_request(self, request_id: str) -> List[AuditEntry]:
        """Get entries filtered by request ID.

        Args:
            request_id: Request ID to filter by

        Returns:
            Filtered list of entries
        """
        return [e for e in self._entries if e.request_id == request_id]

    def get_events_in_timerange(
        self,
        start: datetime,
        end: datetime
    ) -> List[AuditEntry]:
        """Get entries within a time range.

        Args:
            start: Start of time range
            end: End of time range

        Returns:
            Filtered list of entries
        """
        return [
            e for e in self._entries
            if start <= e.timestamp <= end
        ]

    def get_recent_events(
        self,
        hours: float = 24
    ) -> List[AuditEntry]:
        """Get events from the last N hours.

        Args:
            hours: Number of hours to look back

        Returns:
            Filtered list of entries
        """
        cutoff = datetime.now() - timedelta(hours=hours)
        return [e for e in self._entries if e.timestamp >= cutoff]

    def apply_retention_policy(self) -> int:
        """Apply retention policy, removing old entries.

        Returns:
            Number of entries removed
        """
        cutoff = datetime.now() - timedelta(days=self.retention_days)
        original_count = len(self._entries)
        self._entries = [e for e in self._entries if e.timestamp >= cutoff]
        removed = original_count - len(self._entries)

        if removed > 0:
            logger.info(
                f"Retention policy removed {removed} entries "
                f"older than {self.retention_days} days"
            )

        return removed

    def export(self, format: str = "json") -> str:
        """Export audit log in specified format.

        Args:
            format: Output format ("json", "csv", "markdown")

        Returns:
            Formatted audit log string
        """
        if format == "json":
            return self._export_json()
        elif format == "csv":
            return self._export_csv()
        elif format == "markdown":
            return self._export_markdown()
        else:
            raise ValueError(f"Unsupported format: {format}")

    def _export_json(self) -> str:
        """Export as JSON."""
        return json.dumps(
            [e.to_dict() for e in self._entries],
            indent=2,
            default=str
        )

    def _export_csv(self) -> str:
        """Export as CSV."""
        output = io.StringIO()
        fieldnames = [
            "entry_id", "timestamp", "event_type", "actor",
            "action", "target", "outcome", "request_id"
        ]
        writer = csv.DictWriter(output, fieldnames=fieldnames)
        writer.writeheader()

        for entry in self._entries:
            writer.writerow({
                "entry_id": entry.entry_id,
                "timestamp": entry.timestamp.isoformat(),
                "event_type": entry.event_type.value,
                "actor": entry.actor or "",
                "action": entry.action or "",
                "target": entry.target or "",
                "outcome": entry.outcome or "",
                "request_id": entry.request_id or "",
            })

        return output.getvalue()

    def _export_markdown(self) -> str:
        """Export as Markdown."""
        lines = [
            "# Human Oversight Audit Log",
            "",
            f"**Session**: {self.session_id or 'N/A'}",
            f"**Generated**: {datetime.now().isoformat()}",
            f"**Total Events**: {len(self._entries)}",
            "",
            "## Event Log",
            "",
            "| Time | Event | Actor | Action | Outcome |",
            "|------|-------|-------|--------|---------|",
        ]

        for entry in self._entries:
            time_str = entry.timestamp.strftime("%Y-%m-%d %H:%M:%S")
            lines.append(
                f"| {time_str} | {entry.event_type.value} | "
                f"{entry.actor or '-'} | {entry.action or '-'} | "
                f"{entry.outcome or '-'} |"
            )

        # Add statistics
        lines.extend([
            "",
            "## Statistics",
            "",
        ])

        stats = self.get_statistics()
        for key, value in stats.items():
            if isinstance(value, dict):
                lines.append(f"### {key.replace('_', ' ').title()}")
                for k, v in value.items():
                    lines.append(f"- {k}: {v}")
            else:
                lines.append(f"- **{key.replace('_', ' ').title()}**: {value}")

        return "\n".join(lines)

    def save_to_file(self, path: Union[str, Path], format: str = "json") -> None:
        """Save audit log to file.

        Args:
            path: File path to save to
            format: Output format
        """
        path = Path(path)
        content = self.export(format=format)
        path.write_text(content)
        logger.info(f"Saved audit log to {path}")

    def get_statistics(self) -> Dict[str, Any]:
        """Get audit log statistics.

        Returns:
            Dictionary with statistics
        """
        by_type = {}
        by_actor = {}
        by_outcome = {}

        for entry in self._entries:
            # Count by type
            type_val = entry.event_type.value
            by_type[type_val] = by_type.get(type_val, 0) + 1

            # Count by actor
            if entry.actor:
                by_actor[entry.actor] = by_actor.get(entry.actor, 0) + 1

            # Count by outcome
            if entry.outcome:
                by_outcome[entry.outcome] = by_outcome.get(entry.outcome, 0) + 1

        # Time range
        if self._entries:
            first = min(e.timestamp for e in self._entries)
            last = max(e.timestamp for e in self._entries)
            duration = last - first
        else:
            first = last = duration = None

        return {
            "total_entries": len(self._entries),
            "by_event_type": by_type,
            "by_actor": by_actor,
            "by_outcome": by_outcome,
            "time_range": {
                "first": first.isoformat() if first else None,
                "last": last.isoformat() if last else None,
                "duration_hours": (
                    duration.total_seconds() / 3600 if duration else 0
                ),
            },
        }

    def clear(self) -> None:
        """Clear all entries."""
        self._entries.clear()
        self._entry_counter = 0
