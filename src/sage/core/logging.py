"""Structured JSONL logging for experiments.

This module provides JSONL (JSON Lines) logging for:
- Experiment runs
- Agent actions
- Tool invocations
- Evaluation results
- Reproducibility metadata
"""

import json
import logging
import os
from typing import Optional, Dict, Any, List, Union
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
import threading
from enum import Enum

logger = logging.getLogger(__name__)


class LogLevel(Enum):
    """Log levels for structured logging."""
    DEBUG = "debug"
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class EventType(Enum):
    """Types of events to log."""
    EXPERIMENT_START = "experiment_start"
    EXPERIMENT_END = "experiment_end"
    RUN_START = "run_start"
    RUN_END = "run_end"
    AGENT_ACTION = "agent_action"
    TOOL_CALL = "tool_call"
    PLAN_CREATED = "plan_created"
    PLAN_STEP = "plan_step"
    EVALUATION = "evaluation"
    ERROR = "error"
    GUARDRAIL = "guardrail"
    COST = "cost"
    INCIDENT = "incident"
    CUSTOM = "custom"


class IncidentType(Enum):
    """Standardized incident types for paper traceability.

    These incident types map to the evaluation protocol described in the paper:
    - human_intervention: Human approval was required and granted
    - guardrail_triggered: A safety guardrail blocked an action
    - policy_violation: An action violated a security policy
    - task_termination: Task was terminated due to error/timeout/safety
    - approval_denied: Human explicitly denied an approval request
    - constraint_violated: A plan constraint was violated during execution
    - capability_exceeded: Agent attempted to use unauthorized capability
    """
    HUMAN_INTERVENTION = "human_intervention"
    GUARDRAIL_TRIGGERED = "guardrail_triggered"
    POLICY_VIOLATION = "policy_violation"
    TASK_TERMINATION = "task_termination"
    APPROVAL_DENIED = "approval_denied"
    CONSTRAINT_VIOLATED = "constraint_violated"
    CAPABILITY_EXCEEDED = "capability_exceeded"


class IncidentSeverity(Enum):
    """Severity levels for incidents.

    Maps to cost impact in evaluation:
    - info: Logged but no cost impact
    - low: Minor cost impact (e.g., retry)
    - medium: Moderate cost (e.g., human review needed)
    - high: Significant cost (e.g., task blocked)
    - critical: Task termination required
    """
    INFO = "info"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class LogEntry:
    """A single log entry."""
    timestamp: str
    event_type: str
    level: str
    message: str
    data: Dict[str, Any] = field(default_factory=dict)
    run_id: Optional[str] = None
    experiment_id: Optional[str] = None
    agent_name: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {k: v for k, v in asdict(self).items() if v is not None}

    def to_json(self) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict())


class JSONLLogger:
    """JSONL file logger for structured experiment logging.

    Each line in the output file is a valid JSON object, making it
    easy to parse and analyze experiment data.

    Example:
        >>> logger = JSONLLogger("experiment_001", output_dir="logs")
        >>> logger.log_event(
        ...     EventType.AGENT_ACTION,
        ...     "Agent generated plan",
        ...     {"plan_steps": 5}
        ... )
        >>> logger.close()
    """

    def __init__(
        self,
        experiment_id: str,
        output_dir: str = "logs",
        filename: Optional[str] = None,
        buffer_size: int = 100,
    ):
        """Initialize the logger.

        Args:
            experiment_id: Unique experiment identifier
            output_dir: Directory for log files
            filename: Custom filename (default: {experiment_id}_{timestamp}.jsonl)
            buffer_size: Number of entries to buffer before flushing
        """
        self.experiment_id = experiment_id
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.filename = filename or f"{experiment_id}_{timestamp}.jsonl"
        self.filepath = self.output_dir / self.filename

        self._buffer: List[LogEntry] = []
        self._buffer_size = buffer_size
        self._lock = threading.Lock()
        self._run_id: Optional[str] = None
        self._agent_name: Optional[str] = None

        # Write initial entry
        self._write_entry(LogEntry(
            timestamp=datetime.now().isoformat(),
            event_type=EventType.EXPERIMENT_START.value,
            level=LogLevel.INFO.value,
            message=f"Experiment {experiment_id} started",
            experiment_id=experiment_id,
            data={"output_file": str(self.filepath)},
        ))

        logger.info(f"JSONL logger initialized: {self.filepath}")

    def set_run_id(self, run_id: str):
        """Set the current run ID for subsequent entries."""
        self._run_id = run_id

    def set_agent_name(self, agent_name: str):
        """Set the current agent name for subsequent entries."""
        self._agent_name = agent_name

    def log_event(
        self,
        event_type: Union[EventType, str],
        message: str,
        data: Optional[Dict[str, Any]] = None,
        level: LogLevel = LogLevel.INFO,
    ):
        """Log an event.

        Args:
            event_type: Type of event
            message: Human-readable message
            data: Additional structured data
            level: Log level
        """
        if isinstance(event_type, EventType):
            event_type = event_type.value

        entry = LogEntry(
            timestamp=datetime.now().isoformat(),
            event_type=event_type,
            level=level.value,
            message=message,
            data=data or {},
            run_id=self._run_id,
            experiment_id=self.experiment_id,
            agent_name=self._agent_name,
        )

        self._write_entry(entry)

    def log_agent_action(
        self,
        action: str,
        input_data: Any,
        output_data: Any,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        """Log an agent action.

        Args:
            action: Action name/type
            input_data: Input to the action
            output_data: Output from the action
            metadata: Additional metadata
        """
        self.log_event(
            EventType.AGENT_ACTION,
            f"Agent action: {action}",
            {
                "action": action,
                "input": self._serialize(input_data),
                "output": self._serialize(output_data),
                **(metadata or {}),
            },
        )

    def log_tool_call(
        self,
        tool_name: str,
        arguments: Dict[str, Any],
        result: Any,
        latency_ms: float,
        success: bool = True,
    ):
        """Log a tool invocation.

        Args:
            tool_name: Name of the tool
            arguments: Tool arguments
            result: Tool result
            latency_ms: Call latency in milliseconds
            success: Whether the call succeeded
        """
        self.log_event(
            EventType.TOOL_CALL,
            f"Tool call: {tool_name}",
            {
                "tool_name": tool_name,
                "arguments": arguments,
                "result": self._serialize(result),
                "latency_ms": latency_ms,
                "success": success,
            },
        )

    def log_plan(
        self,
        plan_id: str,
        steps: List[Dict[str, Any]],
        metadata: Optional[Dict[str, Any]] = None,
    ):
        """Log a created plan.

        Args:
            plan_id: Unique plan identifier
            steps: List of plan steps
            metadata: Additional metadata
        """
        self.log_event(
            EventType.PLAN_CREATED,
            f"Plan created: {plan_id}",
            {
                "plan_id": plan_id,
                "num_steps": len(steps),
                "steps": steps,
                **(metadata or {}),
            },
        )

    def log_evaluation(
        self,
        metrics: Dict[str, float],
        task_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        """Log evaluation results.

        Args:
            metrics: Evaluation metrics
            task_id: Optional task identifier
            metadata: Additional metadata
        """
        self.log_event(
            EventType.EVALUATION,
            "Evaluation results",
            {
                "task_id": task_id,
                "metrics": metrics,
                **(metadata or {}),
            },
        )

    def log_error(
        self,
        error: Exception,
        context: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        """Log an error.

        Args:
            error: The exception
            context: Error context description
            metadata: Additional metadata
        """
        self.log_event(
            EventType.ERROR,
            f"Error: {str(error)}",
            {
                "error_type": type(error).__name__,
                "error_message": str(error),
                "context": context,
                **(metadata or {}),
            },
            level=LogLevel.ERROR,
        )

    def log_guardrail(
        self,
        guardrail_name: str,
        triggered: bool,
        action_blocked: Optional[str] = None,
        reason: Optional[str] = None,
    ):
        """Log a guardrail check.

        Args:
            guardrail_name: Name of the guardrail
            triggered: Whether the guardrail was triggered
            action_blocked: Action that was blocked (if any)
            reason: Reason for blocking
        """
        self.log_event(
            EventType.GUARDRAIL,
            f"Guardrail {'triggered' if triggered else 'passed'}: {guardrail_name}",
            {
                "guardrail_name": guardrail_name,
                "triggered": triggered,
                "action_blocked": action_blocked,
                "reason": reason,
            },
            level=LogLevel.WARNING if triggered else LogLevel.DEBUG,
        )

    def log_cost(
        self,
        cost_data: Dict[str, Any],
    ):
        """Log cost information.

        Args:
            cost_data: Cost breakdown data
        """
        self.log_event(
            EventType.COST,
            "Cost recorded",
            cost_data,
        )

    def log_incident(
        self,
        incident_type: Union[IncidentType, str],
        severity: Union[IncidentSeverity, str],
        description: str,
        action_attempted: Optional[str] = None,
        action_result: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        """Log a standardized incident for paper traceability.

        Incidents are logged with standardized fields for consistent
        analysis and paper reporting:
        - incident_type: One of the IncidentType enum values
        - severity: One of the IncidentSeverity enum values
        - description: Human-readable description
        - action_attempted: What action triggered the incident
        - action_result: What happened (blocked, approved, terminated)

        Args:
            incident_type: Type of incident (from IncidentType enum)
            severity: Severity level (from IncidentSeverity enum)
            description: Human-readable description
            action_attempted: The action that caused the incident
            action_result: The outcome (blocked, approved, etc.)
            metadata: Additional metadata

        Example:
            >>> logger.log_incident(
            ...     IncidentType.GUARDRAIL_TRIGGERED,
            ...     IncidentSeverity.HIGH,
            ...     "Blocked attempt to delete production database",
            ...     action_attempted="delete_database",
            ...     action_result="blocked",
            ... )
        """
        if isinstance(incident_type, IncidentType):
            incident_type_str = incident_type.value
        else:
            incident_type_str = incident_type

        if isinstance(severity, IncidentSeverity):
            severity_str = severity.value
        else:
            severity_str = severity

        # Map severity to log level
        severity_to_level = {
            "info": LogLevel.INFO,
            "low": LogLevel.INFO,
            "medium": LogLevel.WARNING,
            "high": LogLevel.WARNING,
            "critical": LogLevel.ERROR,
        }
        level = severity_to_level.get(severity_str, LogLevel.WARNING)

        self.log_event(
            EventType.INCIDENT,
            f"Incident [{severity_str.upper()}]: {description}",
            {
                "incident_type": incident_type_str,
                "severity": severity_str,
                "description": description,
                "action_attempted": action_attempted,
                "action_result": action_result,
                **(metadata or {}),
            },
            level=level,
        )

    def _serialize(self, obj: Any) -> Any:
        """Serialize an object for logging."""
        if obj is None:
            return None
        if isinstance(obj, (str, int, float, bool)):
            return obj
        if isinstance(obj, (list, tuple)):
            return [self._serialize(item) for item in obj]
        if isinstance(obj, dict):
            return {k: self._serialize(v) for k, v in obj.items()}
        if hasattr(obj, "to_dict"):
            return obj.to_dict()
        if hasattr(obj, "__dict__"):
            return {k: self._serialize(v) for k, v in obj.__dict__.items()}
        return str(obj)[:1000]  # Truncate long strings

    def _write_entry(self, entry: LogEntry):
        """Write entry to buffer and flush if needed."""
        with self._lock:
            self._buffer.append(entry)
            if len(self._buffer) >= self._buffer_size:
                self._flush()

    def _flush(self):
        """Flush buffer to file."""
        if not self._buffer:
            return

        with open(self.filepath, "a") as f:
            for entry in self._buffer:
                f.write(entry.to_json() + "\n")

        self._buffer = []

    def close(self):
        """Close the logger and flush remaining entries."""
        with self._lock:
            self._write_entry(LogEntry(
                timestamp=datetime.now().isoformat(),
                event_type=EventType.EXPERIMENT_END.value,
                level=LogLevel.INFO.value,
                message=f"Experiment {self.experiment_id} ended",
                experiment_id=self.experiment_id,
            ))
            self._flush()

        logger.info(f"JSONL logger closed: {self.filepath}")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type:
            self.log_error(exc_val, "Exception during experiment")
        self.close()


def load_jsonl_log(filepath: str) -> List[Dict[str, Any]]:
    """Load a JSONL log file.

    Args:
        filepath: Path to the log file

    Returns:
        List of log entries as dictionaries
    """
    entries = []
    with open(filepath, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                entries.append(json.loads(line))
    return entries


def filter_log_entries(
    entries: List[Dict[str, Any]],
    event_type: Optional[str] = None,
    run_id: Optional[str] = None,
    level: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Filter log entries by criteria.

    Args:
        entries: List of log entries
        event_type: Filter by event type
        run_id: Filter by run ID
        level: Filter by log level

    Returns:
        Filtered list of entries
    """
    result = entries

    if event_type:
        result = [e for e in result if e.get("event_type") == event_type]
    if run_id:
        result = [e for e in result if e.get("run_id") == run_id]
    if level:
        result = [e for e in result if e.get("level") == level]

    return result


def summarize_log(filepath: str) -> Dict[str, Any]:
    """Generate summary statistics from a log file.

    Args:
        filepath: Path to the log file

    Returns:
        Summary dictionary
    """
    entries = load_jsonl_log(filepath)

    event_counts = {}
    tool_calls = []
    errors = []
    evaluations = []

    for entry in entries:
        event_type = entry.get("event_type", "unknown")
        event_counts[event_type] = event_counts.get(event_type, 0) + 1

        if event_type == "tool_call":
            tool_calls.append(entry.get("data", {}))
        elif event_type == "error":
            errors.append(entry)
        elif event_type == "evaluation":
            evaluations.append(entry.get("data", {}).get("metrics", {}))

    return {
        "total_entries": len(entries),
        "event_counts": event_counts,
        "num_tool_calls": len(tool_calls),
        "num_errors": len(errors),
        "num_evaluations": len(evaluations),
        "evaluations": evaluations,
    }
