"""Audit logging for tool invocations."""

import json
import logging
from typing import Optional, List, Dict, Any
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
import threading

logger = logging.getLogger(__name__)


@dataclass
class AuditEntry:
    """Single audit log entry."""
    timestamp: str
    tool_name: str
    action: str
    agent_name: Optional[str] = None
    parameters: Dict[str, Any] = field(default_factory=dict)
    result_summary: Optional[str] = None
    success: bool = True
    error: Optional[str] = None
    duration_ms: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "tool_name": self.tool_name,
            "action": self.action,
            "agent_name": self.agent_name,
            "parameters": self._sanitize_params(self.parameters),
            "result_summary": self.result_summary,
            "success": self.success,
            "error": self.error,
            "duration_ms": self.duration_ms,
            "metadata": self.metadata,
        }

    def _sanitize_params(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Sanitize parameters to remove sensitive data."""
        sanitized = {}
        sensitive_keys = {"password", "token", "secret", "key", "credential"}

        for key, value in params.items():
            if any(s in key.lower() for s in sensitive_keys):
                sanitized[key] = "***REDACTED***"
            elif isinstance(value, str) and len(value) > 500:
                sanitized[key] = value[:500] + "..."
            else:
                sanitized[key] = value

        return sanitized


class AuditLogger:
    """Audit logger for tool invocations.

    Provides immutable audit trail of all tool usage.

    Example:
        >>> audit = AuditLogger(output_file="audit.jsonl")
        >>> audit.log_invocation(
        ...     tool_name="search",
        ...     action="execute",
        ...     parameters={"query": "AI papers"},
        ...     success=True,
        ... )
    """

    def __init__(
        self,
        output_file: Optional[str] = None,
        max_entries: int = 10000,
        enabled: bool = True,
    ):
        """Initialize the audit logger.

        Args:
            output_file: Path to output JSONL file
            max_entries: Maximum in-memory entries
            enabled: Whether logging is enabled
        """
        self.output_file = output_file
        self.max_entries = max_entries
        self.enabled = enabled

        self._entries: List[AuditEntry] = []
        self._lock = threading.Lock()

        if output_file:
            Path(output_file).parent.mkdir(parents=True, exist_ok=True)

    def log_invocation(
        self,
        tool_name: str,
        action: str = "execute",
        agent_name: Optional[str] = None,
        parameters: Optional[Dict[str, Any]] = None,
        result_summary: Optional[str] = None,
        success: bool = True,
        error: Optional[str] = None,
        duration_ms: float = 0.0,
        **metadata,
    ):
        """Log a tool invocation.

        Args:
            tool_name: Name of the tool
            action: Action performed
            agent_name: Name of the agent
            parameters: Tool parameters
            result_summary: Brief result summary
            success: Whether invocation succeeded
            error: Error message if failed
            duration_ms: Execution duration
            **metadata: Additional metadata
        """
        if not self.enabled:
            return

        entry = AuditEntry(
            timestamp=datetime.now().isoformat(),
            tool_name=tool_name,
            action=action,
            agent_name=agent_name,
            parameters=parameters or {},
            result_summary=result_summary,
            success=success,
            error=error,
            duration_ms=duration_ms,
            metadata=metadata,
        )

        with self._lock:
            self._entries.append(entry)

            # Trim if over limit
            if len(self._entries) > self.max_entries:
                self._entries = self._entries[-self.max_entries:]

            # Write to file if configured
            if self.output_file:
                self._write_entry(entry)

    def _write_entry(self, entry: AuditEntry):
        """Write entry to file."""
        try:
            with open(self.output_file, "a") as f:
                f.write(json.dumps(entry.to_dict()) + "\n")
        except Exception as e:
            logger.error(f"Failed to write audit entry: {e}")

    def get_entries(
        self,
        tool_name: Optional[str] = None,
        agent_name: Optional[str] = None,
        success_only: bool = False,
        limit: int = 100,
    ) -> List[AuditEntry]:
        """Get audit entries with filtering.

        Args:
            tool_name: Filter by tool name
            agent_name: Filter by agent name
            success_only: Only successful invocations
            limit: Maximum entries to return

        Returns:
            List of matching entries
        """
        with self._lock:
            entries = self._entries.copy()

        # Apply filters
        if tool_name:
            entries = [e for e in entries if e.tool_name == tool_name]
        if agent_name:
            entries = [e for e in entries if e.agent_name == agent_name]
        if success_only:
            entries = [e for e in entries if e.success]

        return entries[-limit:]

    def get_statistics(self) -> Dict[str, Any]:
        """Get audit statistics."""
        with self._lock:
            entries = self._entries.copy()

        if not entries:
            return {"total_entries": 0}

        tool_counts = {}
        success_count = 0
        total_duration = 0.0

        for entry in entries:
            tool_counts[entry.tool_name] = tool_counts.get(entry.tool_name, 0) + 1
            if entry.success:
                success_count += 1
            total_duration += entry.duration_ms

        return {
            "total_entries": len(entries),
            "success_count": success_count,
            "failure_count": len(entries) - success_count,
            "success_rate": success_count / len(entries),
            "total_duration_ms": total_duration,
            "avg_duration_ms": total_duration / len(entries),
            "tool_counts": tool_counts,
        }

    def clear(self):
        """Clear in-memory entries."""
        with self._lock:
            self._entries = []


# Global audit logger
_global_audit = AuditLogger()


def get_audit_logger() -> AuditLogger:
    """Get the global audit logger."""
    return _global_audit


def log_tool_invocation(**kwargs):
    """Convenience function to log to global audit."""
    _global_audit.log_invocation(**kwargs)
