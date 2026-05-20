"""
Escalation Handling for Human Oversight

Manages escalation of agent decisions and incidents to appropriate
human supervisors based on configurable policies.

Supports tiered escalation with automatic routing based on:
- Risk level of the operation
- Time since original request
- Number of previous escalations
- Incident severity

Example:
    >>> escalation = EscalationHandler()
    >>> escalation.add_policy(
    ...     EscalationPolicy(
    ...         trigger_level=EscalationLevel.TIER_1,
    ...         risk_levels=[RiskLevel.HIGH, RiskLevel.CRITICAL],
    ...         timeout_seconds=300,
    ...         notify_channels=["email", "slack"]
    ...     )
    ... )
    >>>
    >>> request = escalation.escalate(
    ...     request_id="req-123",
    ...     reason="High-risk operation pending too long"
    ... )
"""

from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List, Callable
from datetime import datetime, timedelta
from enum import Enum
import logging
import uuid

from .approval_flow import RiskLevel, ApprovalRequest

logger = logging.getLogger(__name__)


class EscalationLevel(Enum):
    """Escalation tiers for human oversight."""
    TIER_1 = "tier_1"  # First-line supervisor
    TIER_2 = "tier_2"  # Manager
    TIER_3 = "tier_3"  # Senior management
    EMERGENCY = "emergency"  # Emergency contacts


@dataclass
class EscalationPolicy:
    """Policy defining when and how to escalate.

    Attributes:
        trigger_level: Escalation level this policy applies to
        risk_levels: Risk levels that trigger this policy
        timeout_seconds: Time before escalation triggers
        notify_channels: Channels to notify (email, slack, pager, etc.)
        auto_escalate: Whether to auto-escalate on timeout
        required_approvers: Number of approvers required at this level
        escalate_to: Next escalation level if not resolved
    """
    trigger_level: EscalationLevel
    risk_levels: List[RiskLevel]
    timeout_seconds: float = 300.0
    notify_channels: List[str] = field(default_factory=list)
    auto_escalate: bool = True
    required_approvers: int = 1
    escalate_to: Optional[EscalationLevel] = None


@dataclass
class EscalationRequest:
    """Represents an escalation request.

    Attributes:
        escalation_id: Unique identifier
        original_request_id: ID of the original approval request
        level: Current escalation level
        reason: Reason for escalation
        created_at: When escalation was created
        resolved: Whether escalation is resolved
        resolved_at: When escalation was resolved
        resolver: Who resolved the escalation
        metadata: Additional context
    """
    escalation_id: str
    original_request_id: str
    level: EscalationLevel
    reason: str
    created_at: datetime
    resolved: bool = False
    resolved_at: Optional[datetime] = None
    resolver: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "escalation_id": self.escalation_id,
            "original_request_id": self.original_request_id,
            "level": self.level.value,
            "reason": self.reason,
            "created_at": self.created_at.isoformat(),
            "resolved": self.resolved,
            "resolved_at": self.resolved_at.isoformat() if self.resolved_at else None,
            "resolver": self.resolver,
            "metadata": self.metadata,
        }


class EscalationHandler:
    """
    Manages escalation of approval requests to human supervisors.

    Provides tiered escalation based on configurable policies,
    with support for multiple notification channels and automatic
    escalation on timeout.

    Example:
        >>> handler = EscalationHandler()
        >>> handler.add_policy(
        ...     EscalationPolicy(
        ...         trigger_level=EscalationLevel.TIER_1,
        ...         risk_levels=[RiskLevel.HIGH],
        ...         timeout_seconds=600
        ...     )
        ... )
        >>>
        >>> escalation = handler.escalate(
        ...     request_id="req-123",
        ...     level=EscalationLevel.TIER_1,
        ...     reason="Approval pending for critical operation"
        ... )

    Attributes:
        policies: List of escalation policies
    """

    def __init__(
        self,
        on_escalate: Optional[Callable[[EscalationRequest], None]] = None,
    ):
        """Initialize escalation handler.

        Args:
            on_escalate: Callback when escalation is created
        """
        self._policies: List[EscalationPolicy] = []
        self._escalations: Dict[str, EscalationRequest] = {}
        self._on_escalate = on_escalate
        self._audit_log: List[Dict[str, Any]] = []

    def add_policy(self, policy: EscalationPolicy) -> None:
        """Add an escalation policy.

        Args:
            policy: EscalationPolicy to add
        """
        self._policies.append(policy)
        logger.info(
            f"Added escalation policy for {policy.trigger_level.value} "
            f"(risk levels: {[r.value for r in policy.risk_levels]})"
        )

    def get_policy(
        self,
        level: EscalationLevel,
        risk_level: RiskLevel
    ) -> Optional[EscalationPolicy]:
        """Get the applicable policy for a level and risk.

        Args:
            level: Escalation level
            risk_level: Risk level of the request

        Returns:
            Applicable EscalationPolicy or None
        """
        for policy in self._policies:
            if policy.trigger_level == level and risk_level in policy.risk_levels:
                return policy
        return None

    def escalate(
        self,
        request_id: str,
        level: EscalationLevel = EscalationLevel.TIER_1,
        reason: str = "",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> EscalationRequest:
        """Create an escalation for a request.

        Args:
            request_id: ID of the original approval request
            level: Escalation level
            reason: Reason for escalation
            metadata: Additional context

        Returns:
            EscalationRequest object
        """
        escalation_id = str(uuid.uuid4())

        escalation = EscalationRequest(
            escalation_id=escalation_id,
            original_request_id=request_id,
            level=level,
            reason=reason,
            created_at=datetime.now(),
            metadata=metadata or {},
        )

        self._escalations[escalation_id] = escalation
        self._log_audit("escalation_created", escalation)

        logger.info(
            f"Created escalation {escalation_id} at {level.value} "
            f"for request {request_id}: {reason}"
        )

        if self._on_escalate:
            self._on_escalate(escalation)

        return escalation

    def resolve(
        self,
        escalation_id: str,
        resolver: Optional[str] = None,
    ) -> EscalationRequest:
        """Mark an escalation as resolved.

        Args:
            escalation_id: ID of escalation to resolve
            resolver: Who resolved it

        Returns:
            Updated EscalationRequest

        Raises:
            KeyError: If escalation not found
        """
        if escalation_id not in self._escalations:
            raise KeyError(f"Escalation {escalation_id} not found")

        escalation = self._escalations[escalation_id]
        escalation.resolved = True
        escalation.resolved_at = datetime.now()
        escalation.resolver = resolver

        self._log_audit("escalation_resolved", escalation)

        logger.info(f"Escalation {escalation_id} resolved by {resolver}")

        return escalation

    def escalate_further(
        self,
        escalation_id: str,
        reason: str = "Escalating to next tier",
    ) -> Optional[EscalationRequest]:
        """Escalate to the next tier.

        Args:
            escalation_id: Current escalation ID
            reason: Reason for further escalation

        Returns:
            New EscalationRequest or None if no next level
        """
        if escalation_id not in self._escalations:
            raise KeyError(f"Escalation {escalation_id} not found")

        current = self._escalations[escalation_id]

        # Determine next level
        level_order = [
            EscalationLevel.TIER_1,
            EscalationLevel.TIER_2,
            EscalationLevel.TIER_3,
            EscalationLevel.EMERGENCY,
        ]

        try:
            current_idx = level_order.index(current.level)
            if current_idx >= len(level_order) - 1:
                logger.warning(
                    f"Escalation {escalation_id} already at highest level"
                )
                return None
            next_level = level_order[current_idx + 1]
        except ValueError:
            next_level = EscalationLevel.TIER_2

        # Mark current as resolved (escalated)
        current.resolved = True
        current.resolved_at = datetime.now()

        # Create new escalation
        return self.escalate(
            request_id=current.original_request_id,
            level=next_level,
            reason=reason,
            metadata={
                "previous_escalation": escalation_id,
                "previous_level": current.level.value,
            },
        )

    def get_escalation(self, escalation_id: str) -> Optional[EscalationRequest]:
        """Get an escalation by ID."""
        return self._escalations.get(escalation_id)

    def get_escalations_for_request(
        self,
        request_id: str
    ) -> List[EscalationRequest]:
        """Get all escalations for a request."""
        return [
            e for e in self._escalations.values()
            if e.original_request_id == request_id
        ]

    def get_active_escalations(self) -> List[EscalationRequest]:
        """Get all unresolved escalations."""
        return [
            e for e in self._escalations.values()
            if not e.resolved
        ]

    def get_escalations_by_level(
        self,
        level: EscalationLevel
    ) -> List[EscalationRequest]:
        """Get all escalations at a specific level."""
        return [
            e for e in self._escalations.values()
            if e.level == level
        ]

    def _log_audit(self, event: str, escalation: EscalationRequest) -> None:
        """Add entry to audit log."""
        self._audit_log.append({
            "timestamp": datetime.now().isoformat(),
            "event": event,
            "escalation_id": escalation.escalation_id,
            "request_id": escalation.original_request_id,
            "level": escalation.level.value,
            "resolved": escalation.resolved,
        })

    def get_audit_log(self) -> List[Dict[str, Any]]:
        """Get the audit log."""
        return self._audit_log.copy()

    def get_statistics(self) -> Dict[str, Any]:
        """Get escalation statistics."""
        total = len(self._escalations)
        active = len(self.get_active_escalations())
        by_level = {}
        resolution_times = []

        for escalation in self._escalations.values():
            level = escalation.level.value
            by_level[level] = by_level.get(level, 0) + 1

            if escalation.resolved and escalation.resolved_at:
                delta = (escalation.resolved_at - escalation.created_at).total_seconds()
                resolution_times.append(delta)

        return {
            "total_escalations": total,
            "active_escalations": active,
            "resolved_escalations": total - active,
            "by_level": by_level,
            "avg_resolution_time_seconds": (
                sum(resolution_times) / len(resolution_times)
                if resolution_times else 0.0
            ),
        }

    def clear(self) -> None:
        """Clear all escalations and audit log."""
        self._escalations.clear()
        self._audit_log.clear()
