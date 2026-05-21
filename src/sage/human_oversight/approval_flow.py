"""
Human Approval Flow for Agentic AI Systems

Implements human-in-the-loop approval mechanisms for sensitive operations
in autonomous agent systems. Provides structured approval requests,
timeout handling, and audit logging.

This module supports the human oversight requirements described in Section IX
of the IEEE TAI paper on Agentic AI.

Key Features:
    - Async approval waiting with configurable timeouts
    - Risk-level based auto-rejection policies
    - Multiple approval channels (CLI, webhook, queue)
    - Comprehensive audit logging
    - Batch approval support
    - Approval delegation and escalation

Example:
    >>> handler = ApprovalHandler(default_timeout=300)
    >>> request = handler.create_request(
    ...     action="deploy_model",
    ...     context={"model": "gpt-4", "environment": "production"},
    ...     risk_level=RiskLevel.HIGH
    ... )
    >>>
    >>> # Async wait for human decision
    >>> result = await handler.wait_for_approval(request.request_id)
    >>> if result.approved:
    ...     print("Deploying model...")
"""

from dataclasses import dataclass, field
from typing import (
    Optional,
    Callable,
    Awaitable,
    Dict,
    Any,
    List,
    Union,
)
from datetime import datetime, timedelta
from enum import Enum
import logging
import asyncio
import uuid
import json

logger = logging.getLogger(__name__)


class RiskLevel(Enum):
    """Risk level for operations requiring approval.

    Risk levels determine default timeout behavior and escalation policies.
    Higher risk operations may have shorter timeouts (auto-reject)
    or require multiple approvers.
    """
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class ApprovalStatus(Enum):
    """Status of an approval request."""
    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"
    TIMEOUT = "timeout"
    CANCELLED = "cancelled"
    ESCALATED = "escalated"


# Type alias for approval callbacks
ApprovalCallback = Callable[["ApprovalRequest"], Awaitable[None]]


class ApprovalTimeoutError(Exception):
    """Exception raised when approval request times out."""

    def __init__(self, request_id: str, timeout: float):
        self.request_id = request_id
        self.timeout = timeout
        super().__init__(
            f"Approval request {request_id} timed out after {timeout}s"
        )


@dataclass
class ApprovalRequest:
    """Represents a pending human approval request.

    Attributes:
        request_id: Unique identifier for this request
        action: Name of the action requiring approval
        context: Additional context about the action
        risk_level: Risk level of the operation
        created_at: When the request was created
        timeout: Timeout in seconds (None for no timeout)
        expires_at: When the request expires
        status: Current status of the request
        requester: Who/what requested the approval
        metadata: Additional metadata
        approver: Who approved/rejected (if decided)
        decision_at: When decision was made
        decision_reason: Reason for the decision
    """
    request_id: str
    action: str
    context: Dict[str, Any]
    risk_level: RiskLevel
    created_at: datetime
    timeout: Optional[float]
    expires_at: Optional[datetime]
    status: ApprovalStatus = ApprovalStatus.PENDING
    requester: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    approver: Optional[str] = None
    decision_at: Optional[datetime] = None
    decision_reason: Optional[str] = None

    def is_expired(self) -> bool:
        """Check if the request has expired."""
        if self.expires_at is None:
            return False
        return datetime.now() > self.expires_at

    def is_pending(self) -> bool:
        """Check if request is still pending."""
        return self.status == ApprovalStatus.PENDING and not self.is_expired()

    def time_remaining(self) -> Optional[float]:
        """Get seconds remaining before timeout."""
        if self.expires_at is None:
            return None
        delta = self.expires_at - datetime.now()
        return max(0.0, delta.total_seconds())

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "request_id": self.request_id,
            "action": self.action,
            "context": self.context,
            "risk_level": self.risk_level.value,
            "created_at": self.created_at.isoformat(),
            "timeout": self.timeout,
            "expires_at": self.expires_at.isoformat() if self.expires_at else None,
            "status": self.status.value,
            "requester": self.requester,
            "metadata": self.metadata,
            "approver": self.approver,
            "decision_at": self.decision_at.isoformat() if self.decision_at else None,
            "decision_reason": self.decision_reason,
        }


@dataclass
class ApprovalResult:
    """Result of an approval request.

    Attributes:
        request_id: ID of the approval request
        approved: Whether the request was approved
        status: Final status
        approver: Who made the decision
        reason: Reason for the decision
        decided_at: When decision was made
        conditions: Any conditions attached to approval
    """
    request_id: str
    approved: bool
    status: ApprovalStatus
    approver: Optional[str] = None
    reason: Optional[str] = None
    decided_at: Optional[datetime] = None
    conditions: Dict[str, Any] = field(default_factory=dict)


class ApprovalHandler:
    """
    Manages human approval requests for sensitive agent operations.

    Provides async approval waiting, timeout handling, and audit logging
    for human-in-the-loop oversight of autonomous agents.

    Features:
        - Async approval waiting with configurable timeouts
        - Risk-level based policies
        - Multiple notification channels
        - Batch approval support
        - Comprehensive audit trail

    Example:
        >>> handler = ApprovalHandler(
        ...     default_timeout=300,
        ...     auto_reject_on_timeout=True
        ... )
        >>>
        >>> # Create a request
        >>> request = handler.create_request(
        ...     action="execute_trade",
        ...     context={"symbol": "AAPL", "quantity": 1000},
        ...     risk_level=RiskLevel.HIGH
        ... )
        >>>
        >>> # Wait for approval (blocks until approved/rejected/timeout)
        >>> result = await handler.wait_for_approval(request.request_id)
        >>>
        >>> if result.approved:
        ...     execute_trade()
        ... else:
        ...     print(f"Rejected: {result.reason}")

    Attributes:
        default_timeout: Default timeout in seconds for requests
        auto_reject_on_timeout: Whether to auto-reject on timeout
        risk_timeouts: Risk-level specific timeouts
    """

    def __init__(
        self,
        default_timeout: float = 300.0,
        auto_reject_on_timeout: bool = True,
        risk_timeouts: Optional[Dict[RiskLevel, float]] = None,
        on_request_created: Optional[ApprovalCallback] = None,
        on_decision: Optional[ApprovalCallback] = None,
    ):
        """Initialize the approval handler.

        Args:
            default_timeout: Default timeout in seconds for approval requests.
                            Set to 0 or None for no timeout.
            auto_reject_on_timeout: If True, automatically reject requests
                                   that timeout. If False, raise exception.
            risk_timeouts: Optional dict mapping RiskLevel to specific timeouts.
                          Overrides default_timeout for that risk level.
            on_request_created: Callback when new request is created.
            on_decision: Callback when decision is made.
        """
        self.default_timeout = default_timeout if default_timeout else None
        self.auto_reject_on_timeout = auto_reject_on_timeout
        self.risk_timeouts = risk_timeouts or {}

        self._on_request_created = on_request_created
        self._on_decision = on_decision

        # Storage for pending requests
        self._requests: Dict[str, ApprovalRequest] = {}
        self._events: Dict[str, asyncio.Event] = {}
        self._results: Dict[str, ApprovalResult] = {}

        # Audit log
        self._audit_log: List[Dict[str, Any]] = []

    def create_request(
        self,
        action: str,
        context: Optional[Dict[str, Any]] = None,
        risk_level: RiskLevel = RiskLevel.MEDIUM,
        timeout: Optional[float] = None,
        requester: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> ApprovalRequest:
        """Create a new approval request.

        Args:
            action: Name of the action requiring approval
            context: Additional context about the action
            risk_level: Risk level of the operation
            timeout: Override timeout for this request (None uses default)
            requester: Who/what is requesting approval
            metadata: Additional metadata to attach

        Returns:
            ApprovalRequest object
        """
        request_id = str(uuid.uuid4())
        created_at = datetime.now()

        # Determine timeout
        if timeout is None:
            timeout = self.risk_timeouts.get(risk_level, self.default_timeout)

        # Calculate expiration
        expires_at = None
        if timeout and timeout > 0:
            expires_at = created_at + timedelta(seconds=timeout)

        request = ApprovalRequest(
            request_id=request_id,
            action=action,
            context=context or {},
            risk_level=risk_level,
            created_at=created_at,
            timeout=timeout,
            expires_at=expires_at,
            requester=requester,
            metadata=metadata or {},
        )

        # Store request
        self._requests[request_id] = request
        self._events[request_id] = asyncio.Event()

        # Audit log
        self._log_audit("request_created", request)

        logger.info(
            f"Created approval request {request_id} for action '{action}' "
            f"(risk: {risk_level.value}, timeout: {timeout}s)"
        )

        return request

    async def wait_for_approval(
        self,
        request_id: str,
        poll_interval: float = 0.5,
    ) -> ApprovalResult:
        """Wait for approval decision on a request.

        Blocks until the request is approved, rejected, or times out.

        Args:
            request_id: ID of the request to wait for
            poll_interval: How often to check for timeout (seconds)

        Returns:
            ApprovalResult with the decision

        Raises:
            KeyError: If request_id not found
            ApprovalTimeoutError: If timeout and auto_reject_on_timeout=False
        """
        if request_id not in self._requests:
            raise KeyError(f"Approval request {request_id} not found")

        request = self._requests[request_id]
        event = self._events[request_id]

        # If already decided, return immediately
        if request_id in self._results:
            return self._results[request_id]

        # Wait for decision or timeout
        while request.is_pending():
            try:
                # Wait with timeout
                timeout = poll_interval
                if request.expires_at:
                    remaining = request.time_remaining()
                    if remaining is not None:
                        timeout = min(poll_interval, remaining)

                await asyncio.wait_for(event.wait(), timeout=timeout)

                # Decision was made
                if request_id in self._results:
                    return self._results[request_id]

            except asyncio.TimeoutError:
                # Check if request expired
                if request.is_expired():
                    return self._handle_timeout(request)

        # Request is no longer pending
        if request_id in self._results:
            return self._results[request_id]

        # Should not reach here
        return self._handle_timeout(request)

    def approve(
        self,
        request_id: str,
        approver: Optional[str] = None,
        reason: Optional[str] = None,
        conditions: Optional[Dict[str, Any]] = None,
    ) -> ApprovalResult:
        """Approve a pending request.

        Args:
            request_id: ID of the request to approve
            approver: Who is approving
            reason: Reason for approval
            conditions: Any conditions attached to approval

        Returns:
            ApprovalResult

        Raises:
            KeyError: If request not found
            ValueError: If request is not pending
        """
        return self._make_decision(
            request_id=request_id,
            approved=True,
            approver=approver,
            reason=reason,
            conditions=conditions,
        )

    def reject(
        self,
        request_id: str,
        approver: Optional[str] = None,
        reason: Optional[str] = None,
    ) -> ApprovalResult:
        """Reject a pending request.

        Args:
            request_id: ID of the request to reject
            approver: Who is rejecting
            reason: Reason for rejection

        Returns:
            ApprovalResult

        Raises:
            KeyError: If request not found
            ValueError: If request is not pending
        """
        return self._make_decision(
            request_id=request_id,
            approved=False,
            approver=approver,
            reason=reason,
        )

    def cancel(self, request_id: str) -> ApprovalResult:
        """Cancel a pending request.

        Args:
            request_id: ID of the request to cancel

        Returns:
            ApprovalResult with cancelled status
        """
        if request_id not in self._requests:
            raise KeyError(f"Approval request {request_id} not found")

        request = self._requests[request_id]

        if not request.is_pending():
            raise ValueError(f"Request {request_id} is not pending")

        request.status = ApprovalStatus.CANCELLED
        request.decision_at = datetime.now()

        result = ApprovalResult(
            request_id=request_id,
            approved=False,
            status=ApprovalStatus.CANCELLED,
            decided_at=datetime.now(),
            reason="Request cancelled",
        )

        self._results[request_id] = result
        self._events[request_id].set()
        self._log_audit("request_cancelled", request)

        logger.info(f"Approval request {request_id} cancelled")

        return result

    def _make_decision(
        self,
        request_id: str,
        approved: bool,
        approver: Optional[str] = None,
        reason: Optional[str] = None,
        conditions: Optional[Dict[str, Any]] = None,
    ) -> ApprovalResult:
        """Make a decision on a request."""
        if request_id not in self._requests:
            raise KeyError(f"Approval request {request_id} not found")

        request = self._requests[request_id]

        if not request.is_pending():
            raise ValueError(
                f"Request {request_id} is not pending (status: {request.status.value})"
            )

        # Update request
        decided_at = datetime.now()
        request.status = ApprovalStatus.APPROVED if approved else ApprovalStatus.REJECTED
        request.approver = approver
        request.decision_at = decided_at
        request.decision_reason = reason

        # Create result
        result = ApprovalResult(
            request_id=request_id,
            approved=approved,
            status=request.status,
            approver=approver,
            reason=reason,
            decided_at=decided_at,
            conditions=conditions or {},
        )

        self._results[request_id] = result
        self._events[request_id].set()

        # Audit log
        self._log_audit(
            "request_approved" if approved else "request_rejected",
            request
        )

        logger.info(
            f"Approval request {request_id} "
            f"{'approved' if approved else 'rejected'} by {approver}"
        )

        return result

    def _handle_timeout(self, request: ApprovalRequest) -> ApprovalResult:
        """Handle request timeout."""
        request.status = ApprovalStatus.TIMEOUT
        request.decision_at = datetime.now()

        result = ApprovalResult(
            request_id=request.request_id,
            approved=False,
            status=ApprovalStatus.TIMEOUT,
            decided_at=datetime.now(),
            reason="Request timed out",
        )

        self._results[request.request_id] = result
        self._events[request.request_id].set()
        self._log_audit("request_timeout", request)

        logger.warning(f"Approval request {request.request_id} timed out")

        if not self.auto_reject_on_timeout:
            raise ApprovalTimeoutError(request.request_id, request.timeout or 0)

        return result

    def get_request(self, request_id: str) -> Optional[ApprovalRequest]:
        """Get a request by ID."""
        return self._requests.get(request_id)

    def get_pending_requests(self) -> List[ApprovalRequest]:
        """Get all pending requests."""
        return [
            req for req in self._requests.values()
            if req.is_pending()
        ]

    def get_requests_by_action(self, action: str) -> List[ApprovalRequest]:
        """Get all requests for a specific action."""
        return [
            req for req in self._requests.values()
            if req.action == action
        ]

    def get_requests_by_risk_level(
        self,
        risk_level: RiskLevel
    ) -> List[ApprovalRequest]:
        """Get all requests with a specific risk level."""
        return [
            req for req in self._requests.values()
            if req.risk_level == risk_level
        ]

    def get_audit_log(self) -> List[Dict[str, Any]]:
        """Get the audit log."""
        return self._audit_log.copy()

    def export_audit_log(self, format: str = "json") -> str:
        """Export audit log in specified format.

        Args:
            format: Output format ("json", "csv", "markdown")

        Returns:
            Formatted audit log string
        """
        if format == "json":
            return json.dumps(self._audit_log, indent=2, default=str)

        elif format == "csv":
            if not self._audit_log:
                return "timestamp,event,request_id,action,risk_level,status,approver"

            lines = ["timestamp,event,request_id,action,risk_level,status,approver"]
            for entry in self._audit_log:
                lines.append(
                    f"{entry['timestamp']},{entry['event']},"
                    f"{entry['request_id']},{entry.get('action', '')},"
                    f"{entry.get('risk_level', '')},{entry.get('status', '')},"
                    f"{entry.get('approver', '')}"
                )
            return "\n".join(lines)

        elif format == "markdown":
            lines = [
                "# Approval Audit Log",
                "",
                "| Timestamp | Event | Request ID | Action | Risk | Status | Approver |",
                "|-----------|-------|------------|--------|------|--------|----------|",
            ]
            for entry in self._audit_log:
                lines.append(
                    f"| {entry['timestamp']} | {entry['event']} | "
                    f"{entry['request_id'][:8]}... | {entry.get('action', '-')} | "
                    f"{entry.get('risk_level', '-')} | {entry.get('status', '-')} | "
                    f"{entry.get('approver', '-')} |"
                )
            return "\n".join(lines)

        else:
            raise ValueError(f"Unsupported format: {format}")

    def _log_audit(self, event: str, request: ApprovalRequest) -> None:
        """Add entry to audit log."""
        entry = {
            "timestamp": datetime.now().isoformat(),
            "event": event,
            "request_id": request.request_id,
            "action": request.action,
            "risk_level": request.risk_level.value,
            "status": request.status.value,
            "approver": request.approver,
            "context": request.context,
        }
        self._audit_log.append(entry)

    def clear(self) -> None:
        """Clear all requests and audit log."""
        self._requests.clear()
        self._events.clear()
        self._results.clear()
        self._audit_log.clear()

    def get_statistics(self) -> Dict[str, Any]:
        """Get approval statistics.

        Returns:
            Dictionary with approval statistics
        """
        total = len(self._requests)
        by_status = {}
        by_risk = {}
        avg_decision_time = []

        for request in self._requests.values():
            # Count by status
            status = request.status.value
            by_status[status] = by_status.get(status, 0) + 1

            # Count by risk
            risk = request.risk_level.value
            by_risk[risk] = by_risk.get(risk, 0) + 1

            # Decision time
            if request.decision_at and request.created_at:
                delta = (request.decision_at - request.created_at).total_seconds()
                avg_decision_time.append(delta)

        return {
            "total_requests": total,
            "by_status": by_status,
            "by_risk_level": by_risk,
            "pending_count": len(self.get_pending_requests()),
            "avg_decision_time_seconds": (
                sum(avg_decision_time) / len(avg_decision_time)
                if avg_decision_time else 0.0
            ),
            "approval_rate": (
                by_status.get("approved", 0) / total if total > 0 else 0.0
            ),
        }


class BatchApprovalHandler:
    """Handler for batch approval operations.

    Allows approving or rejecting multiple requests at once.

    Example:
        >>> batch_handler = BatchApprovalHandler(handler)
        >>> results = batch_handler.approve_all(
        ...     request_ids=["req1", "req2", "req3"],
        ...     approver="admin",
        ...     reason="Batch approved after review"
        ... )
    """

    def __init__(self, handler: ApprovalHandler):
        """Initialize batch handler.

        Args:
            handler: ApprovalHandler instance to use
        """
        self.handler = handler

    def approve_all(
        self,
        request_ids: List[str],
        approver: Optional[str] = None,
        reason: Optional[str] = None,
    ) -> List[ApprovalResult]:
        """Approve multiple requests.

        Args:
            request_ids: List of request IDs to approve
            approver: Who is approving
            reason: Reason for approval

        Returns:
            List of ApprovalResult objects
        """
        results = []
        for request_id in request_ids:
            try:
                result = self.handler.approve(
                    request_id=request_id,
                    approver=approver,
                    reason=reason,
                )
                results.append(result)
            except (KeyError, ValueError) as e:
                logger.warning(f"Could not approve {request_id}: {e}")
        return results

    def reject_all(
        self,
        request_ids: List[str],
        approver: Optional[str] = None,
        reason: Optional[str] = None,
    ) -> List[ApprovalResult]:
        """Reject multiple requests.

        Args:
            request_ids: List of request IDs to reject
            approver: Who is rejecting
            reason: Reason for rejection

        Returns:
            List of ApprovalResult objects
        """
        results = []
        for request_id in request_ids:
            try:
                result = self.handler.reject(
                    request_id=request_id,
                    approver=approver,
                    reason=reason,
                )
                results.append(result)
            except (KeyError, ValueError) as e:
                logger.warning(f"Could not reject {request_id}: {e}")
        return results

    def approve_by_action(
        self,
        action: str,
        approver: Optional[str] = None,
        reason: Optional[str] = None,
    ) -> List[ApprovalResult]:
        """Approve all pending requests for a specific action.

        Args:
            action: Action name to filter by
            approver: Who is approving
            reason: Reason for approval

        Returns:
            List of ApprovalResult objects
        """
        requests = self.handler.get_requests_by_action(action)
        pending_ids = [r.request_id for r in requests if r.is_pending()]
        return self.approve_all(pending_ids, approver, reason)

    def approve_by_risk_level(
        self,
        risk_level: RiskLevel,
        approver: Optional[str] = None,
        reason: Optional[str] = None,
    ) -> List[ApprovalResult]:
        """Approve all pending requests with a specific risk level.

        Args:
            risk_level: Risk level to filter by
            approver: Who is approving
            reason: Reason for approval

        Returns:
            List of ApprovalResult objects
        """
        requests = self.handler.get_requests_by_risk_level(risk_level)
        pending_ids = [r.request_id for r in requests if r.is_pending()]
        return self.approve_all(pending_ids, approver, reason)
