"""
Human Oversight Module

Provides human-in-the-loop oversight mechanisms for agentic AI systems,
including approval flows, escalation handling, and audit logging.

This module implements the human oversight patterns described in Section IX
of the IEEE TAI paper on Agentic AI safety.

Key Components:
    - ApprovalHandler: Manages human approval requests for sensitive operations
    - ApprovalRequest: Represents a pending approval request
    - ApprovalStatus: Enum for approval states
    - EscalationHandler: Handles escalation to human supervisors
    - AuditLogger: Records all human oversight interactions

Example:
    >>> from sage.human_oversight import ApprovalHandler, ApprovalRequest
    >>>
    >>> handler = ApprovalHandler(default_timeout=300)
    >>> request = handler.create_request(
    ...     action="delete_database",
    ...     context={"database": "production", "table": "users"},
    ...     risk_level="high"
    ... )
    >>>
    >>> result = await handler.wait_for_approval(request.request_id)
    >>> if result.approved:
    ...     execute_action()
"""

from .approval_flow import (
    ApprovalCallback,
    ApprovalHandler,
    ApprovalRequest,
    ApprovalResult,
    ApprovalStatus,
    ApprovalTimeoutError,
    RiskLevel,
)
from .audit import (
    AuditEntry,
    AuditEventType,
    AuditLogger,
)
from .escalation import (
    EscalationHandler,
    EscalationLevel,
    EscalationPolicy,
    EscalationRequest,
)

__all__ = [
    # Approval Flow
    "ApprovalHandler",
    "ApprovalRequest",
    "ApprovalResult",
    "ApprovalStatus",
    "RiskLevel",
    "ApprovalCallback",
    "ApprovalTimeoutError",
    # Escalation
    "EscalationHandler",
    "EscalationRequest",
    "EscalationLevel",
    "EscalationPolicy",
    # Audit
    "AuditLogger",
    "AuditEntry",
    "AuditEventType",
]
