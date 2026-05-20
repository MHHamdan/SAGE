"""Guarded executor for policy-enforced plan execution.

Combines validation, policy checking, and controlled execution
to safely run agent plans.
"""

import logging
import time
from typing import Optional, List, Dict, Any, Callable
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

from ..planning.schemas import Plan, PlanStep, StepStatus, PlanStatus
from .constraint_checker import ConstraintChecker
from .plan_validator import PlanValidator, ValidationResult
from .policies import Policy, PolicyDecision, PolicyResult, create_default_policy

logger = logging.getLogger(__name__)


class ExecutionStatus(Enum):
    """Status of guarded execution."""
    SUCCESS = "success"
    BLOCKED = "blocked"
    FAILED = "failed"
    PARTIAL = "partial"
    PENDING_APPROVAL = "pending_approval"


@dataclass
class StepExecutionResult:
    """Result of executing a single step."""
    step_id: str
    step_name: str
    status: ExecutionStatus
    result: Any = None
    error: Optional[str] = None
    policy_result: Optional[PolicyResult] = None
    duration_ms: float = 0.0
    cost: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "step_id": self.step_id,
            "step_name": self.step_name,
            "status": self.status.value,
            "result": str(self.result)[:500] if self.result else None,
            "error": self.error,
            "duration_ms": self.duration_ms,
            "cost": self.cost,
        }


@dataclass
class ExecutionResult:
    """Result of executing an entire plan."""
    plan_id: str
    status: ExecutionStatus
    step_results: List[StepExecutionResult] = field(default_factory=list)
    completed_steps: int = 0
    total_steps: int = 0
    blocked_at_step: Optional[str] = None
    blocked_reason: Optional[str] = None
    validation_result: Optional[ValidationResult] = None
    total_duration_ms: float = 0.0
    total_cost: float = 0.0
    approval_required_steps: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "plan_id": self.plan_id,
            "status": self.status.value,
            "completed_steps": self.completed_steps,
            "total_steps": self.total_steps,
            "blocked_at_step": self.blocked_at_step,
            "blocked_reason": self.blocked_reason,
            "total_duration_ms": self.total_duration_ms,
            "total_cost": self.total_cost,
            "approval_required_steps": self.approval_required_steps,
            "step_results": [s.to_dict() for s in self.step_results],
        }


class GuardedExecutor:
    """Executor with policy enforcement and safety guards.

    Provides:
    - Pre-execution validation
    - Per-step policy checking
    - Approval gates for high-risk actions
    - Budget constraints
    - Audit logging

    Example:
        >>> from sage.verification import GuardedExecutor, Policy
        >>>
        >>> executor = GuardedExecutor()
        >>> result = executor.execute(plan, tool_registry)
        >>>
        >>> if result.status == ExecutionStatus.BLOCKED:
        ...     print(f"Execution blocked: {result.blocked_reason}")
    """

    def __init__(
        self,
        policy: Optional[Policy] = None,
        validator: Optional[PlanValidator] = None,
        budget_limit: Optional[float] = None,
        require_validation: bool = True,
        approval_callback: Optional[Callable[[PlanStep], bool]] = None,
    ):
        """Initialize the guarded executor.

        Args:
            policy: Execution policy (default policy if None)
            validator: Plan validator instance
            budget_limit: Maximum budget for execution
            require_validation: Require plan validation before execution
            approval_callback: Callback for approval requests (returns True to approve)
        """
        self.policy = policy or create_default_policy()
        self.validator = validator or PlanValidator()
        self.budget_limit = budget_limit
        self.require_validation = require_validation
        self.approval_callback = approval_callback or self._default_approval

        self._execution_log: List[Dict[str, Any]] = []
        self._spent_budget: float = 0.0

    def _default_approval(self, step: PlanStep) -> bool:
        """Default approval callback (always denies).

        In production, this should prompt for human approval.
        """
        logger.warning(f"Approval required for step '{step.name}' but no callback configured")
        return False

    def execute(
        self,
        plan: Plan,
        tools: Dict[str, Callable],
        state: Optional[Dict[str, Any]] = None,
    ) -> ExecutionResult:
        """Execute a plan with guards and policy enforcement.

        Args:
            plan: Plan to execute
            tools: Dictionary of tool name -> callable
            state: Initial execution state

        Returns:
            ExecutionResult with complete execution information
        """
        result = ExecutionResult(
            plan_id=plan.plan_id,
            status=ExecutionStatus.SUCCESS,
            total_steps=len(plan.steps),
        )

        state = state or {}
        self._spent_budget = 0.0

        # Step 1: Validate plan
        if self.require_validation:
            validation = self.validator.validate(plan, state)
            result.validation_result = validation

            if not validation.is_valid:
                result.status = ExecutionStatus.BLOCKED
                result.blocked_reason = f"Validation failed: {len(validation.errors)} errors"
                logger.warning(f"Plan {plan.plan_id} failed validation")
                return result

        # Step 2: Execute steps
        for step in plan.steps:
            step_result = self._execute_step(step, tools, state)
            result.step_results.append(step_result)
            result.total_duration_ms += step_result.duration_ms
            result.total_cost += step_result.cost

            if step_result.status == ExecutionStatus.SUCCESS:
                result.completed_steps += 1
                step.status = StepStatus.COMPLETED

            elif step_result.status == ExecutionStatus.BLOCKED:
                result.status = ExecutionStatus.BLOCKED
                result.blocked_at_step = step.name
                result.blocked_reason = step_result.error or "Blocked by policy"
                step.status = StepStatus.BLOCKED
                break

            elif step_result.status == ExecutionStatus.PENDING_APPROVAL:
                result.approval_required_steps.append(step.name)
                step.status = StepStatus.BLOCKED

                # If no approval, block execution
                if not step_result.result:
                    result.status = ExecutionStatus.BLOCKED
                    result.blocked_at_step = step.name
                    result.blocked_reason = "Approval denied"
                    break

            elif step_result.status == ExecutionStatus.FAILED:
                result.status = ExecutionStatus.FAILED
                result.blocked_at_step = step.name
                result.blocked_reason = step_result.error
                step.status = StepStatus.FAILED
                break

        # Update plan status
        if result.status == ExecutionStatus.SUCCESS:
            plan.status = PlanStatus.COMPLETED
        elif result.completed_steps > 0:
            result.status = ExecutionStatus.PARTIAL

        self._log_execution(result)
        return result

    def _execute_step(
        self,
        step: PlanStep,
        tools: Dict[str, Callable],
        state: Dict[str, Any],
    ) -> StepExecutionResult:
        """Execute a single step with guards.

        Args:
            step: Step to execute
            tools: Available tools
            state: Current execution state

        Returns:
            StepExecutionResult
        """
        result = StepExecutionResult(
            step_id=step.step_id,
            step_name=step.name,
            status=ExecutionStatus.SUCCESS,
        )

        start_time = time.time()

        # Check policy
        policy_context = {
            "action": step.action,
            "parameters": step.parameters,
            "risk_score": step.risk_score,
            "step_name": step.name,
        }
        policy_result = self.policy.evaluate(policy_context)
        result.policy_result = policy_result

        if policy_result.denied:
            result.status = ExecutionStatus.BLOCKED
            result.error = f"Blocked by policy: {policy_result.reason}"
            logger.warning(f"Step '{step.name}' blocked by policy: {policy_result.rule_name}")
            return result

        # Check approval requirement
        if policy_result.needs_approval or step.requires_approval:
            logger.info(f"Step '{step.name}' requires approval")
            approved = self.approval_callback(step)

            if not approved:
                result.status = ExecutionStatus.PENDING_APPROVAL
                result.result = False
                result.error = "Approval denied"
                return result

        # Check budget
        if self.budget_limit is not None:
            estimated_cost = step.estimated_cost
            if self._spent_budget + estimated_cost > self.budget_limit:
                result.status = ExecutionStatus.BLOCKED
                result.error = f"Budget exceeded: ${self._spent_budget + estimated_cost:.2f} > ${self.budget_limit:.2f}"
                return result

        # Execute the tool
        tool = tools.get(step.action)
        if not tool:
            # Try to find partial match
            for tool_name, tool_fn in tools.items():
                if tool_name.lower() in step.action.lower() or step.action.lower() in tool_name.lower():
                    tool = tool_fn
                    break

        if not tool:
            result.status = ExecutionStatus.FAILED
            result.error = f"Tool not found: {step.action}"
            return result

        try:
            # Execute
            tool_result = tool(**step.parameters) if step.parameters else tool()
            result.result = tool_result
            result.status = ExecutionStatus.SUCCESS

            # Update spent budget
            result.cost = step.estimated_cost
            self._spent_budget += result.cost

        except Exception as e:
            result.status = ExecutionStatus.FAILED
            result.error = str(e)
            logger.error(f"Step '{step.name}' failed: {e}")

        result.duration_ms = (time.time() - start_time) * 1000
        return result

    def _log_execution(self, result: ExecutionResult):
        """Log execution for audit."""
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "plan_id": result.plan_id,
            "status": result.status.value,
            "completed_steps": result.completed_steps,
            "total_steps": result.total_steps,
            "total_cost": result.total_cost,
        }
        self._execution_log.append(log_entry)
        logger.info(f"Execution logged: {result.plan_id} - {result.status.value}")

    def get_execution_log(self) -> List[Dict[str, Any]]:
        """Get the execution audit log.

        Returns:
            List of execution log entries
        """
        return self._execution_log.copy()

    def set_budget(self, budget: float):
        """Set the execution budget.

        Args:
            budget: Maximum budget in USD
        """
        self.budget_limit = budget

    def set_approval_callback(self, callback: Callable[[PlanStep], bool]):
        """Set the approval callback.

        Args:
            callback: Function that takes a step and returns True to approve
        """
        self.approval_callback = callback


def create_executor(
    policy_path: Optional[str] = None,
    budget_limit: Optional[float] = None,
    **kwargs,
) -> GuardedExecutor:
    """Factory function to create a guarded executor.

    Args:
        policy_path: Path to policy YAML file
        budget_limit: Maximum budget
        **kwargs: Additional arguments

    Returns:
        Configured GuardedExecutor
    """
    policy = None
    if policy_path:
        policy = Policy.from_yaml(policy_path)

    return GuardedExecutor(
        policy=policy,
        budget_limit=budget_limit,
        **kwargs,
    )
