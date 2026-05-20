"""Constraint checker for pre/post condition verification.

Checks that plan steps satisfy their specified conditions
before and after execution.
"""

import logging
from typing import Optional, List, Dict, Any, Callable
from dataclasses import dataclass, field
from enum import Enum

from ..planning.schemas import Plan, PlanStep, Condition, ConditionType

logger = logging.getLogger(__name__)


class ConstraintStatus(Enum):
    """Status of a constraint check."""
    SATISFIED = "satisfied"
    VIOLATED = "violated"
    UNKNOWN = "unknown"
    SKIPPED = "skipped"


@dataclass
class ConstraintResult:
    """Result of checking a constraint."""
    condition: Condition
    status: ConstraintStatus
    message: str = ""
    details: Dict[str, Any] = field(default_factory=dict)

    @property
    def is_satisfied(self) -> bool:
        return self.status == ConstraintStatus.SATISFIED

    @property
    def is_violated(self) -> bool:
        return self.status == ConstraintStatus.VIOLATED


@dataclass
class CheckResult:
    """Result of checking all constraints for a step."""
    step_id: str
    step_name: str
    precondition_results: List[ConstraintResult] = field(default_factory=list)
    postcondition_results: List[ConstraintResult] = field(default_factory=list)
    all_satisfied: bool = True
    violations: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "step_id": self.step_id,
            "step_name": self.step_name,
            "all_satisfied": self.all_satisfied,
            "num_violations": len(self.violations),
            "violations": self.violations,
        }


class ConstraintChecker:
    """Checker for plan step constraints.

    Verifies that preconditions are met before step execution
    and postconditions are satisfied after.

    Example:
        >>> checker = ConstraintChecker()
        >>> checker.register_check("file_exists", lambda p: os.path.exists(p["path"]))
        >>>
        >>> result = checker.check_preconditions(step, state)
        >>> if not result.all_satisfied:
        ...     print(f"Preconditions violated: {result.violations}")
    """

    def __init__(self):
        """Initialize the constraint checker."""
        self._checks: Dict[str, Callable] = {}
        self._register_default_checks()

    def _register_default_checks(self):
        """Register default constraint check functions."""
        # Always true/false for testing
        self._checks["always_true"] = lambda params, state: True
        self._checks["always_false"] = lambda params, state: False

        # Basic checks
        self._checks["state_has_key"] = lambda params, state: params.get("key") in state
        self._checks["state_value_equals"] = lambda params, state: (
            state.get(params.get("key")) == params.get("value")
        )
        self._checks["state_value_not_empty"] = lambda params, state: (
            bool(state.get(params.get("key")))
        )

        # Budget check
        self._checks["within_budget"] = lambda params, state: (
            state.get("spent", 0) <= params.get("budget", float("inf"))
        )

        # Risk check
        self._checks["risk_acceptable"] = lambda params, state: (
            params.get("risk", 0) <= params.get("threshold", 0.7)
        )

    def register_check(
        self,
        name: str,
        check_fn: Callable[[Dict[str, Any], Dict[str, Any]], bool],
    ):
        """Register a custom constraint check function.

        Args:
            name: Name of the check
            check_fn: Function that takes (params, state) and returns bool
        """
        self._checks[name] = check_fn
        logger.debug(f"Registered constraint check: {name}")

    def check_condition(
        self,
        condition: Condition,
        state: Dict[str, Any],
    ) -> ConstraintResult:
        """Check a single condition.

        Args:
            condition: Condition to check
            state: Current execution state

        Returns:
            ConstraintResult
        """
        if condition.check_fn and condition.check_fn in self._checks:
            check_fn = self._checks[condition.check_fn]
            try:
                is_satisfied = check_fn(condition.parameters, state)
                return ConstraintResult(
                    condition=condition,
                    status=ConstraintStatus.SATISFIED if is_satisfied else ConstraintStatus.VIOLATED,
                    message=f"Check '{condition.check_fn}' {'passed' if is_satisfied else 'failed'}",
                )
            except Exception as e:
                return ConstraintResult(
                    condition=condition,
                    status=ConstraintStatus.VIOLATED,
                    message=f"Check '{condition.check_fn}' raised exception: {e}",
                )
        else:
            # No check function - assume satisfied but log warning
            logger.warning(f"No check function for condition: {condition.name}")
            return ConstraintResult(
                condition=condition,
                status=ConstraintStatus.UNKNOWN,
                message=f"No check function registered for '{condition.check_fn}'",
            )

    def check_preconditions(
        self,
        step: PlanStep,
        state: Dict[str, Any],
    ) -> CheckResult:
        """Check all preconditions for a step.

        Args:
            step: Plan step to check
            state: Current execution state

        Returns:
            CheckResult with all precondition results
        """
        result = CheckResult(
            step_id=step.step_id,
            step_name=step.name,
        )

        for condition in step.preconditions:
            check_result = self.check_condition(condition, state)
            result.precondition_results.append(check_result)

            if check_result.is_violated:
                result.all_satisfied = False
                result.violations.append(
                    f"Precondition '{condition.name}' violated: {check_result.message}"
                )

        return result

    def check_postconditions(
        self,
        step: PlanStep,
        state: Dict[str, Any],
    ) -> CheckResult:
        """Check all postconditions for a step.

        Args:
            step: Plan step to check
            state: Current execution state after step execution

        Returns:
            CheckResult with all postcondition results
        """
        result = CheckResult(
            step_id=step.step_id,
            step_name=step.name,
        )

        for condition in step.postconditions:
            check_result = self.check_condition(condition, state)
            result.postcondition_results.append(check_result)

            if check_result.is_violated:
                result.all_satisfied = False
                result.violations.append(
                    f"Postcondition '{condition.name}' violated: {check_result.message}"
                )

        return result

    def check_plan(
        self,
        plan: Plan,
        initial_state: Optional[Dict[str, Any]] = None,
    ) -> List[CheckResult]:
        """Check all preconditions for all steps in a plan.

        Args:
            plan: Plan to check
            initial_state: Initial execution state

        Returns:
            List of CheckResults for each step
        """
        state = initial_state or {}
        results = []

        for step in plan.steps:
            result = self.check_preconditions(step, state)
            results.append(result)

        return results

    def get_available_checks(self) -> List[str]:
        """Get list of available check functions.

        Returns:
            List of check function names
        """
        return list(self._checks.keys())
