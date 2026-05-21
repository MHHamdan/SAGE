"""Verification module for plan validation and guarded execution.

Provides:
- ConstraintChecker: Pre/post condition verification
- PlanSimulator: Dry-run tool execution
- PlanValidator: Comprehensive plan validation
- GuardedExecutor: Policy-enforced execution
- Policy DSL: Policy-as-code definitions

Usage:
    >>> from sage.verification import GuardedExecutor, Policy
    >>> from sage.planning import Plan
    >>>
    >>> policy = Policy.from_yaml("security_policy.yaml")
    >>> executor = GuardedExecutor(policy)
    >>>
    >>> result = executor.execute(plan, tools)
    >>> if result.blocked:
    ...     print(f"Blocked by policy: {result.reason}")
"""

from .constraint_checker import ConstraintChecker, ConstraintResult
from .guarded_executor import ExecutionResult, GuardedExecutor
from .plan_validator import PlanValidator, ValidationResult
from .policies import Policy, PolicyDecision, PolicyRule
from .simulator import PlanSimulator, SimulationResult

__all__ = [
    "ConstraintChecker",
    "ConstraintResult",
    "PlanSimulator",
    "SimulationResult",
    "PlanValidator",
    "ValidationResult",
    "GuardedExecutor",
    "ExecutionResult",
    "Policy",
    "PolicyRule",
    "PolicyDecision",
]
