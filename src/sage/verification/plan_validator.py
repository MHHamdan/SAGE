"""Plan validator for comprehensive plan verification.

Combines constraint checking, simulation, and policy verification
to validate plans before execution.
"""

import logging
from typing import Optional, List, Dict, Any
from dataclasses import dataclass, field
from enum import Enum

from ..planning.schemas import Plan, PlanStep
from .constraint_checker import ConstraintChecker, CheckResult
from .simulator import PlanSimulator, SimulationResult

logger = logging.getLogger(__name__)


class ValidationSeverity(Enum):
    """Severity of validation issues."""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class ValidationIssue:
    """A single validation issue."""
    severity: ValidationSeverity
    category: str
    message: str
    step_id: Optional[str] = None
    details: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "severity": self.severity.value,
            "category": self.category,
            "message": self.message,
            "step_id": self.step_id,
            "details": self.details,
        }


@dataclass
class ValidationResult:
    """Result of plan validation."""
    plan_id: str
    is_valid: bool
    issues: List[ValidationIssue] = field(default_factory=list)
    constraint_results: List[CheckResult] = field(default_factory=list)
    simulation_result: Optional[SimulationResult] = None
    estimated_cost: float = 0.0
    estimated_duration_ms: float = 0.0
    risk_score: float = 0.0

    @property
    def errors(self) -> List[ValidationIssue]:
        return [i for i in self.issues if i.severity in [ValidationSeverity.ERROR, ValidationSeverity.CRITICAL]]

    @property
    def warnings(self) -> List[ValidationIssue]:
        return [i for i in self.issues if i.severity == ValidationSeverity.WARNING]

    @property
    def has_critical(self) -> bool:
        return any(i.severity == ValidationSeverity.CRITICAL for i in self.issues)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "plan_id": self.plan_id,
            "is_valid": self.is_valid,
            "num_issues": len(self.issues),
            "num_errors": len(self.errors),
            "num_warnings": len(self.warnings),
            "issues": [i.to_dict() for i in self.issues],
            "estimated_cost": self.estimated_cost,
            "estimated_duration_ms": self.estimated_duration_ms,
            "risk_score": self.risk_score,
        }


class PlanValidator:
    """Comprehensive plan validator.

    Performs multiple validation checks:
    1. Structural validation (plan format, dependencies)
    2. Constraint checking (pre/postconditions)
    3. Simulation (dry-run, cost estimation)
    4. Policy compliance
    5. Risk assessment

    Example:
        >>> validator = PlanValidator()
        >>> result = validator.validate(plan)
        >>> if not result.is_valid:
        ...     for issue in result.errors:
        ...         print(f"Error: {issue.message}")
    """

    def __init__(
        self,
        constraint_checker: Optional[ConstraintChecker] = None,
        simulator: Optional[PlanSimulator] = None,
        max_steps: int = 20,
        max_cost: float = 100.0,
        max_risk: float = 0.8,
    ):
        """Initialize the validator.

        Args:
            constraint_checker: Constraint checker instance
            simulator: Plan simulator instance
            max_steps: Maximum allowed steps
            max_cost: Maximum allowed cost
            max_risk: Maximum allowed risk score
        """
        self.constraint_checker = constraint_checker or ConstraintChecker()
        self.simulator = simulator or PlanSimulator()
        self.max_steps = max_steps
        self.max_cost = max_cost
        self.max_risk = max_risk

    def validate(
        self,
        plan: Plan,
        state: Optional[Dict[str, Any]] = None,
    ) -> ValidationResult:
        """Validate a plan comprehensively.

        Args:
            plan: Plan to validate
            state: Current execution state

        Returns:
            ValidationResult with all findings
        """
        result = ValidationResult(plan_id=plan.plan_id, is_valid=True)
        state = state or {}

        # 1. Structural validation
        self._validate_structure(plan, result)

        # 2. Constraint checking
        self._validate_constraints(plan, state, result)

        # 3. Simulation
        self._validate_simulation(plan, state, result)

        # 4. Risk assessment
        self._validate_risk(plan, result)

        # 5. Determine overall validity
        result.is_valid = not result.has_critical and len(result.errors) == 0

        logger.info(
            f"Plan validation complete: valid={result.is_valid}, "
            f"issues={len(result.issues)}"
        )

        return result

    def _validate_structure(self, plan: Plan, result: ValidationResult):
        """Validate plan structure."""
        # Check for goal
        if not plan.goal:
            result.issues.append(ValidationIssue(
                severity=ValidationSeverity.ERROR,
                category="structure",
                message="Plan has no goal defined",
            ))

        # Check for steps
        if not plan.steps:
            result.issues.append(ValidationIssue(
                severity=ValidationSeverity.ERROR,
                category="structure",
                message="Plan has no steps",
            ))
            return

        # Check step count
        if len(plan.steps) > self.max_steps:
            result.issues.append(ValidationIssue(
                severity=ValidationSeverity.WARNING,
                category="structure",
                message=f"Plan has {len(plan.steps)} steps, exceeding recommended max of {self.max_steps}",
            ))

        # Check for duplicate step IDs
        step_ids = [s.step_id for s in plan.steps]
        if len(step_ids) != len(set(step_ids)):
            result.issues.append(ValidationIssue(
                severity=ValidationSeverity.ERROR,
                category="structure",
                message="Plan has duplicate step IDs",
            ))

        # Check dependencies
        for step in plan.steps:
            for dep in step.dependencies:
                if dep not in step_ids:
                    result.issues.append(ValidationIssue(
                        severity=ValidationSeverity.ERROR,
                        category="dependencies",
                        message=f"Step '{step.name}' has unknown dependency: {dep}",
                        step_id=step.step_id,
                    ))

        # Check for circular dependencies
        if self._has_circular_dependencies(plan):
            result.issues.append(ValidationIssue(
                severity=ValidationSeverity.CRITICAL,
                category="dependencies",
                message="Plan has circular dependencies",
            ))

    def _validate_constraints(
        self,
        plan: Plan,
        state: Dict[str, Any],
        result: ValidationResult,
    ):
        """Validate plan constraints."""
        constraint_results = self.constraint_checker.check_plan(plan, state)
        result.constraint_results = constraint_results

        for check_result in constraint_results:
            if not check_result.all_satisfied:
                for violation in check_result.violations:
                    result.issues.append(ValidationIssue(
                        severity=ValidationSeverity.WARNING,
                        category="constraints",
                        message=violation,
                        step_id=check_result.step_id,
                    ))

    def _validate_simulation(
        self,
        plan: Plan,
        state: Dict[str, Any],
        result: ValidationResult,
    ):
        """Validate via simulation."""
        sim_result = self.simulator.simulate(plan, state)
        result.simulation_result = sim_result
        result.estimated_cost = sim_result.total_estimated_cost
        result.estimated_duration_ms = sim_result.total_estimated_duration_ms

        # Check cost limit
        if sim_result.total_estimated_cost > self.max_cost:
            result.issues.append(ValidationIssue(
                severity=ValidationSeverity.ERROR,
                category="cost",
                message=f"Estimated cost ${sim_result.total_estimated_cost:.2f} exceeds max ${self.max_cost:.2f}",
            ))

        # Check if simulation would complete
        if not sim_result.would_complete:
            result.issues.append(ValidationIssue(
                severity=ValidationSeverity.ERROR,
                category="simulation",
                message=f"Plan would not complete - blocked at step: {sim_result.blocking_step}",
            ))

        # Add simulation warnings
        for warning in sim_result.all_warnings:
            result.issues.append(ValidationIssue(
                severity=ValidationSeverity.WARNING,
                category="simulation",
                message=warning,
            ))

        # Note side effects
        for side_effect in sim_result.all_side_effects:
            result.issues.append(ValidationIssue(
                severity=ValidationSeverity.INFO,
                category="side_effects",
                message=f"Plan has side effect: {side_effect}",
            ))

    def _validate_risk(self, plan: Plan, result: ValidationResult):
        """Validate plan risk levels."""
        max_step_risk = max((s.risk_score for s in plan.steps), default=0)
        avg_risk = sum(s.risk_score for s in plan.steps) / len(plan.steps) if plan.steps else 0
        result.risk_score = max_step_risk

        if max_step_risk > self.max_risk:
            result.issues.append(ValidationIssue(
                severity=ValidationSeverity.ERROR,
                category="risk",
                message=f"Plan contains step with risk {max_step_risk:.2f} exceeding max {self.max_risk:.2f}",
            ))

        # Find high-risk steps
        for step in plan.steps:
            if step.risk_score >= 0.7:
                result.issues.append(ValidationIssue(
                    severity=ValidationSeverity.WARNING,
                    category="risk",
                    message=f"Step '{step.name}' has high risk score: {step.risk_score:.2f}",
                    step_id=step.step_id,
                ))

    def _has_circular_dependencies(self, plan: Plan) -> bool:
        """Check for circular dependencies in plan."""
        # Build adjacency list
        graph: Dict[str, List[str]] = {s.step_id: s.dependencies for s in plan.steps}

        # DFS for cycle detection
        visited = set()
        rec_stack = set()

        def has_cycle(node: str) -> bool:
            visited.add(node)
            rec_stack.add(node)

            for neighbor in graph.get(node, []):
                if neighbor not in visited:
                    if has_cycle(neighbor):
                        return True
                elif neighbor in rec_stack:
                    return True

            rec_stack.remove(node)
            return False

        for step_id in graph:
            if step_id not in visited:
                if has_cycle(step_id):
                    return True

        return False

    def quick_validate(self, plan: Plan) -> bool:
        """Quick validation check (structure only).

        Args:
            plan: Plan to validate

        Returns:
            True if plan passes basic validation
        """
        errors = plan.validate()
        return len(errors) == 0
