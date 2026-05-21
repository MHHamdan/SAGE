"""Schemas for planning module.

Defines the core data structures for plans, steps, conditions,
and plan validation.
"""

from typing import Optional, List, Dict, Any, Union
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime
import uuid


class StepStatus(Enum):
    """Status of a plan step."""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"
    BLOCKED = "blocked"


class PlanStatus(Enum):
    """Status of an entire plan."""
    DRAFT = "draft"
    VALIDATED = "validated"
    EXECUTING = "executing"
    COMPLETED = "completed"
    FAILED = "failed"
    REPLANNING = "replanning"


class ConditionType(Enum):
    """Types of conditions for steps."""
    PRECONDITION = "precondition"
    POSTCONDITION = "postcondition"
    INVARIANT = "invariant"


@dataclass
class Condition:
    """A condition that must be satisfied.

    Used for preconditions, postconditions, and invariants.
    """
    name: str
    description: str
    condition_type: ConditionType
    check_fn: Optional[str] = None  # Name of function to check
    parameters: Dict[str, Any] = field(default_factory=dict)
    is_satisfied: Optional[bool] = None
    failure_message: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "description": self.description,
            "condition_type": self.condition_type.value,
            "check_fn": self.check_fn,
            "parameters": self.parameters,
            "is_satisfied": self.is_satisfied,
            "failure_message": self.failure_message,
        }


@dataclass
class PlanStep:
    """A single step in a plan.

    Represents an atomic action to be taken by the agent.
    """
    step_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    name: str = ""
    description: str = ""
    action: str = ""  # Tool or action to execute
    parameters: Dict[str, Any] = field(default_factory=dict)

    # Ordering and dependencies
    order: int = 0
    dependencies: List[str] = field(default_factory=list)  # step_ids

    # Conditions
    preconditions: List[Condition] = field(default_factory=list)
    postconditions: List[Condition] = field(default_factory=list)

    # Execution state
    status: StepStatus = StepStatus.PENDING
    result: Optional[Any] = None
    error: Optional[str] = None
    start_time: Optional[str] = None
    end_time: Optional[str] = None

    # Cost and risk estimates
    estimated_cost: float = 0.0
    risk_score: float = 0.0  # 0.0 to 1.0
    requires_approval: bool = False

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "step_id": self.step_id,
            "name": self.name,
            "description": self.description,
            "action": self.action,
            "parameters": self.parameters,
            "order": self.order,
            "dependencies": self.dependencies,
            "preconditions": [c.to_dict() for c in self.preconditions],
            "postconditions": [c.to_dict() for c in self.postconditions],
            "status": self.status.value,
            "result": str(self.result)[:500] if self.result else None,
            "error": self.error,
            "estimated_cost": self.estimated_cost,
            "risk_score": self.risk_score,
            "requires_approval": self.requires_approval,
        }


@dataclass
class Plan:
    """A complete plan with multiple steps.

    Represents the agent's strategy for accomplishing a task.
    """
    plan_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    name: str = ""
    goal: str = ""
    description: str = ""

    # Steps
    steps: List[PlanStep] = field(default_factory=list)

    # Status
    status: PlanStatus = PlanStatus.DRAFT
    current_step_idx: int = 0

    # Metadata
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    updated_at: Optional[str] = None
    planner_type: str = ""

    # Cost and risk
    total_estimated_cost: float = 0.0
    total_risk_score: float = 0.0
    max_steps_allowed: int = 10

    # Replanning
    parent_plan_id: Optional[str] = None
    replan_count: int = 0
    replan_reason: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "plan_id": self.plan_id,
            "name": self.name,
            "goal": self.goal,
            "description": self.description,
            "steps": [s.to_dict() for s in self.steps],
            "status": self.status.value,
            "current_step_idx": self.current_step_idx,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "planner_type": self.planner_type,
            "total_estimated_cost": self.total_estimated_cost,
            "total_risk_score": self.total_risk_score,
            "replan_count": self.replan_count,
        }

    @property
    def num_steps(self) -> int:
        """Number of steps in the plan."""
        return len(self.steps)

    @property
    def completed_steps(self) -> int:
        """Number of completed steps."""
        return sum(1 for s in self.steps if s.status == StepStatus.COMPLETED)

    @property
    def progress(self) -> float:
        """Progress as fraction (0.0 to 1.0)."""
        if not self.steps:
            return 0.0
        return self.completed_steps / len(self.steps)

    @property
    def current_step(self) -> Optional[PlanStep]:
        """Get the current step."""
        if 0 <= self.current_step_idx < len(self.steps):
            return self.steps[self.current_step_idx]
        return None

    def get_step(self, step_id: str) -> Optional[PlanStep]:
        """Get step by ID."""
        for step in self.steps:
            if step.step_id == step_id:
                return step
        return None

    def add_step(self, step: PlanStep):
        """Add a step to the plan."""
        step.order = len(self.steps)
        self.steps.append(step)
        self._update_estimates()

    def _update_estimates(self):
        """Update total cost and risk estimates."""
        self.total_estimated_cost = sum(s.estimated_cost for s in self.steps)
        if self.steps:
            self.total_risk_score = max(s.risk_score for s in self.steps)
        self.updated_at = datetime.now().isoformat()

    def validate(self) -> List[str]:
        """Validate plan structure.

        Returns:
            List of validation error messages (empty if valid)
        """
        errors = []

        if not self.goal:
            errors.append("Plan has no goal")

        if not self.steps:
            errors.append("Plan has no steps")

        if len(self.steps) > self.max_steps_allowed:
            errors.append(f"Plan exceeds max steps ({len(self.steps)} > {self.max_steps_allowed})")

        # Check for dependency cycles
        step_ids = {s.step_id for s in self.steps}
        for step in self.steps:
            for dep in step.dependencies:
                if dep not in step_ids:
                    errors.append(f"Step {step.step_id} has unknown dependency: {dep}")

        # Check order consistency
        orders = [s.order for s in self.steps]
        if sorted(orders) != orders:
            errors.append("Step order is not sequential")

        return errors


@dataclass
class PlanningContext:
    """Context for planning operations."""
    task: str
    available_tools: List[str] = field(default_factory=list)
    constraints: List[str] = field(default_factory=list)
    preferences: Dict[str, Any] = field(default_factory=dict)
    history: List[Dict[str, Any]] = field(default_factory=list)
    budget_limit: Optional[float] = None
    risk_tolerance: float = 0.5


@dataclass
class PlanningResult:
    """Result from a planning operation."""
    plan: Optional[Plan] = None
    success: bool = False
    error: Optional[str] = None
    planning_time_ms: float = 0.0
    tokens_used: int = 0


def create_step(
    name: str,
    action: str,
    description: str = "",
    parameters: Optional[Dict[str, Any]] = None,
    risk_score: float = 0.1,
    requires_approval: bool = False,
) -> PlanStep:
    """Factory function to create a plan step.

    Args:
        name: Step name
        action: Action/tool to execute
        description: Step description
        parameters: Action parameters
        risk_score: Risk score (0.0 to 1.0)
        requires_approval: Whether step needs human approval

    Returns:
        PlanStep instance
    """
    return PlanStep(
        name=name,
        action=action,
        description=description,
        parameters=parameters or {},
        risk_score=risk_score,
        requires_approval=requires_approval,
    )


def create_plan(
    goal: str,
    steps: Optional[List[PlanStep]] = None,
    name: str = "",
    planner_type: str = "manual",
) -> Plan:
    """Factory function to create a plan.

    Args:
        goal: The goal of the plan
        steps: List of steps
        name: Plan name
        planner_type: Type of planner used

    Returns:
        Plan instance
    """
    plan = Plan(
        name=name or f"Plan for: {goal[:50]}",
        goal=goal,
        planner_type=planner_type,
    )

    if steps:
        for i, step in enumerate(steps):
            step.order = i
            plan.steps.append(step)
        plan._update_estimates()

    return plan
