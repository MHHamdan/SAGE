"""Plan simulator for dry-run execution.

Simulates plan execution without actually running tools,
useful for validation and cost estimation.
"""

import logging
import time
from typing import Optional, List, Dict, Any, Callable
from dataclasses import dataclass, field

from ..planning.schemas import Plan, PlanStep, StepStatus

logger = logging.getLogger(__name__)


@dataclass
class StepSimulation:
    """Result of simulating a single step."""
    step_id: str
    step_name: str
    action: str
    would_succeed: bool = True
    estimated_duration_ms: float = 100.0
    estimated_cost: float = 0.0
    side_effects: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    blocked_reason: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "step_id": self.step_id,
            "step_name": self.step_name,
            "action": self.action,
            "would_succeed": self.would_succeed,
            "estimated_duration_ms": self.estimated_duration_ms,
            "estimated_cost": self.estimated_cost,
            "side_effects": self.side_effects,
            "warnings": self.warnings,
            "blocked_reason": self.blocked_reason,
        }


@dataclass
class SimulationResult:
    """Result of simulating an entire plan."""
    plan_id: str
    plan_goal: str
    step_simulations: List[StepSimulation] = field(default_factory=list)
    total_estimated_duration_ms: float = 0.0
    total_estimated_cost: float = 0.0
    would_complete: bool = True
    blocking_step: Optional[str] = None
    all_warnings: List[str] = field(default_factory=list)
    all_side_effects: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "plan_id": self.plan_id,
            "plan_goal": self.plan_goal,
            "num_steps": len(self.step_simulations),
            "would_complete": self.would_complete,
            "blocking_step": self.blocking_step,
            "total_estimated_duration_ms": self.total_estimated_duration_ms,
            "total_estimated_cost": self.total_estimated_cost,
            "warnings": self.all_warnings,
            "side_effects": self.all_side_effects,
        }


class ToolSimulator:
    """Simulates individual tool behavior."""

    def __init__(self):
        """Initialize the tool simulator."""
        self._tool_configs: Dict[str, Dict[str, Any]] = {}
        self._register_default_tools()

    def _register_default_tools(self):
        """Register default tool simulation configs."""
        # Read operations - low risk, fast
        self._tool_configs["read"] = {
            "duration_ms": 50,
            "cost": 0.0,
            "side_effects": [],
            "success_rate": 0.95,
        }
        self._tool_configs["search"] = {
            "duration_ms": 200,
            "cost": 0.001,
            "side_effects": [],
            "success_rate": 0.9,
        }

        # Write operations - higher risk
        self._tool_configs["write"] = {
            "duration_ms": 100,
            "cost": 0.0,
            "side_effects": ["modifies_file"],
            "success_rate": 0.9,
        }
        self._tool_configs["delete"] = {
            "duration_ms": 50,
            "cost": 0.0,
            "side_effects": ["deletes_data"],
            "success_rate": 0.95,
            "warnings": ["Destructive operation"],
        }

        # Execute operations - highest risk
        self._tool_configs["execute"] = {
            "duration_ms": 500,
            "cost": 0.001,
            "side_effects": ["runs_code"],
            "success_rate": 0.8,
            "warnings": ["Executes arbitrary code"],
        }
        self._tool_configs["deploy"] = {
            "duration_ms": 5000,
            "cost": 0.01,
            "side_effects": ["modifies_production"],
            "success_rate": 0.85,
            "warnings": ["Production deployment"],
        }

        # Default for unknown tools
        self._tool_configs["default"] = {
            "duration_ms": 100,
            "cost": 0.0,
            "side_effects": [],
            "success_rate": 0.9,
        }

    def register_tool(
        self,
        tool_name: str,
        config: Dict[str, Any],
    ):
        """Register a tool simulation config.

        Args:
            tool_name: Name of the tool
            config: Configuration dictionary with:
                - duration_ms: Estimated execution time
                - cost: Estimated cost
                - side_effects: List of side effect descriptions
                - success_rate: Expected success rate (0-1)
                - warnings: List of warnings
        """
        self._tool_configs[tool_name] = config

    def simulate_tool(
        self,
        tool_name: str,
        parameters: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Simulate a tool execution.

        Args:
            tool_name: Name of the tool
            parameters: Tool parameters

        Returns:
            Simulation result dictionary
        """
        # Find matching config
        config = self._tool_configs.get(tool_name)
        if not config:
            # Try to find partial match
            for name, cfg in self._tool_configs.items():
                if name in tool_name.lower():
                    config = cfg
                    break

        if not config:
            config = self._tool_configs["default"]

        return {
            "duration_ms": config.get("duration_ms", 100),
            "cost": config.get("cost", 0.0),
            "side_effects": config.get("side_effects", []),
            "warnings": config.get("warnings", []),
            "success_rate": config.get("success_rate", 0.9),
        }


class PlanSimulator:
    """Simulates plan execution without side effects.

    Useful for:
    - Estimating plan costs and duration
    - Identifying potential issues before execution
    - Validating plans against policies

    Example:
        >>> simulator = PlanSimulator()
        >>> result = simulator.simulate(plan)
        >>> print(f"Estimated duration: {result.total_estimated_duration_ms}ms")
        >>> print(f"Warnings: {result.all_warnings}")
    """

    def __init__(
        self,
        tool_simulator: Optional[ToolSimulator] = None,
        blocked_actions: Optional[List[str]] = None,
    ):
        """Initialize the plan simulator.

        Args:
            tool_simulator: Custom tool simulator
            blocked_actions: List of action patterns to block
        """
        self.tool_simulator = tool_simulator or ToolSimulator()
        self.blocked_actions = blocked_actions or []

    def simulate_step(
        self,
        step: PlanStep,
        state: Optional[Dict[str, Any]] = None,
    ) -> StepSimulation:
        """Simulate a single step.

        Args:
            step: Step to simulate
            state: Current simulation state

        Returns:
            StepSimulation result
        """
        # Check if action is blocked
        for blocked in self.blocked_actions:
            if blocked.lower() in step.action.lower():
                return StepSimulation(
                    step_id=step.step_id,
                    step_name=step.name,
                    action=step.action,
                    would_succeed=False,
                    blocked_reason=f"Action '{step.action}' is blocked by policy",
                )

        # Simulate the tool
        tool_result = self.tool_simulator.simulate_tool(
            step.action,
            step.parameters,
        )

        return StepSimulation(
            step_id=step.step_id,
            step_name=step.name,
            action=step.action,
            would_succeed=True,
            estimated_duration_ms=tool_result["duration_ms"],
            estimated_cost=tool_result["cost"],
            side_effects=tool_result["side_effects"],
            warnings=tool_result.get("warnings", []),
        )

    def simulate(
        self,
        plan: Plan,
        initial_state: Optional[Dict[str, Any]] = None,
    ) -> SimulationResult:
        """Simulate an entire plan.

        Args:
            plan: Plan to simulate
            initial_state: Initial state

        Returns:
            SimulationResult with aggregated results
        """
        result = SimulationResult(
            plan_id=plan.plan_id,
            plan_goal=plan.goal,
        )

        state = initial_state or {}

        for step in plan.steps:
            step_sim = self.simulate_step(step, state)
            result.step_simulations.append(step_sim)

            result.total_estimated_duration_ms += step_sim.estimated_duration_ms
            result.total_estimated_cost += step_sim.estimated_cost
            result.all_warnings.extend(step_sim.warnings)
            result.all_side_effects.extend(step_sim.side_effects)

            if not step_sim.would_succeed:
                result.would_complete = False
                result.blocking_step = step.name
                break

        return result

    def add_blocked_action(self, pattern: str):
        """Add an action pattern to block.

        Args:
            pattern: Action pattern to block
        """
        self.blocked_actions.append(pattern)

    def estimate_cost(self, plan: Plan) -> float:
        """Quick cost estimation for a plan.

        Args:
            plan: Plan to estimate

        Returns:
            Total estimated cost
        """
        result = self.simulate(plan)
        return result.total_estimated_cost

    def estimate_duration(self, plan: Plan) -> float:
        """Quick duration estimation for a plan.

        Args:
            plan: Plan to estimate

        Returns:
            Total estimated duration in milliseconds
        """
        result = self.simulate(plan)
        return result.total_estimated_duration_ms
