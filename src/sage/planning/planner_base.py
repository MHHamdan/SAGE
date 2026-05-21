"""Base planner class for all planning strategies.

Provides the abstract interface and common functionality for
deliberative, reactive, hybrid, and HTN planners.
"""

import logging
import time
from abc import ABC, abstractmethod
from typing import Optional, List, Dict, Any, Callable

from .schemas import (
    Plan, PlanStep, PlanningContext, PlanningResult,
    PlanStatus, StepStatus, create_plan, create_step,
)

logger = logging.getLogger(__name__)


class BasePlanner(ABC):
    """Abstract base class for all planners.

    Planners are responsible for generating action plans from goals
    and available tools.

    Example:
        >>> planner = DeliberativePlanner(llm_client)
        >>> context = PlanningContext(task="Summarize the document")
        >>> result = planner.plan(context)
        >>> if result.success:
        ...     print(f"Created plan with {result.plan.num_steps} steps")
    """

    def __init__(
        self,
        llm_client: Any,
        max_steps: int = 10,
        max_retries: int = 3,
        verbose: bool = False,
    ):
        """Initialize the planner.

        Args:
            llm_client: LLM client for plan generation
            max_steps: Maximum number of steps in a plan
            max_retries: Maximum retries for plan generation
            verbose: Enable verbose logging
        """
        self.llm_client = llm_client
        self.max_steps = max_steps
        self.max_retries = max_retries
        self.verbose = verbose
        self.planner_type = "base"

    @abstractmethod
    def plan(self, context: PlanningContext) -> PlanningResult:
        """Generate a plan for the given context.

        Args:
            context: Planning context with task and constraints

        Returns:
            PlanningResult with the generated plan
        """
        pass

    @abstractmethod
    def replan(
        self,
        context: PlanningContext,
        current_plan: Plan,
        failure_reason: str,
    ) -> PlanningResult:
        """Replan after a failure or change in context.

        Args:
            context: Updated planning context
            current_plan: The current (possibly partial) plan
            failure_reason: Reason for replanning

        Returns:
            PlanningResult with the new plan
        """
        pass

    def validate_plan(self, plan: Plan) -> List[str]:
        """Validate a plan.

        Args:
            plan: Plan to validate

        Returns:
            List of validation errors (empty if valid)
        """
        return plan.validate()

    def _build_planning_prompt(
        self,
        context: PlanningContext,
        current_state: Optional[str] = None,
    ) -> str:
        """Build the prompt for plan generation.

        Args:
            context: Planning context
            current_state: Optional description of current state

        Returns:
            Formatted prompt string
        """
        tools_str = ", ".join(context.available_tools) if context.available_tools else "None specified"
        constraints_str = "\n".join(f"- {c}" for c in context.constraints) if context.constraints else "None"

        prompt = f"""You are an AI planning assistant. Create a detailed step-by-step plan to accomplish the following task.

TASK: {context.task}

AVAILABLE TOOLS: {tools_str}

CONSTRAINTS:
{constraints_str}

REQUIREMENTS:
1. Each step should be atomic and executable
2. Steps should be in logical order
3. Consider dependencies between steps
4. Estimate risk for each step (low, medium, high)
5. Keep the plan under {self.max_steps} steps

"""

        if current_state:
            prompt += f"\nCURRENT STATE:\n{current_state}\n"

        if context.history:
            history_str = "\n".join(
                f"- {h.get('action', 'Unknown')}: {h.get('result', 'No result')}"
                for h in context.history[-5:]  # Last 5 actions
            )
            prompt += f"\nRECENT HISTORY:\n{history_str}\n"

        prompt += """
OUTPUT FORMAT:
Provide the plan as a numbered list with the following format for each step:
STEP N: [Step Name]
ACTION: [tool/action to use]
DESCRIPTION: [what this step does]
PARAMETERS: [key=value pairs]
RISK: [low/medium/high]
DEPENDS_ON: [step numbers this depends on, or "none"]

Begin your plan:"""

        return prompt

    def _parse_plan_response(
        self,
        response: str,
        context: PlanningContext,
    ) -> Plan:
        """Parse LLM response into a Plan.

        Args:
            response: Raw LLM response
            context: Planning context

        Returns:
            Parsed Plan object
        """
        plan = create_plan(
            goal=context.task,
            planner_type=self.planner_type,
        )

        # Parse steps from response
        lines = response.strip().split("\n")
        current_step = None
        step_data = {}

        for line in lines:
            line = line.strip()
            if not line:
                continue

            # Check for step marker
            if line.upper().startswith("STEP"):
                # Save previous step
                if step_data:
                    step = self._create_step_from_data(step_data)
                    plan.add_step(step)
                    step_data = {}

                # Parse step name
                parts = line.split(":", 1)
                if len(parts) > 1:
                    step_data["name"] = parts[1].strip()

            elif line.upper().startswith("ACTION:"):
                step_data["action"] = line.split(":", 1)[1].strip()

            elif line.upper().startswith("DESCRIPTION:"):
                step_data["description"] = line.split(":", 1)[1].strip()

            elif line.upper().startswith("PARAMETERS:"):
                params_str = line.split(":", 1)[1].strip()
                step_data["parameters"] = self._parse_parameters(params_str)

            elif line.upper().startswith("RISK:"):
                risk_str = line.split(":", 1)[1].strip().lower()
                step_data["risk"] = {"low": 0.2, "medium": 0.5, "high": 0.8}.get(risk_str, 0.3)

            elif line.upper().startswith("DEPENDS_ON:"):
                deps_str = line.split(":", 1)[1].strip()
                if deps_str.lower() != "none":
                    step_data["dependencies"] = self._parse_dependencies(deps_str)

        # Don't forget the last step
        if step_data:
            step = self._create_step_from_data(step_data)
            plan.add_step(step)

        plan._update_estimates()
        plan.status = PlanStatus.DRAFT

        return plan

    def _create_step_from_data(self, data: Dict[str, Any]) -> PlanStep:
        """Create a PlanStep from parsed data."""
        return create_step(
            name=data.get("name", "Unnamed step"),
            action=data.get("action", "unknown"),
            description=data.get("description", ""),
            parameters=data.get("parameters", {}),
            risk_score=data.get("risk", 0.3),
            requires_approval=data.get("risk", 0.3) >= 0.7,
        )

    def _parse_parameters(self, params_str: str) -> Dict[str, Any]:
        """Parse parameter string into dictionary."""
        params = {}
        if not params_str or params_str.lower() == "none":
            return params

        # Try key=value format
        for part in params_str.split(","):
            part = part.strip()
            if "=" in part:
                key, value = part.split("=", 1)
                params[key.strip()] = value.strip()

        return params

    def _parse_dependencies(self, deps_str: str) -> List[str]:
        """Parse dependencies string into list."""
        deps = []
        for part in deps_str.replace(",", " ").split():
            part = part.strip()
            if part.isdigit():
                deps.append(f"step_{int(part)}")
            elif part:
                deps.append(part)
        return deps

    def _log(self, message: str, level: str = "info"):
        """Log a message if verbose."""
        if self.verbose:
            getattr(logger, level)(f"[{self.planner_type}] {message}")


class SimplePlanner(BasePlanner):
    """Simple single-shot planner for basic tasks.

    Generates a complete plan in one LLM call. Good for
    straightforward tasks with clear structure.
    """

    def __init__(self, llm_client: Any, **kwargs):
        super().__init__(llm_client, **kwargs)
        self.planner_type = "simple"

    def plan(self, context: PlanningContext) -> PlanningResult:
        """Generate a simple plan.

        Args:
            context: Planning context

        Returns:
            PlanningResult
        """
        start_time = time.time()

        try:
            prompt = self._build_planning_prompt(context)
            response = self.llm_client.generate(prompt)

            plan = self._parse_plan_response(response.content, context)

            # Validate
            errors = self.validate_plan(plan)
            if errors:
                self._log(f"Plan validation errors: {errors}", "warning")

            plan.status = PlanStatus.VALIDATED

            return PlanningResult(
                plan=plan,
                success=True,
                planning_time_ms=(time.time() - start_time) * 1000,
                tokens_used=response.total_tokens,
            )

        except Exception as e:
            self._log(f"Planning failed: {e}", "error")
            return PlanningResult(
                success=False,
                error=str(e),
                planning_time_ms=(time.time() - start_time) * 1000,
            )

    def replan(
        self,
        context: PlanningContext,
        current_plan: Plan,
        failure_reason: str,
    ) -> PlanningResult:
        """Replan by generating a new plan.

        Args:
            context: Updated context
            current_plan: Current plan
            failure_reason: Why replanning is needed

        Returns:
            PlanningResult with new plan
        """
        # Add failure context
        context.history.append({
            "action": "replan",
            "result": f"Previous plan failed: {failure_reason}",
        })

        result = self.plan(context)

        if result.plan:
            result.plan.parent_plan_id = current_plan.plan_id
            result.plan.replan_count = current_plan.replan_count + 1
            result.plan.replan_reason = failure_reason

        return result
