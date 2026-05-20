"""Deliberative planner - full plan before execution.

Implements classical AI planning where a complete plan is generated
before any execution begins. Best for well-defined tasks where
the full solution can be reasoned about upfront.
"""

import logging
import time
from typing import Optional, List, Dict, Any

from .planner_base import BasePlanner
from .schemas import (
    Plan, PlanStep, PlanningContext, PlanningResult,
    PlanStatus, StepStatus, Condition, ConditionType,
    create_plan, create_step,
)

logger = logging.getLogger(__name__)


class DeliberativePlanner(BasePlanner):
    """Deliberative planner that creates complete plans upfront.

    This planner:
    1. Analyzes the goal comprehensively
    2. Generates a complete plan before execution
    3. Validates all steps and dependencies
    4. Estimates costs and risks for the entire plan

    Best used when:
    - Task structure is well-defined
    - All information is available upfront
    - Plan verification is important
    - Rollback/undo capability is needed

    Example:
        >>> planner = DeliberativePlanner(llm_client)
        >>> context = PlanningContext(
        ...     task="Deploy the application to production",
        ...     available_tools=["git", "docker", "kubectl"],
        ... )
        >>> result = planner.plan(context)
    """

    def __init__(
        self,
        llm_client: Any,
        enable_decomposition: bool = True,
        enable_condition_generation: bool = True,
        **kwargs,
    ):
        """Initialize the deliberative planner.

        Args:
            llm_client: LLM client for planning
            enable_decomposition: Enable hierarchical task decomposition
            enable_condition_generation: Generate pre/postconditions
            **kwargs: Additional arguments for BasePlanner
        """
        super().__init__(llm_client, **kwargs)
        self.planner_type = "deliberative"
        self.enable_decomposition = enable_decomposition
        self.enable_condition_generation = enable_condition_generation

    def plan(self, context: PlanningContext) -> PlanningResult:
        """Generate a complete deliberative plan.

        Args:
            context: Planning context with task and constraints

        Returns:
            PlanningResult with the complete plan
        """
        start_time = time.time()
        total_tokens = 0

        try:
            self._log(f"Starting deliberative planning for: {context.task}")

            # Step 1: Decompose the task if enabled
            if self.enable_decomposition:
                subtasks = self._decompose_task(context)
                total_tokens += subtasks.get("tokens", 0)
            else:
                subtasks = {"tasks": [context.task]}

            # Step 2: Generate detailed steps for each subtask
            all_steps = []
            for subtask in subtasks.get("tasks", [context.task]):
                steps_result = self._generate_steps(subtask, context)
                all_steps.extend(steps_result.get("steps", []))
                total_tokens += steps_result.get("tokens", 0)

            # Step 3: Order and link dependencies
            ordered_steps = self._order_steps(all_steps)

            # Step 4: Generate conditions if enabled
            if self.enable_condition_generation:
                for step in ordered_steps:
                    conditions = self._generate_conditions(step, context)
                    step.preconditions = conditions.get("preconditions", [])
                    step.postconditions = conditions.get("postconditions", [])
                    total_tokens += conditions.get("tokens", 0)

            # Step 5: Estimate costs and risks
            for step in ordered_steps:
                step.estimated_cost = self._estimate_step_cost(step)
                step.risk_score = self._estimate_step_risk(step, context)
                step.requires_approval = step.risk_score >= 0.7

            # Step 6: Create and validate plan
            plan = create_plan(
                goal=context.task,
                steps=ordered_steps,
                planner_type=self.planner_type,
            )

            # Validate
            errors = self.validate_plan(plan)
            if errors:
                self._log(f"Validation errors: {errors}", "warning")
                # Try to fix simple errors
                plan = self._fix_plan_errors(plan, errors)

            plan.status = PlanStatus.VALIDATED
            self._log(f"Created plan with {plan.num_steps} steps")

            return PlanningResult(
                plan=plan,
                success=True,
                planning_time_ms=(time.time() - start_time) * 1000,
                tokens_used=total_tokens,
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
        """Replan after failure with analysis.

        Args:
            context: Updated planning context
            current_plan: The current plan
            failure_reason: Reason for replanning

        Returns:
            PlanningResult with new plan
        """
        self._log(f"Replanning due to: {failure_reason}")

        # Analyze what went wrong
        analysis = self._analyze_failure(current_plan, failure_reason)

        # Update context with failure information
        context.history.append({
            "action": "plan_failure",
            "plan_id": current_plan.plan_id,
            "failure_reason": failure_reason,
            "completed_steps": current_plan.completed_steps,
            "analysis": analysis,
        })

        # Add constraint to avoid the same failure
        if analysis.get("avoid_action"):
            context.constraints.append(
                f"Avoid: {analysis['avoid_action']} - caused failure"
            )

        # Generate new plan
        result = self.plan(context)

        if result.plan:
            result.plan.parent_plan_id = current_plan.plan_id
            result.plan.replan_count = current_plan.replan_count + 1
            result.plan.replan_reason = failure_reason

        return result

    def _decompose_task(self, context: PlanningContext) -> Dict[str, Any]:
        """Decompose a complex task into subtasks.

        Args:
            context: Planning context

        Returns:
            Dictionary with subtasks and token count
        """
        prompt = f"""Decompose the following task into 2-4 high-level subtasks.

TASK: {context.task}

Output each subtask on a new line, prefixed with "SUBTASK:".
If the task is simple and doesn't need decomposition, output "SUBTASK: {context.task}".

Example output:
SUBTASK: Gather requirements
SUBTASK: Implement solution
SUBTASK: Test and verify
"""

        response = self.llm_client.generate(prompt)
        content = response.content

        subtasks = []
        for line in content.split("\n"):
            if line.strip().upper().startswith("SUBTASK:"):
                subtask = line.split(":", 1)[1].strip()
                if subtask:
                    subtasks.append(subtask)

        if not subtasks:
            subtasks = [context.task]

        return {
            "tasks": subtasks,
            "tokens": response.total_tokens,
        }

    def _generate_steps(
        self,
        subtask: str,
        context: PlanningContext,
    ) -> Dict[str, Any]:
        """Generate detailed steps for a subtask.

        Args:
            subtask: The subtask to plan for
            context: Planning context

        Returns:
            Dictionary with steps and token count
        """
        prompt = self._build_planning_prompt(
            PlanningContext(
                task=subtask,
                available_tools=context.available_tools,
                constraints=context.constraints,
            )
        )

        response = self.llm_client.generate(prompt)
        temp_plan = self._parse_plan_response(response.content, context)

        return {
            "steps": temp_plan.steps,
            "tokens": response.total_tokens,
        }

    def _order_steps(self, steps: List[PlanStep]) -> List[PlanStep]:
        """Order steps and resolve dependencies.

        Args:
            steps: Unordered list of steps

        Returns:
            Ordered list of steps
        """
        # Simple ordering by current order values
        ordered = sorted(steps, key=lambda s: s.order)

        # Reassign order values
        for i, step in enumerate(ordered):
            step.order = i
            step.step_id = f"step_{i}"

        return ordered

    def _generate_conditions(
        self,
        step: PlanStep,
        context: PlanningContext,
    ) -> Dict[str, Any]:
        """Generate pre and postconditions for a step.

        Args:
            step: The step to generate conditions for
            context: Planning context

        Returns:
            Dictionary with conditions and token count
        """
        prompt = f"""For the following action step, identify:
1. PRECONDITIONS: What must be true before this step can execute?
2. POSTCONDITIONS: What should be true after this step completes?

STEP: {step.name}
ACTION: {step.action}
DESCRIPTION: {step.description}

Output format:
PRECONDITION: [condition description]
POSTCONDITION: [condition description]

List 1-3 of each type."""

        response = self.llm_client.generate(prompt)
        content = response.content

        preconditions = []
        postconditions = []

        for line in content.split("\n"):
            line = line.strip()
            if line.upper().startswith("PRECONDITION:"):
                desc = line.split(":", 1)[1].strip()
                if desc:
                    preconditions.append(Condition(
                        name=f"pre_{len(preconditions)}",
                        description=desc,
                        condition_type=ConditionType.PRECONDITION,
                    ))
            elif line.upper().startswith("POSTCONDITION:"):
                desc = line.split(":", 1)[1].strip()
                if desc:
                    postconditions.append(Condition(
                        name=f"post_{len(postconditions)}",
                        description=desc,
                        condition_type=ConditionType.POSTCONDITION,
                    ))

        return {
            "preconditions": preconditions,
            "postconditions": postconditions,
            "tokens": response.total_tokens,
        }

    def _estimate_step_cost(self, step: PlanStep) -> float:
        """Estimate the cost of a step.

        Args:
            step: The step to estimate

        Returns:
            Estimated cost in USD (0 for local models)
        """
        # Base cost for local Ollama models is 0
        # Add small cost for complexity
        base_cost = 0.0

        # Add nominal cost based on action type
        action_costs = {
            "search": 0.001,
            "read": 0.0005,
            "write": 0.001,
            "execute": 0.002,
            "analyze": 0.002,
        }

        action_lower = step.action.lower()
        for action_type, cost in action_costs.items():
            if action_type in action_lower:
                base_cost += cost
                break

        return base_cost

    def _estimate_step_risk(
        self,
        step: PlanStep,
        context: PlanningContext,
    ) -> float:
        """Estimate the risk of a step.

        Args:
            step: The step to estimate
            context: Planning context

        Returns:
            Risk score (0.0 to 1.0)
        """
        risk = 0.1  # Base risk

        # Increase risk for certain actions
        high_risk_keywords = ["delete", "remove", "modify", "execute", "write", "update"]
        action_lower = step.action.lower()
        desc_lower = step.description.lower()

        for keyword in high_risk_keywords:
            if keyword in action_lower or keyword in desc_lower:
                risk += 0.15

        # Cap at 1.0
        return min(risk, 1.0)

    def _analyze_failure(
        self,
        plan: Plan,
        failure_reason: str,
    ) -> Dict[str, Any]:
        """Analyze why a plan failed.

        Args:
            plan: The failed plan
            failure_reason: Reason for failure

        Returns:
            Analysis dictionary
        """
        analysis = {
            "failed_at_step": plan.current_step_idx,
            "completed_steps": plan.completed_steps,
            "failure_reason": failure_reason,
        }

        # Identify the problematic step
        current = plan.current_step
        if current:
            analysis["failed_step"] = current.name
            analysis["failed_action"] = current.action
            analysis["avoid_action"] = current.action if "error" in failure_reason.lower() else None

        return analysis

    def _fix_plan_errors(self, plan: Plan, errors: List[str]) -> Plan:
        """Attempt to fix simple plan errors.

        Args:
            plan: Plan with errors
            errors: List of error messages

        Returns:
            Fixed plan
        """
        for error in errors:
            if "no steps" in error.lower():
                # Add a default step
                plan.add_step(create_step(
                    name="Execute task",
                    action="execute",
                    description=plan.goal,
                ))

            if "exceeds max steps" in error.lower():
                # Truncate to max steps
                plan.steps = plan.steps[:plan.max_steps_allowed]

        return plan
