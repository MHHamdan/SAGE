"""Hybrid planner - plan skeleton with reactive refinement.

Combines deliberative and reactive planning:
1. Creates a high-level plan skeleton upfront
2. Refines each step reactively during execution
3. Replans when significant deviations occur

This approach balances foresight with adaptability.
"""

import logging
import time
from typing import Optional, List, Dict, Any

from .planner_base import BasePlanner
from .deliberative import DeliberativePlanner
from .reactive import ReactivePlanner
from .schemas import (
    Plan, PlanStep, PlanningContext, PlanningResult,
    PlanStatus, StepStatus, create_plan, create_step,
)

logger = logging.getLogger(__name__)


class HybridPlanner(BasePlanner):
    """Hybrid planner combining deliberative skeleton with reactive execution.

    This planner:
    1. Creates a high-level plan skeleton (deliberative)
    2. Refines each step before execution (reactive)
    3. Monitors execution and replans if needed

    Best used when:
    - Task has known structure but details are uncertain
    - Some planning is beneficial but adaptation is needed
    - You want both predictability and flexibility

    Example:
        >>> planner = HybridPlanner(llm_client)
        >>> context = PlanningContext(
        ...     task="Build and deploy a web application",
        ...     available_tools=["code", "test", "deploy"],
        ... )
        >>> result = planner.plan(context)
        >>> # Skeleton plan created, now refine each step during execution
        >>> refined = planner.refine_step(context, result.plan, 0)
    """

    def __init__(
        self,
        llm_client: Any,
        replan_threshold: float = 0.3,
        max_refinement_depth: int = 3,
        **kwargs,
    ):
        """Initialize the hybrid planner.

        Args:
            llm_client: LLM client for planning
            replan_threshold: Threshold for triggering replanning (0-1)
            max_refinement_depth: Maximum depth for step refinement
            **kwargs: Additional arguments for BasePlanner
        """
        super().__init__(llm_client, **kwargs)
        self.planner_type = "hybrid"
        self.replan_threshold = replan_threshold
        self.max_refinement_depth = max_refinement_depth

        # Create sub-planners
        self._deliberative = DeliberativePlanner(
            llm_client,
            enable_decomposition=True,
            enable_condition_generation=False,  # Skip for skeleton
            max_steps=kwargs.get("max_steps", 10),
            verbose=kwargs.get("verbose", False),
        )
        self._reactive = ReactivePlanner(
            llm_client,
            max_iterations=5,
            verbose=kwargs.get("verbose", False),
        )

    def plan(self, context: PlanningContext) -> PlanningResult:
        """Generate a hybrid plan (skeleton + refinement hooks).

        Args:
            context: Planning context

        Returns:
            PlanningResult with skeleton plan
        """
        start_time = time.time()
        total_tokens = 0

        try:
            self._log(f"Starting hybrid planning for: {context.task}")

            # Step 1: Generate high-level skeleton
            skeleton_result = self._generate_skeleton(context)
            total_tokens += skeleton_result.get("tokens", 0)

            if not skeleton_result.get("steps"):
                return PlanningResult(
                    success=False,
                    error="Failed to generate plan skeleton",
                    planning_time_ms=(time.time() - start_time) * 1000,
                )

            # Step 2: Create plan with skeleton steps
            plan = create_plan(
                goal=context.task,
                planner_type=self.planner_type,
            )

            for i, skeleton_step in enumerate(skeleton_result["steps"]):
                step = create_step(
                    name=skeleton_step.get("name", f"Step {i+1}"),
                    action=skeleton_step.get("action", "execute"),
                    description=skeleton_step.get("description", ""),
                    parameters={
                        "skeleton": True,  # Mark as needing refinement
                        "refinement_depth": 0,
                    },
                    risk_score=skeleton_step.get("risk", 0.3),
                )
                plan.add_step(step)

            plan.status = PlanStatus.VALIDATED
            self._log(f"Created skeleton plan with {plan.num_steps} high-level steps")

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

    def refine_step(
        self,
        context: PlanningContext,
        plan: Plan,
        step_idx: int,
        observation: Optional[str] = None,
    ) -> PlanningResult:
        """Refine a skeleton step into concrete actions.

        Args:
            context: Planning context
            plan: Current plan
            step_idx: Index of step to refine
            observation: Current observation/state

        Returns:
            PlanningResult with refined step(s)
        """
        start_time = time.time()

        try:
            if step_idx >= len(plan.steps):
                return PlanningResult(
                    plan=plan,
                    success=False,
                    error=f"Invalid step index: {step_idx}",
                )

            step = plan.steps[step_idx]

            # Check if already refined
            if not step.parameters.get("skeleton", False):
                return PlanningResult(
                    plan=plan,
                    success=True,
                    planning_time_ms=(time.time() - start_time) * 1000,
                )

            # Check refinement depth
            depth = step.parameters.get("refinement_depth", 0)
            if depth >= self.max_refinement_depth:
                step.parameters["skeleton"] = False  # Mark as refined
                return PlanningResult(
                    plan=plan,
                    success=True,
                    planning_time_ms=(time.time() - start_time) * 1000,
                )

            self._log(f"Refining step {step_idx}: {step.name}")

            # Create sub-context for this step
            sub_context = PlanningContext(
                task=f"Execute: {step.name}\nDetails: {step.description}",
                available_tools=context.available_tools,
                constraints=context.constraints,
                history=context.history.copy(),
            )

            if observation:
                sub_context.history.append({
                    "action": "observation",
                    "result": observation,
                })

            # Use reactive planner to refine
            refinement = self._reactive.plan(sub_context)

            if refinement.success and refinement.plan:
                # Replace skeleton step with refined steps
                refined_steps = refinement.plan.steps

                if refined_steps:
                    # Update step with refinement
                    step.action = refined_steps[0].action
                    step.description = refined_steps[0].description
                    step.parameters = {
                        **refined_steps[0].parameters,
                        "skeleton": False,
                        "refinement_depth": depth + 1,
                    }

                    # Add additional refined steps if any
                    for i, extra_step in enumerate(refined_steps[1:], 1):
                        extra_step.parameters["skeleton"] = False
                        extra_step.parameters["refinement_depth"] = depth + 1
                        extra_step.order = step.order + (i * 0.1)
                        plan.steps.insert(step_idx + i, extra_step)

            return PlanningResult(
                plan=plan,
                success=True,
                planning_time_ms=(time.time() - start_time) * 1000,
                tokens_used=refinement.tokens_used if refinement else 0,
            )

        except Exception as e:
            self._log(f"Step refinement failed: {e}", "error")
            return PlanningResult(
                plan=plan,
                success=False,
                error=str(e),
                planning_time_ms=(time.time() - start_time) * 1000,
            )

    def should_replan(
        self,
        context: PlanningContext,
        plan: Plan,
        observation: str,
    ) -> bool:
        """Determine if replanning is needed.

        Args:
            context: Planning context
            plan: Current plan
            observation: Current observation

        Returns:
            True if replanning should occur
        """
        # Check for explicit failure indicators
        failure_keywords = ["error", "failed", "cannot", "unable", "impossible"]
        obs_lower = observation.lower()

        for keyword in failure_keywords:
            if keyword in obs_lower:
                return True

        # Check progress - if stuck too long
        if plan.current_step_idx > 0:
            completed_ratio = plan.completed_steps / plan.current_step_idx
            if completed_ratio < (1 - self.replan_threshold):
                return True

        return False

    def replan(
        self,
        context: PlanningContext,
        current_plan: Plan,
        failure_reason: str,
    ) -> PlanningResult:
        """Replan with consideration of completed work.

        Args:
            context: Updated context
            current_plan: Current plan
            failure_reason: Reason for replanning

        Returns:
            PlanningResult with new plan
        """
        self._log(f"Replanning due to: {failure_reason}")

        # Keep completed steps, replan remaining
        completed_steps = [
            s for s in current_plan.steps
            if s.status == StepStatus.COMPLETED
        ]

        # Update context with what's done
        context.history.append({
            "action": "partial_completion",
            "result": f"Completed {len(completed_steps)} steps before failure: {failure_reason}",
        })

        # Add constraint about the failure
        context.constraints.append(
            f"Previous approach failed at step {current_plan.current_step_idx}: {failure_reason}"
        )

        # Generate new plan
        result = self.plan(context)

        if result.plan:
            result.plan.parent_plan_id = current_plan.plan_id
            result.plan.replan_count = current_plan.replan_count + 1
            result.plan.replan_reason = failure_reason

        return result

    def _generate_skeleton(self, context: PlanningContext) -> Dict[str, Any]:
        """Generate the high-level plan skeleton.

        Args:
            context: Planning context

        Returns:
            Dictionary with steps and token count
        """
        prompt = f"""Create a high-level plan skeleton for the following task.
Each step should be a major milestone, not detailed actions.
We will refine each step later during execution.

TASK: {context.task}

AVAILABLE TOOLS: {', '.join(context.available_tools) if context.available_tools else 'general actions'}

Create 3-5 high-level steps. For each step provide:
STEP: [Step name - short phrase]
DESCRIPTION: [What this step accomplishes]
RISK: [low/medium/high]

Begin:"""

        response = self.llm_client.generate(prompt)
        content = response.content

        steps = []
        current_step = {}

        for line in content.split("\n"):
            line = line.strip()
            if not line:
                continue

            if line.upper().startswith("STEP:"):
                if current_step:
                    steps.append(current_step)
                current_step = {"name": line.split(":", 1)[1].strip()}

            elif line.upper().startswith("DESCRIPTION:"):
                current_step["description"] = line.split(":", 1)[1].strip()

            elif line.upper().startswith("RISK:"):
                risk_str = line.split(":", 1)[1].strip().lower()
                current_step["risk"] = {"low": 0.2, "medium": 0.5, "high": 0.8}.get(risk_str, 0.3)

        if current_step:
            steps.append(current_step)

        return {
            "steps": steps,
            "tokens": response.total_tokens,
        }
