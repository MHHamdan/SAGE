"""Reactive planner - interleaved reasoning and acting (ReAct).

Implements the ReAct pattern where planning and execution are
interleaved. The agent reasons about the next action based on
current observations.
"""

import logging
import time
from typing import Optional, List, Dict, Any, Callable

from .planner_base import BasePlanner
from .schemas import (
    Plan, PlanStep, PlanningContext, PlanningResult,
    PlanStatus, StepStatus, create_plan, create_step,
)

logger = logging.getLogger(__name__)


class ReactivePlanner(BasePlanner):
    """Reactive planner using the ReAct pattern.

    This planner interleaves:
    1. THOUGHT: Reasoning about the current state
    2. ACTION: Deciding what to do next
    3. OBSERVATION: Processing the result

    Best used when:
    - Task requires adaptation to observations
    - Full plan cannot be determined upfront
    - Environment is dynamic or uncertain
    - Step-by-step reasoning is beneficial

    Example:
        >>> planner = ReactivePlanner(llm_client)
        >>> context = PlanningContext(
        ...     task="Find and summarize recent news about AI",
        ...     available_tools=["search", "read", "summarize"],
        ... )
        >>> # Get first step
        >>> result = planner.plan(context)
        >>> # Execute and get next step
        >>> context.history.append({"action": "search", "result": "Found 5 articles"})
        >>> next_result = planner.get_next_step(context, result.plan)
    """

    def __init__(
        self,
        llm_client: Any,
        max_iterations: int = 10,
        stop_phrases: Optional[List[str]] = None,
        **kwargs,
    ):
        """Initialize the reactive planner.

        Args:
            llm_client: LLM client for reasoning
            max_iterations: Maximum ReAct iterations
            stop_phrases: Phrases indicating task completion
            **kwargs: Additional arguments for BasePlanner
        """
        super().__init__(llm_client, **kwargs)
        self.planner_type = "reactive"
        self.max_iterations = max_iterations
        self.stop_phrases = stop_phrases or [
            "FINISH",
            "DONE",
            "COMPLETE",
            "TASK COMPLETED",
        ]

    def plan(self, context: PlanningContext) -> PlanningResult:
        """Generate the first step using ReAct reasoning.

        Unlike deliberative planning, this only returns the first step.
        Call get_next_step() after execution to get subsequent steps.

        Args:
            context: Planning context

        Returns:
            PlanningResult with initial plan (one step)
        """
        start_time = time.time()

        try:
            self._log(f"Starting reactive planning for: {context.task}")

            # Generate first thought and action
            react_result = self._react_step(context)

            if react_result.get("is_complete"):
                # Task determined to be complete or trivial
                plan = create_plan(
                    goal=context.task,
                    planner_type=self.planner_type,
                )
                plan.status = PlanStatus.COMPLETED
                return PlanningResult(
                    plan=plan,
                    success=True,
                    planning_time_ms=(time.time() - start_time) * 1000,
                    tokens_used=react_result.get("tokens", 0),
                )

            # Create plan with first step
            step = create_step(
                name=f"Step 1: {react_result.get('action', 'Unknown')}",
                action=react_result.get("action", "think"),
                description=react_result.get("thought", ""),
                parameters=react_result.get("parameters", {}),
                risk_score=0.3,
            )

            plan = create_plan(
                goal=context.task,
                steps=[step],
                planner_type=self.planner_type,
            )
            plan.status = PlanStatus.EXECUTING

            self._log(f"Generated first step: {step.action}")

            return PlanningResult(
                plan=plan,
                success=True,
                planning_time_ms=(time.time() - start_time) * 1000,
                tokens_used=react_result.get("tokens", 0),
            )

        except Exception as e:
            self._log(f"Planning failed: {e}", "error")
            return PlanningResult(
                success=False,
                error=str(e),
                planning_time_ms=(time.time() - start_time) * 1000,
            )

    def get_next_step(
        self,
        context: PlanningContext,
        current_plan: Plan,
        observation: Optional[str] = None,
    ) -> PlanningResult:
        """Get the next step based on observation.

        Args:
            context: Updated planning context
            current_plan: Current plan
            observation: Result of last action

        Returns:
            PlanningResult with updated plan
        """
        start_time = time.time()

        try:
            # Check iteration limit
            if current_plan.num_steps >= self.max_iterations:
                self._log("Max iterations reached")
                current_plan.status = PlanStatus.COMPLETED
                return PlanningResult(
                    plan=current_plan,
                    success=True,
                    planning_time_ms=(time.time() - start_time) * 1000,
                )

            # Add observation to context
            if observation:
                context.history.append({
                    "action": current_plan.steps[-1].action if current_plan.steps else "unknown",
                    "result": observation,
                })

            # Generate next thought and action
            react_result = self._react_step(context)

            if react_result.get("is_complete"):
                current_plan.status = PlanStatus.COMPLETED
                return PlanningResult(
                    plan=current_plan,
                    success=True,
                    planning_time_ms=(time.time() - start_time) * 1000,
                    tokens_used=react_result.get("tokens", 0),
                )

            # Add new step
            step_num = current_plan.num_steps + 1
            step = create_step(
                name=f"Step {step_num}: {react_result.get('action', 'Unknown')}",
                action=react_result.get("action", "think"),
                description=react_result.get("thought", ""),
                parameters=react_result.get("parameters", {}),
                risk_score=0.3,
            )

            current_plan.add_step(step)

            return PlanningResult(
                plan=current_plan,
                success=True,
                planning_time_ms=(time.time() - start_time) * 1000,
                tokens_used=react_result.get("tokens", 0),
            )

        except Exception as e:
            self._log(f"Get next step failed: {e}", "error")
            return PlanningResult(
                plan=current_plan,
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
        """Replan by adding a recovery step.

        Args:
            context: Updated context
            current_plan: Current plan
            failure_reason: Reason for replanning

        Returns:
            PlanningResult with recovery step
        """
        # Add failure to history
        context.history.append({
            "action": "error",
            "result": f"Previous action failed: {failure_reason}",
        })

        # Get next step considering the failure
        return self.get_next_step(context, current_plan, f"Error: {failure_reason}")

    def _react_step(self, context: PlanningContext) -> Dict[str, Any]:
        """Perform one ReAct step (thought + action).

        Args:
            context: Planning context

        Returns:
            Dictionary with thought, action, parameters, is_complete, tokens
        """
        prompt = self._build_react_prompt(context)
        response = self.llm_client.generate(prompt)
        content = response.content

        # Parse the response
        result = self._parse_react_response(content)
        result["tokens"] = response.total_tokens

        # Check for completion
        for phrase in self.stop_phrases:
            if phrase.upper() in content.upper():
                result["is_complete"] = True
                break

        return result

    def _build_react_prompt(self, context: PlanningContext) -> str:
        """Build the ReAct prompt.

        Args:
            context: Planning context

        Returns:
            Formatted prompt
        """
        tools_str = ", ".join(context.available_tools) if context.available_tools else "think, respond"

        prompt = f"""You are an AI assistant using the ReAct (Reasoning and Acting) framework.

TASK: {context.task}

AVAILABLE ACTIONS: {tools_str}

"""

        if context.history:
            prompt += "PREVIOUS STEPS:\n"
            for i, h in enumerate(context.history[-10:], 1):  # Last 10
                action = h.get("action", "unknown")
                result = h.get("result", "no result")[:200]
                prompt += f"{i}. Action: {action}\n   Observation: {result}\n"
            prompt += "\n"

        prompt += """Now reason about the next step.

Format your response as:
THOUGHT: [Your reasoning about what to do next]
ACTION: [The action to take - use one of the available actions]
PARAMETERS: [key=value pairs for the action, or "none"]

If the task is complete, respond with:
THOUGHT: [Why the task is complete]
ACTION: FINISH
PARAMETERS: none

Your response:"""

        return prompt

    def _parse_react_response(self, content: str) -> Dict[str, Any]:
        """Parse ReAct response.

        Args:
            content: LLM response

        Returns:
            Parsed result dictionary
        """
        result = {
            "thought": "",
            "action": "think",
            "parameters": {},
            "is_complete": False,
        }

        lines = content.strip().split("\n")

        for line in lines:
            line = line.strip()

            if line.upper().startswith("THOUGHT:"):
                result["thought"] = line.split(":", 1)[1].strip()

            elif line.upper().startswith("ACTION:"):
                action = line.split(":", 1)[1].strip()
                result["action"] = action

                if action.upper() in ["FINISH", "DONE", "COMPLETE"]:
                    result["is_complete"] = True

            elif line.upper().startswith("PARAMETERS:"):
                params_str = line.split(":", 1)[1].strip()
                result["parameters"] = self._parse_parameters(params_str)

        return result


class ReActAgent:
    """Complete ReAct agent combining planner and executor.

    Provides a higher-level interface that handles the full
    ReAct loop including tool execution.

    Example:
        >>> agent = ReActAgent(llm_client, tools=[search, read])
        >>> result = agent.run("Find information about Python")
    """

    def __init__(
        self,
        llm_client: Any,
        tools: Optional[List[Callable]] = None,
        max_iterations: int = 10,
        verbose: bool = False,
    ):
        """Initialize the ReAct agent.

        Args:
            llm_client: LLM client
            tools: List of available tools
            max_iterations: Maximum iterations
            verbose: Enable verbose output
        """
        self.llm_client = llm_client
        self.tools = {t.__name__: t for t in (tools or [])}
        self.max_iterations = max_iterations
        self.verbose = verbose

        self.planner = ReactivePlanner(
            llm_client,
            max_iterations=max_iterations,
            verbose=verbose,
        )

    def run(self, task: str) -> Dict[str, Any]:
        """Run the ReAct agent on a task.

        Args:
            task: Task to accomplish

        Returns:
            Result dictionary with output and history
        """
        context = PlanningContext(
            task=task,
            available_tools=list(self.tools.keys()),
        )

        history = []
        result = self.planner.plan(context)

        for iteration in range(self.max_iterations):
            if not result.success or not result.plan:
                break

            plan = result.plan

            if plan.status == PlanStatus.COMPLETED:
                break

            # Get current step
            current_step = plan.steps[-1] if plan.steps else None
            if not current_step:
                break

            # Execute the action
            action = current_step.action
            params = current_step.parameters

            if action.lower() in ["finish", "done", "complete"]:
                break

            observation = self._execute_action(action, params)

            history.append({
                "thought": current_step.description,
                "action": action,
                "parameters": params,
                "observation": observation,
            })

            # Get next step
            result = self.planner.get_next_step(context, plan, observation)

        return {
            "success": result.success if result else False,
            "history": history,
            "final_answer": history[-1]["observation"] if history else "No answer",
            "iterations": len(history),
        }

    def _execute_action(
        self,
        action: str,
        params: Dict[str, Any],
    ) -> str:
        """Execute an action.

        Args:
            action: Action name
            params: Action parameters

        Returns:
            Observation string
        """
        if action in self.tools:
            try:
                tool = self.tools[action]
                result = tool(**params) if params else tool()
                return str(result)
            except Exception as e:
                return f"Error executing {action}: {e}"

        return f"Unknown action: {action}. Available: {list(self.tools.keys())}"
