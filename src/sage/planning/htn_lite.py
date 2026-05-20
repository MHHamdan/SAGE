"""HTN-Lite planner - Hierarchical Task Network template decomposition.

Implements a lightweight version of HTN planning where:
1. Tasks are decomposed using predefined templates
2. Templates specify how to break down complex tasks
3. Primitive actions are executed directly

This is "lite" because it uses simple pattern matching rather than
formal precondition/effect reasoning.
"""

import logging
import time
import re
from typing import Optional, List, Dict, Any, Callable

from .planner_base import BasePlanner
from .schemas import (
    Plan, PlanStep, PlanningContext, PlanningResult,
    PlanStatus, StepStatus, create_plan, create_step,
)

logger = logging.getLogger(__name__)


class TaskTemplate:
    """Template for decomposing a task type."""

    def __init__(
        self,
        name: str,
        pattern: str,
        subtasks: List[Dict[str, Any]],
        description: str = "",
    ):
        """Initialize a task template.

        Args:
            name: Template name
            pattern: Regex pattern to match tasks
            subtasks: List of subtask specifications
            description: Template description
        """
        self.name = name
        self.pattern = pattern
        self._regex = re.compile(pattern, re.IGNORECASE)
        self.subtasks = subtasks
        self.description = description

    def matches(self, task: str) -> Optional[Dict[str, str]]:
        """Check if task matches this template.

        Args:
            task: Task description

        Returns:
            Match groups if matched, None otherwise
        """
        match = self._regex.search(task)
        if match:
            return match.groupdict()
        return None

    def decompose(self, task: str, groups: Dict[str, str]) -> List[Dict[str, Any]]:
        """Decompose task into subtasks.

        Args:
            task: Original task
            groups: Regex match groups

        Returns:
            List of subtask specifications
        """
        result = []
        for subtask in self.subtasks:
            # Substitute variables
            st = subtask.copy()
            for key, value in st.items():
                if isinstance(value, str):
                    for group_name, group_value in groups.items():
                        st[key] = st[key].replace(f"{{{group_name}}}", group_value or "")
            result.append(st)
        return result


# Default templates for common task patterns
DEFAULT_TEMPLATES = [
    TaskTemplate(
        name="search_and_summarize",
        pattern=r"(?:find|search|look up).*(?:about|for|on)\s+(?P<topic>.+?)(?:\s+and\s+(?:summarize|summarise))?",
        subtasks=[
            {"action": "search", "description": "Search for information about {topic}", "params": {"query": "{topic}"}},
            {"action": "read", "description": "Read and extract relevant information", "params": {}},
            {"action": "summarize", "description": "Summarize the findings about {topic}", "params": {}},
        ],
        description="Search for a topic and summarize results",
    ),
    TaskTemplate(
        name="analyze_data",
        pattern=r"(?:analyze|analyse|examine)\s+(?P<data>.+)",
        subtasks=[
            {"action": "load", "description": "Load the data: {data}", "params": {"source": "{data}"}},
            {"action": "analyze", "description": "Perform analysis", "params": {}},
            {"action": "report", "description": "Generate analysis report", "params": {}},
        ],
        description="Load, analyze, and report on data",
    ),
    TaskTemplate(
        name="create_document",
        pattern=r"(?:create|write|generate)\s+(?:a\s+)?(?P<doctype>report|summary|document|email)\s+(?:about|on|for)\s+(?P<topic>.+)",
        subtasks=[
            {"action": "research", "description": "Gather information about {topic}", "params": {"topic": "{topic}"}},
            {"action": "outline", "description": "Create outline for {doctype}", "params": {}},
            {"action": "write", "description": "Write the {doctype}", "params": {"type": "{doctype}"}},
            {"action": "review", "description": "Review and finalize", "params": {}},
        ],
        description="Research and create a document",
    ),
    TaskTemplate(
        name="code_task",
        pattern=r"(?:implement|code|build|develop)\s+(?P<feature>.+)",
        subtasks=[
            {"action": "understand", "description": "Understand requirements for {feature}", "params": {}},
            {"action": "design", "description": "Design the implementation", "params": {}},
            {"action": "implement", "description": "Implement {feature}", "params": {"feature": "{feature}"}},
            {"action": "test", "description": "Test the implementation", "params": {}},
        ],
        description="Software development workflow",
    ),
    TaskTemplate(
        name="deploy_task",
        pattern=r"(?:deploy|release|ship)\s+(?P<artifact>.+?)(?:\s+to\s+(?P<environment>.+))?",
        subtasks=[
            {"action": "verify", "description": "Verify {artifact} is ready", "params": {}},
            {"action": "backup", "description": "Create backup/snapshot", "params": {}},
            {"action": "deploy", "description": "Deploy {artifact} to {environment}", "params": {"target": "{environment}"}},
            {"action": "validate", "description": "Validate deployment", "params": {}},
        ],
        description="Deployment workflow",
    ),
]


class HTNLitePlanner(BasePlanner):
    """Hierarchical Task Network Lite planner.

    Uses predefined templates to decompose tasks hierarchically.
    Falls back to LLM-based decomposition for unmatched tasks.

    Best used when:
    - Task patterns are well-known
    - Consistent decomposition is desired
    - You want predictable planning

    Example:
        >>> planner = HTNLitePlanner(llm_client)
        >>> # Add custom template
        >>> planner.add_template(TaskTemplate(
        ...     name="custom",
        ...     pattern=r"custom task (?P<param>.+)",
        ...     subtasks=[{"action": "do", "description": "Do {param}"}],
        ... ))
        >>> result = planner.plan(PlanningContext(task="custom task foo"))
    """

    def __init__(
        self,
        llm_client: Any,
        templates: Optional[List[TaskTemplate]] = None,
        use_default_templates: bool = True,
        max_decomposition_depth: int = 3,
        **kwargs,
    ):
        """Initialize the HTN-Lite planner.

        Args:
            llm_client: LLM client for fallback planning
            templates: Custom task templates
            use_default_templates: Include default templates
            max_decomposition_depth: Maximum decomposition depth
            **kwargs: Additional arguments for BasePlanner
        """
        super().__init__(llm_client, **kwargs)
        self.planner_type = "htn_lite"
        self.max_decomposition_depth = max_decomposition_depth

        self.templates: List[TaskTemplate] = []
        if use_default_templates:
            self.templates.extend(DEFAULT_TEMPLATES)
        if templates:
            self.templates.extend(templates)

    def add_template(self, template: TaskTemplate):
        """Add a task template.

        Args:
            template: Template to add
        """
        self.templates.append(template)
        self._log(f"Added template: {template.name}")

    def plan(self, context: PlanningContext) -> PlanningResult:
        """Generate a plan using HTN decomposition.

        Args:
            context: Planning context

        Returns:
            PlanningResult with decomposed plan
        """
        start_time = time.time()
        total_tokens = 0

        try:
            self._log(f"Starting HTN planning for: {context.task}")

            # Try template-based decomposition
            steps, tokens = self._decompose_task(
                context.task,
                context,
                depth=0,
            )
            total_tokens += tokens

            if not steps:
                # Fallback to LLM-based planning
                self._log("No template matched, using LLM fallback")
                fallback = self._llm_decompose(context)
                steps = fallback.get("steps", [])
                total_tokens += fallback.get("tokens", 0)

            # Create plan
            plan = create_plan(
                goal=context.task,
                planner_type=self.planner_type,
            )

            for step_data in steps:
                step = create_step(
                    name=step_data.get("action", "step"),
                    action=step_data.get("action", "execute"),
                    description=step_data.get("description", ""),
                    parameters=step_data.get("params", {}),
                    risk_score=step_data.get("risk", 0.2),
                )
                plan.add_step(step)

            plan.status = PlanStatus.VALIDATED
            self._log(f"Created HTN plan with {plan.num_steps} steps")

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
        """Replan after failure.

        Args:
            context: Updated context
            current_plan: Current plan
            failure_reason: Reason for replanning

        Returns:
            PlanningResult with new plan
        """
        # Add failure context
        context.history.append({
            "action": "htn_failure",
            "result": failure_reason,
        })

        # Try alternative templates or LLM fallback
        result = self.plan(context)

        if result.plan:
            result.plan.parent_plan_id = current_plan.plan_id
            result.plan.replan_count = current_plan.replan_count + 1
            result.plan.replan_reason = failure_reason

        return result

    def _decompose_task(
        self,
        task: str,
        context: PlanningContext,
        depth: int,
    ) -> tuple[List[Dict[str, Any]], int]:
        """Recursively decompose a task using templates.

        Args:
            task: Task to decompose
            context: Planning context
            depth: Current decomposition depth

        Returns:
            Tuple of (steps, tokens_used)
        """
        if depth >= self.max_decomposition_depth:
            # Treat as primitive action
            return [{"action": "execute", "description": task, "params": {}}], 0

        # Try to match a template
        for template in self.templates:
            groups = template.matches(task)
            if groups is not None:
                self._log(f"Matched template '{template.name}' for: {task[:50]}...")

                subtasks = template.decompose(task, groups)
                all_steps = []
                total_tokens = 0

                for subtask in subtasks:
                    # Check if subtask is complex (needs further decomposition)
                    if self._is_complex_task(subtask.get("description", "")):
                        sub_steps, tokens = self._decompose_task(
                            subtask.get("description", ""),
                            context,
                            depth + 1,
                        )
                        all_steps.extend(sub_steps)
                        total_tokens += tokens
                    else:
                        all_steps.append(subtask)

                return all_steps, total_tokens

        # No template matched
        return [], 0

    def _is_complex_task(self, task: str) -> bool:
        """Check if a task is complex and needs decomposition.

        Args:
            task: Task description

        Returns:
            True if task is complex
        """
        # Simple heuristic: long tasks or tasks with "and" are complex
        if len(task) > 100:
            return True
        if " and " in task.lower():
            return True
        return False

    def _llm_decompose(self, context: PlanningContext) -> Dict[str, Any]:
        """Use LLM to decompose a task.

        Args:
            context: Planning context

        Returns:
            Dictionary with steps and token count
        """
        prompt = f"""Decompose the following task into 3-6 sequential steps.

TASK: {context.task}

AVAILABLE TOOLS: {', '.join(context.available_tools) if context.available_tools else 'general actions'}

For each step, provide:
ACTION: [action name]
DESCRIPTION: [what this step does]

Steps should be concrete and executable.

Begin:"""

        response = self.llm_client.generate(prompt)
        content = response.content

        steps = []
        current = {}

        for line in content.split("\n"):
            line = line.strip()
            if not line:
                continue

            if line.upper().startswith("ACTION:"):
                if current:
                    steps.append(current)
                current = {"action": line.split(":", 1)[1].strip()}

            elif line.upper().startswith("DESCRIPTION:"):
                current["description"] = line.split(":", 1)[1].strip()

        if current:
            steps.append(current)

        return {
            "steps": steps,
            "tokens": response.total_tokens,
        }

    def get_template_info(self) -> List[Dict[str, str]]:
        """Get information about available templates.

        Returns:
            List of template info dictionaries
        """
        return [
            {
                "name": t.name,
                "pattern": t.pattern,
                "description": t.description,
                "num_subtasks": len(t.subtasks),
            }
            for t in self.templates
        ]
