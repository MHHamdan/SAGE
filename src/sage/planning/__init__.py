"""Planning module for agentic AI systems.

Provides multiple planning strategies:
- DeliberativePlanner: Full plan before execution
- ReactivePlanner: Interleaved reasoning and acting (ReAct)
- HybridPlanner: Plan skeleton with reactive refinement
- HTNLitePlanner: Hierarchical task network template decomposition

Usage:
    >>> from sage.planning import HybridPlanner, PlanningContext
    >>> from sage.core.ollama_client import OllamaClient
    >>>
    >>> client = OllamaClient(model="llama3.1:8b")
    >>> planner = HybridPlanner(client)
    >>>
    >>> context = PlanningContext(
    ...     task="Analyze the sales data and create a report",
    ...     available_tools=["read", "analyze", "write"],
    ... )
    >>> result = planner.plan(context)
    >>> print(f"Plan has {result.plan.num_steps} steps")
"""

from .schemas import (
    Plan,
    PlanStep,
    PlanStatus,
    StepStatus,
    Condition,
    ConditionType,
    PlanningContext,
    PlanningResult,
    create_plan,
    create_step,
)

from .planner_base import BasePlanner, SimplePlanner
from .deliberative import DeliberativePlanner
from .reactive import ReactivePlanner, ReActAgent
from .hybrid import HybridPlanner
from .htn_lite import HTNLitePlanner, TaskTemplate

__all__ = [
    # Schemas
    "Plan",
    "PlanStep",
    "PlanStatus",
    "StepStatus",
    "Condition",
    "ConditionType",
    "PlanningContext",
    "PlanningResult",
    "create_plan",
    "create_step",
    # Planners
    "BasePlanner",
    "SimplePlanner",
    "DeliberativePlanner",
    "ReactivePlanner",
    "ReActAgent",
    "HybridPlanner",
    "HTNLitePlanner",
    "TaskTemplate",
]


def create_planner(
    planner_type: str,
    llm_client,
    **kwargs,
):
    """Factory function to create a planner.

    Args:
        planner_type: Type of planner ("deliberative", "reactive", "hybrid", "htn_lite")
        llm_client: LLM client for planning
        **kwargs: Additional arguments for the planner

    Returns:
        Configured planner instance

    Example:
        >>> planner = create_planner("hybrid", llm_client, max_steps=10)
    """
    planners = {
        "simple": SimplePlanner,
        "deliberative": DeliberativePlanner,
        "reactive": ReactivePlanner,
        "hybrid": HybridPlanner,
        "htn_lite": HTNLitePlanner,
        "htn": HTNLitePlanner,
    }

    planner_class = planners.get(planner_type.lower())
    if not planner_class:
        raise ValueError(f"Unknown planner type: {planner_type}. Available: {list(planners.keys())}")

    return planner_class(llm_client, **kwargs)
