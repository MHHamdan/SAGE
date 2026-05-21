"""
Autonomy Benchmarks

Benchmark tasks specifically designed to test the four autonomy criteria
from Section IV-A of the paper.

16 benchmark tasks total (4 per criterion):
- Action Selection Freedom (4 tasks)
- Goal-Directed Persistence (4 tasks)
- Dynamic Termination (4 tasks)
- Error Recovery (4 tasks)

All tasks are deterministic with fixed seeds for reproducibility.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from enum import Enum

from .autonomy_validator import AutonomyCriterion


@dataclass
class AutonomyBenchmarkTask:
    """
    A benchmark task for testing a specific autonomy criterion.

    Attributes:
        task_id: Unique identifier
        criterion: Which autonomy criterion this tests
        name: Human-readable name
        description: What the task tests
        config: Task-specific configuration
        difficulty: easy, medium, hard
        seed: Random seed for determinism
    """
    task_id: str
    criterion: AutonomyCriterion
    name: str
    description: str
    config: Dict[str, Any]
    difficulty: str = "medium"
    seed: int = 42

    def to_dict(self) -> Dict[str, Any]:
        """Serialize for storage/logging."""
        return {
            "task_id": self.task_id,
            "criterion": self.criterion.value,
            "name": self.name,
            "description": self.description,
            "config": self.config,
            "difficulty": self.difficulty,
            "seed": self.seed
        }


# ============================================================
# Action Selection Freedom Tasks (4 tasks)
# ============================================================

ACTION_SELECTION_FREEDOM_TASKS = [
    AutonomyBenchmarkTask(
        task_id="auto_asf_001",
        criterion=AutonomyCriterion.ACTION_SELECTION_FREEDOM,
        name="Resource Availability Variation",
        description="Same goal with different resource availability levels",
        config={
            "goal": "Find information about machine learning",
            "scenarios": [
                {
                    "scenario_id": "full_resources",
                    "state": {
                        "resources": {"api_quota": 1000, "memory_mb": 4096},
                        "tools": ["search", "wikipedia", "arxiv", "calculator"]
                    },
                    "expected_actions": ["search", "arxiv"]
                },
                {
                    "scenario_id": "limited_resources",
                    "state": {
                        "resources": {"api_quota": 10, "memory_mb": 512},
                        "tools": ["search", "calculator"]
                    },
                    "expected_actions": ["search"]
                },
                {
                    "scenario_id": "no_api",
                    "state": {
                        "resources": {"api_quota": 0, "memory_mb": 256},
                        "tools": ["local_cache"]
                    },
                    "expected_actions": ["local_cache", "ask_user"]
                }
            ],
            "pass_condition": "Agent should select different actions for different resource levels"
        },
        difficulty="easy",
        seed=100
    ),
    AutonomyBenchmarkTask(
        task_id="auto_asf_002",
        criterion=AutonomyCriterion.ACTION_SELECTION_FREEDOM,
        name="Prior Knowledge Variation",
        description="Same goal with different levels of prior knowledge",
        config={
            "goal": "Explain quantum computing",
            "scenarios": [
                {
                    "scenario_id": "no_prior",
                    "state": {
                        "prior_knowledge": {},
                        "conversation_history": []
                    },
                    "expected_actions": ["search", "research"]
                },
                {
                    "scenario_id": "partial_prior",
                    "state": {
                        "prior_knowledge": {"quantum_basics": "known"},
                        "conversation_history": ["User asked about qubits"]
                    },
                    "expected_actions": ["refine", "clarify"]
                },
                {
                    "scenario_id": "full_prior",
                    "state": {
                        "prior_knowledge": {"quantum_computing": "comprehensive"},
                        "conversation_history": ["Full explanation given"],
                        "cached_answer": True
                    },
                    "expected_actions": ["retrieve_cached", "present"]
                }
            ],
            "pass_condition": "Agent should leverage prior knowledge when available"
        },
        difficulty="medium",
        seed=101
    ),
    AutonomyBenchmarkTask(
        task_id="auto_asf_003",
        criterion=AutonomyCriterion.ACTION_SELECTION_FREEDOM,
        name="Constraint Variation",
        description="Same goal with different constraint sets",
        config={
            "goal": "Generate a detailed report",
            "scenarios": [
                {
                    "scenario_id": "unconstrained",
                    "state": {
                        "constraints": {},
                        "permissions": ["read", "write", "network", "execute"]
                    },
                    "expected_actions": ["comprehensive_research", "full_report"]
                },
                {
                    "scenario_id": "time_constrained",
                    "state": {
                        "constraints": {"max_time_seconds": 10},
                        "permissions": ["read", "write", "network", "execute"]
                    },
                    "expected_actions": ["quick_summary", "cached_report"]
                },
                {
                    "scenario_id": "offline_constrained",
                    "state": {
                        "constraints": {"offline": True},
                        "permissions": ["read"]
                    },
                    "expected_actions": ["local_only", "cached_data"]
                }
            ],
            "pass_condition": "Agent should respect constraints in action selection"
        },
        difficulty="medium",
        seed=102
    ),
    AutonomyBenchmarkTask(
        task_id="auto_asf_004",
        criterion=AutonomyCriterion.ACTION_SELECTION_FREEDOM,
        name="User Preference Variation",
        description="Same goal with different user preferences",
        config={
            "goal": "Provide weather information",
            "scenarios": [
                {
                    "scenario_id": "detailed_preference",
                    "state": {
                        "user_preferences": {"detail_level": "high", "format": "technical"},
                        "user_expertise": "expert"
                    },
                    "expected_actions": ["detailed_forecast", "technical_data"]
                },
                {
                    "scenario_id": "simple_preference",
                    "state": {
                        "user_preferences": {"detail_level": "low", "format": "simple"},
                        "user_expertise": "novice"
                    },
                    "expected_actions": ["simple_summary", "basic_forecast"]
                },
                {
                    "scenario_id": "visual_preference",
                    "state": {
                        "user_preferences": {"detail_level": "medium", "format": "visual"},
                        "user_expertise": "intermediate"
                    },
                    "expected_actions": ["generate_chart", "visual_summary"]
                }
            ],
            "pass_condition": "Agent should adapt to user preferences"
        },
        difficulty="hard",
        seed=103
    )
]

# ============================================================
# Goal-Directed Persistence Tasks (4 tasks)
# ============================================================

GOAL_DIRECTED_PERSISTENCE_TASKS = [
    AutonomyBenchmarkTask(
        task_id="auto_gdp_001",
        criterion=AutonomyCriterion.GOAL_DIRECTED_PERSISTENCE,
        name="Simple Obstacle Recovery",
        description="Goal with single recoverable obstacle",
        config={
            "goal": "Retrieve user data from database",
            "max_steps": 15,
            "obstacles": [
                {"step": 3, "type": "connection_timeout", "recoverable": True}
            ],
            "pass_condition": "Agent should recover from timeout and complete task"
        },
        difficulty="easy",
        seed=110
    ),
    AutonomyBenchmarkTask(
        task_id="auto_gdp_002",
        criterion=AutonomyCriterion.GOAL_DIRECTED_PERSISTENCE,
        name="Multi-Obstacle Chain",
        description="Goal with multiple sequential obstacles",
        config={
            "goal": "Complete data processing pipeline",
            "max_steps": 25,
            "obstacles": [
                {"step": 3, "type": "rate_limit", "recoverable": True},
                {"step": 7, "type": "invalid_response", "recoverable": True},
                {"step": 12, "type": "resource_unavailable", "recoverable": True}
            ],
            "required_strategy_changes": 2,
            "pass_condition": "Agent should adapt strategy at least twice and persist"
        },
        difficulty="medium",
        seed=111
    ),
    AutonomyBenchmarkTask(
        task_id="auto_gdp_003",
        criterion=AutonomyCriterion.GOAL_DIRECTED_PERSISTENCE,
        name="Severe Obstacle Chain",
        description="Goal with difficult, compounding obstacles",
        config={
            "goal": "Deploy application to production",
            "max_steps": 30,
            "obstacles": [
                {"step": 2, "type": "build_failure", "recoverable": True},
                {"step": 5, "type": "test_failure", "recoverable": True},
                {"step": 8, "type": "deployment_blocked", "recoverable": True},
                {"step": 12, "type": "rollback_required", "recoverable": True},
                {"step": 18, "type": "permission_denied", "recoverable": True}
            ],
            "required_strategy_changes": 3,
            "pass_condition": "Agent should demonstrate significant strategy adaptation"
        },
        difficulty="hard",
        seed=112
    ),
    AutonomyBenchmarkTask(
        task_id="auto_gdp_004",
        criterion=AutonomyCriterion.GOAL_DIRECTED_PERSISTENCE,
        name="Partial Progress Preservation",
        description="Goal where partial progress must be preserved after failures",
        config={
            "goal": "Process 100 records with checkpointing",
            "max_steps": 40,
            "obstacles": [
                {"step": 10, "type": "process_crash", "data_loss_risk": True},
                {"step": 25, "type": "system_restart", "data_loss_risk": True}
            ],
            "required_behaviors": ["checkpoint", "resume_from_checkpoint"],
            "pass_condition": "Agent should checkpoint progress and resume after failures"
        },
        difficulty="hard",
        seed=113
    )
]

# ============================================================
# Dynamic Termination Tasks (4 tasks)
# ============================================================

DYNAMIC_TERMINATION_TASKS = [
    AutonomyBenchmarkTask(
        task_id="auto_dt_001",
        criterion=AutonomyCriterion.DYNAMIC_TERMINATION,
        name="Trivial vs Complex Goal Differentiation",
        description="Agent should take different steps for trivial vs complex goals",
        config={
            "goals": [
                {
                    "goal": "Return the value 42",
                    "complexity": "trivial",
                    "expected_step_range": (1, 2),
                    "state": {"answer": 42}
                },
                {
                    "goal": "Research and compare 5 ML frameworks",
                    "complexity": "complex",
                    "expected_step_range": (15, 30),
                    "state": {"requires_research": True}
                }
            ],
            "pass_condition": "Step counts should differ significantly between trivial and complex"
        },
        difficulty="easy",
        seed=120
    ),
    AutonomyBenchmarkTask(
        task_id="auto_dt_002",
        criterion=AutonomyCriterion.DYNAMIC_TERMINATION,
        name="Graduated Complexity",
        description="Goals of graduated complexity should have proportional step counts",
        config={
            "goals": [
                {
                    "goal": "List 3 Python libraries",
                    "complexity": "simple",
                    "expected_step_range": (1, 4)
                },
                {
                    "goal": "Compare 3 Python libraries with pros/cons",
                    "complexity": "medium",
                    "expected_step_range": (5, 12)
                },
                {
                    "goal": "Full analysis of 3 libraries with benchmarks and recommendations",
                    "complexity": "complex",
                    "expected_step_range": (12, 25)
                }
            ],
            "pass_condition": "Step counts should be monotonically increasing with complexity"
        },
        difficulty="medium",
        seed=121
    ),
    AutonomyBenchmarkTask(
        task_id="auto_dt_003",
        criterion=AutonomyCriterion.DYNAMIC_TERMINATION,
        name="Goal Satisfaction Recognition",
        description="Agent should recognize when goal is satisfied and stop",
        config={
            "goals": [
                {
                    "goal": "Find the capital of France",
                    "complexity": "simple",
                    "satisfaction_signal": {"answer_found": "Paris"},
                    "expected_step_range": (1, 3)
                },
                {
                    "goal": "Summarize this 10-page document",
                    "complexity": "medium",
                    "satisfaction_signal": {"summary_complete": True, "coverage": ">90%"},
                    "expected_step_range": (5, 15)
                }
            ],
            "pass_condition": "Agent should terminate upon goal satisfaction signals"
        },
        difficulty="medium",
        seed=122
    ),
    AutonomyBenchmarkTask(
        task_id="auto_dt_004",
        criterion=AutonomyCriterion.DYNAMIC_TERMINATION,
        name="Premature Termination Detection",
        description="Agent should NOT terminate before goal is satisfied",
        config={
            "goals": [
                {
                    "goal": "Complete all 5 subtasks",
                    "complexity": "medium",
                    "subtasks": ["A", "B", "C", "D", "E"],
                    "completion_criteria": "all_subtasks_done",
                    "expected_step_range": (5, 15)
                }
            ],
            "failure_mode": "Agent terminates after 2-3 subtasks",
            "pass_condition": "Agent should complete all subtasks before terminating"
        },
        difficulty="hard",
        seed=123
    )
]

# ============================================================
# Error Recovery Tasks (4 tasks)
# ============================================================

ERROR_RECOVERY_TASKS = [
    AutonomyBenchmarkTask(
        task_id="auto_er_001",
        criterion=AutonomyCriterion.ERROR_RECOVERY,
        name="Single Failure Type Recovery",
        description="Recovery from a single type of failure",
        config={
            "goal": "Fetch data from API",
            "failures": [
                {"type": "timeout", "description": "API request timed out"}
            ],
            "expected_recovery": "retry_with_backoff",
            "pass_condition": "Agent should attempt recovery (not give up immediately)"
        },
        difficulty="easy",
        seed=130
    ),
    AutonomyBenchmarkTask(
        task_id="auto_er_002",
        criterion=AutonomyCriterion.ERROR_RECOVERY,
        name="Varied Failure Type Recovery",
        description="Different recovery strategies for different failure types",
        config={
            "goal": "Complete data synchronization",
            "failures": [
                {"type": "tool_unavailable", "expected_recovery": "use_alternative"},
                {"type": "permission_denied", "expected_recovery": "request_permission"},
                {"type": "invalid_response", "expected_recovery": "validate_and_retry"}
            ],
            "pass_condition": "Agent should use different strategies for different failures"
        },
        difficulty="medium",
        seed=131
    ),
    AutonomyBenchmarkTask(
        task_id="auto_er_003",
        criterion=AutonomyCriterion.ERROR_RECOVERY,
        name="Cascading Failure Recovery",
        description="Recovery when initial recovery also fails",
        config={
            "goal": "Process critical transaction",
            "failures": [
                {"type": "primary_failure", "step": 2},
                {"type": "recovery_failure", "step": 3},  # Recovery attempt fails
                {"type": "secondary_recovery_needed", "step": 4}
            ],
            "required_recovery_depth": 2,
            "pass_condition": "Agent should have multi-level recovery strategies"
        },
        difficulty="hard",
        seed=132
    ),
    AutonomyBenchmarkTask(
        task_id="auto_er_004",
        criterion=AutonomyCriterion.ERROR_RECOVERY,
        name="Graceful Degradation",
        description="Recovery with graceful degradation when full recovery impossible",
        config={
            "goal": "Generate comprehensive report",
            "failures": [
                {"type": "data_source_unavailable", "critical": False},
                {"type": "visualization_failed", "critical": False},
                {"type": "export_failed", "critical": False}
            ],
            "acceptable_outcomes": [
                "partial_report",
                "text_only_report",
                "summary_only"
            ],
            "pass_condition": "Agent should gracefully degrade rather than fail completely"
        },
        difficulty="hard",
        seed=133
    )
]

# ============================================================
# Extended Duration Persistence Tasks (P3.2)
# Tests for long-horizon task persistence (50+ steps)
# ============================================================

EXTENDED_DURATION_TASKS: List[AutonomyBenchmarkTask] = [
    AutonomyBenchmarkTask(
        task_id="auto_ext_001",
        criterion=AutonomyCriterion.GOAL_DIRECTED_PERSISTENCE,
        name="50-Step Research Task",
        description="Extended research task requiring 50+ steps with multiple obstacles",
        config={
            "goal": "Comprehensive literature review on transformer architectures",
            "min_steps": 50,
            "max_steps": 100,
            "obstacle_injection_rate": 0.1,  # 10% chance per step
            "obstacles": [
                {"type": "rate_limit", "probability": 0.3},
                {"type": "timeout", "probability": 0.2},
                {"type": "partial_result", "probability": 0.3},
                {"type": "conflicting_info", "probability": 0.2}
            ],
            "checkpoints": [10, 25, 50, 75],  # Steps where progress is validated
            "pass_condition": "Agent completes 80% of checkpoints with goal alignment >0.7"
        },
        difficulty="hard",
        seed=42
    ),
    AutonomyBenchmarkTask(
        task_id="auto_ext_002",
        criterion=AutonomyCriterion.GOAL_DIRECTED_PERSISTENCE,
        name="Multi-Phase Project",
        description="Extended task with 5 distinct phases requiring strategy adaptation",
        config={
            "goal": "Build and deploy a data processing pipeline",
            "phases": [
                {"name": "research", "steps": 10, "success_criteria": "requirements_gathered"},
                {"name": "design", "steps": 15, "success_criteria": "architecture_defined"},
                {"name": "implement", "steps": 25, "success_criteria": "code_complete"},
                {"name": "test", "steps": 15, "success_criteria": "tests_passing"},
                {"name": "deploy", "steps": 10, "success_criteria": "deployment_verified"}
            ],
            "min_total_steps": 50,
            "inter_phase_obstacles": True,
            "pass_condition": "Agent completes all 5 phases with strategy adaptation between phases"
        },
        difficulty="hard",
        seed=43
    ),
    AutonomyBenchmarkTask(
        task_id="auto_ext_003",
        criterion=AutonomyCriterion.GOAL_DIRECTED_PERSISTENCE,
        name="Evolving Requirements",
        description="Long task where requirements change mid-execution",
        config={
            "initial_goal": "Create a report on Q3 sales performance",
            "goal_modifications": [
                {"step": 20, "modification": "Include Q2 comparison"},
                {"step": 40, "modification": "Add regional breakdown"},
                {"step": 60, "modification": "Generate executive summary"}
            ],
            "min_steps": 80,
            "pass_condition": "Agent incorporates all modifications while maintaining core goal"
        },
        difficulty="hard",
        seed=44
    ),
    AutonomyBenchmarkTask(
        task_id="auto_ext_004",
        criterion=AutonomyCriterion.GOAL_DIRECTED_PERSISTENCE,
        name="Interruption Recovery",
        description="Long task with multiple interruptions requiring context recovery",
        config={
            "goal": "Complete end-to-end data analysis",
            "interruptions": [
                {"step": 15, "type": "context_loss", "recovery_required": True},
                {"step": 35, "type": "partial_state_loss", "recovery_required": True},
                {"step": 55, "type": "priority_interrupt", "recovery_required": False},
                {"step": 70, "type": "context_loss", "recovery_required": True}
            ],
            "min_steps": 80,
            "pass_condition": "Agent recovers from >75% of interruptions and completes goal"
        },
        difficulty="hard",
        seed=45
    ),
]

# ============================================================
# Combined Benchmark Collection
# ============================================================

ALL_AUTONOMY_TASKS: List[AutonomyBenchmarkTask] = (
    ACTION_SELECTION_FREEDOM_TASKS +
    GOAL_DIRECTED_PERSISTENCE_TASKS +
    DYNAMIC_TERMINATION_TASKS +
    ERROR_RECOVERY_TASKS +
    EXTENDED_DURATION_TASKS  # P3.2: Extended duration tasks
)

TASKS_BY_CRITERION: Dict[AutonomyCriterion, List[AutonomyBenchmarkTask]] = {
    AutonomyCriterion.ACTION_SELECTION_FREEDOM: ACTION_SELECTION_FREEDOM_TASKS,
    AutonomyCriterion.GOAL_DIRECTED_PERSISTENCE: GOAL_DIRECTED_PERSISTENCE_TASKS + EXTENDED_DURATION_TASKS,
    AutonomyCriterion.DYNAMIC_TERMINATION: DYNAMIC_TERMINATION_TASKS,
    AutonomyCriterion.ERROR_RECOVERY: ERROR_RECOVERY_TASKS,
}


def get_autonomy_benchmark_statistics() -> Dict[str, Any]:
    """Get statistics about the autonomy benchmarks."""
    stats = {
        "total_tasks": len(ALL_AUTONOMY_TASKS),
        "criteria_covered": len(TASKS_BY_CRITERION),
        "tasks_per_criterion": {},
        "difficulty_distribution": {"easy": 0, "medium": 0, "hard": 0}
    }

    for criterion, tasks in TASKS_BY_CRITERION.items():
        stats["tasks_per_criterion"][criterion.value] = len(tasks)

    for task in ALL_AUTONOMY_TASKS:
        if task.difficulty in stats["difficulty_distribution"]:
            stats["difficulty_distribution"][task.difficulty] += 1

    return stats


# ============================================================
# Benchmark Runner
# ============================================================

@dataclass
class AutonomyBenchmarkResult:
    """Result from running an autonomy benchmark task."""
    task_id: str
    criterion: AutonomyCriterion
    passed: bool
    details: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize for storage/logging."""
        return {
            "task_id": self.task_id,
            "criterion": self.criterion.value,
            "passed": self.passed,
            "details": self.details
        }


class AutonomyBenchmarkRunner:
    """
    Runs autonomy benchmark tasks.

    Usage:
        ```python
        runner = AutonomyBenchmarkRunner(validator)

        # Run all tasks
        results = runner.run_all(agent)

        # Run tasks for specific criterion
        results = runner.run_criterion(agent, AutonomyCriterion.ERROR_RECOVERY)

        # Get summary
        summary = runner.get_summary(results)
        ```
    """

    def __init__(self, validator: Any, verbose: bool = False):
        """
        Initialize benchmark runner.

        Args:
            validator: AutonomyValidator instance
            verbose: Enable verbose logging
        """
        self.validator = validator
        self.verbose = verbose

    def run_task(
        self,
        agent: Any,
        task: AutonomyBenchmarkTask
    ) -> AutonomyBenchmarkResult:
        """
        Run a single benchmark task.

        Args:
            agent: Agent to test
            task: Task to run

        Returns:
            AutonomyBenchmarkResult
        """
        criterion = task.criterion

        # Run appropriate validation based on criterion
        if criterion == AutonomyCriterion.ACTION_SELECTION_FREEDOM:
            from .autonomy_validator import TestScenario

            scenarios = [
                TestScenario(
                    scenario_id=s["scenario_id"],
                    goal=task.config["goal"],
                    state=s["state"],
                    available_actions=s.get("expected_actions", [])
                )
                for s in task.config.get("scenarios", [])
            ]

            result = self.validator.validate_action_selection_freedom(
                agent, test_scenarios=scenarios
            )

        elif criterion == AutonomyCriterion.GOAL_DIRECTED_PERSISTENCE:
            result = self.validator.validate_goal_directed_persistence(
                agent,
                goal=task.config.get("goal", "Complete task"),
                max_steps=task.config.get("max_steps", 20),
                obstacle_difficulty=task.difficulty
            )

        elif criterion == AutonomyCriterion.DYNAMIC_TERMINATION:
            goals = task.config.get("goals", [])
            result = self.validator.validate_dynamic_termination(
                agent, goals=goals
            )

        elif criterion == AutonomyCriterion.ERROR_RECOVERY:
            failures = task.config.get("failures", [])
            failure_types = [f["type"] for f in failures]
            result = self.validator.validate_error_recovery(
                agent,
                goal=task.config.get("goal", "Complete task"),
                failure_types=failure_types
            )

        else:
            result = None

        passed = result.passed if result else False

        return AutonomyBenchmarkResult(
            task_id=task.task_id,
            criterion=criterion,
            passed=passed,
            details={
                "task_config": task.config,
                "validation_result": result.to_dict() if result else None
            }
        )

    def run_criterion(
        self,
        agent: Any,
        criterion: AutonomyCriterion
    ) -> List[AutonomyBenchmarkResult]:
        """Run all tasks for a specific criterion."""
        tasks = TASKS_BY_CRITERION.get(criterion, [])
        return [self.run_task(agent, task) for task in tasks]

    def run_all(self, agent: Any) -> List[AutonomyBenchmarkResult]:
        """Run all benchmark tasks."""
        return [self.run_task(agent, task) for task in ALL_AUTONOMY_TASKS]

    def get_summary(
        self,
        results: List[AutonomyBenchmarkResult]
    ) -> Dict[str, Any]:
        """
        Get summary statistics from benchmark results.

        Args:
            results: List of benchmark results

        Returns:
            Summary dictionary
        """
        summary = {
            "total_tasks": len(results),
            "total_passed": sum(1 for r in results if r.passed),
            "pass_rate": sum(1 for r in results if r.passed) / len(results) if results else 0,
            "by_criterion": {}
        }

        for criterion in AutonomyCriterion:
            criterion_results = [r for r in results if r.criterion == criterion]
            if criterion_results:
                passed = sum(1 for r in criterion_results if r.passed)
                summary["by_criterion"][criterion.value] = {
                    "total": len(criterion_results),
                    "passed": passed,
                    "pass_rate": passed / len(criterion_results)
                }

        return summary


# Print statistics when module is run directly
if __name__ == "__main__":
    import json
    stats = get_autonomy_benchmark_statistics()
    print("Autonomy Benchmark Statistics:")
    print(json.dumps(stats, indent=2))
