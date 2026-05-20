"""Evaluation framework for agentic AI systems.

This module provides comprehensive evaluation metrics and tools for assessing
agent performance, including cost-efficiency metrics, long-horizon evaluation,
and safety incident tracking.

Key Features:
- Cost-Normalized Success Rate (CNSR) for cost-efficiency evaluation
- Rolling window metrics for temporal performance analysis
- Goal drift detection for long-horizon task monitoring
- Incident tracking for safety evaluation
"""

from typing import List, Optional
from dataclasses import dataclass


@dataclass
class EvaluationResult:
    """Container for comprehensive evaluation results.

    Attributes:
        success_rate: Proportion of successful tasks (0.0 to 1.0)
        mean_cost: Average cost per task in USD
        cnsr: Cost-Normalized Success Rate (higher is better)
        total_tasks: Total number of tasks evaluated
        total_successes: Number of successful tasks
        total_cost: Total cost across all tasks in USD
    """

    success_rate: float
    mean_cost: float
    cnsr: float
    total_tasks: int
    total_successes: int
    total_cost: float


def calculate_cnsr(
    successes: int,
    total_tasks: int,
    total_cost: float,
) -> float:
    """Calculate Cost-Normalized Success Rate (CNSR).

    CNSR balances task success against economic cost, providing a more
    realistic measure of agent efficiency than raw success rate alone.

    Formula:
        CNSR = Success Rate / Mean Cost per Task

    A higher CNSR indicates better cost-efficiency. This metric penalizes
    systems that achieve high success through expensive brute-force approaches.

    Args:
        successes: Number of successful tasks
        total_tasks: Total number of tasks attempted
        total_cost: Total cost in USD across all tasks

    Returns:
        CNSR value (higher is better). Returns inf if cost is 0 with successes,
        or 0.0 if no tasks were attempted.

    Example:
        >>> # System A: 80% success at $0.50/task = CNSR 1.6
        >>> cnsr_a = calculate_cnsr(80, 100, 50.0)
        >>> # System B: 90% success at $2.00/task = CNSR 0.45
        >>> cnsr_b = calculate_cnsr(90, 100, 200.0)
        >>> # System A is more cost-effective despite lower raw success rate
    """
    if total_tasks == 0:
        return 0.0
    if total_cost == 0:
        return float("inf") if successes > 0 else 0.0

    success_rate = successes / total_tasks
    mean_cost = total_cost / total_tasks
    return success_rate / mean_cost


def evaluate_agent(
    successes: int,
    total_tasks: int,
    total_cost: float,
) -> EvaluationResult:
    """Perform comprehensive agent evaluation.

    Computes all standard evaluation metrics in a single call.

    Args:
        successes: Number of successful tasks
        total_tasks: Total number of tasks
        total_cost: Total cost in USD

    Returns:
        EvaluationResult with all computed metrics

    Example:
        >>> result = evaluate_agent(successes=80, total_tasks=100, total_cost=50.0)
        >>> print(f"Success: {result.success_rate:.2%}")  # 80.00%
        >>> print(f"CNSR: {result.cnsr:.2f}")            # 1.60
    """
    success_rate = successes / total_tasks if total_tasks > 0 else 0.0
    mean_cost = total_cost / total_tasks if total_tasks > 0 else 0.0
    cnsr = calculate_cnsr(successes, total_tasks, total_cost)

    return EvaluationResult(
        success_rate=success_rate,
        mean_cost=mean_cost,
        cnsr=cnsr,
        total_tasks=total_tasks,
        total_successes=successes,
        total_cost=total_cost,
    )


def rolling_window_success(
    results: List[bool],
    window_size: int = 10,
) -> List[float]:
    """Calculate rolling-window success rate over time.

    Computes success rate within a sliding window, revealing performance
    trends and degradation patterns that single aggregate metrics miss.

    This is essential for evaluating long-running agents where performance
    may degrade over time due to memory issues, goal drift, or other factors.

    Args:
        results: List of task outcomes (True=success, False=failure)
        window_size: Size of the rolling window (default: 10)

    Returns:
        List of rolling success rates, one per task. Early values use
        smaller windows until window_size tasks have accumulated.

    Example:
        >>> results = [True, True, False, True, False, True, True, True, False, True]
        >>> rolling = rolling_window_success(results, window_size=5)
        >>> print(f"Recent performance: {rolling[-1]:.2%}")
        >>> # Detect degradation: compare rolling[-1] vs rolling[0]
    """
    if not results:
        return []

    rolling = []
    for i in range(len(results)):
        start = max(0, i - window_size + 1)
        window = results[start : i + 1]
        success_rate = sum(window) / len(window)
        rolling.append(success_rate)

    return rolling


def goal_drift_score(
    original_goal_embedding: List[float],
    current_goal_embedding: List[float],
) -> float:
    """Calculate goal drift score between original and current objectives.

    Measures how much an agent's current behavior has diverged from its
    original goal. Uses cosine distance between goal embeddings.

    Formula:
        Drift = 1 - cosine_similarity(original_goal, current_goal)

    A score of 0 indicates no drift (goals aligned), while 1 indicates
    complete divergence. Increasing drift over time suggests the agent
    is losing track of its original objectives.

    Args:
        original_goal_embedding: Embedding vector of original goal
        current_goal_embedding: Embedding vector of current inferred goal

    Returns:
        Drift score between 0 (no drift) and 1 (complete drift)

    Example:
        >>> # Use your preferred embedding model
        >>> g0 = embed("Summarize the document")
        >>> gt = embed("Rewrite the entire document")  # Drifted!
        >>> drift = goal_drift_score(g0, gt)
        >>> if drift > 0.3:
        ...     print("Warning: significant goal drift detected")
    """
    # Cosine similarity computation
    dot_product = sum(a * b for a, b in zip(original_goal_embedding, current_goal_embedding))
    norm_a = sum(a * a for a in original_goal_embedding) ** 0.5
    norm_b = sum(b * b for b in current_goal_embedding) ** 0.5

    if norm_a == 0 or norm_b == 0:
        return 1.0  # Maximum drift if embeddings are zero vectors

    similarity = dot_product / (norm_a * norm_b)
    return 1.0 - similarity


@dataclass
class IncidentTracker:
    """Track safety-relevant incidents during agent operation.

    Monitors various types of safety events that occur during agent
    execution. A well-behaved agent should have stable or decreasing
    incident rates over time.

    Attributes:
        human_interventions: Count of times human intervention was required
        guardrail_activations: Count of guardrail/safety filter triggers
        constraint_violations: Count of policy or constraint violations
        unexpected_terminations: Count of unexpected agent terminations
    """

    human_interventions: int = 0
    guardrail_activations: int = 0
    constraint_violations: int = 0
    unexpected_terminations: int = 0

    @property
    def total_incidents(self) -> int:
        """Total number of incidents across all categories."""
        return (
            self.human_interventions
            + self.guardrail_activations
            + self.constraint_violations
            + self.unexpected_terminations
        )

    def incident_rate(self, total_tasks: int) -> float:
        """Calculate incident rate per task.

        Args:
            total_tasks: Total number of tasks executed

        Returns:
            Incidents per task (0.0 if no tasks)
        """
        if total_tasks == 0:
            return 0.0
        return self.total_incidents / total_tasks

    def record_incident(self, incident_type: str) -> None:
        """Record a new incident.

        Args:
            incident_type: One of 'human_intervention', 'guardrail',
                          'violation', or 'termination'
        """
        if incident_type == "human_intervention":
            self.human_interventions += 1
        elif incident_type == "guardrail":
            self.guardrail_activations += 1
        elif incident_type == "violation":
            self.constraint_violations += 1
        elif incident_type == "termination":
            self.unexpected_terminations += 1


@dataclass
class CostTrajectory:
    """Track cost evolution over time for budget analysis.

    Monitors how costs accumulate relative to successes, enabling
    detection of cost inefficiency patterns and budget planning.

    Attributes:
        costs: List of costs for each task
        successes: List of success indicators for each task
    """

    costs: List[float]
    successes: List[bool]

    def cost_per_success(self) -> List[float]:
        """Calculate cumulative cost per successful task over time.

        Returns:
            List of cumulative cost per success values. Values are inf
            until the first success occurs.
        """
        trajectory = []
        cumulative_cost = 0.0
        cumulative_successes = 0

        for cost, success in zip(self.costs, self.successes):
            cumulative_cost += cost
            if success:
                cumulative_successes += 1

            if cumulative_successes > 0:
                trajectory.append(cumulative_cost / cumulative_successes)
            else:
                trajectory.append(float("inf"))

        return trajectory

    def cost_variance(self) -> float:
        """Calculate cost variance (unpredictability).

        Higher variance indicates less predictable costs, which can
        complicate budget planning.

        Returns:
            Variance of costs (0.0 if fewer than 2 tasks)
        """
        if len(self.costs) < 2:
            return 0.0
        mean = sum(self.costs) / len(self.costs)
        return sum((c - mean) ** 2 for c in self.costs) / len(self.costs)


class LongHorizonEvaluator:
    """Comprehensive evaluator for long-running agent deployments.

    Tracks performance over extended operation periods (50-500+ tasks),
    revealing patterns invisible in single-task evaluation such as
    performance degradation, increasing costs, and incident trends.

    Example:
        >>> evaluator = LongHorizonEvaluator(window_size=50)
        >>> for success, cost in agent_results:
        ...     evaluator.record_task(success=success, cost=cost)
        >>> metrics = evaluator.get_metrics()
        >>> print(f"CNSR: {metrics['cnsr']:.2f}")
        >>> print(f"Trend: {metrics['rolling_success'][-1]:.2%}")
    """

    def __init__(self, window_size: int = 50):
        """Initialize the long-horizon evaluator.

        Args:
            window_size: Size of rolling window for trend analysis
        """
        self.window_size = window_size
        self.results: List[bool] = []
        self.costs: List[float] = []
        self.incident_tracker = IncidentTracker()

    def record_task(
        self,
        success: bool,
        cost: float,
        incident_type: Optional[str] = None,
    ) -> None:
        """Record a task result.

        Args:
            success: Whether the task succeeded
            cost: Cost of the task in USD
            incident_type: Type of incident if any occurred.
                          One of: 'human_intervention', 'guardrail',
                          'violation', 'termination', or None
        """
        self.results.append(success)
        self.costs.append(cost)

        if incident_type:
            self.incident_tracker.record_incident(incident_type)

    def get_metrics(self) -> dict:
        """Get comprehensive evaluation metrics.

        Returns:
            Dictionary containing:
            - total_tasks: Number of tasks evaluated
            - success_rate: Overall success rate
            - mean_cost: Average cost per task
            - cnsr: Cost-Normalized Success Rate
            - rolling_success: List of rolling success rates
            - incident_rate: Incidents per task
            - cost_variance: Cost unpredictability measure
        """
        total_tasks = len(self.results)
        total_successes = sum(self.results)
        total_cost = sum(self.costs)

        # Calculate cost variance
        if len(self.costs) >= 2:
            mean_cost = total_cost / total_tasks if total_tasks else 0
            cost_variance = sum((c - mean_cost) ** 2 for c in self.costs) / len(self.costs)
        else:
            cost_variance = 0.0

        return {
            "total_tasks": total_tasks,
            "success_rate": total_successes / total_tasks if total_tasks else 0,
            "mean_cost": total_cost / total_tasks if total_tasks else 0,
            "cnsr": calculate_cnsr(total_successes, total_tasks, total_cost),
            "rolling_success": rolling_window_success(self.results, self.window_size),
            "incident_rate": self.incident_tracker.incident_rate(total_tasks),
            "cost_variance": cost_variance,
            "incidents": {
                "human_interventions": self.incident_tracker.human_interventions,
                "guardrail_activations": self.incident_tracker.guardrail_activations,
                "constraint_violations": self.incident_tracker.constraint_violations,
                "unexpected_terminations": self.incident_tracker.unexpected_terminations,
            },
        }

    def reset(self) -> None:
        """Reset all tracked metrics."""
        self.results = []
        self.costs = []
        self.incident_tracker = IncidentTracker()


from .metrics import (
    compute_cnsr,
    compute_accuracy,
    compute_task_completion_rate,
    compute_efficiency_score,
    compute_f1_score,
    compute_mean_reciprocal_rank,
    MetricsCollector,
    AggregatedMetrics,
    # Full cost model (Section XI-C)
    TaskCostBreakdown,
    TaskResult,
    compute_cnsr_from_results,
    compute_cost_from_usage,
    DEFAULT_COST_RATES,
)
from .benchmarks import (
    Benchmark,
    BenchmarkTask,
    BenchmarkResult,
    TaskDifficulty,
    TaskCategory,
    ReasoningBenchmark,
    ToolUseBenchmark,
    PlanningBenchmark,
    MultiAgentBenchmark,
    create_benchmark_suite,
)
from .harness import (
    EvaluationHarness,
    EvaluationConfig,
    run_evaluation,
)
from .failure_taxonomy import (
    FailurePathology,
    PathologySeverity,
    PathologyIncident,
    PathologyMitigation,
    FailureDetector,
    MITIGATION_STRATEGIES,
    map_incident_type_to_pathology,
)
from .pathology_benchmarks import (
    PathologyBenchmarkTask,
    PathologyBenchmarkResult,
    PathologyBenchmarkRunner,
    ALL_PATHOLOGY_TASKS,
    TASKS_BY_PATHOLOGY,
    get_benchmark_statistics as get_pathology_benchmark_statistics,
)
from .autonomy_validator import (
    AutonomyValidator,
    AutonomyLevel,
    AutonomyCriterion,
    AutonomyCriteria,
    AutonomyThresholds,  # P2.1: Configurable thresholds
    AutonomyValidationResult,
    TestResult,
    TestScenario,
    TestScenarioGenerator,
    ObstacleInjector,
    FailureInjector,
    GenuineAgent,
    ScriptedAgent,
    FragileAgent,
    FixedStepAgent,
)
from .autonomy_benchmarks import (
    AutonomyBenchmarkTask,
    AutonomyBenchmarkResult,
    AutonomyBenchmarkRunner,
    ALL_AUTONOMY_TASKS,
    TASKS_BY_CRITERION,
    get_autonomy_benchmark_statistics,
)
from .cnsr_benchmark import (
    CNSRBenchmark,
    BenchmarkConfig as CNSRBenchmarkConfig,
    AgentEvaluationResult,
    CNSRResults,
    ParetoAnalysis,
    ParetoPoint,
    DivergenceReport,
    RankingDivergence,
    RankingMethod,
    SensitivityReport,
    SensitivityResult,
    MODEL_COST_RATES,
    create_cnsr_benchmark,
    quick_cnsr_comparison,
)

__all__ = [
    # Base metrics
    "EvaluationResult",
    "calculate_cnsr",
    "evaluate_agent",
    "rolling_window_success",
    "goal_drift_score",
    "IncidentTracker",
    "CostTrajectory",
    "LongHorizonEvaluator",
    # Extended metrics
    "compute_cnsr",
    "compute_accuracy",
    "compute_task_completion_rate",
    "compute_efficiency_score",
    "compute_f1_score",
    "compute_mean_reciprocal_rank",
    "MetricsCollector",
    "AggregatedMetrics",
    # Full cost model (Section XI-C)
    "TaskCostBreakdown",
    "TaskResult",
    "compute_cnsr_from_results",
    "compute_cost_from_usage",
    "DEFAULT_COST_RATES",
    # Benchmarks
    "Benchmark",
    "BenchmarkTask",
    "BenchmarkResult",
    "TaskDifficulty",
    "TaskCategory",
    "ReasoningBenchmark",
    "ToolUseBenchmark",
    "PlanningBenchmark",
    "MultiAgentBenchmark",
    "create_benchmark_suite",
    # Harness
    "EvaluationHarness",
    "EvaluationConfig",
    "run_evaluation",
    # Failure Taxonomy (Section XV)
    "FailurePathology",
    "PathologySeverity",
    "PathologyIncident",
    "PathologyMitigation",
    "FailureDetector",
    "MITIGATION_STRATEGIES",
    "map_incident_type_to_pathology",
    # Pathology Benchmarks
    "PathologyBenchmarkTask",
    "PathologyBenchmarkResult",
    "PathologyBenchmarkRunner",
    "ALL_PATHOLOGY_TASKS",
    "TASKS_BY_PATHOLOGY",
    "get_pathology_benchmark_statistics",
    # Autonomy Validator (Section IV-A)
    "AutonomyValidator",
    "AutonomyLevel",
    "AutonomyCriterion",
    "AutonomyCriteria",
    "AutonomyThresholds",  # P2.1: Configurable thresholds
    "AutonomyValidationResult",
    "TestResult",
    "TestScenario",
    "TestScenarioGenerator",
    "ObstacleInjector",
    "FailureInjector",
    # Mock Agents for Testing
    "GenuineAgent",
    "ScriptedAgent",
    "FragileAgent",
    "FixedStepAgent",
    # Autonomy Benchmarks
    "AutonomyBenchmarkTask",
    "AutonomyBenchmarkResult",
    "AutonomyBenchmarkRunner",
    "ALL_AUTONOMY_TASKS",
    "TASKS_BY_CRITERION",
    "get_autonomy_benchmark_statistics",
    # CNSR Benchmark (Section XI-C-3)
    "CNSRBenchmark",
    "CNSRBenchmarkConfig",
    "AgentEvaluationResult",
    "CNSRResults",
    "ParetoAnalysis",
    "ParetoPoint",
    "DivergenceReport",
    "RankingDivergence",
    "RankingMethod",
    "SensitivityReport",
    "SensitivityResult",
    "MODEL_COST_RATES",
    "create_cnsr_benchmark",
    "quick_cnsr_comparison",
]
