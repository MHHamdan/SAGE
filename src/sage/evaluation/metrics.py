"""Evaluation metrics for agent benchmarking.

Implements the cost model from Section XI-C of the paper:
    C_total = C_inference + C_tools + C_latency + C_human

And the CNSR metric from Equation 6:
    CNSR = Success_Rate / Mean_Cost
"""

import logging
import math
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
from collections import defaultdict

logger = logging.getLogger(__name__)


# =============================================================================
# TASK RESULT WITH FULL COST MODEL (Section XI-C)
# =============================================================================

@dataclass
class TaskCostBreakdown:
    """
    Full cost breakdown for a single task per Equation 5 (Section XI-C).

    C_total = C_inference + C_tools + C_latency + C_human

    Attributes:
        inference_cost: Token costs (input + output tokens * pricing)
        tool_cost: Tool invocation costs (API fees, compute)
        latency_cost: Time cost (seconds * opportunity cost rate)
        human_cost: Human intervention costs (approvals, corrections)
    """
    inference_cost: float = 0.0  # C_inference: Token costs
    tool_cost: float = 0.0       # C_tools: Tool invocation costs
    latency_cost: float = 0.0    # C_latency: Time costs
    human_cost: float = 0.0      # C_human: Human intervention costs

    @property
    def total_cost(self) -> float:
        """Total cost per Equation 5."""
        return self.inference_cost + self.tool_cost + self.latency_cost + self.human_cost

    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary."""
        return {
            "inference_cost": self.inference_cost,
            "tool_cost": self.tool_cost,
            "latency_cost": self.latency_cost,
            "human_cost": self.human_cost,
            "total_cost": self.total_cost
        }

    def __add__(self, other: "TaskCostBreakdown") -> "TaskCostBreakdown":
        """Add two cost breakdowns."""
        return TaskCostBreakdown(
            inference_cost=self.inference_cost + other.inference_cost,
            tool_cost=self.tool_cost + other.tool_cost,
            latency_cost=self.latency_cost + other.latency_cost,
            human_cost=self.human_cost + other.human_cost
        )


@dataclass
class TaskResult:
    """
    Complete result for a single task including full cost breakdown.

    Used for computing CNSR with the proper 4-component cost model.
    """
    task_id: str
    success: bool
    cost: TaskCostBreakdown = field(default_factory=TaskCostBreakdown)
    duration_seconds: float = 0.0
    steps_taken: int = 0
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "task_id": self.task_id,
            "success": self.success,
            "cost": self.cost.to_dict(),
            "duration_seconds": self.duration_seconds,
            "steps_taken": self.steps_taken,
            "error": self.error,
            "metadata": self.metadata
        }


def compute_cnsr_from_results(
    results: List[TaskResult],
    epsilon: float = 1e-6
) -> Dict[str, float]:
    """
    Compute CNSR using the full 4-component cost model (Section XI-C).

    CNSR = Success_Rate / Mean_Cost (Equation 6)

    Where Mean_Cost uses:
        C_total = C_inference + C_tools + C_latency + C_human (Equation 5)

    Args:
        results: List of TaskResult with full cost breakdowns
        epsilon: Small value to prevent division by zero

    Returns:
        Dictionary with CNSR and breakdown metrics
    """
    if not results:
        return {
            "cnsr": 0.0,
            "success_rate": 0.0,
            "mean_total_cost": 0.0,
            "total_tasks": 0,
            "cost_breakdown": TaskCostBreakdown().to_dict()
        }

    total_tasks = len(results)
    successes = sum(1 for r in results if r.success)
    success_rate = successes / total_tasks

    # Aggregate costs using full model
    total_cost = TaskCostBreakdown()
    for r in results:
        total_cost = total_cost + r.cost

    mean_total_cost = total_cost.total_cost / total_tasks

    # Compute CNSR
    if mean_total_cost < epsilon:
        cnsr = float('inf') if success_rate > 0 else 0.0
    else:
        cnsr = success_rate / mean_total_cost

    return {
        "cnsr": cnsr,
        "success_rate": success_rate,
        "mean_total_cost": mean_total_cost,
        "total_tasks": total_tasks,
        "total_successes": successes,
        "cost_breakdown": {
            "mean_inference_cost": total_cost.inference_cost / total_tasks,
            "mean_tool_cost": total_cost.tool_cost / total_tasks,
            "mean_latency_cost": total_cost.latency_cost / total_tasks,
            "mean_human_cost": total_cost.human_cost / total_tasks,
            "total_inference_cost": total_cost.inference_cost,
            "total_tool_cost": total_cost.tool_cost,
            "total_latency_cost": total_cost.latency_cost,
            "total_human_cost": total_cost.human_cost,
        }
    }


# Default cost rates for compute_cost_from_usage()
DEFAULT_COST_RATES = {
    "token_input_per_1k": 0.0,      # $0 for Ollama (local)
    "token_output_per_1k": 0.0,     # $0 for Ollama (local)
    "tool_call_cost": 0.001,        # $0.001 per tool call
    "latency_per_second": 0.0001,   # $0.0001 per second
    "human_intervention_cost": 5.0  # $5 per intervention
}


def compute_cost_from_usage(
    input_tokens: int = 0,
    output_tokens: int = 0,
    tool_calls: int = 0,
    latency_seconds: float = 0.0,
    human_interventions: int = 0,
    rates: Optional[Dict[str, float]] = None
) -> TaskCostBreakdown:
    """
    Compute cost breakdown from usage metrics.

    This is a convenience function for computing costs from raw usage data.

    Args:
        input_tokens: Number of input tokens
        output_tokens: Number of output tokens
        tool_calls: Number of tool invocations
        latency_seconds: Total execution time
        human_interventions: Number of human interventions
        rates: Custom cost rates (uses DEFAULT_COST_RATES if None)

    Returns:
        TaskCostBreakdown with all components
    """
    rates = rates or DEFAULT_COST_RATES

    inference_cost = (
        (input_tokens / 1000) * rates.get("token_input_per_1k", 0) +
        (output_tokens / 1000) * rates.get("token_output_per_1k", 0)
    )

    tool_cost = tool_calls * rates.get("tool_call_cost", 0.001)
    latency_cost = latency_seconds * rates.get("latency_per_second", 0.0001)
    human_cost = human_interventions * rates.get("human_intervention_cost", 5.0)

    return TaskCostBreakdown(
        inference_cost=inference_cost,
        tool_cost=tool_cost,
        latency_cost=latency_cost,
        human_cost=human_cost
    )


def compute_cnsr(
    success_rate: float,
    mean_cost: float,
    epsilon: float = 1e-6,
) -> float:
    """Compute Cost-Normalized Success Rate.

    CNSR = Success_Rate / Mean_Cost

    Args:
        success_rate: Task success rate (0-1)
        mean_cost: Mean cost per task in USD
        epsilon: Small value to prevent division by zero

    Returns:
        CNSR score (higher is better)
    """
    if mean_cost < epsilon:
        return float('inf') if success_rate > 0 else 0.0
    return success_rate / mean_cost


def compute_accuracy(
    predictions: List[Any],
    ground_truth: List[Any],
) -> float:
    """Compute accuracy between predictions and ground truth.

    Args:
        predictions: List of predicted values
        ground_truth: List of true values

    Returns:
        Accuracy (0-1)
    """
    if not predictions or not ground_truth:
        return 0.0

    if len(predictions) != len(ground_truth):
        raise ValueError("Predictions and ground truth must have same length")

    correct = sum(p == g for p, g in zip(predictions, ground_truth))
    return correct / len(predictions)


def compute_task_completion_rate(
    completed: int,
    total: int,
) -> float:
    """Compute task completion rate.

    Args:
        completed: Number of completed tasks
        total: Total number of tasks

    Returns:
        Completion rate (0-1)
    """
    if total == 0:
        return 0.0
    return completed / total


def compute_efficiency_score(
    success_rate: float,
    avg_steps: float,
    optimal_steps: float,
    cost_weight: float = 0.5,
) -> float:
    """Compute efficiency score combining success and step efficiency.

    Score = success_rate * (1 - cost_weight + cost_weight * (optimal_steps / avg_steps))

    Args:
        success_rate: Task success rate
        avg_steps: Average steps taken
        optimal_steps: Optimal number of steps
        cost_weight: Weight for step efficiency (0-1)

    Returns:
        Efficiency score (0-1)
    """
    if avg_steps <= 0:
        return success_rate

    step_efficiency = min(1.0, optimal_steps / avg_steps)
    return success_rate * (1 - cost_weight + cost_weight * step_efficiency)


def compute_f1_score(
    predictions: List[bool],
    ground_truth: List[bool],
) -> Tuple[float, float, float]:
    """Compute precision, recall, and F1 score.

    Args:
        predictions: List of binary predictions
        ground_truth: List of binary ground truth

    Returns:
        Tuple of (precision, recall, f1_score)
    """
    if len(predictions) != len(ground_truth):
        raise ValueError("Predictions and ground truth must have same length")

    tp = sum(p and g for p, g in zip(predictions, ground_truth))
    fp = sum(p and not g for p, g in zip(predictions, ground_truth))
    fn = sum(not p and g for p, g in zip(predictions, ground_truth))

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0

    if precision + recall == 0:
        f1 = 0.0
    else:
        f1 = 2 * precision * recall / (precision + recall)

    return precision, recall, f1


def compute_mean_reciprocal_rank(
    rankings: List[int],
) -> float:
    """Compute Mean Reciprocal Rank (MRR).

    Args:
        rankings: List of ranks for correct answers (1-indexed)

    Returns:
        MRR score
    """
    if not rankings:
        return 0.0

    reciprocals = [1.0 / r if r > 0 else 0.0 for r in rankings]
    return sum(reciprocals) / len(reciprocals)


@dataclass
class MetricValue:
    """A metric value with metadata."""
    name: str
    value: float
    unit: str = ""
    lower_is_better: bool = False


@dataclass
class MetricsCollector:
    """Collects and aggregates evaluation metrics.

    Example:
        >>> collector = MetricsCollector()
        >>> collector.record("success", 1.0)
        >>> collector.record("cost", 0.05)
        >>> collector.record("steps", 5)
        >>> summary = collector.get_summary()
    """

    _metrics: Dict[str, List[float]] = field(default_factory=lambda: defaultdict(list))
    _metadata: Dict[str, Dict[str, Any]] = field(default_factory=dict)

    def record(
        self,
        name: str,
        value: float,
        unit: str = "",
        lower_is_better: bool = False,
    ):
        """Record a metric value.

        Args:
            name: Metric name
            value: Metric value
            unit: Optional unit
            lower_is_better: Whether lower values are better
        """
        self._metrics[name].append(value)
        self._metadata[name] = {
            "unit": unit,
            "lower_is_better": lower_is_better,
        }

    def get_metric(self, name: str) -> List[float]:
        """Get all values for a metric.

        Args:
            name: Metric name

        Returns:
            List of recorded values
        """
        return list(self._metrics.get(name, []))

    def get_mean(self, name: str) -> float:
        """Get mean value for a metric.

        Args:
            name: Metric name

        Returns:
            Mean value or 0.0 if no values
        """
        values = self._metrics.get(name, [])
        if not values:
            return 0.0
        return sum(values) / len(values)

    def get_std(self, name: str) -> float:
        """Get standard deviation for a metric.

        Args:
            name: Metric name

        Returns:
            Standard deviation or 0.0 if insufficient values
        """
        values = self._metrics.get(name, [])
        if len(values) < 2:
            return 0.0

        mean = sum(values) / len(values)
        variance = sum((x - mean) ** 2 for x in values) / len(values)
        return math.sqrt(variance)

    def get_min(self, name: str) -> float:
        """Get minimum value for a metric."""
        values = self._metrics.get(name, [])
        return min(values) if values else 0.0

    def get_max(self, name: str) -> float:
        """Get maximum value for a metric."""
        values = self._metrics.get(name, [])
        return max(values) if values else 0.0

    def get_summary(self) -> Dict[str, Dict[str, Any]]:
        """Get summary statistics for all metrics.

        Returns:
            Dictionary mapping metric names to statistics
        """
        summary = {}

        for name, values in self._metrics.items():
            if not values:
                continue

            summary[name] = {
                "count": len(values),
                "mean": sum(values) / len(values),
                "std": self.get_std(name),
                "min": min(values),
                "max": max(values),
                **self._metadata.get(name, {}),
            }

        return summary

    def to_dict(self) -> Dict[str, List[float]]:
        """Convert to dictionary."""
        return dict(self._metrics)

    def reset(self):
        """Clear all recorded metrics."""
        self._metrics.clear()
        self._metadata.clear()


class AggregatedMetrics:
    """Aggregate metrics across multiple runs.

    Example:
        >>> agg = AggregatedMetrics()
        >>> for run in runs:
        ...     agg.add_run(run.metrics)
        >>> final = agg.get_aggregated()
    """

    def __init__(self):
        """Initialize aggregated metrics."""
        self._runs: List[Dict[str, float]] = []

    def add_run(self, metrics: Dict[str, float]):
        """Add metrics from a single run.

        Args:
            metrics: Dictionary of metric name -> value
        """
        self._runs.append(metrics)

    def get_aggregated(self) -> Dict[str, Dict[str, float]]:
        """Get aggregated statistics.

        Returns:
            Dictionary with mean, std, min, max for each metric
        """
        if not self._runs:
            return {}

        # Collect all metric names
        all_metrics = set()
        for run in self._runs:
            all_metrics.update(run.keys())

        aggregated = {}

        for metric in all_metrics:
            values = [run.get(metric) for run in self._runs if metric in run]

            if not values:
                continue

            mean = sum(values) / len(values)
            variance = sum((x - mean) ** 2 for x in values) / len(values) if len(values) > 1 else 0

            aggregated[metric] = {
                "mean": mean,
                "std": math.sqrt(variance),
                "min": min(values),
                "max": max(values),
                "count": len(values),
            }

        return aggregated

    @property
    def num_runs(self) -> int:
        """Number of recorded runs."""
        return len(self._runs)

    def reset(self):
        """Clear all runs."""
        self._runs.clear()
