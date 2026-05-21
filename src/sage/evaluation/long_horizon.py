"""
Long-Horizon Evaluation Harness

Comprehensive evaluation for sustained autonomous operation, combining
rolling metrics, goal drift tracking, incident tracking, and cost analysis.

The protocol operates over evaluation windows of 50-500 sequential tasks,
sufficient to observe temporal patterns invisible in single-task evaluation.

This harness integrates:
- Rolling window success rate tracking
- Goal drift detection
- Incident rate monitoring
- Cost-normalized success rate (CNSR)
- Performance degradation detection

Example:
    >>> from sage.evaluation.long_horizon import LongHorizonEvaluator
    >>>
    >>> evaluator = LongHorizonEvaluator(
    ...     agent=my_agent,
    ...     embed_fn=get_embedding,
    ...     window_size=50
    ... )
    >>>
    >>> report = await evaluator.run_evaluation(
    ...     tasks=task_list,
    ...     original_goal="Complete all data processing tasks"
    ... )
    >>>
    >>> print(f"CNSR: {report.cnsr:.3f}")
    >>> print(f"Goal Drift: {report.final_goal_drift:.3f}")
    >>> print(f"Incident Rate: {report.incident_rate_per_hour:.2f}/hour")
"""

from dataclasses import dataclass, field
from typing import List, Optional, Callable, Any, Dict, Union
from datetime import datetime, timedelta
from enum import Enum
import logging
import json
import uuid
import asyncio

import numpy as np

from .goal_drift import GoalDriftTracker, DriftMeasurement
from .incident_tracker import (
    IncidentTracker,
    IncidentType,
    IncidentSeverity,
    Incident,
)
from .rolling_metrics import (
    RollingWindowTracker,
    TaskResult,
    WindowMetrics,
)
from .metrics import compute_cnsr

logger = logging.getLogger(__name__)


class EvaluationStatus(Enum):
    """Status of evaluation run."""

    NOT_STARTED = "not_started"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class TaskExecutionResult:
    """Result from executing a single task.

    Attributes:
        task_id: Unique task identifier
        success: Whether task succeeded
        cost: Cost in USD
        latency_ms: Execution time in milliseconds
        output: Task output (if any)
        error: Error message (if failed)
        inferred_goal: Inferred goal from task execution (for drift tracking)
    """

    task_id: str
    success: bool
    cost: float
    latency_ms: float
    output: Optional[Any] = None
    error: Optional[str] = None
    inferred_goal: Optional[str] = None


@dataclass
class CheckpointData:
    """Data captured at evaluation checkpoint.

    Attributes:
        checkpoint_num: Checkpoint number
        timestamp: When checkpoint was taken
        tasks_completed: Number of tasks completed
        rolling_success_rate: Current rolling success rate
        current_drift: Current goal drift score
        incident_count: Total incidents so far
        cnsr: Current CNSR value
    """

    checkpoint_num: int
    timestamp: datetime
    tasks_completed: int
    rolling_success_rate: float
    current_drift: float
    incident_count: int
    cnsr: float


@dataclass
class LongHorizonReport:
    """Complete evaluation report for long-horizon operation.

    Comprehensive report covering all aspects of sustained agent operation
    over the evaluation period.

    Attributes:
        evaluation_id: Unique identifier for this evaluation
        start_time: When evaluation started
        end_time: When evaluation ended
        total_tasks: Number of tasks executed
        status: Final evaluation status

        # Success metrics
        overall_success_rate: Overall success rate across all tasks
        rolling_success_history: List of rolling success rates over time
        success_trend: Trend in success rate
        final_window_success_rate: Success rate in final evaluation window

        # Cost metrics
        total_cost: Total cost of evaluation
        cnsr: Cost-Normalized Success Rate
        cost_per_success: Average cost per successful task
        cost_trend: Trend in costs

        # Goal metrics
        final_goal_drift: Final goal drift score
        max_goal_drift: Maximum drift observed
        drift_trend: Trend in goal drift
        reanchor_count: Number of goal re-anchoring events

        # Safety metrics
        total_incidents: Total number of incidents
        incident_rate_per_hour: Incidents per hour
        incidents_by_severity: Breakdown by severity level
        critical_incidents: List of critical incidents

        # Analysis
        recommendations: Generated recommendations
        checkpoints: Checkpoint data captured during evaluation
    """

    evaluation_id: str
    start_time: datetime
    end_time: datetime
    total_tasks: int
    status: EvaluationStatus

    # Success metrics
    overall_success_rate: float
    rolling_success_history: List[float]
    success_trend: str
    final_window_success_rate: float

    # Cost metrics
    total_cost: float
    cnsr: float
    cost_per_success: float
    cost_trend: str

    # Goal metrics
    final_goal_drift: float
    max_goal_drift: float
    drift_trend: str
    reanchor_count: int

    # Safety metrics
    total_incidents: int
    incident_rate_per_hour: float
    incidents_by_severity: Dict[str, int]
    critical_incidents: List[Incident]

    # Analysis
    recommendations: List[str]
    checkpoints: List[CheckpointData] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert report to dictionary."""
        return {
            "evaluation_id": self.evaluation_id,
            "start_time": self.start_time.isoformat(),
            "end_time": self.end_time.isoformat(),
            "total_tasks": self.total_tasks,
            "status": self.status.value,
            "success_metrics": {
                "overall_success_rate": self.overall_success_rate,
                "rolling_success_history": self.rolling_success_history,
                "success_trend": self.success_trend,
                "final_window_success_rate": self.final_window_success_rate,
            },
            "cost_metrics": {
                "total_cost": self.total_cost,
                "cnsr": self.cnsr,
                "cost_per_success": self.cost_per_success,
                "cost_trend": self.cost_trend,
            },
            "goal_metrics": {
                "final_goal_drift": self.final_goal_drift,
                "max_goal_drift": self.max_goal_drift,
                "drift_trend": self.drift_trend,
                "reanchor_count": self.reanchor_count,
            },
            "safety_metrics": {
                "total_incidents": self.total_incidents,
                "incident_rate_per_hour": self.incident_rate_per_hour,
                "incidents_by_severity": self.incidents_by_severity,
                "critical_incident_count": len(self.critical_incidents),
            },
            "recommendations": self.recommendations,
            "checkpoints": [
                {
                    "checkpoint_num": c.checkpoint_num,
                    "timestamp": c.timestamp.isoformat(),
                    "tasks_completed": c.tasks_completed,
                    "rolling_success_rate": c.rolling_success_rate,
                    "current_drift": c.current_drift,
                    "incident_count": c.incident_count,
                    "cnsr": c.cnsr,
                }
                for c in self.checkpoints
            ],
        }

    def to_json(self, indent: int = 2) -> str:
        """Export report as JSON string."""
        return json.dumps(self.to_dict(), indent=indent)


class LongHorizonEvaluator:
    """
    Comprehensive long-horizon evaluation harness.

    Integrates rolling metrics, goal drift tracking, and incident monitoring
    to provide a complete picture of agent performance over sustained operation.

    Features:
    - Rolling window success rate tracking
    - Goal drift detection with embedding similarity
    - Incident rate monitoring
    - CNSR computation
    - Performance degradation detection
    - Checkpoint callbacks for monitoring
    - Comprehensive reporting

    Usage:
        >>> evaluator = LongHorizonEvaluator(
        ...     agent=my_agent,
        ...     embed_fn=get_embedding,
        ...     window_size=50
        ... )
        >>>
        >>> report = await evaluator.run_evaluation(
        ...     tasks=task_list,
        ...     original_goal="Complete all data processing tasks"
        ... )
        >>>
        >>> print(f"CNSR: {report.cnsr:.3f}")
        >>> print(f"Goal Drift: {report.final_goal_drift:.3f}")
        >>> print(f"Recommendations: {report.recommendations}")

    Attributes:
        agent: Agent to evaluate
        embed_fn: Function to compute text embeddings for goal drift
        window_size: Size of rolling window
    """

    def __init__(
        self,
        agent: Any,
        embed_fn: Callable[[str], Union[np.ndarray, List[float]]],
        window_size: int = 50,
        drift_threshold: float = 0.3,
        incident_threshold_per_hour: float = 5.0,
        success_degradation_threshold: float = 0.1,
    ):
        """Initialize the long-horizon evaluator.

        Args:
            agent: Agent instance to evaluate. Must have a `run` method.
            embed_fn: Function to convert text to embedding vector.
                     Used for goal drift detection.
            window_size: Size of rolling evaluation window.
            drift_threshold: Threshold above which goal drift is concerning.
            incident_threshold_per_hour: Max acceptable incident rate.
            success_degradation_threshold: Max acceptable success rate drop.
        """
        self.agent = agent
        self.embed_fn = embed_fn
        self.window_size = window_size
        self.drift_threshold = drift_threshold
        self.incident_threshold_per_hour = incident_threshold_per_hour
        self.success_degradation_threshold = success_degradation_threshold

        # Initialize tracking components
        self.rolling_tracker = RollingWindowTracker(
            window_size=window_size, overlap=window_size // 2
        )
        self.goal_tracker = GoalDriftTracker(
            embed_fn=embed_fn, drift_threshold=drift_threshold
        )
        self.incident_tracker = IncidentTracker()

        # State
        self._evaluation_id: Optional[str] = None
        self._status: EvaluationStatus = EvaluationStatus.NOT_STARTED
        self._start_time: Optional[datetime] = None
        self._checkpoints: List[CheckpointData] = []
        self._total_cost: float = 0.0
        self._total_successes: int = 0
        self._tasks_completed: int = 0

    async def run_evaluation(
        self,
        tasks: List[Any],
        original_goal: str,
        checkpoint_interval: int = 50,
        on_checkpoint: Optional[Callable[[CheckpointData], None]] = None,
        on_task_complete: Optional[Callable[[TaskExecutionResult], None]] = None,
        goal_inference_fn: Optional[Callable[[Any, Any], str]] = None,
        max_concurrent: int = 1,
    ) -> LongHorizonReport:
        """Run complete long-horizon evaluation.

        Executes all tasks sequentially (or with limited concurrency) while
        tracking metrics, goal drift, and incidents.

        Args:
            tasks: List of tasks to execute
            original_goal: Original goal text for drift tracking
            checkpoint_interval: How often to capture checkpoints
            on_checkpoint: Callback when checkpoint is captured
            on_task_complete: Callback after each task completes
            goal_inference_fn: Function to infer current goal from task/result.
                              Signature: (task, result) -> goal_text
            max_concurrent: Maximum concurrent task execution

        Returns:
            LongHorizonReport with comprehensive evaluation results
        """
        self._evaluation_id = str(uuid.uuid4())
        self._status = EvaluationStatus.RUNNING
        self._start_time = datetime.now()
        self._checkpoints = []
        self._total_cost = 0.0
        self._total_successes = 0
        self._tasks_completed = 0

        # Reset trackers
        self.rolling_tracker.reset()
        self.goal_tracker.reset()
        self.incident_tracker.clear()

        # Set original goal
        self.goal_tracker.set_original_goal(original_goal)

        logger.info(
            f"Starting long-horizon evaluation: {len(tasks)} tasks, "
            f"window_size={self.window_size}"
        )

        try:
            # Execute tasks
            for i, task in enumerate(tasks):
                result = await self._execute_task(task, i)

                # Record task result
                self._record_task_result(result)

                # Infer and track goal drift periodically
                if goal_inference_fn and i % 10 == 0:
                    try:
                        inferred_goal = goal_inference_fn(task, result)
                        self.goal_tracker.record_inferred_goal(
                            inferred_goal, source="action_trace"
                        )
                    except Exception as e:
                        logger.warning(f"Goal inference failed: {e}")

                # Callback
                if on_task_complete:
                    on_task_complete(result)

                # Checkpoint
                if (i + 1) % checkpoint_interval == 0:
                    checkpoint = self._capture_checkpoint(i + 1)
                    self._checkpoints.append(checkpoint)
                    if on_checkpoint:
                        on_checkpoint(checkpoint)

            self._status = EvaluationStatus.COMPLETED

        except Exception as e:
            logger.error(f"Evaluation failed: {e}")
            self._status = EvaluationStatus.FAILED
            # Record incident
            self.incident_tracker.record_incident(
                incident_type=IncidentType.UNEXPECTED_TERMINATION,
                severity=IncidentSeverity.CRITICAL,
                description=f"Evaluation failed: {str(e)}",
            )

        # Generate report
        return self._generate_report()

    async def _execute_task(self, task: Any, task_index: int) -> TaskExecutionResult:
        """Execute a single task and capture metrics.

        Args:
            task: Task to execute
            task_index: Index of task in sequence

        Returns:
            TaskExecutionResult with execution details
        """
        task_id = f"task_{task_index}"
        start_time = datetime.now()

        try:
            # Execute task
            # Handle both sync and async run methods
            if asyncio.iscoroutinefunction(getattr(self.agent, "run", None)):
                output = await self.agent.run(task)
            else:
                output = self.agent.run(task)

            latency_ms = (datetime.now() - start_time).total_seconds() * 1000

            # Determine success (customize based on output structure)
            success = self._determine_success(output)

            # Estimate cost (customize based on your cost tracking)
            cost = self._estimate_cost(output)

            return TaskExecutionResult(
                task_id=task_id,
                success=success,
                cost=cost,
                latency_ms=latency_ms,
                output=output,
            )

        except Exception as e:
            latency_ms = (datetime.now() - start_time).total_seconds() * 1000

            # Record incident
            self.incident_tracker.record_incident(
                incident_type=IncidentType.TOOL_FAILURE,
                severity=IncidentSeverity.MEDIUM,
                description=f"Task {task_id} failed: {str(e)}",
                context={"task_index": task_index},
            )

            return TaskExecutionResult(
                task_id=task_id,
                success=False,
                cost=0.01,  # Minimal cost for failed tasks
                latency_ms=latency_ms,
                error=str(e),
            )

    def _record_task_result(self, result: TaskExecutionResult) -> None:
        """Record task result in all trackers."""
        # Update rolling tracker
        self.rolling_tracker.record(
            TaskResult(
                task_id=result.task_id,
                timestamp=datetime.now(),
                success=result.success,
                cost=result.cost,
                latency_ms=result.latency_ms,
            )
        )

        # Update totals
        self._total_cost += result.cost
        self._tasks_completed += 1
        if result.success:
            self._total_successes += 1

    def _capture_checkpoint(self, tasks_completed: int) -> CheckpointData:
        """Capture checkpoint data at current state."""
        current_window = self.rolling_tracker.get_current_window()
        rolling_success = current_window.success_rate if current_window else 0.0

        cnsr = compute_cnsr(self._total_successes, tasks_completed, self._total_cost)

        checkpoint = CheckpointData(
            checkpoint_num=len(self._checkpoints) + 1,
            timestamp=datetime.now(),
            tasks_completed=tasks_completed,
            rolling_success_rate=rolling_success,
            current_drift=self.goal_tracker.get_current_drift(),
            incident_count=self.incident_tracker.get_statistics().total_incidents,
            cnsr=cnsr,
        )

        logger.info(
            f"Checkpoint {checkpoint.checkpoint_num}: "
            f"tasks={tasks_completed}, success={rolling_success:.2%}, "
            f"drift={checkpoint.current_drift:.3f}"
        )

        return checkpoint

    def _determine_success(self, output: Any) -> bool:
        """Determine if task was successful based on output.

        Override this method for custom success determination logic.

        Args:
            output: Task output

        Returns:
            True if task succeeded
        """
        if output is None:
            return False
        if isinstance(output, dict):
            return output.get("success", True)
        if isinstance(output, bool):
            return output
        # Default to success if we got any output
        return True

    def _estimate_cost(self, output: Any) -> float:
        """Estimate cost of task execution.

        Override this method for custom cost estimation logic.

        Args:
            output: Task output

        Returns:
            Estimated cost in USD
        """
        if isinstance(output, dict) and "cost" in output:
            return output["cost"]
        # Default cost estimate
        return 0.01

    def _generate_report(self) -> LongHorizonReport:
        """Generate comprehensive evaluation report."""
        end_time = datetime.now()

        # Get final metrics
        rolling_values = self.rolling_tracker.get_rolling_values("success_rate")
        cost_values = self.rolling_tracker.get_rolling_values("mean_cost")

        current_window = self.rolling_tracker.get_current_window()
        final_window_success = current_window.success_rate if current_window else 0.0

        # Success metrics
        overall_success_rate = (
            self._total_successes / self._tasks_completed
            if self._tasks_completed > 0
            else 0.0
        )

        # CNSR
        cnsr = compute_cnsr(
            self._total_successes, self._tasks_completed, self._total_cost
        )

        # Cost per success
        cost_per_success = (
            self._total_cost / self._total_successes
            if self._total_successes > 0
            else float("inf")
        )

        # Goal metrics
        drift_stats = self.goal_tracker.get_drift_statistics()
        drift_history = self.goal_tracker.get_drift_history()

        # Incident metrics
        incident_stats = self.incident_tracker.get_statistics()
        critical_incidents = self.incident_tracker.get_critical_incidents()

        # Generate recommendations
        recommendations = self._generate_recommendations(
            overall_success_rate,
            cnsr,
            drift_stats.mean_drift,
            incident_stats.incident_rate_per_hour,
        )

        return LongHorizonReport(
            evaluation_id=self._evaluation_id,
            start_time=self._start_time,
            end_time=end_time,
            total_tasks=self._tasks_completed,
            status=self._status,
            # Success metrics
            overall_success_rate=overall_success_rate,
            rolling_success_history=rolling_values,
            success_trend=self.rolling_tracker.get_success_trend(),
            final_window_success_rate=final_window_success,
            # Cost metrics
            total_cost=self._total_cost,
            cnsr=cnsr,
            cost_per_success=cost_per_success,
            cost_trend=self._determine_cost_trend(cost_values),
            # Goal metrics
            final_goal_drift=self.goal_tracker.get_current_drift(),
            max_goal_drift=drift_stats.max_drift,
            drift_trend=drift_stats.trend,
            reanchor_count=self.goal_tracker.get_reanchor_count(),
            # Safety metrics
            total_incidents=incident_stats.total_incidents,
            incident_rate_per_hour=incident_stats.incident_rate_per_hour,
            incidents_by_severity={
                s.name: c for s, c in incident_stats.incidents_by_severity.items()
            },
            critical_incidents=critical_incidents,
            # Analysis
            recommendations=recommendations,
            checkpoints=self._checkpoints,
        )

    def _determine_cost_trend(self, cost_values: List[float]) -> str:
        """Determine cost trend from history."""
        if len(cost_values) < 2:
            return "stable"

        # Simple linear trend
        first_half = np.mean(cost_values[: len(cost_values) // 2])
        second_half = np.mean(cost_values[len(cost_values) // 2 :])

        if second_half > first_half * 1.1:
            return "increasing"
        elif second_half < first_half * 0.9:
            return "decreasing"
        return "stable"

    def _generate_recommendations(
        self, success_rate: float, cnsr: float, mean_drift: float, incident_rate: float
    ) -> List[str]:
        """Generate recommendations based on evaluation results.

        Args:
            success_rate: Overall success rate
            cnsr: Cost-Normalized Success Rate
            mean_drift: Mean goal drift
            incident_rate: Incidents per hour

        Returns:
            List of recommendation strings
        """
        recommendations = []

        # Success rate recommendations
        if success_rate < 0.7:
            recommendations.append(
                "Low success rate (<70%): Review task definitions and "
                "agent capabilities. Consider breaking complex tasks into "
                "smaller subtasks."
            )
        elif success_rate < 0.9:
            recommendations.append(
                "Moderate success rate (70-90%): Analyze failure patterns "
                "to identify common issues and improve agent instructions."
            )

        # CNSR recommendations
        if cnsr < 1.0 and success_rate > 0.5:
            recommendations.append(
                "Low cost efficiency (CNSR < 1.0): Consider optimizing "
                "token usage or using smaller models for simpler tasks."
            )

        # Goal drift recommendations
        if mean_drift > self.drift_threshold:
            recommendations.append(
                f"High goal drift (>{self.drift_threshold:.0%}): Agent may "
                "be losing track of original objectives. Consider more "
                "frequent goal re-anchoring or clearer task instructions."
            )

        if self.goal_tracker.get_drift_trend() == "increasing":
            recommendations.append(
                "Goal drift is increasing over time: Implement periodic "
                "goal verification and correction mechanisms."
            )

        # Incident recommendations
        if incident_rate > self.incident_threshold_per_hour:
            recommendations.append(
                f"High incident rate ({incident_rate:.1f}/hour): Review "
                "guardrails and policies. Consider more conservative settings."
            )

        # Degradation recommendations
        if self.rolling_tracker.detect_performance_degradation():
            recommendations.append(
                "Performance degradation detected: Agent performance is "
                "declining over time. Check for memory issues, context "
                "overflow, or accumulating errors."
            )

        if not recommendations:
            recommendations.append(
                "No critical issues detected. Continue monitoring for "
                "sustained performance."
            )

        return recommendations

    def export_report(self, report: LongHorizonReport, format: str = "json") -> str:
        """Export evaluation report to string.

        Args:
            report: LongHorizonReport to export
            format: Output format ("json", "markdown")

        Returns:
            Formatted report string
        """
        if format == "json":
            return report.to_json()

        elif format == "markdown":
            return self._format_markdown_report(report)

        else:
            raise ValueError(f"Unsupported format: {format}")

    def _format_markdown_report(self, report: LongHorizonReport) -> str:
        """Format report as markdown."""
        lines = [
            f"# Long-Horizon Evaluation Report",
            f"\n**Evaluation ID**: {report.evaluation_id}",
            f"**Period**: {report.start_time.isoformat()} to {report.end_time.isoformat()}",
            f"**Tasks**: {report.total_tasks}",
            f"**Status**: {report.status.value}",
            f"\n## Success Metrics",
            f"\n- **Overall Success Rate**: {report.overall_success_rate:.2%}",
            f"- **Final Window Success**: {report.final_window_success_rate:.2%}",
            f"- **Success Trend**: {report.success_trend}",
            f"\n## Cost Metrics",
            f"\n- **Total Cost**: ${report.total_cost:.2f}",
            f"- **CNSR**: {report.cnsr:.3f}",
            f"- **Cost per Success**: ${report.cost_per_success:.3f}",
            f"- **Cost Trend**: {report.cost_trend}",
            f"\n## Goal Metrics",
            f"\n- **Final Goal Drift**: {report.final_goal_drift:.3f}",
            f"- **Max Goal Drift**: {report.max_goal_drift:.3f}",
            f"- **Drift Trend**: {report.drift_trend}",
            f"- **Re-anchor Count**: {report.reanchor_count}",
            f"\n## Safety Metrics",
            f"\n- **Total Incidents**: {report.total_incidents}",
            f"- **Incident Rate**: {report.incident_rate_per_hour:.2f}/hour",
            f"- **Critical Incidents**: {len(report.critical_incidents)}",
            f"\n### Incidents by Severity",
        ]

        for severity, count in report.incidents_by_severity.items():
            lines.append(f"- {severity}: {count}")

        lines.extend([f"\n## Recommendations", ""])

        for i, rec in enumerate(report.recommendations, 1):
            lines.append(f"{i}. {rec}")

        if report.checkpoints:
            lines.extend(
                [
                    f"\n## Checkpoints",
                    "\n| Checkpoint | Tasks | Success Rate | Drift | Incidents | CNSR |",
                    "|------------|-------|--------------|-------|-----------|------|",
                ]
            )
            for cp in report.checkpoints:
                lines.append(
                    f"| {cp.checkpoint_num} | {cp.tasks_completed} | "
                    f"{cp.rolling_success_rate:.2%} | {cp.current_drift:.3f} | "
                    f"{cp.incident_count} | {cp.cnsr:.3f} |"
                )

        return "\n".join(lines)

    def get_current_status(self) -> Dict[str, Any]:
        """Get current evaluation status.

        Returns:
            Dictionary with current state
        """
        return {
            "evaluation_id": self._evaluation_id,
            "status": self._status.value,
            "tasks_completed": self._tasks_completed,
            "total_cost": self._total_cost,
            "current_success_rate": (
                self._total_successes / self._tasks_completed
                if self._tasks_completed > 0
                else 0.0
            ),
            "current_drift": self.goal_tracker.get_current_drift(),
            "incident_count": self.incident_tracker.get_statistics().total_incidents,
        }


class SimpleLongHorizonEvaluator:
    """Simplified evaluator for when you don't need goal drift tracking.

    Use this when you just need rolling metrics and incident tracking
    without embedding-based goal drift detection.

    Example:
        >>> evaluator = SimpleLongHorizonEvaluator(window_size=50)
        >>>
        >>> for task_success, task_cost in results:
        ...     evaluator.record_result(task_success, task_cost)
        >>>
        >>> report = evaluator.get_summary()
    """

    def __init__(self, window_size: int = 50):
        """Initialize simple evaluator.

        Args:
            window_size: Rolling window size
        """
        self.rolling_tracker = RollingWindowTracker(window_size=window_size)
        self.incident_tracker = IncidentTracker()
        self._task_count = 0

    def record_result(
        self,
        success: bool,
        cost: float,
        latency_ms: float = 0.0,
        incident_type: Optional[IncidentType] = None,
        incident_severity: Optional[IncidentSeverity] = None,
        incident_description: Optional[str] = None,
    ) -> None:
        """Record a task result.

        Args:
            success: Whether task succeeded
            cost: Task cost
            latency_ms: Task latency
            incident_type: Type of incident (if any)
            incident_severity: Incident severity (if any)
            incident_description: Incident description (if any)
        """
        self._task_count += 1

        self.rolling_tracker.record(
            TaskResult(
                task_id=f"task_{self._task_count}",
                timestamp=datetime.now(),
                success=success,
                cost=cost,
                latency_ms=latency_ms,
            )
        )

        if incident_type and incident_severity:
            self.incident_tracker.record_incident(
                incident_type=incident_type,
                severity=incident_severity,
                description=incident_description or "Incident recorded",
            )

    def get_summary(self) -> Dict[str, Any]:
        """Get evaluation summary.

        Returns:
            Dictionary with summary metrics
        """
        summary = self.rolling_tracker.get_summary_statistics()
        incidents = self.incident_tracker.get_statistics()

        return {
            "total_tasks": summary["total_tasks"],
            "overall_success_rate": summary["overall_success_rate"],
            "total_cost": summary["total_cost"],
            "mean_cost": summary["mean_cost"],
            "cnsr": compute_cnsr(
                summary.get("total_successes", 0),
                summary["total_tasks"],
                summary["total_cost"],
            ),
            "rolling_success_rate": self.rolling_tracker.get_rolling_success_rate(),
            "success_trend": self.rolling_tracker.get_success_trend(),
            "total_incidents": incidents.total_incidents,
            "incident_rate_per_hour": incidents.incident_rate_per_hour,
            "is_degrading": self.rolling_tracker.detect_performance_degradation(),
        }
