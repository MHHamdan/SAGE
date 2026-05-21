"""
Rolling Window Metrics Module

Provides sliding window analysis of agent performance over time,
revealing performance degradation patterns invisible in aggregate metrics.

The rolling-window success metric tracks performance over sliding windows:

    Success_[t-w, t] = (1/w) * Σ 𝟙[task_i succeeded]

This is essential for evaluating long-running agents where performance
may degrade over time due to memory issues, goal drift, or other factors.

Example:
    >>> from sage.evaluation.rolling_metrics import RollingWindowTracker
    >>>
    >>> tracker = RollingWindowTracker(window_size=50)
    >>>
    >>> for task in tasks:
    ...     result = agent.run(task)
    ...     tracker.record(TaskResult(
    ...         task_id=task.id,
    ...         timestamp=datetime.now(),
    ...         success=result.success,
    ...         cost=result.cost,
    ...         latency_ms=result.latency
    ...     ))
    >>>
    >>> metrics = tracker.get_current_window()
    >>> print(f"Rolling success rate: {metrics.success_rate:.2%}")
"""

from dataclasses import dataclass, field
from typing import List, Optional, Deque, Tuple, Dict, Any
from collections import deque
from datetime import datetime, timedelta
import logging
import uuid
import math

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class TaskResult:
    """Result of a single task execution.

    Attributes:
        task_id: Unique identifier for the task
        timestamp: When the task completed
        success: Whether the task succeeded
        cost: Cost in USD (API calls, compute, etc.)
        latency_ms: Execution time in milliseconds
        metadata: Additional task-specific data
    """

    task_id: str
    timestamp: datetime
    success: bool
    cost: float
    latency_ms: float
    metadata: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "task_id": self.task_id,
            "timestamp": self.timestamp.isoformat(),
            "success": self.success,
            "cost": self.cost,
            "latency_ms": self.latency_ms,
            "metadata": self.metadata or {},
        }


@dataclass
class WindowMetrics:
    """Metrics for a single window.

    Attributes:
        window_id: Unique identifier for this window
        window_start: Start timestamp of the window
        window_end: End timestamp of the window
        success_rate: Proportion of successful tasks
        task_count: Number of tasks in window
        success_count: Number of successful tasks
        failure_count: Number of failed tasks
        mean_cost: Average cost per task
        total_cost: Total cost for window
        mean_latency_ms: Average latency
        cost_variance: Variance of costs
        latency_variance: Variance of latencies
    """

    window_id: str
    window_start: datetime
    window_end: datetime
    success_rate: float
    task_count: int
    success_count: int
    failure_count: int
    mean_cost: float
    total_cost: float
    mean_latency_ms: float
    cost_variance: float
    latency_variance: float

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "window_id": self.window_id,
            "window_start": self.window_start.isoformat(),
            "window_end": self.window_end.isoformat(),
            "success_rate": self.success_rate,
            "task_count": self.task_count,
            "success_count": self.success_count,
            "failure_count": self.failure_count,
            "mean_cost": self.mean_cost,
            "total_cost": self.total_cost,
            "mean_latency_ms": self.mean_latency_ms,
            "cost_variance": self.cost_variance,
            "latency_variance": self.latency_variance,
        }

    @property
    def cost_per_success(self) -> float:
        """Cost per successful task."""
        if self.success_count == 0:
            return float("inf")
        return self.total_cost / self.success_count


@dataclass
class PerformanceTrend:
    """Analysis of performance trends over time.

    Attributes:
        success_trend: Direction of success rate change
        cost_trend: Direction of cost change
        latency_trend: Direction of latency change
        success_slope: Rate of change in success rate
        cost_slope: Rate of change in cost
        latency_slope: Rate of change in latency
        is_degrading: Whether overall performance is degrading
        degradation_metrics: Which metrics are degrading
    """

    success_trend: str  # "stable", "improving", "degrading"
    cost_trend: str  # "stable", "increasing", "decreasing"
    latency_trend: str  # "stable", "increasing", "decreasing"
    success_slope: float
    cost_slope: float
    latency_slope: float
    is_degrading: bool
    degradation_metrics: List[str]


class RollingWindowTracker:
    """
    Tracks agent performance using sliding windows.

    Provides comprehensive rolling window analysis for:
    - Success rate tracking over time
    - Cost trajectory analysis
    - Latency monitoring
    - Performance degradation detection
    - Window-by-window comparison

    Usage:
        >>> tracker = RollingWindowTracker(window_size=50)
        >>>
        >>> for task in tasks:
        ...     result = agent.run(task)
        ...     tracker.record(TaskResult(
        ...         task_id=task.id,
        ...         timestamp=datetime.now(),
        ...         success=result.success,
        ...         cost=result.cost,
        ...         latency_ms=result.latency
        ...     ))
        >>>
        >>> # Get current window metrics
        >>> metrics = tracker.get_current_window()
        >>> print(f"Rolling success rate: {metrics.success_rate:.2%}")
        >>>
        >>> # Check for degradation
        >>> if tracker.detect_performance_degradation():
        ...     print("Warning: Performance is degrading!")

    Attributes:
        window_size: Number of tasks per window
        overlap: Sliding overlap between windows
    """

    def __init__(
        self, window_size: int = 50, overlap: int = 25, max_history_windows: int = 100
    ):
        """Initialize the rolling window tracker.

        Args:
            window_size: Number of tasks in each window
            overlap: Number of tasks overlap between consecutive windows.
                    Set to window_size-1 for fully sliding window.
            max_history_windows: Maximum number of historical windows to keep
        """
        if window_size < 1:
            raise ValueError("window_size must be at least 1")
        if overlap < 0 or overlap >= window_size:
            raise ValueError("overlap must be between 0 and window_size-1")

        self.window_size = window_size
        self.overlap = overlap
        self.max_history_windows = max_history_windows

        self._results: Deque[TaskResult] = deque()
        self._window_history: List[WindowMetrics] = []
        self._total_tasks: int = 0

    def record(self, result: TaskResult) -> Optional[WindowMetrics]:
        """Record a task result.

        Args:
            result: TaskResult to record

        Returns:
            WindowMetrics if a new window was completed, None otherwise
        """
        self._results.append(result)
        self._total_tasks += 1

        # Check if we've completed a window
        step = self.window_size - self.overlap
        if (
            len(self._results) >= self.window_size
            and (len(self._results) - self.window_size) % step == 0
        ):
            # Compute metrics for the current window
            metrics = self._compute_window_metrics()
            self._window_history.append(metrics)

            # Trim history if needed
            if len(self._window_history) > self.max_history_windows:
                self._window_history = self._window_history[-self.max_history_windows :]

            logger.debug(f"Window completed: success_rate={metrics.success_rate:.2%}")
            return metrics

        return None

    def get_current_window(self) -> Optional[WindowMetrics]:
        """Get metrics for current window.

        Returns the metrics for the most recent window_size tasks,
        even if a full window hasn't been completed yet.

        Returns:
            WindowMetrics for current window, or None if no results
        """
        if not self._results:
            return None

        return self._compute_window_metrics()

    def get_rolling_success_rate(self) -> float:
        """Get success rate over current window.

        Returns:
            Success rate from 0.0 to 1.0
        """
        current = self.get_current_window()
        if current is None:
            return 0.0
        return current.success_rate

    def get_rolling_values(self, metric: str = "success_rate") -> List[float]:
        """Get list of rolling values for a metric.

        Args:
            metric: Metric name ("success_rate", "mean_cost", "mean_latency_ms")

        Returns:
            List of metric values for each window
        """
        if not self._window_history:
            current = self.get_current_window()
            if current:
                return [getattr(current, metric)]
            return []

        values = []
        for window in self._window_history:
            values.append(getattr(window, metric))

        # Add current window if it's different from last recorded
        current = self.get_current_window()
        if current and (
            not self._window_history
            or current.window_id != self._window_history[-1].window_id
        ):
            values.append(getattr(current, metric))

        return values

    def get_success_trend(self, num_windows: int = 5) -> str:
        """Analyze success rate trend over recent windows.

        Args:
            num_windows: Number of recent windows to analyze

        Returns:
            Trend string: "stable", "improving", or "degrading"
        """
        values = self.get_rolling_values("success_rate")

        if len(values) < 2:
            return "stable"

        recent = values[-num_windows:]
        if len(recent) < 2:
            return "stable"

        slope = self._compute_slope(recent)

        # Use 5% threshold for trend detection
        if slope > 0.05:
            return "improving"
        elif slope < -0.05:
            return "degrading"
        else:
            return "stable"

    def get_cost_trajectory(self) -> List[float]:
        """Get cost per successful task over time.

        Returns:
            List of cost-per-success values for each window
        """
        if not self._window_history:
            current = self.get_current_window()
            if current:
                return [current.cost_per_success]
            return []

        trajectory = []
        for window in self._window_history:
            trajectory.append(window.cost_per_success)

        return trajectory

    def detect_performance_degradation(
        self,
        success_threshold: float = 0.1,
        cost_threshold: float = 0.2,
        latency_threshold: float = 0.2,
    ) -> bool:
        """Detect if performance is degrading beyond threshold.

        Compares recent windows against earlier windows to detect
        significant degradation in success rate, cost, or latency.

        Args:
            success_threshold: Maximum acceptable success rate drop
            cost_threshold: Maximum acceptable cost increase ratio
            latency_threshold: Maximum acceptable latency increase ratio

        Returns:
            True if performance is degrading significantly
        """
        trend = self.analyze_trend()

        if trend is None:
            return False

        return trend.is_degrading

    def analyze_trend(self, num_windows: int = 5) -> Optional[PerformanceTrend]:
        """Analyze performance trend over recent windows.

        Args:
            num_windows: Number of windows to analyze

        Returns:
            PerformanceTrend analysis, or None if insufficient data
        """
        success_values = self.get_rolling_values("success_rate")
        cost_values = self.get_rolling_values("mean_cost")
        latency_values = self.get_rolling_values("mean_latency_ms")

        if len(success_values) < 2:
            return None

        # Compute slopes
        success_slope = self._compute_slope(success_values[-num_windows:])
        cost_slope = self._compute_slope(cost_values[-num_windows:])
        latency_slope = self._compute_slope(latency_values[-num_windows:])

        # Determine trends
        success_trend = "stable"
        if success_slope > 0.01:
            success_trend = "improving"
        elif success_slope < -0.01:
            success_trend = "degrading"

        cost_trend = "stable"
        if cost_slope > 0.01:
            cost_trend = "increasing"
        elif cost_slope < -0.01:
            cost_trend = "decreasing"

        latency_trend = "stable"
        if latency_slope > 0.01:
            latency_trend = "increasing"
        elif latency_slope < -0.01:
            latency_trend = "decreasing"

        # Check for degradation
        degradation_metrics = []
        if success_trend == "degrading":
            degradation_metrics.append("success_rate")
        if cost_trend == "increasing":
            degradation_metrics.append("cost")
        if latency_trend == "increasing":
            degradation_metrics.append("latency")

        is_degrading = (
            len(degradation_metrics) > 0 and "success_rate" in degradation_metrics
        )

        return PerformanceTrend(
            success_trend=success_trend,
            cost_trend=cost_trend,
            latency_trend=latency_trend,
            success_slope=success_slope,
            cost_slope=cost_slope,
            latency_slope=latency_slope,
            is_degrading=is_degrading,
            degradation_metrics=degradation_metrics,
        )

    def get_window_history(self) -> List[WindowMetrics]:
        """Get history of all computed windows.

        Returns:
            List of WindowMetrics for historical windows
        """
        return list(self._window_history)

    def get_all_results(self) -> List[TaskResult]:
        """Get all recorded task results.

        Returns:
            List of all TaskResults
        """
        return list(self._results)

    def get_summary_statistics(self) -> Dict[str, Any]:
        """Get summary statistics across all tasks.

        Returns:
            Dictionary with aggregate statistics
        """
        if not self._results:
            return {
                "total_tasks": 0,
                "overall_success_rate": 0.0,
                "total_cost": 0.0,
                "mean_cost": 0.0,
                "mean_latency_ms": 0.0,
            }

        successes = sum(1 for r in self._results if r.success)
        total_cost = sum(r.cost for r in self._results)
        total_latency = sum(r.latency_ms for r in self._results)

        return {
            "total_tasks": len(self._results),
            "total_successes": successes,
            "total_failures": len(self._results) - successes,
            "overall_success_rate": successes / len(self._results),
            "total_cost": total_cost,
            "mean_cost": total_cost / len(self._results),
            "mean_latency_ms": total_latency / len(self._results),
            "windows_completed": len(self._window_history),
        }

    def _compute_window_metrics(self) -> WindowMetrics:
        """Compute metrics for the current window."""
        # Get the most recent window_size results
        window_results = list(self._results)[-self.window_size :]

        if not window_results:
            now = datetime.now()
            return WindowMetrics(
                window_id=str(uuid.uuid4()),
                window_start=now,
                window_end=now,
                success_rate=0.0,
                task_count=0,
                success_count=0,
                failure_count=0,
                mean_cost=0.0,
                total_cost=0.0,
                mean_latency_ms=0.0,
                cost_variance=0.0,
                latency_variance=0.0,
            )

        # Calculate metrics
        success_count = sum(1 for r in window_results if r.success)
        failure_count = len(window_results) - success_count
        success_rate = success_count / len(window_results)

        costs = [r.cost for r in window_results]
        latencies = [r.latency_ms for r in window_results]

        total_cost = sum(costs)
        mean_cost = total_cost / len(costs)
        mean_latency = sum(latencies) / len(latencies)

        # Compute variances
        cost_variance = (
            sum((c - mean_cost) ** 2 for c in costs) / len(costs)
            if len(costs) > 1
            else 0.0
        )
        latency_variance = (
            sum((l - mean_latency) ** 2 for l in latencies) / len(latencies)
            if len(latencies) > 1
            else 0.0
        )

        return WindowMetrics(
            window_id=str(uuid.uuid4()),
            window_start=window_results[0].timestamp,
            window_end=window_results[-1].timestamp,
            success_rate=success_rate,
            task_count=len(window_results),
            success_count=success_count,
            failure_count=failure_count,
            mean_cost=mean_cost,
            total_cost=total_cost,
            mean_latency_ms=mean_latency,
            cost_variance=cost_variance,
            latency_variance=latency_variance,
        )

    def _compute_slope(self, values: List[float]) -> float:
        """Compute slope of values using linear regression.

        Args:
            values: List of values

        Returns:
            Slope (rate of change per unit)
        """
        if len(values) < 2:
            return 0.0

        n = len(values)
        x = np.arange(n)
        y = np.array(values)

        # Handle NaN and inf
        mask = np.isfinite(y)
        if not np.any(mask):
            return 0.0

        x = x[mask]
        y = y[mask]

        if len(x) < 2:
            return 0.0

        # Least squares slope
        n = len(x)
        slope = (n * np.sum(x * y) - np.sum(x) * np.sum(y)) / (
            n * np.sum(x**2) - np.sum(x) ** 2
        )

        return float(slope)

    def reset(self) -> None:
        """Reset all tracking state."""
        self._results.clear()
        self._window_history.clear()
        self._total_tasks = 0
        logger.info("Rolling window tracker reset")

    def to_dict(self) -> Dict[str, Any]:
        """Export tracker state to dictionary."""
        return {
            "window_size": self.window_size,
            "overlap": self.overlap,
            "total_tasks": self._total_tasks,
            "current_buffer_size": len(self._results),
            "windows_completed": len(self._window_history),
            "current_window": (
                self.get_current_window().to_dict()
                if self.get_current_window()
                else None
            ),
            "summary": self.get_summary_statistics(),
        }


class TimeBasedRollingTracker:
    """Alternative tracker that uses time-based windows instead of task count.

    Useful when task rate varies significantly over time.

    Example:
        >>> tracker = TimeBasedRollingTracker(window_minutes=60)
        >>> for result in results:
        ...     tracker.record(result)
        >>> metrics = tracker.get_current_window()
    """

    def __init__(self, window_minutes: int = 60, overlap_minutes: int = 30):
        """Initialize time-based tracker.

        Args:
            window_minutes: Window size in minutes
            overlap_minutes: Overlap between windows
        """
        self.window_duration = timedelta(minutes=window_minutes)
        self.overlap_duration = timedelta(minutes=overlap_minutes)

        self._results: List[TaskResult] = []
        self._window_history: List[WindowMetrics] = []

    def record(self, result: TaskResult) -> None:
        """Record a task result."""
        self._results.append(result)

    def get_current_window(self) -> Optional[WindowMetrics]:
        """Get metrics for current time window."""
        if not self._results:
            return None

        now = datetime.now()
        window_start = now - self.window_duration
        window_results = [r for r in self._results if r.timestamp >= window_start]

        if not window_results:
            return None

        # Calculate metrics
        success_count = sum(1 for r in window_results if r.success)
        costs = [r.cost for r in window_results]
        latencies = [r.latency_ms for r in window_results]

        total_cost = sum(costs)
        mean_cost = total_cost / len(costs)
        mean_latency = sum(latencies) / len(latencies)

        cost_variance = (
            sum((c - mean_cost) ** 2 for c in costs) / len(costs)
            if len(costs) > 1
            else 0.0
        )
        latency_variance = (
            sum((l - mean_latency) ** 2 for l in latencies) / len(latencies)
            if len(latencies) > 1
            else 0.0
        )

        return WindowMetrics(
            window_id=str(uuid.uuid4()),
            window_start=window_start,
            window_end=now,
            success_rate=success_count / len(window_results),
            task_count=len(window_results),
            success_count=success_count,
            failure_count=len(window_results) - success_count,
            mean_cost=mean_cost,
            total_cost=total_cost,
            mean_latency_ms=mean_latency,
            cost_variance=cost_variance,
            latency_variance=latency_variance,
        )

    def get_success_rate_over_time(
        self, bucket_minutes: int = 15
    ) -> List[Tuple[datetime, float]]:
        """Get success rates bucketed by time.

        Args:
            bucket_minutes: Size of each time bucket

        Returns:
            List of (bucket_start, success_rate) tuples
        """
        if not self._results:
            return []

        bucket_duration = timedelta(minutes=bucket_minutes)
        min_time = min(r.timestamp for r in self._results)
        max_time = max(r.timestamp for r in self._results)

        buckets = []
        current_start = min_time

        while current_start <= max_time:
            current_end = current_start + bucket_duration
            bucket_results = [
                r for r in self._results if current_start <= r.timestamp < current_end
            ]

            if bucket_results:
                success_rate = sum(1 for r in bucket_results if r.success) / len(
                    bucket_results
                )
                buckets.append((current_start, success_rate))

            current_start = current_end

        return buckets
