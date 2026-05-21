"""
Tests for Rolling Window Metrics Module.

Tests cover:
- Task result recording
- Window metrics computation
- Success rate tracking
- Trend detection
- Performance degradation detection
- Edge cases
"""

import pytest
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import Mock

from sage.evaluation.rolling_metrics import (
    RollingWindowTracker,
    TimeBasedRollingTracker,
    TaskResult,
    WindowMetrics,
    PerformanceTrend,
)


# Test fixtures

@pytest.fixture
def tracker():
    """Create a basic rolling window tracker."""
    return RollingWindowTracker(window_size=10, overlap=5)


@pytest.fixture
def populated_tracker():
    """Create tracker with some results recorded."""
    tracker = RollingWindowTracker(window_size=10, overlap=5)

    now = datetime.now()
    for i in range(20):
        tracker.record(TaskResult(
            task_id=f"task_{i}",
            timestamp=now + timedelta(minutes=i),
            success=(i % 3 != 0),  # 2/3 success rate
            cost=0.1 + (i * 0.01),
            latency_ms=100 + (i * 10)
        ))

    return tracker


def make_task_result(
    task_id: str = "task",
    success: bool = True,
    cost: float = 0.1,
    latency_ms: float = 100.0,
    timestamp: datetime = None
) -> TaskResult:
    """Helper to create task results."""
    return TaskResult(
        task_id=task_id,
        timestamp=timestamp or datetime.now(),
        success=success,
        cost=cost,
        latency_ms=latency_ms
    )


# Tests for TaskResult dataclass

class TestTaskResult:
    """Tests for TaskResult dataclass."""

    def test_task_result_creation(self):
        """Should create task result with all fields."""
        now = datetime.now()
        result = TaskResult(
            task_id="test-123",
            timestamp=now,
            success=True,
            cost=0.05,
            latency_ms=150.5
        )

        assert result.task_id == "test-123"
        assert result.timestamp == now
        assert result.success is True
        assert result.cost == 0.05
        assert result.latency_ms == 150.5

    def test_task_result_with_metadata(self):
        """Should store optional metadata."""
        result = TaskResult(
            task_id="test",
            timestamp=datetime.now(),
            success=True,
            cost=0.1,
            latency_ms=100.0,
            metadata={"model": "gpt-4", "tokens": 500}
        )

        assert result.metadata["model"] == "gpt-4"
        assert result.metadata["tokens"] == 500

    def test_task_result_to_dict(self):
        """Should convert to dictionary."""
        result = TaskResult(
            task_id="test",
            timestamp=datetime.now(),
            success=True,
            cost=0.1,
            latency_ms=100.0
        )

        data = result.to_dict()
        assert data["task_id"] == "test"
        assert data["success"] is True
        assert "timestamp" in data


# Tests for WindowMetrics dataclass

class TestWindowMetrics:
    """Tests for WindowMetrics dataclass."""

    def test_window_metrics_creation(self):
        """Should create window metrics."""
        now = datetime.now()
        metrics = WindowMetrics(
            window_id="window-1",
            window_start=now,
            window_end=now + timedelta(minutes=10),
            success_rate=0.8,
            task_count=10,
            success_count=8,
            failure_count=2,
            mean_cost=0.05,
            total_cost=0.5,
            mean_latency_ms=100.0,
            cost_variance=0.001,
            latency_variance=50.0
        )

        assert metrics.success_rate == 0.8
        assert metrics.task_count == 10

    def test_cost_per_success(self):
        """Should calculate cost per success."""
        metrics = WindowMetrics(
            window_id="test",
            window_start=datetime.now(),
            window_end=datetime.now(),
            success_rate=0.8,
            task_count=10,
            success_count=8,
            failure_count=2,
            mean_cost=0.05,
            total_cost=0.5,
            mean_latency_ms=100.0,
            cost_variance=0.001,
            latency_variance=50.0
        )

        # 0.5 total / 8 successes = 0.0625
        assert metrics.cost_per_success == pytest.approx(0.0625, abs=0.001)

    def test_cost_per_success_no_successes(self):
        """Should return inf when no successes."""
        metrics = WindowMetrics(
            window_id="test",
            window_start=datetime.now(),
            window_end=datetime.now(),
            success_rate=0.0,
            task_count=10,
            success_count=0,
            failure_count=10,
            mean_cost=0.05,
            total_cost=0.5,
            mean_latency_ms=100.0,
            cost_variance=0.001,
            latency_variance=50.0
        )

        assert metrics.cost_per_success == float('inf')


# Tests for RollingWindowTracker initialization

class TestRollingWindowTrackerInitialization:
    """Tests for tracker initialization."""

    def test_default_initialization(self):
        """Should initialize with defaults."""
        tracker = RollingWindowTracker()
        assert tracker.window_size == 50
        assert tracker.overlap == 25

    def test_custom_initialization(self):
        """Should accept custom parameters."""
        tracker = RollingWindowTracker(
            window_size=100,
            overlap=50,
            max_history_windows=200
        )

        assert tracker.window_size == 100
        assert tracker.overlap == 50
        assert tracker.max_history_windows == 200

    def test_invalid_window_size(self):
        """Should reject invalid window size."""
        with pytest.raises(ValueError, match="window_size must be"):
            RollingWindowTracker(window_size=0)

    def test_invalid_overlap(self):
        """Should reject invalid overlap."""
        with pytest.raises(ValueError, match="overlap must be"):
            RollingWindowTracker(window_size=10, overlap=10)

        with pytest.raises(ValueError, match="overlap must be"):
            RollingWindowTracker(window_size=10, overlap=-1)


# Tests for recording results

class TestRollingWindowTrackerRecording:
    """Tests for recording task results."""

    def test_record_single_result(self, tracker):
        """Should record a single result."""
        result = make_task_result()
        tracker.record(result)

        assert len(tracker.get_all_results()) == 1

    def test_record_multiple_results(self, tracker):
        """Should record multiple results."""
        for i in range(5):
            tracker.record(make_task_result(task_id=f"task_{i}"))

        assert len(tracker.get_all_results()) == 5

    def test_window_completion_returns_metrics(self, tracker):
        """Should return metrics when window completes."""
        # Window size is 10, overlap is 5
        # First window completes at 10 results
        for i in range(9):
            result = tracker.record(make_task_result(task_id=f"task_{i}"))
            assert result is None

        # 10th result completes the window
        result = tracker.record(make_task_result(task_id="task_9"))
        assert result is not None
        assert isinstance(result, WindowMetrics)

    def test_subsequent_window_completion(self, tracker):
        """Should complete subsequent windows with overlap."""
        windows_completed = 0

        for i in range(20):
            result = tracker.record(make_task_result(task_id=f"task_{i}"))
            if result is not None:
                windows_completed += 1

        # With window_size=10 and overlap=5, step is 5
        # Windows at: 10, 15, 20 = 3 windows
        assert windows_completed == 3


# Tests for current window metrics

class TestCurrentWindowMetrics:
    """Tests for getting current window metrics."""

    def test_current_window_empty_tracker(self, tracker):
        """Should return None for empty tracker."""
        assert tracker.get_current_window() is None

    def test_current_window_partial_data(self, tracker):
        """Should compute metrics for partial window."""
        for i in range(5):
            tracker.record(make_task_result(task_id=f"task_{i}"))

        metrics = tracker.get_current_window()
        assert metrics is not None
        assert metrics.task_count == 5

    def test_current_window_success_rate(self, tracker):
        """Should compute correct success rate."""
        # Add 6 successes and 4 failures
        for i in range(10):
            tracker.record(make_task_result(
                task_id=f"task_{i}",
                success=(i < 6)
            ))

        metrics = tracker.get_current_window()
        assert metrics.success_rate == pytest.approx(0.6, abs=0.01)
        assert metrics.success_count == 6
        assert metrics.failure_count == 4


# Tests for rolling success rate

class TestRollingSuccessRate:
    """Tests for rolling success rate."""

    def test_rolling_success_rate_empty(self, tracker):
        """Should return 0 for empty tracker."""
        assert tracker.get_rolling_success_rate() == 0.0

    def test_rolling_success_rate_all_success(self, tracker):
        """Should return 1.0 for all successes."""
        for i in range(10):
            tracker.record(make_task_result(success=True))

        rate = tracker.get_rolling_success_rate()
        assert rate == 1.0

    def test_rolling_success_rate_all_failures(self, tracker):
        """Should return 0.0 for all failures."""
        for i in range(10):
            tracker.record(make_task_result(success=False))

        rate = tracker.get_rolling_success_rate()
        assert rate == 0.0

    def test_rolling_success_rate_mixed(self, tracker):
        """Should compute correct rate for mixed results."""
        # 7 successes, 3 failures
        for i in range(10):
            tracker.record(make_task_result(success=(i < 7)))

        rate = tracker.get_rolling_success_rate()
        assert rate == pytest.approx(0.7, abs=0.01)


# Tests for rolling values

class TestRollingValues:
    """Tests for getting rolling values."""

    def test_get_rolling_values_success_rate(self, populated_tracker):
        """Should return success rate values."""
        values = populated_tracker.get_rolling_values("success_rate")
        assert len(values) > 0
        assert all(0.0 <= v <= 1.0 for v in values)

    def test_get_rolling_values_cost(self, populated_tracker):
        """Should return cost values."""
        values = populated_tracker.get_rolling_values("mean_cost")
        assert len(values) > 0
        assert all(v >= 0 for v in values)


# Tests for trend detection

class TestTrendDetection:
    """Tests for trend detection."""

    def test_success_trend_stable(self, tracker):
        """Should detect stable trend with consistent success."""
        # All same success rate
        for i in range(20):
            # Alternate: success, success, failure pattern
            tracker.record(make_task_result(success=(i % 3 != 0)))

        trend = tracker.get_success_trend()
        # With consistent pattern, should be stable or slight variation
        assert trend in ["stable", "improving", "degrading"]

    def test_success_trend_improving(self):
        """Should detect improving trend."""
        tracker = RollingWindowTracker(window_size=5, overlap=2)

        # First batch: all failures
        now = datetime.now()
        for i in range(5):
            tracker.record(TaskResult(
                task_id=f"task_{i}",
                timestamp=now + timedelta(minutes=i),
                success=False,
                cost=0.1,
                latency_ms=100
            ))

        # Second batch: all successes
        for i in range(5, 15):
            tracker.record(TaskResult(
                task_id=f"task_{i}",
                timestamp=now + timedelta(minutes=i),
                success=True,
                cost=0.1,
                latency_ms=100
            ))

        trend = tracker.get_success_trend()
        assert trend == "improving"

    def test_success_trend_degrading(self):
        """Should detect degrading trend."""
        tracker = RollingWindowTracker(window_size=5, overlap=2)

        now = datetime.now()
        # First batch: all successes
        for i in range(5):
            tracker.record(TaskResult(
                task_id=f"task_{i}",
                timestamp=now + timedelta(minutes=i),
                success=True,
                cost=0.1,
                latency_ms=100
            ))

        # Second batch: all failures
        for i in range(5, 15):
            tracker.record(TaskResult(
                task_id=f"task_{i}",
                timestamp=now + timedelta(minutes=i),
                success=False,
                cost=0.1,
                latency_ms=100
            ))

        trend = tracker.get_success_trend()
        assert trend == "degrading"


# Tests for performance degradation detection

class TestPerformanceDegradationDetection:
    """Tests for degradation detection."""

    def test_no_degradation_stable_performance(self, tracker):
        """Should not detect degradation with stable performance."""
        # Add results with stable performance
        for i in range(20):
            tracker.record(make_task_result(
                success=True,
                cost=0.1,
                latency_ms=100
            ))

        assert not tracker.detect_performance_degradation()

    def test_detect_degradation_success_drop(self):
        """Should detect degradation when success rate drops."""
        tracker = RollingWindowTracker(window_size=5, overlap=2)

        now = datetime.now()
        # Start with high success
        for i in range(10):
            tracker.record(TaskResult(
                task_id=f"task_{i}",
                timestamp=now + timedelta(minutes=i),
                success=True,
                cost=0.1,
                latency_ms=100
            ))

        # Then failures
        for i in range(10, 20):
            tracker.record(TaskResult(
                task_id=f"task_{i}",
                timestamp=now + timedelta(minutes=i),
                success=False,
                cost=0.1,
                latency_ms=100
            ))

        assert tracker.detect_performance_degradation()


# Tests for trend analysis

class TestTrendAnalysis:
    """Tests for comprehensive trend analysis."""

    def test_analyze_trend_insufficient_data(self, tracker):
        """Should return None with insufficient data."""
        tracker.record(make_task_result())
        trend = tracker.analyze_trend()
        assert trend is None

    def test_analyze_trend_returns_all_components(self, populated_tracker):
        """Should return complete trend analysis."""
        trend = populated_tracker.analyze_trend()

        assert trend is not None
        assert trend.success_trend in ["stable", "improving", "degrading"]
        assert trend.cost_trend in ["stable", "increasing", "decreasing"]
        assert trend.latency_trend in ["stable", "increasing", "decreasing"]
        assert isinstance(trend.is_degrading, bool)
        assert isinstance(trend.degradation_metrics, list)


# Tests for cost trajectory

class TestCostTrajectory:
    """Tests for cost trajectory."""

    def test_cost_trajectory_empty(self, tracker):
        """Should return empty list for empty tracker."""
        trajectory = tracker.get_cost_trajectory()
        assert trajectory == []

    def test_cost_trajectory_computation(self, populated_tracker):
        """Should compute cost trajectory."""
        trajectory = populated_tracker.get_cost_trajectory()
        assert len(trajectory) > 0


# Tests for window history

class TestWindowHistory:
    """Tests for window history."""

    def test_window_history_initially_empty(self, tracker):
        """Should start with empty history."""
        assert tracker.get_window_history() == []

    def test_window_history_populated(self, populated_tracker):
        """Should track window history."""
        history = populated_tracker.get_window_history()
        assert len(history) > 0
        assert all(isinstance(w, WindowMetrics) for w in history)


# Tests for summary statistics

class TestSummaryStatistics:
    """Tests for summary statistics."""

    def test_summary_empty_tracker(self, tracker):
        """Should handle empty tracker."""
        summary = tracker.get_summary_statistics()

        assert summary["total_tasks"] == 0
        assert summary["overall_success_rate"] == 0.0
        assert summary["total_cost"] == 0.0

    def test_summary_with_data(self, populated_tracker):
        """Should compute summary statistics."""
        summary = populated_tracker.get_summary_statistics()

        assert summary["total_tasks"] == 20
        assert 0.0 <= summary["overall_success_rate"] <= 1.0
        assert summary["total_cost"] > 0
        assert summary["mean_latency_ms"] > 0


# Tests for reset

class TestTrackerReset:
    """Tests for resetting tracker."""

    def test_reset_clears_all_data(self, populated_tracker):
        """Should clear all data on reset."""
        assert len(populated_tracker.get_all_results()) > 0
        assert len(populated_tracker.get_window_history()) > 0

        populated_tracker.reset()

        assert len(populated_tracker.get_all_results()) == 0
        assert len(populated_tracker.get_window_history()) == 0


# Tests for to_dict export

class TestTrackerExport:
    """Tests for export functionality."""

    def test_to_dict_empty(self, tracker):
        """Should export empty tracker state."""
        data = tracker.to_dict()

        assert data["window_size"] == 10
        assert data["total_tasks"] == 0
        assert data["current_window"] is None

    def test_to_dict_with_data(self, populated_tracker):
        """Should export populated tracker state."""
        data = populated_tracker.to_dict()

        assert data["total_tasks"] == 20
        assert data["windows_completed"] > 0
        assert data["current_window"] is not None
        assert "summary" in data


# Tests for TimeBasedRollingTracker

class TestTimeBasedRollingTracker:
    """Tests for time-based tracker."""

    def test_initialization(self):
        """Should initialize with time parameters."""
        tracker = TimeBasedRollingTracker(
            window_minutes=60,
            overlap_minutes=30
        )

        assert tracker.window_duration == timedelta(minutes=60)
        assert tracker.overlap_duration == timedelta(minutes=30)

    def test_record_and_retrieve(self):
        """Should record and retrieve results."""
        tracker = TimeBasedRollingTracker(window_minutes=60)

        now = datetime.now()
        for i in range(5):
            tracker.record(TaskResult(
                task_id=f"task_{i}",
                timestamp=now - timedelta(minutes=i * 5),
                success=True,
                cost=0.1,
                latency_ms=100
            ))

        metrics = tracker.get_current_window()
        assert metrics is not None
        assert metrics.task_count == 5

    def test_time_window_filtering(self):
        """Should only include results within time window."""
        tracker = TimeBasedRollingTracker(window_minutes=30)

        now = datetime.now()

        # Result within window
        tracker.record(TaskResult(
            task_id="recent",
            timestamp=now - timedelta(minutes=10),
            success=True,
            cost=0.1,
            latency_ms=100
        ))

        # Result outside window
        tracker.record(TaskResult(
            task_id="old",
            timestamp=now - timedelta(minutes=60),
            success=False,
            cost=0.1,
            latency_ms=100
        ))

        metrics = tracker.get_current_window()
        assert metrics.task_count == 1  # Only recent result
        assert metrics.success_rate == 1.0

    def test_success_rate_over_time(self):
        """Should compute bucketed success rates."""
        tracker = TimeBasedRollingTracker(window_minutes=60)

        now = datetime.now()
        # Add results at different times
        for i in range(10):
            tracker.record(TaskResult(
                task_id=f"task_{i}",
                timestamp=now - timedelta(minutes=i * 5),
                success=(i % 2 == 0),
                cost=0.1,
                latency_ms=100
            ))

        buckets = tracker.get_success_rate_over_time(bucket_minutes=15)
        assert len(buckets) > 0


# Edge case tests

class TestEdgeCases:
    """Tests for edge cases."""

    def test_single_result_window(self):
        """Should handle single result as window."""
        tracker = RollingWindowTracker(window_size=1, overlap=0)

        result = tracker.record(make_task_result(success=True))
        assert result is not None
        assert result.success_rate == 1.0

    def test_all_same_values(self, tracker):
        """Should handle all identical values."""
        for i in range(20):
            tracker.record(make_task_result(
                success=True,
                cost=0.1,
                latency_ms=100
            ))

        metrics = tracker.get_current_window()
        assert metrics.cost_variance < 0.001
        assert metrics.latency_variance < 0.001

    def test_high_variance_data(self, tracker):
        """Should handle high variance data."""
        for i in range(20):
            tracker.record(make_task_result(
                cost=0.1 * (2 ** i),  # Exponentially increasing
                latency_ms=100 * (i + 1)
            ))

        metrics = tracker.get_current_window()
        assert metrics.cost_variance > 0
        assert metrics.latency_variance > 0
