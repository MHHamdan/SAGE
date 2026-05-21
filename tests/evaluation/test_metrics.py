"""Tests for evaluation metrics."""

import pytest


class TestCNSR:
    """Test CNSR calculation."""

    def test_cnsr_basic(self):
        """Test basic CNSR calculation."""
        from sage.evaluation import calculate_cnsr

        # 80% success at $0.50/task = CNSR 1.6
        cnsr = calculate_cnsr(successes=80, total_tasks=100, total_cost=50.0)
        assert abs(cnsr - 1.6) < 0.001

    def test_cnsr_zero_cost(self):
        """Test CNSR with zero cost (local models)."""
        from sage.evaluation import calculate_cnsr

        cnsr = calculate_cnsr(successes=80, total_tasks=100, total_cost=0.0)
        assert cnsr == float("inf")

    def test_cnsr_no_successes(self):
        """Test CNSR with no successes."""
        from sage.evaluation import calculate_cnsr

        cnsr = calculate_cnsr(successes=0, total_tasks=100, total_cost=10.0)
        assert cnsr == 0.0

    def test_cnsr_no_tasks(self):
        """Test CNSR with no tasks."""
        from sage.evaluation import calculate_cnsr

        cnsr = calculate_cnsr(successes=0, total_tasks=0, total_cost=0.0)
        assert cnsr == 0.0


class TestRollingWindow:
    """Test rolling window success rate."""

    def test_rolling_window_basic(self):
        """Test basic rolling window calculation."""
        from sage.evaluation import rolling_window_success

        results = [True, True, False, True, False]
        rolling = rolling_window_success(results, window_size=3)

        assert len(rolling) == 5
        # Last window [True, False, True]: 2/3
        assert abs(rolling[-1] - 2 / 3) < 0.001

    def test_rolling_window_all_success(self):
        """Test rolling window with all successes."""
        from sage.evaluation import rolling_window_success

        results = [True] * 10
        rolling = rolling_window_success(results, window_size=5)

        assert all(r == 1.0 for r in rolling)

    def test_rolling_window_empty(self):
        """Test rolling window with empty results."""
        from sage.evaluation import rolling_window_success

        rolling = rolling_window_success([], window_size=5)
        assert rolling == []


class TestGoalDrift:
    """Test goal drift score."""

    def test_goal_drift_identical(self):
        """Test drift with identical embeddings."""
        from sage.evaluation import goal_drift_score

        embedding = [1.0, 0.5, 0.3]
        drift = goal_drift_score(embedding, embedding)

        assert abs(drift) < 0.001

    def test_goal_drift_orthogonal(self):
        """Test drift with orthogonal embeddings."""
        from sage.evaluation import goal_drift_score

        e1 = [1.0, 0.0]
        e2 = [0.0, 1.0]
        drift = goal_drift_score(e1, e2)

        assert abs(drift - 1.0) < 0.001


class TestIncidentTracker:
    """Test incident tracking."""

    def test_incident_recording(self):
        """Test incident recording."""
        from sage.evaluation import IncidentTracker

        tracker = IncidentTracker()
        tracker.record_incident("human_intervention")
        tracker.record_incident("guardrail")
        tracker.record_incident("violation")

        assert tracker.human_interventions == 1
        assert tracker.guardrail_activations == 1
        assert tracker.constraint_violations == 1
        assert tracker.total_incidents == 3

    def test_incident_rate(self):
        """Test incident rate calculation."""
        from sage.evaluation import IncidentTracker

        tracker = IncidentTracker()
        tracker.record_incident("human_intervention")
        tracker.record_incident("guardrail")

        rate = tracker.incident_rate(total_tasks=100)
        assert rate == 0.02


class TestEvaluateAgent:
    """Test comprehensive agent evaluation."""

    def test_evaluate_agent(self):
        """Test full agent evaluation."""
        from sage.evaluation import evaluate_agent

        result = evaluate_agent(
            successes=80,
            total_tasks=100,
            total_cost=50.0,
        )

        assert result.success_rate == 0.8
        assert result.mean_cost == 0.5
        assert abs(result.cnsr - 1.6) < 0.001
        assert result.total_tasks == 100
        assert result.total_successes == 80
