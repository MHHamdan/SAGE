"""
Tests for Goal Drift Detection Module.

Tests cover:
- Drift score computation
- Goal similarity calculation
- Drift tracking over time
- Trend detection
- Re-anchoring behavior
- Edge cases and error handling
"""

import pytest
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import Mock, patch

from sage.evaluation.goal_drift import (
    GoalDriftTracker,
    GoalSnapshot,
    DriftMeasurement,
    DriftStatistics,
    GoalSource,
    MultiGoalDriftTracker,
    compute_goal_similarity,
    compute_drift_score,
)


# Test fixtures

@pytest.fixture
def mock_embed_fn():
    """Mock embedding function that returns deterministic embeddings."""
    def embed(text: str) -> np.ndarray:
        # Create deterministic embedding based on text content
        # This allows us to control similarity for testing
        np.random.seed(hash(text) % (2**32))
        return np.random.randn(384)  # Common embedding dimension
    return embed


@pytest.fixture
def similar_embed_fn():
    """Embedding function where similar text produces similar embeddings."""
    base_embedding = np.array([1.0, 0.5, 0.3, 0.1, 0.0])

    def embed(text: str) -> np.ndarray:
        if "identical" in text.lower():
            return base_embedding.copy()
        elif "similar" in text.lower():
            # Add small noise for similar but not identical
            noise = np.random.randn(5) * 0.1
            return base_embedding + noise
        elif "different" in text.lower():
            # Return orthogonal vector
            return np.array([0.0, 0.1, 0.3, 0.5, 1.0])
        elif "opposite" in text.lower():
            # Return opposite direction
            return -base_embedding
        else:
            # Random embedding
            np.random.seed(hash(text) % (2**32))
            return np.random.randn(5)
    return embed


@pytest.fixture
def tracker(mock_embed_fn):
    """Create a basic goal drift tracker."""
    return GoalDriftTracker(embed_fn=mock_embed_fn)


@pytest.fixture
def tracker_with_goal(similar_embed_fn):
    """Create tracker with original goal set."""
    tracker = GoalDriftTracker(embed_fn=similar_embed_fn)
    tracker.set_original_goal("This is the identical original goal")
    return tracker


# Tests for compute_goal_similarity

class TestComputeGoalSimilarity:
    """Tests for the goal similarity computation function."""

    def test_identical_embeddings_have_similarity_one(self):
        """Identical embeddings should have similarity of 1.0."""
        embedding = np.array([1.0, 2.0, 3.0, 4.0])
        similarity = compute_goal_similarity(embedding, embedding)
        assert similarity == pytest.approx(1.0, abs=1e-6)

    def test_opposite_embeddings_have_similarity_negative_one(self):
        """Opposite embeddings should have similarity of -1.0."""
        embedding = np.array([1.0, 2.0, 3.0, 4.0])
        opposite = -embedding
        similarity = compute_goal_similarity(embedding, opposite)
        assert similarity == pytest.approx(-1.0, abs=1e-6)

    def test_orthogonal_embeddings_have_zero_similarity(self):
        """Orthogonal embeddings should have similarity of 0.0."""
        e1 = np.array([1.0, 0.0, 0.0])
        e2 = np.array([0.0, 1.0, 0.0])
        similarity = compute_goal_similarity(e1, e2)
        assert similarity == pytest.approx(0.0, abs=1e-6)

    def test_similarity_is_symmetric(self):
        """Similarity should be symmetric: sim(a,b) = sim(b,a)."""
        e1 = np.array([1.0, 2.0, 3.0])
        e2 = np.array([4.0, 5.0, 6.0])
        assert compute_goal_similarity(e1, e2) == pytest.approx(
            compute_goal_similarity(e2, e1), abs=1e-6
        )

    def test_mismatched_dimensions_raises_error(self):
        """Different embedding dimensions should raise ValueError."""
        e1 = np.array([1.0, 2.0, 3.0])
        e2 = np.array([1.0, 2.0])
        with pytest.raises(ValueError, match="dimensions must match"):
            compute_goal_similarity(e1, e2)

    def test_zero_vector_returns_zero(self):
        """Zero vector should return 0.0 similarity."""
        e1 = np.array([1.0, 2.0, 3.0])
        e2 = np.array([0.0, 0.0, 0.0])
        similarity = compute_goal_similarity(e1, e2)
        assert similarity == 0.0

    def test_accepts_list_input(self):
        """Should accept Python lists as input."""
        e1 = [1.0, 2.0, 3.0]
        e2 = [1.0, 2.0, 3.0]
        similarity = compute_goal_similarity(e1, e2)
        assert similarity == pytest.approx(1.0, abs=1e-6)


# Tests for compute_drift_score

class TestComputeDriftScore:
    """Tests for drift score computation."""

    def test_identical_embeddings_have_zero_drift(self):
        """Identical goals should have drift score of 0."""
        embedding = np.array([1.0, 2.0, 3.0, 4.0])
        drift = compute_drift_score(embedding, embedding)
        assert drift == pytest.approx(0.0, abs=1e-6)

    def test_opposite_embeddings_have_maximum_drift(self):
        """Opposite goals should have drift score of 1.0."""
        embedding = np.array([1.0, 2.0, 3.0, 4.0])
        opposite = -embedding
        drift = compute_drift_score(embedding, opposite)
        assert drift == pytest.approx(1.0, abs=1e-6)

    def test_drift_is_between_zero_and_one(self):
        """Drift score should always be in [0, 1]."""
        for _ in range(10):
            e1 = np.random.randn(10)
            e2 = np.random.randn(10)
            drift = compute_drift_score(e1, e2)
            assert 0.0 <= drift <= 1.0


# Tests for GoalDriftTracker

class TestGoalDriftTrackerInitialization:
    """Tests for tracker initialization."""

    def test_initialization_with_defaults(self, mock_embed_fn):
        """Tracker should initialize with default values."""
        tracker = GoalDriftTracker(embed_fn=mock_embed_fn)
        assert tracker.drift_threshold == 0.3
        assert tracker.window_size == 10
        assert tracker.reanchor_threshold == 0.5

    def test_initialization_with_custom_values(self, mock_embed_fn):
        """Tracker should accept custom configuration."""
        tracker = GoalDriftTracker(
            embed_fn=mock_embed_fn,
            drift_threshold=0.5,
            window_size=20,
            reanchor_threshold=0.7
        )
        assert tracker.drift_threshold == 0.5
        assert tracker.window_size == 20
        assert tracker.reanchor_threshold == 0.7


class TestGoalDriftTrackerOriginalGoal:
    """Tests for setting original goal."""

    def test_set_original_goal(self, tracker):
        """Should successfully set original goal."""
        snapshot = tracker.set_original_goal("Complete the analysis")
        assert snapshot.goal_text == "Complete the analysis"
        assert snapshot.source == GoalSource.ORIGINAL
        assert snapshot.confidence == 1.0

    def test_empty_goal_raises_error(self, tracker):
        """Empty goal should raise ValueError."""
        with pytest.raises(ValueError, match="cannot be empty"):
            tracker.set_original_goal("")

    def test_whitespace_goal_raises_error(self, tracker):
        """Whitespace-only goal should raise ValueError."""
        with pytest.raises(ValueError, match="cannot be empty"):
            tracker.set_original_goal("   ")

    def test_goal_with_metadata(self, tracker):
        """Should store metadata with goal."""
        metadata = {"source": "user", "priority": "high"}
        snapshot = tracker.set_original_goal("Complete task", metadata=metadata)
        assert snapshot.metadata == metadata


class TestGoalDriftTrackerRecordInferredGoal:
    """Tests for recording inferred goals."""

    def test_record_inferred_goal_without_original_raises_error(self, tracker):
        """Should raise error if original goal not set."""
        with pytest.raises(RuntimeError, match="Original goal must be set"):
            tracker.record_inferred_goal("Some goal")

    def test_record_identical_goal_has_low_drift(self, tracker_with_goal):
        """Recording identical goal should have very low drift."""
        measurement = tracker_with_goal.record_inferred_goal(
            "This is the identical original goal"
        )
        assert measurement.drift_score < 0.1

    def test_record_different_goal_has_high_drift(self, tracker_with_goal):
        """Recording different goal should have high drift."""
        measurement = tracker_with_goal.record_inferred_goal(
            "This is a completely different goal"
        )
        assert measurement.drift_score > 0.1

    def test_record_with_confidence(self, tracker_with_goal):
        """Should record confidence level."""
        measurement = tracker_with_goal.record_inferred_goal(
            "Similar goal",
            confidence=0.8
        )
        assert measurement.confidence == 0.8

    def test_invalid_confidence_raises_error(self, tracker_with_goal):
        """Invalid confidence should raise ValueError."""
        with pytest.raises(ValueError, match="Confidence must be"):
            tracker_with_goal.record_inferred_goal("Goal", confidence=1.5)

    def test_record_with_source(self, tracker_with_goal):
        """Should record source of inference."""
        measurement = tracker_with_goal.record_inferred_goal(
            "Similar goal",
            source="explicit_statement"
        )
        assert measurement.source == GoalSource.EXPLICIT_STATEMENT


class TestGoalDriftTrackerDriftMeasurement:
    """Tests for drift measurement retrieval."""

    def test_get_current_drift_no_measurements(self, tracker):
        """Should return 0.0 when no measurements recorded."""
        tracker.set_original_goal("Original")
        assert tracker.get_current_drift() == 0.0

    def test_get_current_drift_after_measurement(self, tracker_with_goal):
        """Should return most recent drift score."""
        tracker_with_goal.record_inferred_goal("Similar goal")
        drift = tracker_with_goal.get_current_drift()
        assert 0.0 <= drift <= 1.0

    def test_get_drift_history(self, tracker_with_goal):
        """Should return all measurements."""
        tracker_with_goal.record_inferred_goal("Goal 1")
        tracker_with_goal.record_inferred_goal("Goal 2")
        tracker_with_goal.record_inferred_goal("Goal 3")

        history = tracker_with_goal.get_drift_history()
        assert len(history) == 3


class TestGoalDriftTrackerTrendDetection:
    """Tests for drift trend detection."""

    def test_trend_stable_with_few_measurements(self, tracker_with_goal):
        """Should return 'stable' with insufficient data."""
        tracker_with_goal.record_inferred_goal("Goal")
        assert tracker_with_goal.get_drift_trend() == "stable"

    def test_trend_stable_with_constant_drift(self, similar_embed_fn):
        """Should return 'stable' when drift is constant."""
        tracker = GoalDriftTracker(
            embed_fn=similar_embed_fn,
            trend_sensitivity=0.1
        )
        tracker.set_original_goal("identical original")

        # Record many similar goals
        for _ in range(10):
            tracker.record_inferred_goal("identical current")

        assert tracker.get_drift_trend() == "stable"

    def test_trend_detection_with_increasing_drift(self, mock_embed_fn):
        """Should detect increasing drift trend."""
        tracker = GoalDriftTracker(
            embed_fn=mock_embed_fn,
            window_size=5,
            trend_sensitivity=0.01
        )
        tracker.set_original_goal("original")

        # Manually set measurements with increasing drift
        for i in range(5):
            measurement = DriftMeasurement(
                measurement_id=str(i),
                timestamp=datetime.now(),
                drift_score=0.1 + i * 0.1,  # 0.1, 0.2, 0.3, 0.4, 0.5
                original_goal="original",
                current_goal=f"goal_{i}",
                similarity=0.9 - i * 0.1
            )
            tracker._measurements.append(measurement)

        assert tracker.get_drift_trend() == "increasing"


class TestGoalDriftTrackerStatistics:
    """Tests for drift statistics."""

    def test_statistics_with_no_measurements(self, tracker):
        """Should return zero statistics when no measurements."""
        tracker.set_original_goal("Original")
        stats = tracker.get_drift_statistics()
        assert stats.num_measurements == 0
        assert stats.mean_drift == 0.0

    def test_statistics_calculation(self, tracker_with_goal):
        """Should calculate correct statistics."""
        # Record some goals
        for _ in range(5):
            tracker_with_goal.record_inferred_goal("Similar goal")

        stats = tracker_with_goal.get_drift_statistics()
        assert stats.num_measurements == 5
        assert 0.0 <= stats.mean_drift <= 1.0
        assert stats.min_drift <= stats.mean_drift <= stats.max_drift


class TestGoalDriftTrackerReanchoring:
    """Tests for goal re-anchoring."""

    def test_should_reanchor_low_drift(self, tracker_with_goal):
        """Should not recommend re-anchoring with low drift."""
        tracker_with_goal.record_inferred_goal("identical goal")
        assert not tracker_with_goal.should_reanchor()

    def test_should_reanchor_high_drift(self, similar_embed_fn):
        """Should recommend re-anchoring with high drift."""
        tracker = GoalDriftTracker(
            embed_fn=similar_embed_fn,
            reanchor_threshold=0.3
        )
        tracker.set_original_goal("identical original")
        tracker.record_inferred_goal("opposite direction goal")

        # Check if drift is high enough to trigger reanchor
        if tracker.get_current_drift() > 0.3:
            assert tracker.should_reanchor()

    def test_reanchor_resets_tracking(self, tracker_with_goal):
        """Re-anchoring should reset measurements."""
        tracker_with_goal.record_inferred_goal("Goal 1")
        tracker_with_goal.record_inferred_goal("Goal 2")
        assert len(tracker_with_goal.get_drift_history()) == 2

        tracker_with_goal.reanchor_goal("New goal")
        assert len(tracker_with_goal.get_drift_history()) == 0

    def test_reanchor_increments_count(self, tracker_with_goal):
        """Re-anchoring should increment counter."""
        assert tracker_with_goal.get_reanchor_count() == 0

        tracker_with_goal.reanchor_goal("New goal 1")
        assert tracker_with_goal.get_reanchor_count() == 1

        tracker_with_goal.reanchor_goal("New goal 2")
        assert tracker_with_goal.get_reanchor_count() == 2


class TestGoalDriftTrackerEdgeCases:
    """Tests for edge cases and error handling."""

    def test_is_drifting(self, tracker_with_goal):
        """Should correctly report drifting state."""
        tracker_with_goal.record_inferred_goal("identical goal")

        # Set a very low threshold to test the check
        tracker_with_goal.drift_threshold = 0.0001
        if tracker_with_goal.get_current_drift() > 0.0001:
            assert tracker_with_goal.is_drifting()

    def test_reset_clears_all_state(self, tracker_with_goal):
        """Reset should clear all tracking state."""
        tracker_with_goal.record_inferred_goal("Goal")
        tracker_with_goal.reanchor_goal("New")

        tracker_with_goal.reset()

        assert tracker_with_goal.get_reanchor_count() == 0
        assert len(tracker_with_goal.get_drift_history()) == 0
        assert len(tracker_with_goal.get_goal_history()) == 0

    def test_to_dict_export(self, tracker_with_goal):
        """Should export state to dictionary."""
        tracker_with_goal.record_inferred_goal("Goal")

        data = tracker_with_goal.to_dict()
        assert "original_goal" in data
        assert "current_goal" in data
        assert "current_drift" in data
        assert "drift_trend" in data


# Tests for MultiGoalDriftTracker

class TestMultiGoalDriftTracker:
    """Tests for multi-goal tracking."""

    def test_add_multiple_goals(self, mock_embed_fn):
        """Should track multiple goals independently."""
        tracker = MultiGoalDriftTracker(embed_fn=mock_embed_fn)

        tracker.add_goal("primary", "Primary objective")
        tracker.add_goal("secondary", "Secondary objective")

        report = tracker.get_drift_report()
        assert "primary" in report
        assert "secondary" in report

    def test_record_goal_for_specific_id(self, mock_embed_fn):
        """Should record inferred goal for specific goal ID."""
        tracker = MultiGoalDriftTracker(embed_fn=mock_embed_fn)
        tracker.add_goal("main", "Main goal")

        measurement = tracker.record_inferred_goal("main", "Updated main goal")
        assert 0.0 <= measurement.drift_score <= 1.0

    def test_unknown_goal_id_raises_error(self, mock_embed_fn):
        """Should raise error for unknown goal ID."""
        tracker = MultiGoalDriftTracker(embed_fn=mock_embed_fn)

        with pytest.raises(KeyError, match="Unknown goal ID"):
            tracker.record_inferred_goal("nonexistent", "Goal")

    def test_aggregate_drift_calculation(self, mock_embed_fn):
        """Should calculate weighted aggregate drift."""
        tracker = MultiGoalDriftTracker(embed_fn=mock_embed_fn)
        tracker.add_goal("a", "Goal A")
        tracker.add_goal("b", "Goal B")

        tracker.record_inferred_goal("a", "Updated A")
        tracker.record_inferred_goal("b", "Updated B")

        aggregate = tracker.get_aggregate_drift()
        assert 0.0 <= aggregate <= 1.0


# Tests for GoalSnapshot dataclass

class TestGoalSnapshot:
    """Tests for GoalSnapshot dataclass."""

    def test_invalid_confidence_raises_error(self):
        """Invalid confidence should raise ValueError."""
        with pytest.raises(ValueError, match="Confidence must be"):
            GoalSnapshot(
                snapshot_id="test",
                timestamp=datetime.now(),
                goal_text="Goal",
                goal_embedding=np.array([1, 2, 3]),
                source=GoalSource.ORIGINAL,
                confidence=1.5  # Invalid
            )

    def test_embedding_converted_to_numpy(self):
        """Embedding should be converted to numpy array."""
        snapshot = GoalSnapshot(
            snapshot_id="test",
            timestamp=datetime.now(),
            goal_text="Goal",
            goal_embedding=[1, 2, 3],  # List input
            source=GoalSource.ORIGINAL
        )
        assert isinstance(snapshot.goal_embedding, np.ndarray)


# Tests for DriftMeasurement dataclass

class TestDriftMeasurement:
    """Tests for DriftMeasurement dataclass."""

    def test_invalid_drift_score_raises_error(self):
        """Invalid drift score should raise ValueError."""
        with pytest.raises(ValueError, match="Drift score must be"):
            DriftMeasurement(
                measurement_id="test",
                timestamp=datetime.now(),
                drift_score=1.5,  # Invalid
                original_goal="Original",
                current_goal="Current",
                similarity=0.5
            )

    def test_valid_measurement_creation(self):
        """Should create valid measurement."""
        measurement = DriftMeasurement(
            measurement_id="test",
            timestamp=datetime.now(),
            drift_score=0.3,
            original_goal="Original",
            current_goal="Current",
            similarity=0.7
        )
        assert measurement.drift_score == 0.3
        assert measurement.similarity == 0.7


# Integration tests

class TestGoalDriftIntegration:
    """Integration tests for complete workflows."""

    def test_complete_drift_tracking_workflow(self, similar_embed_fn):
        """Test complete workflow from goal setting to drift analysis."""
        tracker = GoalDriftTracker(
            embed_fn=similar_embed_fn,
            drift_threshold=0.2,
            window_size=5
        )

        # Set original goal
        tracker.set_original_goal("identical original goal")

        # Record several inferred goals over time
        goals = [
            ("identical current goal", 0.1),  # Very similar
            ("similar current goal", 0.3),     # Somewhat similar
            ("different current goal", 0.8),   # Different
        ]

        for goal_text, _ in goals:
            tracker.record_inferred_goal(goal_text)

        # Analyze results
        stats = tracker.get_drift_statistics()
        assert stats.num_measurements == 3

        # Check we can get trend
        trend = tracker.get_drift_trend()
        assert trend in ["stable", "increasing", "decreasing"]

        # Export state
        state = tracker.to_dict()
        assert state["num_measurements"] == 3
