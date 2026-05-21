"""
Goal Drift Detection Module

Tracks divergence between original objectives and agent's current pursued goals
over long-horizon operation. Uses embedding similarity to quantify drift.

The goal drift score measures how much an agent's behavior has diverged from
its original objectives:

    Drift_t = 1 - sim(g_0, ĝ_t)

where g_0 is the original goal embedding and ĝ_t is the inferred current goal
at time t.

A well-behaved long-horizon agent should maintain low drift scores over time.
Increasing drift suggests the agent is losing track of its original objectives,
which is a key failure mode in autonomous systems.

Example:
    >>> from sage.evaluation.goal_drift import GoalDriftTracker
    >>>
    >>> tracker = GoalDriftTracker(embed_fn=get_embedding)
    >>> tracker.set_original_goal("Complete the data analysis report")
    >>>
    >>> # During operation, periodically infer current goal from actions
    >>> tracker.record_inferred_goal("Analyzing dataset", source="action_trace")
    >>> tracker.record_inferred_goal("Creating visualizations", source="action_trace")
    >>> tracker.record_inferred_goal("Writing documentation", source="explicit_statement")
    >>>
    >>> # Check drift
    >>> drift = tracker.get_current_drift()
    >>> if tracker.should_reanchor():
    ...     print("Warning: Consider re-anchoring goal")
"""

import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable, List, Optional, Union

import numpy as np

logger = logging.getLogger(__name__)


class GoalSource(Enum):
    """Source of goal inference."""

    ORIGINAL = "original"
    ACTION_TRACE = "action_trace"
    EXPLICIT_STATEMENT = "explicit_statement"
    USER_FEEDBACK = "user_feedback"
    SUMMARY = "summary"


@dataclass
class GoalSnapshot:
    """Snapshot of goal state at a point in time.

    Attributes:
        snapshot_id: Unique identifier for this snapshot
        timestamp: When the snapshot was taken
        goal_text: Text representation of the goal
        goal_embedding: Vector embedding of the goal
        source: How the goal was inferred
        confidence: Confidence in the goal inference (0.0 to 1.0)
        metadata: Additional context about the snapshot
    """

    snapshot_id: str
    timestamp: datetime
    goal_text: str
    goal_embedding: np.ndarray
    source: GoalSource
    confidence: float = 1.0
    metadata: dict = field(default_factory=dict)

    def __post_init__(self):
        """Validate snapshot data."""
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(
                f"Confidence must be between 0 and 1, got {self.confidence}"
            )
        if self.goal_embedding is not None:
            self.goal_embedding = np.asarray(self.goal_embedding)


@dataclass
class DriftMeasurement:
    """Single drift measurement at a point in time.

    Attributes:
        measurement_id: Unique identifier for this measurement
        timestamp: When the measurement was taken
        drift_score: Drift value from 0 (no drift) to 1 (complete drift)
        original_goal: Text of the original goal
        current_goal: Text of the current inferred goal
        similarity: Cosine similarity between goal embeddings
        confidence: Confidence in the current goal inference
        source: How the current goal was inferred
    """

    measurement_id: str
    timestamp: datetime
    drift_score: float
    original_goal: str
    current_goal: str
    similarity: float
    confidence: float = 1.0
    source: GoalSource = GoalSource.ACTION_TRACE

    def __post_init__(self):
        """Validate measurement data."""
        if not 0.0 <= self.drift_score <= 1.0:
            raise ValueError(
                f"Drift score must be between 0 and 1, got {self.drift_score}"
            )


@dataclass
class DriftStatistics:
    """Aggregated drift statistics.

    Attributes:
        mean_drift: Average drift score
        max_drift: Maximum drift score observed
        min_drift: Minimum drift score observed
        std_drift: Standard deviation of drift scores
        trend: Drift trend direction
        num_measurements: Number of measurements
        time_span_seconds: Time span covered by measurements
    """

    mean_drift: float
    max_drift: float
    min_drift: float
    std_drift: float
    trend: str  # "stable", "increasing", "decreasing"
    num_measurements: int
    time_span_seconds: float


def compute_goal_similarity(
    embedding1: np.ndarray, embedding2: np.ndarray, epsilon: float = 1e-8
) -> float:
    """Compute cosine similarity between goal embeddings.

    Args:
        embedding1: First embedding vector
        embedding2: Second embedding vector
        epsilon: Small value to prevent division by zero

    Returns:
        Cosine similarity between -1 and 1 (typically 0 to 1 for goal embeddings)

    Raises:
        ValueError: If embeddings have different dimensions
    """
    e1 = np.asarray(embedding1).flatten()
    e2 = np.asarray(embedding2).flatten()

    if e1.shape != e2.shape:
        raise ValueError(f"Embedding dimensions must match: {e1.shape} vs {e2.shape}")

    norm1 = np.linalg.norm(e1)
    norm2 = np.linalg.norm(e2)

    if norm1 < epsilon or norm2 < epsilon:
        logger.warning("One or both embeddings have near-zero norm")
        return 0.0

    similarity = np.dot(e1, e2) / (norm1 * norm2)
    # Clamp to valid range (handles floating point errors)
    return float(np.clip(similarity, -1.0, 1.0))


def compute_drift_score(
    original_embedding: np.ndarray, current_embedding: np.ndarray
) -> float:
    """Compute goal drift score from embeddings.

    Drift = 1 - similarity(original, current)

    Args:
        original_embedding: Embedding of original goal
        current_embedding: Embedding of current inferred goal

    Returns:
        Drift score between 0 (no drift) and 1 (complete drift)
    """
    similarity = compute_goal_similarity(original_embedding, current_embedding)
    # Map similarity from [-1, 1] to drift [0, 1]
    # similarity = 1 -> drift = 0 (identical goals)
    # similarity = -1 -> drift = 1 (opposite goals)
    # similarity = 0 -> drift = 0.5 (unrelated goals)
    drift = (1.0 - similarity) / 2.0
    return drift


class GoalDriftTracker:
    """
    Tracks goal drift over agent operation.

    This tracker monitors how an agent's pursued objectives diverge from
    its original goal over time. It supports:

    - Setting and tracking the original goal
    - Recording inferred current goals from various sources
    - Computing drift scores and trends
    - Recommending when goal re-anchoring is needed

    Usage:
        >>> tracker = GoalDriftTracker(embed_fn=get_embedding)
        >>> tracker.set_original_goal("Complete the data analysis report")
        >>>
        >>> # During operation, periodically infer current goal
        >>> measurement = tracker.record_inferred_goal(
        ...     "Working on visualization",
        ...     source="action_trace"
        ... )
        >>> print(f"Current drift: {measurement.drift_score:.3f}")
        >>>
        >>> # Check if re-anchoring is recommended
        >>> if tracker.should_reanchor():
        ...     tracker.reanchor_goal("Updated goal based on feedback")

    Attributes:
        embed_fn: Function to convert text to embedding vector
        drift_threshold: Threshold above which drift is concerning
        window_size: Number of recent measurements for trend analysis
    """

    def __init__(
        self,
        embed_fn: Callable[[str], Union[np.ndarray, List[float]]],
        drift_threshold: float = 0.3,
        window_size: int = 10,
        reanchor_threshold: float = 0.5,
        trend_sensitivity: float = 0.05,
    ):
        """Initialize the goal drift tracker.

        Args:
            embed_fn: Function to convert text to embedding vector.
                      Should accept a string and return a numpy array or list.
            drift_threshold: Threshold above which drift is concerning (0-1).
                            Default 0.3 means 30% drift triggers warnings.
            window_size: Number of recent measurements to consider for trend
                        analysis. Default 10.
            reanchor_threshold: Drift threshold above which re-anchoring is
                               recommended. Default 0.5.
            trend_sensitivity: Minimum slope to detect increasing/decreasing
                              trend. Default 0.05.
        """
        self.embed_fn = embed_fn
        self.drift_threshold = drift_threshold
        self.window_size = window_size
        self.reanchor_threshold = reanchor_threshold
        self.trend_sensitivity = trend_sensitivity

        self._original_goal: Optional[GoalSnapshot] = None
        self._current_goal: Optional[GoalSnapshot] = None
        self._measurements: List[DriftMeasurement] = []
        self._goal_history: List[GoalSnapshot] = []
        self._reanchor_count: int = 0

    def set_original_goal(
        self, goal: str, metadata: Optional[dict] = None
    ) -> GoalSnapshot:
        """Set the original goal at start of operation.

        This establishes the baseline against which all drift is measured.
        Should be called once at the beginning of agent operation.

        Args:
            goal: Text description of the original goal
            metadata: Optional metadata about the goal

        Returns:
            GoalSnapshot of the original goal

        Raises:
            ValueError: If goal is empty
        """
        if not goal or not goal.strip():
            raise ValueError("Goal text cannot be empty")

        embedding = self._get_embedding(goal)

        snapshot = GoalSnapshot(
            snapshot_id=str(uuid.uuid4()),
            timestamp=datetime.now(),
            goal_text=goal,
            goal_embedding=embedding,
            source=GoalSource.ORIGINAL,
            confidence=1.0,
            metadata=metadata or {},
        )

        self._original_goal = snapshot
        self._current_goal = snapshot
        self._goal_history.append(snapshot)

        logger.info(f"Original goal set: {goal[:100]}...")
        return snapshot

    def record_inferred_goal(
        self,
        goal: str,
        source: str = "action_trace",
        confidence: float = 1.0,
        metadata: Optional[dict] = None,
    ) -> DriftMeasurement:
        """Record an inferred current goal and compute drift.

        Call this periodically during agent operation to track how the
        agent's current objective compares to the original goal.

        Args:
            goal: Text description of the inferred current goal
            source: How the goal was inferred. One of:
                   - "action_trace": Inferred from agent actions
                   - "explicit_statement": Agent explicitly stated goal
                   - "user_feedback": Based on user feedback
                   - "summary": From agent's state summary
            confidence: Confidence in the inference (0.0 to 1.0)
            metadata: Optional metadata about the inference

        Returns:
            DriftMeasurement with computed drift score

        Raises:
            RuntimeError: If original goal not set
            ValueError: If goal is empty or confidence invalid
        """
        if self._original_goal is None:
            raise RuntimeError(
                "Original goal must be set before recording inferred goals"
            )

        if not goal or not goal.strip():
            raise ValueError("Goal text cannot be empty")

        if not 0.0 <= confidence <= 1.0:
            raise ValueError(f"Confidence must be between 0 and 1, got {confidence}")

        # Map source string to enum
        source_map = {
            "original": GoalSource.ORIGINAL,
            "action_trace": GoalSource.ACTION_TRACE,
            "explicit_statement": GoalSource.EXPLICIT_STATEMENT,
            "user_feedback": GoalSource.USER_FEEDBACK,
            "summary": GoalSource.SUMMARY,
        }
        goal_source = source_map.get(source.lower(), GoalSource.ACTION_TRACE)

        # Compute embedding and drift
        embedding = self._get_embedding(goal)
        similarity = compute_goal_similarity(
            self._original_goal.goal_embedding, embedding
        )
        drift_score = compute_drift_score(self._original_goal.goal_embedding, embedding)

        # Create snapshot
        snapshot = GoalSnapshot(
            snapshot_id=str(uuid.uuid4()),
            timestamp=datetime.now(),
            goal_text=goal,
            goal_embedding=embedding,
            source=goal_source,
            confidence=confidence,
            metadata=metadata or {},
        )
        self._current_goal = snapshot
        self._goal_history.append(snapshot)

        # Create measurement
        measurement = DriftMeasurement(
            measurement_id=str(uuid.uuid4()),
            timestamp=datetime.now(),
            drift_score=drift_score,
            original_goal=self._original_goal.goal_text,
            current_goal=goal,
            similarity=similarity,
            confidence=confidence,
            source=goal_source,
        )
        self._measurements.append(measurement)

        # Log warnings if drift is concerning
        if drift_score > self.drift_threshold:
            logger.warning(
                f"Goal drift detected: {drift_score:.3f} exceeds threshold "
                f"{self.drift_threshold:.3f}"
            )

        return measurement

    def get_current_drift(self) -> float:
        """Get the most recent drift score.

        Returns:
            Most recent drift score, or 0.0 if no measurements
        """
        if not self._measurements:
            return 0.0
        return self._measurements[-1].drift_score

    def get_current_similarity(self) -> float:
        """Get the most recent similarity score.

        Returns:
            Most recent similarity score, or 1.0 if no measurements
        """
        if not self._measurements:
            return 1.0
        return self._measurements[-1].similarity

    def get_drift_trend(self) -> str:
        """Analyze drift trend over recent measurements.

        Uses linear regression over the window to determine if drift
        is stable, increasing, or decreasing.

        Returns:
            Trend string: "stable", "increasing", or "decreasing"
        """
        if len(self._measurements) < 2:
            return "stable"

        # Get recent measurements
        recent = self._measurements[-self.window_size :]
        if len(recent) < 2:
            return "stable"

        # Compute trend using simple linear regression
        n = len(recent)
        x = np.arange(n)
        y = np.array([m.drift_score for m in recent])

        # Slope of least squares fit
        slope = (n * np.sum(x * y) - np.sum(x) * np.sum(y)) / (
            n * np.sum(x**2) - np.sum(x) ** 2
        )

        if slope > self.trend_sensitivity:
            return "increasing"
        elif slope < -self.trend_sensitivity:
            return "decreasing"
        else:
            return "stable"

    def get_drift_history(self) -> List[DriftMeasurement]:
        """Get full history of drift measurements.

        Returns:
            List of all drift measurements in chronological order
        """
        return list(self._measurements)

    def get_drift_statistics(self) -> DriftStatistics:
        """Get aggregated statistics for drift measurements.

        Returns:
            DriftStatistics with mean, max, min, std, trend, and counts
        """
        if not self._measurements:
            return DriftStatistics(
                mean_drift=0.0,
                max_drift=0.0,
                min_drift=0.0,
                std_drift=0.0,
                trend="stable",
                num_measurements=0,
                time_span_seconds=0.0,
            )

        scores = [m.drift_score for m in self._measurements]

        # Calculate time span
        first_time = self._measurements[0].timestamp
        last_time = self._measurements[-1].timestamp
        time_span = (last_time - first_time).total_seconds()

        return DriftStatistics(
            mean_drift=float(np.mean(scores)),
            max_drift=float(np.max(scores)),
            min_drift=float(np.min(scores)),
            std_drift=float(np.std(scores)) if len(scores) > 1 else 0.0,
            trend=self.get_drift_trend(),
            num_measurements=len(scores),
            time_span_seconds=time_span,
        )

    def should_reanchor(self) -> bool:
        """Check if goal re-anchoring is recommended.

        Re-anchoring is recommended when:
        - Current drift exceeds the reanchor threshold
        - Drift trend is consistently increasing

        Returns:
            True if re-anchoring is recommended
        """
        if not self._measurements:
            return False

        current_drift = self.get_current_drift()
        trend = self.get_drift_trend()

        # Recommend re-anchor if drift is high
        if current_drift > self.reanchor_threshold:
            return True

        # Recommend re-anchor if drift is increasing and above threshold
        if trend == "increasing" and current_drift > self.drift_threshold:
            return True

        return False

    def reanchor_goal(
        self, new_goal: str, metadata: Optional[dict] = None
    ) -> GoalSnapshot:
        """Re-anchor to a new goal, resetting drift tracking.

        Use this when the agent's objective has legitimately changed
        and you want to reset the baseline for drift measurement.

        Args:
            new_goal: New goal text to anchor to
            metadata: Optional metadata about the re-anchor

        Returns:
            GoalSnapshot of the new anchored goal
        """
        self._reanchor_count += 1

        logger.info(
            f"Re-anchoring goal (count: {self._reanchor_count}): "
            f"{new_goal[:100]}..."
        )

        # Store old measurements but start fresh
        old_measurements = self._measurements
        self._measurements = []

        # Set new goal
        snapshot = self.set_original_goal(new_goal, metadata)

        # Add reanchor info to metadata
        snapshot.metadata["reanchor_count"] = self._reanchor_count
        snapshot.metadata["previous_measurement_count"] = len(old_measurements)

        return snapshot

    def get_reanchor_count(self) -> int:
        """Get number of times goal has been re-anchored.

        Returns:
            Number of re-anchoring events
        """
        return self._reanchor_count

    def is_drifting(self) -> bool:
        """Check if current drift exceeds the threshold.

        Returns:
            True if current drift > drift_threshold
        """
        return self.get_current_drift() > self.drift_threshold

    def get_goal_history(self) -> List[GoalSnapshot]:
        """Get history of all goal snapshots.

        Returns:
            List of all goal snapshots in chronological order
        """
        return list(self._goal_history)

    def _get_embedding(self, text: str) -> np.ndarray:
        """Get embedding for text using the embed function.

        Args:
            text: Text to embed

        Returns:
            Embedding as numpy array
        """
        embedding = self.embed_fn(text)
        return np.asarray(embedding)

    def to_dict(self) -> dict:
        """Export tracker state to dictionary.

        Returns:
            Dictionary representation of tracker state
        """
        return {
            "drift_threshold": self.drift_threshold,
            "reanchor_threshold": self.reanchor_threshold,
            "window_size": self.window_size,
            "reanchor_count": self._reanchor_count,
            "original_goal": (
                self._original_goal.goal_text if self._original_goal else None
            ),
            "current_goal": (
                self._current_goal.goal_text if self._current_goal else None
            ),
            "current_drift": self.get_current_drift(),
            "drift_trend": self.get_drift_trend(),
            "num_measurements": len(self._measurements),
            "statistics": (
                {
                    "mean_drift": self.get_drift_statistics().mean_drift,
                    "max_drift": self.get_drift_statistics().max_drift,
                }
                if self._measurements
                else None
            ),
        }

    def reset(self) -> None:
        """Reset all tracking state.

        Clears all goals, measurements, and counters.
        """
        self._original_goal = None
        self._current_goal = None
        self._measurements = []
        self._goal_history = []
        self._reanchor_count = 0
        logger.info("Goal drift tracker reset")


class MultiGoalDriftTracker:
    """Track drift for multiple goals simultaneously.

    Useful when an agent has multiple objectives that should all
    be tracked independently.

    Example:
        >>> tracker = MultiGoalDriftTracker(embed_fn=get_embedding)
        >>> tracker.add_goal("primary", "Complete the main task")
        >>> tracker.add_goal("secondary", "Maintain data quality")
        >>>
        >>> tracker.record_inferred_goal("primary", "Working on main task")
        >>> tracker.record_inferred_goal("secondary", "Validating data")
        >>>
        >>> report = tracker.get_drift_report()
    """

    def __init__(
        self,
        embed_fn: Callable[[str], Union[np.ndarray, List[float]]],
        drift_threshold: float = 0.3,
    ):
        """Initialize multi-goal drift tracker.

        Args:
            embed_fn: Function to convert text to embedding
            drift_threshold: Drift threshold for all goals
        """
        self.embed_fn = embed_fn
        self.drift_threshold = drift_threshold
        self._trackers: dict[str, GoalDriftTracker] = {}

    def add_goal(
        self, goal_id: str, goal_text: str, weight: float = 1.0
    ) -> GoalDriftTracker:
        """Add a goal to track.

        Args:
            goal_id: Unique identifier for this goal
            goal_text: Text description of the goal
            weight: Importance weight for aggregate drift calculation

        Returns:
            GoalDriftTracker for this goal
        """
        tracker = GoalDriftTracker(
            embed_fn=self.embed_fn, drift_threshold=self.drift_threshold
        )
        tracker.set_original_goal(goal_text)
        self._trackers[goal_id] = tracker
        return tracker

    def record_inferred_goal(
        self,
        goal_id: str,
        goal_text: str,
        source: str = "action_trace",
        confidence: float = 1.0,
    ) -> DriftMeasurement:
        """Record inferred goal for a specific goal ID.

        Args:
            goal_id: ID of the goal to update
            goal_text: Inferred current goal text
            source: Source of inference
            confidence: Confidence in inference

        Returns:
            DriftMeasurement for this goal

        Raises:
            KeyError: If goal_id not found
        """
        if goal_id not in self._trackers:
            raise KeyError(f"Unknown goal ID: {goal_id}")
        return self._trackers[goal_id].record_inferred_goal(
            goal_text, source, confidence
        )

    def get_drift_report(self) -> dict:
        """Get drift report for all goals.

        Returns:
            Dictionary with drift info for each goal
        """
        return {
            goal_id: tracker.to_dict() for goal_id, tracker in self._trackers.items()
        }

    def get_aggregate_drift(self, weights: Optional[dict] = None) -> float:
        """Get weighted aggregate drift across all goals.

        Args:
            weights: Optional dict of goal_id -> weight. If None, equal weights.

        Returns:
            Weighted average drift score
        """
        if not self._trackers:
            return 0.0

        if weights is None:
            weights = {k: 1.0 for k in self._trackers}

        total_weight = sum(weights.get(k, 1.0) for k in self._trackers)
        if total_weight == 0:
            return 0.0

        weighted_drift = sum(
            tracker.get_current_drift() * weights.get(goal_id, 1.0)
            for goal_id, tracker in self._trackers.items()
        )

        return weighted_drift / total_weight
