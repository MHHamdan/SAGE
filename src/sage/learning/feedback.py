"""
Feedback Collection for Agent Learning

Collects and processes feedback signals from various sources to enable
continuous improvement of agent behavior.

Supports feedback from:
- Human evaluators
- Automated metrics
- Environment signals
- Peer agents

Example:
    >>> collector = FeedbackCollector()
    >>> collector.add_feedback(
    ...     Feedback(
    ...         feedback_type=FeedbackType.HUMAN,
    ...         source=FeedbackSource.EVALUATOR,
    ...         task_id="task-123",
    ...         rating=4.0,
    ...         comment="Good response but could be more concise"
    ...     )
    ... )
    >>>
    >>> aggregated = collector.aggregate()
    >>> print(f"Average rating: {aggregated.average_rating:.2f}")
"""

from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List, Callable
from datetime import datetime, timedelta
from enum import Enum
import logging
import json

logger = logging.getLogger(__name__)


class FeedbackType(Enum):
    """Types of feedback signals."""
    HUMAN = "human"  # Human evaluator feedback
    AUTOMATED = "automated"  # Automated metrics
    ENVIRONMENT = "environment"  # Environment rewards
    PEER = "peer"  # Peer agent feedback
    SYSTEM = "system"  # System-level feedback


class FeedbackSource(Enum):
    """Sources of feedback."""
    EVALUATOR = "evaluator"  # Human evaluator
    USER = "user"  # End user
    SUPERVISOR = "supervisor"  # Human supervisor
    METRIC = "metric"  # Automated metric
    REWARD = "reward"  # Environment reward signal
    POLICY = "policy"  # Policy compliance check
    PEER_AGENT = "peer_agent"  # Another agent


@dataclass
class Feedback:
    """Single feedback signal.

    Attributes:
        feedback_id: Unique identifier
        feedback_type: Type of feedback
        source: Source of feedback
        task_id: Associated task ID (if any)
        timestamp: When feedback was received
        rating: Numeric rating (0-5 scale)
        comment: Text comment
        metadata: Additional metadata
        dimensions: Multi-dimensional ratings
    """
    feedback_id: str = ""
    feedback_type: FeedbackType = FeedbackType.HUMAN
    source: FeedbackSource = FeedbackSource.EVALUATOR
    task_id: Optional[str] = None
    timestamp: datetime = field(default_factory=datetime.now)
    rating: Optional[float] = None
    comment: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    dimensions: Dict[str, float] = field(default_factory=dict)

    def __post_init__(self):
        if not self.feedback_id:
            self.feedback_id = f"fb-{id(self)}"

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "feedback_id": self.feedback_id,
            "feedback_type": self.feedback_type.value,
            "source": self.source.value,
            "task_id": self.task_id,
            "timestamp": self.timestamp.isoformat(),
            "rating": self.rating,
            "comment": self.comment,
            "metadata": self.metadata,
            "dimensions": self.dimensions,
        }


@dataclass
class AggregatedFeedback:
    """Aggregated feedback statistics.

    Attributes:
        period_start: Start of aggregation period
        period_end: End of aggregation period
        total_count: Total number of feedback items
        average_rating: Average rating
        rating_distribution: Distribution of ratings
        by_type: Breakdown by feedback type
        by_source: Breakdown by source
        dimension_averages: Average per dimension
        common_themes: Common themes from comments
    """
    period_start: datetime
    period_end: datetime
    total_count: int
    average_rating: float
    rating_distribution: Dict[int, int]
    by_type: Dict[str, int]
    by_source: Dict[str, int]
    dimension_averages: Dict[str, float]
    common_themes: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "period_start": self.period_start.isoformat(),
            "period_end": self.period_end.isoformat(),
            "total_count": self.total_count,
            "average_rating": self.average_rating,
            "rating_distribution": self.rating_distribution,
            "by_type": self.by_type,
            "by_source": self.by_source,
            "dimension_averages": self.dimension_averages,
            "common_themes": self.common_themes,
        }


class FeedbackCollector:
    """
    Collects and processes feedback from multiple sources.

    Provides aggregation, filtering, and analysis of feedback signals
    to support agent learning and improvement.

    Example:
        >>> collector = FeedbackCollector()
        >>>
        >>> # Add human feedback
        >>> collector.add_feedback(
        ...     Feedback(
        ...         feedback_type=FeedbackType.HUMAN,
        ...         rating=4.0,
        ...         comment="Good work"
        ...     )
        ... )
        >>>
        >>> # Add automated feedback
        >>> collector.add_automated_feedback(
        ...     task_id="task-123",
        ...     success=True,
        ...     latency_ms=150
        ... )
        >>>
        >>> # Aggregate
        >>> summary = collector.aggregate()

    Attributes:
        feedback_items: List of collected feedback
    """

    def __init__(
        self,
        on_feedback: Optional[Callable[[Feedback], None]] = None,
    ):
        """Initialize feedback collector.

        Args:
            on_feedback: Callback when feedback is added
        """
        self._feedback_items: List[Feedback] = []
        self._on_feedback = on_feedback
        self._feedback_counter = 0

    def add_feedback(self, feedback: Feedback) -> Feedback:
        """Add a feedback item.

        Args:
            feedback: Feedback to add

        Returns:
            Added feedback with ID assigned
        """
        self._feedback_counter += 1
        if not feedback.feedback_id or feedback.feedback_id.startswith("fb-"):
            feedback.feedback_id = f"feedback-{self._feedback_counter:06d}"

        self._feedback_items.append(feedback)

        logger.debug(
            f"Feedback added: {feedback.feedback_type.value} "
            f"rating={feedback.rating}"
        )

        if self._on_feedback:
            self._on_feedback(feedback)

        return feedback

    def add_human_feedback(
        self,
        rating: float,
        comment: Optional[str] = None,
        task_id: Optional[str] = None,
        evaluator: str = "anonymous",
        dimensions: Optional[Dict[str, float]] = None,
    ) -> Feedback:
        """Add human evaluator feedback.

        Args:
            rating: Rating (0-5)
            comment: Optional comment
            task_id: Associated task ID
            evaluator: Evaluator identifier
            dimensions: Multi-dimensional ratings

        Returns:
            Created feedback
        """
        feedback = Feedback(
            feedback_type=FeedbackType.HUMAN,
            source=FeedbackSource.EVALUATOR,
            task_id=task_id,
            rating=rating,
            comment=comment,
            metadata={"evaluator": evaluator},
            dimensions=dimensions or {},
        )
        return self.add_feedback(feedback)

    def add_automated_feedback(
        self,
        task_id: str,
        success: bool,
        latency_ms: Optional[float] = None,
        cost: Optional[float] = None,
        metrics: Optional[Dict[str, float]] = None,
    ) -> Feedback:
        """Add automated metric feedback.

        Args:
            task_id: Task identifier
            success: Whether task succeeded
            latency_ms: Task latency
            cost: Task cost
            metrics: Additional metrics

        Returns:
            Created feedback
        """
        dimensions = metrics or {}
        if latency_ms is not None:
            dimensions["latency_ms"] = latency_ms
        if cost is not None:
            dimensions["cost"] = cost

        feedback = Feedback(
            feedback_type=FeedbackType.AUTOMATED,
            source=FeedbackSource.METRIC,
            task_id=task_id,
            rating=5.0 if success else 0.0,
            metadata={"success": success},
            dimensions=dimensions,
        )
        return self.add_feedback(feedback)

    def add_environment_feedback(
        self,
        reward: float,
        task_id: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> Feedback:
        """Add environment reward feedback.

        Args:
            reward: Environment reward signal
            task_id: Associated task ID
            context: Environment context

        Returns:
            Created feedback
        """
        feedback = Feedback(
            feedback_type=FeedbackType.ENVIRONMENT,
            source=FeedbackSource.REWARD,
            task_id=task_id,
            rating=reward,
            metadata=context or {},
        )
        return self.add_feedback(feedback)

    def get_feedback(
        self,
        feedback_type: Optional[FeedbackType] = None,
        source: Optional[FeedbackSource] = None,
        task_id: Optional[str] = None,
        since: Optional[datetime] = None,
    ) -> List[Feedback]:
        """Get filtered feedback items.

        Args:
            feedback_type: Filter by type
            source: Filter by source
            task_id: Filter by task ID
            since: Filter by time

        Returns:
            Filtered list of feedback
        """
        items = self._feedback_items

        if feedback_type:
            items = [f for f in items if f.feedback_type == feedback_type]

        if source:
            items = [f for f in items if f.source == source]

        if task_id:
            items = [f for f in items if f.task_id == task_id]

        if since:
            items = [f for f in items if f.timestamp >= since]

        return items

    def aggregate(
        self,
        since: Optional[datetime] = None,
        until: Optional[datetime] = None,
    ) -> AggregatedFeedback:
        """Aggregate feedback into summary statistics.

        Args:
            since: Start of aggregation period
            until: End of aggregation period

        Returns:
            Aggregated feedback statistics
        """
        items = self._feedback_items

        # Filter by time
        if since:
            items = [f for f in items if f.timestamp >= since]
        if until:
            items = [f for f in items if f.timestamp <= until]

        # Calculate aggregations
        total_count = len(items)
        ratings = [f.rating for f in items if f.rating is not None]
        average_rating = sum(ratings) / len(ratings) if ratings else 0.0

        # Rating distribution
        rating_distribution = {i: 0 for i in range(6)}
        for r in ratings:
            bucket = min(5, max(0, int(r)))
            rating_distribution[bucket] += 1

        # By type
        by_type = {}
        for f in items:
            t = f.feedback_type.value
            by_type[t] = by_type.get(t, 0) + 1

        # By source
        by_source = {}
        for f in items:
            s = f.source.value
            by_source[s] = by_source.get(s, 0) + 1

        # Dimension averages
        dimension_totals: Dict[str, List[float]] = {}
        for f in items:
            for dim, val in f.dimensions.items():
                if dim not in dimension_totals:
                    dimension_totals[dim] = []
                dimension_totals[dim].append(val)

        dimension_averages = {
            dim: sum(vals) / len(vals)
            for dim, vals in dimension_totals.items()
            if vals
        }

        # Period bounds
        period_start = min(f.timestamp for f in items) if items else datetime.now()
        period_end = max(f.timestamp for f in items) if items else datetime.now()

        return AggregatedFeedback(
            period_start=period_start,
            period_end=period_end,
            total_count=total_count,
            average_rating=average_rating,
            rating_distribution=rating_distribution,
            by_type=by_type,
            by_source=by_source,
            dimension_averages=dimension_averages,
        )

    def get_recent_trend(
        self,
        window_hours: float = 24,
        buckets: int = 6,
    ) -> List[Dict[str, Any]]:
        """Get recent feedback trend.

        Args:
            window_hours: Hours to analyze
            buckets: Number of time buckets

        Returns:
            List of bucket statistics
        """
        now = datetime.now()
        bucket_duration = timedelta(hours=window_hours / buckets)
        trends = []

        for i in range(buckets):
            start = now - bucket_duration * (buckets - i)
            end = now - bucket_duration * (buckets - i - 1)

            items = [
                f for f in self._feedback_items
                if start <= f.timestamp < end
            ]

            ratings = [f.rating for f in items if f.rating is not None]

            trends.append({
                "bucket": i,
                "start": start.isoformat(),
                "end": end.isoformat(),
                "count": len(items),
                "average_rating": sum(ratings) / len(ratings) if ratings else None,
            })

        return trends

    def export(self, format: str = "json") -> str:
        """Export feedback as string.

        Args:
            format: Output format ("json", "csv")

        Returns:
            Formatted feedback string
        """
        if format == "json":
            return json.dumps(
                [f.to_dict() for f in self._feedback_items],
                indent=2,
                default=str
            )

        elif format == "csv":
            lines = ["feedback_id,type,source,task_id,rating,timestamp"]
            for f in self._feedback_items:
                lines.append(
                    f"{f.feedback_id},{f.feedback_type.value},"
                    f"{f.source.value},{f.task_id or ''},"
                    f"{f.rating or ''},{f.timestamp.isoformat()}"
                )
            return "\n".join(lines)

        else:
            raise ValueError(f"Unsupported format: {format}")

    def clear(self) -> None:
        """Clear all feedback."""
        self._feedback_items.clear()
        self._feedback_counter = 0

    def __len__(self) -> int:
        """Get number of feedback items."""
        return len(self._feedback_items)
