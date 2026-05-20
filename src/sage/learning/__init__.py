"""
Learning Module for Agentic AI Systems

Provides continuous learning and improvement mechanisms for autonomous agents,
including feedback collection, experience replay, and deployment optimization.

This module implements the learning patterns described in Section VII
of the IEEE TAI paper on Agentic AI.

Key Components:
    - DeploymentLoop: Continuous deployment and learning cycle
    - FeedbackCollector: Collects and processes feedback signals
    - ExperienceBuffer: Stores experiences for learning
    - PerformanceTracker: Tracks performance over deployments

Example:
    >>> from sage.learning import DeploymentLoop, FeedbackCollector
    >>>
    >>> loop = DeploymentLoop(
    ...     agent=my_agent,
    ...     feedback_collector=FeedbackCollector(),
    ...     evaluation_interval=100
    ... )
    >>>
    >>> await loop.run(tasks=task_stream)
"""

from .deployment_loop import (
    DeploymentLoop,
    DeploymentConfig,
    DeploymentState,
    DeploymentStatus,
    DeploymentMetrics,
)

from .feedback import (
    FeedbackCollector,
    Feedback,
    FeedbackType,
    FeedbackSource,
    AggregatedFeedback,
)

from .experience import (
    ExperienceBuffer,
    Experience,
    ExperienceBatch,
    PrioritizedExperienceBuffer,
)

__all__ = [
    # Deployment Loop
    "DeploymentLoop",
    "DeploymentConfig",
    "DeploymentState",
    "DeploymentStatus",
    "DeploymentMetrics",
    # Feedback
    "FeedbackCollector",
    "Feedback",
    "FeedbackType",
    "FeedbackSource",
    "AggregatedFeedback",
    # Experience
    "ExperienceBuffer",
    "Experience",
    "ExperienceBatch",
    "PrioritizedExperienceBuffer",
]
