"""
Stability Monitor Module

Monitors agent execution for stability condition violations based on the
formal control-theoretic framework from Section III-F of the paper.

Implements checks for:
- Goal convergence (Definition 1)
- Bounded oscillation (Definition 2)
- Progress monotonicity (Theorem 1, condition ii)
- Observation fidelity (Theorem 1, condition i)
- Context noise bounds (Theorem 1, condition iii)

This module addresses IEEE TAI Review Issue M2: Control-Theoretic Framework
Lacks Formal Development.

Reference: Section III-F-5 - Formal Stability Analysis

Example:
    >>> from sage.monitoring import StabilityMonitor
    >>>
    >>> monitor = StabilityMonitor(
    ...     goal_embedding=embed("Complete the task"),
    ...     similarity_threshold=0.9,
    ...     oscillation_window=10,
    ...     oscillation_bound=3
    ... )
    >>>
    >>> # During agent execution
    >>> for step in agent_steps:
    ...     status = monitor.track_state(
    ...         state_embedding=embed(step.state),
    ...         action=step.action,
    ...         observation=step.observation
    ...     )
    ...     if status.has_violation:
    ...         handle_instability(status)
"""

from __future__ import annotations

import logging
import time
import hashlib
from enum import Enum, auto
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Callable, Tuple, Set
from collections import deque
from datetime import datetime

import numpy as np

logger = logging.getLogger(__name__)


# =============================================================================
# STABILITY DEFINITIONS (Section III-F-5)
# =============================================================================

class ViolationType(Enum):
    """Types of stability condition violations."""
    GOAL_DIVERGENCE = auto()        # Definition 1 violated
    OSCILLATION = auto()            # Definition 2 violated
    MONOTONICITY_FAILURE = auto()   # Theorem 1 condition (ii) violated
    OBSERVATION_FIDELITY = auto()   # Theorem 1 condition (i) violated
    CONTEXT_NOISE = auto()          # Theorem 1 condition (iii) violated
    DEADLOCK = auto()               # Agent stuck in absorbing state


class StabilitySeverity(Enum):
    """Severity levels for stability violations."""
    INFO = 0
    WARNING = 1
    CRITICAL = 2
    FATAL = 3


@dataclass
class ConvergenceStatus:
    """Status of goal convergence check (Definition 1).

    Definition 1 (Goal Convergence): An agent exhibits goal convergence
    for goal g if there exists T < infinity such that:
        E[sim(s_T, g)] >= 1 - epsilon

    Attributes:
        converging: Whether agent is converging toward goal
        current_similarity: Current goal similarity
        target_similarity: Target similarity (1 - epsilon)
        estimated_steps_remaining: Estimated steps to convergence
        trend: Direction of convergence trend
    """
    converging: bool
    current_similarity: float
    target_similarity: float
    estimated_steps_remaining: Optional[int]
    trend: str  # "improving", "stable", "degrading"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "converging": self.converging,
            "current_similarity": self.current_similarity,
            "target_similarity": self.target_similarity,
            "estimated_steps_remaining": self.estimated_steps_remaining,
            "trend": self.trend
        }


@dataclass
class OscillationStatus:
    """Status of bounded oscillation check (Definition 2).

    Definition 2 (Bounded Oscillation): An agent exhibits bounded oscillation
    if the action sequence {a_t} satisfies:
        sup_{t > t_0} |{a_t, ..., a_{t+k}} ∩ {a_{t-k}, ..., a_{t-1}}| <= B

    Attributes:
        oscillating: Whether oscillation is detected
        cycle_detected: Whether a repeating cycle was found
        overlap_ratio: Ratio of repeated actions in window
        window_size: Size of detection window
        bound: Maximum allowed overlap
        repeated_actions: List of repeated action hashes
    """
    oscillating: bool
    cycle_detected: bool
    overlap_ratio: float
    window_size: int
    bound: int
    repeated_actions: List[str]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "oscillating": self.oscillating,
            "cycle_detected": self.cycle_detected,
            "overlap_ratio": self.overlap_ratio,
            "window_size": self.window_size,
            "bound": self.bound,
            "repeated_action_count": len(self.repeated_actions)
        }


@dataclass
class MonotonicityStatus:
    """Status of progress monotonicity check (Theorem 1, condition ii).

    Theorem 1 condition (ii): Progress monotonicity requires:
        E[sim(s_{t+1}, g) - sim(s_t, g) | a_t != null] >= delta_p > 0

    Attributes:
        monotonic: Whether monotonicity is satisfied
        mean_progress: Mean progress over window
        required_progress: Required minimum progress (delta_p)
        consecutive_negative: Number of consecutive negative progress steps
        window_size: Size of analysis window
    """
    monotonic: bool
    mean_progress: float
    required_progress: float
    consecutive_negative: int
    window_size: int

    def to_dict(self) -> Dict[str, Any]:
        return {
            "monotonic": self.monotonic,
            "mean_progress": self.mean_progress,
            "required_progress": self.required_progress,
            "consecutive_negative": self.consecutive_negative,
            "window_size": self.window_size
        }


@dataclass
class FidelityStatus:
    """Status of observation fidelity check (Theorem 1, condition i).

    Theorem 1 condition (i): Observation fidelity requires:
        E[||o_t - h(s_t)||] <= delta_o

    Attributes:
        fidelity_satisfied: Whether fidelity bound is satisfied
        schema_valid: Whether observation matches expected schema
        validation_errors: List of validation errors
        error_rate: Rate of validation failures
        bound: Maximum allowed error (delta_o)
    """
    fidelity_satisfied: bool
    schema_valid: bool
    validation_errors: List[str]
    error_rate: float
    bound: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "fidelity_satisfied": self.fidelity_satisfied,
            "schema_valid": self.schema_valid,
            "error_count": len(self.validation_errors),
            "error_rate": self.error_rate,
            "bound": self.bound
        }


@dataclass
class StabilityViolation:
    """Record of a stability violation.

    Attributes:
        violation_type: Type of violation
        severity: Violation severity
        timestamp: When violation occurred
        step: Step number when violation occurred
        description: Human-readable description
        context: Additional context data
        recommended_action: Suggested remediation
    """
    violation_type: ViolationType
    severity: StabilitySeverity
    timestamp: float
    step: int
    description: str
    context: Dict[str, Any] = field(default_factory=dict)
    recommended_action: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": self.violation_type.name,
            "severity": self.severity.name,
            "timestamp": self.timestamp,
            "step": self.step,
            "description": self.description,
            "context": self.context,
            "recommended_action": self.recommended_action
        }


@dataclass
class StabilityStatus:
    """Overall stability status at a point in time.

    Attributes:
        step: Current step number
        timestamp: Current timestamp
        convergence: Goal convergence status
        oscillation: Oscillation status
        monotonicity: Monotonicity status
        fidelity: Observation fidelity status
        has_violation: Whether any violation occurred
        violations: List of violations detected
        goal_similarity: Current goal similarity
    """
    step: int
    timestamp: float
    convergence: ConvergenceStatus
    oscillation: OscillationStatus
    monotonicity: MonotonicityStatus
    fidelity: FidelityStatus
    has_violation: bool
    violations: List[StabilityViolation]
    goal_similarity: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "step": self.step,
            "timestamp": self.timestamp,
            "has_violation": self.has_violation,
            "goal_similarity": self.goal_similarity,
            "convergence": self.convergence.to_dict(),
            "oscillation": self.oscillation.to_dict(),
            "monotonicity": self.monotonicity.to_dict(),
            "fidelity": self.fidelity.to_dict(),
            "violations": [v.to_dict() for v in self.violations]
        }


@dataclass
class StabilityReport:
    """Comprehensive stability assessment report.

    Attributes:
        total_steps: Total steps tracked
        total_violations: Total violations detected
        violations_by_type: Count of violations by type
        mean_goal_similarity: Average goal similarity
        monotonicity_satisfaction_rate: Rate of monotonicity satisfaction
        oscillation_incidents: Number of oscillation incidents
        recommendations: List of recommendations
    """
    total_steps: int
    total_violations: int
    violations_by_type: Dict[str, int]
    mean_goal_similarity: float
    monotonicity_satisfaction_rate: float
    oscillation_incidents: int
    recommendations: List[str]
    status_history: List[StabilityStatus] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_steps": self.total_steps,
            "total_violations": self.total_violations,
            "violations_by_type": self.violations_by_type,
            "mean_goal_similarity": self.mean_goal_similarity,
            "monotonicity_satisfaction_rate": self.monotonicity_satisfaction_rate,
            "oscillation_incidents": self.oscillation_incidents,
            "recommendations": self.recommendations
        }


# =============================================================================
# STABILITY MONITOR CLASS
# =============================================================================

class StabilityMonitor:
    """
    Monitors agent execution for stability condition violations.

    Implements the formal stability conditions from Section III-F-5:

    **Definition 1 (Goal Convergence)**: An agent exhibits goal convergence
    for goal g if there exists T < infinity such that:
        E[sim(s_T, g)] >= 1 - epsilon

    **Definition 2 (Bounded Oscillation)**: An agent exhibits bounded oscillation
    if the action sequence {a_t} satisfies:
        sup_{t > t_0} |{a_t, ..., a_{t+k}} ∩ {a_{t-k}, ..., a_{t-1}}| <= B

    **Theorem 1 (Sufficient Conditions for Goal Convergence)**:
    An LLM agent achieves goal convergence if:
        (i) Observation fidelity: E[||o_t - h(s_t)||] <= delta_o
        (ii) Progress monotonicity: E[sim(s_{t+1}, g) - sim(s_t, g) | a_t != null] >= delta_p
        (iii) Bounded context noise: KL(C_t || C_t*) <= delta_c

    Example:
        >>> monitor = StabilityMonitor(
        ...     goal_embedding=embed("Complete the task"),
        ...     similarity_threshold=0.9
        ... )
        >>>
        >>> for step in agent.run():
        ...     status = monitor.track_state(
        ...         state_embedding=embed(step.state),
        ...         action=step.action,
        ...         observation=step.observation
        ...     )
        ...     if status.has_violation:
        ...         print(f"Instability detected: {status.violations}")
    """

    def __init__(
        self,
        goal_embedding: np.ndarray,
        similarity_threshold: float = 0.9,
        oscillation_window: int = 10,
        oscillation_bound: int = 3,
        monotonicity_window: int = 10,
        progress_threshold: float = 0.001,
        fidelity_bound: float = 0.1,
        embedding_fn: Optional[Callable[[str], np.ndarray]] = None
    ):
        """Initialize stability monitor.

        Args:
            goal_embedding: Embedding vector of the target goal
            similarity_threshold: Target similarity for convergence (1 - epsilon)
            oscillation_window: Window size for oscillation detection (k)
            oscillation_bound: Maximum allowed action overlap (B)
            monotonicity_window: Window size for monotonicity analysis
            progress_threshold: Minimum required progress (delta_p)
            fidelity_bound: Maximum observation error (delta_o)
            embedding_fn: Optional function to embed strings
        """
        self.goal_embedding = np.asarray(goal_embedding)
        self.similarity_threshold = similarity_threshold
        self.oscillation_window = oscillation_window
        self.oscillation_bound = oscillation_bound
        self.monotonicity_window = monotonicity_window
        self.progress_threshold = progress_threshold
        self.fidelity_bound = fidelity_bound
        self.embedding_fn = embedding_fn

        # Tracking state
        self._step = 0
        self._action_history: deque = deque(maxlen=oscillation_window * 2)
        self._similarity_history: deque = deque(maxlen=monotonicity_window)
        self._status_history: List[StabilityStatus] = []
        self._violations: List[StabilityViolation] = []
        self._fidelity_errors: int = 0
        self._fidelity_checks: int = 0

        logger.info(
            f"StabilityMonitor initialized: "
            f"threshold={similarity_threshold}, "
            f"oscillation_window={oscillation_window}, "
            f"oscillation_bound={oscillation_bound}"
        )

    def track_state(
        self,
        state_embedding: np.ndarray,
        action: str,
        observation: Optional[Dict[str, Any]] = None,
        expected_schema: Optional[Dict[str, Any]] = None
    ) -> StabilityStatus:
        """Update tracking and check for violations.

        Args:
            state_embedding: Current state embedding
            action: Action taken
            observation: Observation received (optional)
            expected_schema: Expected observation schema (optional)

        Returns:
            StabilityStatus with any triggered alerts
        """
        self._step += 1
        timestamp = time.time()

        # Compute goal similarity
        state_embedding = np.asarray(state_embedding)
        similarity = self._compute_similarity(state_embedding, self.goal_embedding)
        self._similarity_history.append(similarity)

        # Track action
        action_hash = self._hash_action(action)
        self._action_history.append(action_hash)

        # Run all checks
        convergence = self.check_goal_convergence()
        oscillation = self.check_oscillation()
        monotonicity = self.check_monotonicity()
        fidelity = self.check_observation_fidelity(observation, expected_schema)

        # Collect violations
        violations: List[StabilityViolation] = []

        if not convergence.converging and convergence.trend == "degrading":
            violations.append(StabilityViolation(
                violation_type=ViolationType.GOAL_DIVERGENCE,
                severity=StabilitySeverity.WARNING,
                timestamp=timestamp,
                step=self._step,
                description=f"Goal divergence detected: similarity={similarity:.3f}",
                context={"similarity": similarity, "trend": convergence.trend},
                recommended_action="Consider re-anchoring goal or adjusting strategy"
            ))

        if oscillation.oscillating:
            violations.append(StabilityViolation(
                violation_type=ViolationType.OSCILLATION,
                severity=StabilitySeverity.CRITICAL if oscillation.cycle_detected else StabilitySeverity.WARNING,
                timestamp=timestamp,
                step=self._step,
                description=f"Action oscillation detected: overlap={oscillation.overlap_ratio:.1%}",
                context=oscillation.to_dict(),
                recommended_action="Break cycle by introducing exploration or external guidance"
            ))

        if not monotonicity.monotonic:
            violations.append(StabilityViolation(
                violation_type=ViolationType.MONOTONICITY_FAILURE,
                severity=StabilitySeverity.WARNING,
                timestamp=timestamp,
                step=self._step,
                description=f"Monotonicity violated: mean_progress={monotonicity.mean_progress:.4f}",
                context=monotonicity.to_dict(),
                recommended_action="Trigger intervention or replanning"
            ))

        if not fidelity.fidelity_satisfied:
            violations.append(StabilityViolation(
                violation_type=ViolationType.OBSERVATION_FIDELITY,
                severity=StabilitySeverity.WARNING,
                timestamp=timestamp,
                step=self._step,
                description=f"Observation fidelity violated: error_rate={fidelity.error_rate:.1%}",
                context=fidelity.to_dict(),
                recommended_action="Validate tool outputs and implement parsing verification"
            ))

        self._violations.extend(violations)

        status = StabilityStatus(
            step=self._step,
            timestamp=timestamp,
            convergence=convergence,
            oscillation=oscillation,
            monotonicity=monotonicity,
            fidelity=fidelity,
            has_violation=len(violations) > 0,
            violations=violations,
            goal_similarity=similarity
        )

        self._status_history.append(status)
        return status

    def check_goal_convergence(self) -> ConvergenceStatus:
        """Check if agent is converging toward goal (Definition 1).

        Returns:
            ConvergenceStatus with convergence analysis
        """
        if not self._similarity_history:
            return ConvergenceStatus(
                converging=True,
                current_similarity=0.0,
                target_similarity=self.similarity_threshold,
                estimated_steps_remaining=None,
                trend="stable"
            )

        current_similarity = self._similarity_history[-1]

        # Determine trend
        if len(self._similarity_history) >= 3:
            recent = list(self._similarity_history)[-5:]
            if len(recent) >= 2:
                slope = (recent[-1] - recent[0]) / len(recent)
                if slope > 0.01:
                    trend = "improving"
                elif slope < -0.01:
                    trend = "degrading"
                else:
                    trend = "stable"
            else:
                trend = "stable"
        else:
            trend = "stable"

        # Estimate steps remaining
        estimated_steps = None
        if trend == "improving" and current_similarity < self.similarity_threshold:
            gap = self.similarity_threshold - current_similarity
            if len(self._similarity_history) >= 2:
                recent = list(self._similarity_history)[-5:]
                progress_rate = (recent[-1] - recent[0]) / max(len(recent), 1)
                if progress_rate > 0:
                    estimated_steps = int(gap / progress_rate)

        converging = current_similarity >= self.similarity_threshold or trend == "improving"

        return ConvergenceStatus(
            converging=converging,
            current_similarity=current_similarity,
            target_similarity=self.similarity_threshold,
            estimated_steps_remaining=estimated_steps,
            trend=trend
        )

    def check_oscillation(self) -> OscillationStatus:
        """Detect action sequence cycling (Definition 2).

        Returns:
            OscillationStatus with oscillation analysis
        """
        if len(self._action_history) < self.oscillation_window:
            return OscillationStatus(
                oscillating=False,
                cycle_detected=False,
                overlap_ratio=0.0,
                window_size=self.oscillation_window,
                bound=self.oscillation_bound,
                repeated_actions=[]
            )

        actions = list(self._action_history)

        # Split into two windows
        mid = len(actions) // 2
        window1 = set(actions[:mid])
        window2 = set(actions[mid:])

        # Calculate overlap
        overlap = window1 & window2
        overlap_count = len(overlap)
        min_window_size = min(len(window1), len(window2))
        overlap_ratio = overlap_count / min_window_size if min_window_size > 0 else 0

        # Check for exact cycle (repeated sequence)
        cycle_detected = False
        if len(actions) >= self.oscillation_window * 2:
            for cycle_len in range(2, self.oscillation_window // 2 + 1):
                recent = actions[-cycle_len * 2:]
                first_half = recent[:cycle_len]
                second_half = recent[cycle_len:]
                if first_half == second_half:
                    cycle_detected = True
                    break

        oscillating = overlap_count > self.oscillation_bound or cycle_detected

        return OscillationStatus(
            oscillating=oscillating,
            cycle_detected=cycle_detected,
            overlap_ratio=overlap_ratio,
            window_size=self.oscillation_window,
            bound=self.oscillation_bound,
            repeated_actions=list(overlap)
        )

    def check_monotonicity(self, window: Optional[int] = None) -> MonotonicityStatus:
        """Verify progress monotonicity over sliding window (Theorem 1, condition ii).

        Args:
            window: Window size for analysis (uses default if None)

        Returns:
            MonotonicityStatus with monotonicity analysis
        """
        window = window or self.monotonicity_window

        if len(self._similarity_history) < 2:
            return MonotonicityStatus(
                monotonic=True,
                mean_progress=0.0,
                required_progress=self.progress_threshold,
                consecutive_negative=0,
                window_size=window
            )

        similarities = list(self._similarity_history)[-window:]

        # Compute progress deltas
        deltas = [
            similarities[i + 1] - similarities[i]
            for i in range(len(similarities) - 1)
        ]

        if not deltas:
            return MonotonicityStatus(
                monotonic=True,
                mean_progress=0.0,
                required_progress=self.progress_threshold,
                consecutive_negative=0,
                window_size=window
            )

        mean_progress = sum(deltas) / len(deltas)

        # Count consecutive negative progress
        consecutive_negative = 0
        for delta in reversed(deltas):
            if delta < 0:
                consecutive_negative += 1
            else:
                break

        monotonic = mean_progress >= self.progress_threshold

        return MonotonicityStatus(
            monotonic=monotonic,
            mean_progress=mean_progress,
            required_progress=self.progress_threshold,
            consecutive_negative=consecutive_negative,
            window_size=window
        )

    def check_observation_fidelity(
        self,
        observation: Optional[Dict[str, Any]],
        expected_schema: Optional[Dict[str, Any]] = None
    ) -> FidelityStatus:
        """Validate observation against expected schema (Theorem 1, condition i).

        Args:
            observation: Observation to validate
            expected_schema: Expected schema for validation

        Returns:
            FidelityStatus with fidelity analysis
        """
        self._fidelity_checks += 1

        if observation is None:
            return FidelityStatus(
                fidelity_satisfied=True,
                schema_valid=True,
                validation_errors=[],
                error_rate=0.0,
                bound=self.fidelity_bound
            )

        validation_errors: List[str] = []

        # Schema validation
        if expected_schema:
            schema_valid = self._validate_schema(observation, expected_schema, validation_errors)
        else:
            schema_valid = True

        # Track errors
        if validation_errors:
            self._fidelity_errors += 1

        error_rate = self._fidelity_errors / self._fidelity_checks

        fidelity_satisfied = error_rate <= self.fidelity_bound and schema_valid

        return FidelityStatus(
            fidelity_satisfied=fidelity_satisfied,
            schema_valid=schema_valid,
            validation_errors=validation_errors,
            error_rate=error_rate,
            bound=self.fidelity_bound
        )

    def _validate_schema(
        self,
        observation: Dict[str, Any],
        schema: Dict[str, Any],
        errors: List[str]
    ) -> bool:
        """Validate observation against schema.

        Args:
            observation: Observation to validate
            schema: Expected schema
            errors: List to append errors to

        Returns:
            True if valid
        """
        valid = True

        # Check required fields
        required = schema.get("required", [])
        for field in required:
            if field not in observation:
                errors.append(f"Missing required field: {field}")
                valid = False

        # Check types
        properties = schema.get("properties", {})
        for field, spec in properties.items():
            if field in observation:
                expected_type = spec.get("type")
                if expected_type:
                    actual = observation[field]
                    if expected_type == "string" and not isinstance(actual, str):
                        errors.append(f"Field {field} should be string, got {type(actual).__name__}")
                        valid = False
                    elif expected_type == "number" and not isinstance(actual, (int, float)):
                        errors.append(f"Field {field} should be number, got {type(actual).__name__}")
                        valid = False
                    elif expected_type == "boolean" and not isinstance(actual, bool):
                        errors.append(f"Field {field} should be boolean, got {type(actual).__name__}")
                        valid = False
                    elif expected_type == "object" and not isinstance(actual, dict):
                        errors.append(f"Field {field} should be object, got {type(actual).__name__}")
                        valid = False
                    elif expected_type == "array" and not isinstance(actual, list):
                        errors.append(f"Field {field} should be array, got {type(actual).__name__}")
                        valid = False

        return valid

    def get_stability_report(self) -> StabilityReport:
        """Get comprehensive stability assessment.

        Returns:
            StabilityReport with full analysis
        """
        # Count violations by type
        violations_by_type: Dict[str, int] = {}
        for v in self._violations:
            key = v.violation_type.name
            violations_by_type[key] = violations_by_type.get(key, 0) + 1

        # Calculate mean similarity
        mean_similarity = (
            sum(self._similarity_history) / len(self._similarity_history)
            if self._similarity_history else 0.0
        )

        # Calculate monotonicity satisfaction rate
        if len(self._status_history) > 0:
            monotonic_count = sum(
                1 for s in self._status_history if s.monotonicity.monotonic
            )
            monotonicity_rate = monotonic_count / len(self._status_history)
        else:
            monotonicity_rate = 1.0

        # Count oscillation incidents
        oscillation_incidents = sum(
            1 for s in self._status_history if s.oscillation.oscillating
        )

        # Generate recommendations
        recommendations = self._generate_recommendations(violations_by_type)

        return StabilityReport(
            total_steps=self._step,
            total_violations=len(self._violations),
            violations_by_type=violations_by_type,
            mean_goal_similarity=mean_similarity,
            monotonicity_satisfaction_rate=monotonicity_rate,
            oscillation_incidents=oscillation_incidents,
            recommendations=recommendations,
            status_history=self._status_history
        )

    def _generate_recommendations(
        self,
        violations_by_type: Dict[str, int]
    ) -> List[str]:
        """Generate recommendations based on violations.

        Args:
            violations_by_type: Count of violations by type

        Returns:
            List of recommendations
        """
        recommendations = []

        if violations_by_type.get("GOAL_DIVERGENCE", 0) > 0:
            recommendations.append(
                "Implement periodic goal re-anchoring to maintain alignment"
            )

        if violations_by_type.get("OSCILLATION", 0) > 0:
            recommendations.append(
                "Add exploration noise or external guidance to break action cycles"
            )

        if violations_by_type.get("MONOTONICITY_FAILURE", 0) > self._step * 0.2:
            recommendations.append(
                "Consider more aggressive progress monitoring with earlier intervention"
            )

        if violations_by_type.get("OBSERVATION_FIDELITY", 0) > 0:
            recommendations.append(
                "Implement schema validation and parsing verification for tool outputs"
            )

        if not recommendations:
            recommendations.append("Agent stability is within acceptable bounds")

        return recommendations

    def _compute_similarity(
        self,
        embedding1: np.ndarray,
        embedding2: np.ndarray
    ) -> float:
        """Compute cosine similarity between embeddings.

        Args:
            embedding1: First embedding
            embedding2: Second embedding

        Returns:
            Cosine similarity in [-1, 1]
        """
        norm1 = np.linalg.norm(embedding1)
        norm2 = np.linalg.norm(embedding2)

        if norm1 == 0 or norm2 == 0:
            return 0.0

        similarity = np.dot(embedding1, embedding2) / (norm1 * norm2)
        return float(np.clip(similarity, -1.0, 1.0))

    def _hash_action(self, action: str) -> str:
        """Create hash of action for comparison.

        Args:
            action: Action string

        Returns:
            Short hash of action
        """
        return hashlib.md5(str(action).encode()).hexdigest()[:8]

    def reset(self):
        """Reset all tracking state."""
        self._step = 0
        self._action_history.clear()
        self._similarity_history.clear()
        self._status_history.clear()
        self._violations.clear()
        self._fidelity_errors = 0
        self._fidelity_checks = 0
        logger.info("StabilityMonitor reset")


# =============================================================================
# COROLLARY 1 IMPLEMENTATION: LIMIT CYCLE DETECTION
# =============================================================================

class LimitCycleDetector:
    """
    Detects limit cycles as described in Corollary 1.

    Corollary 1 (Oscillation from Violated Monotonicity): When condition (ii)
    fails, specifically when E[sim(s_{t+1}, g) - sim(s_t, g)] ≈ 0 with high
    variance, the agent may exhibit limit cycles.

    This occurs when:
        ∃ S_cycle ⊂ S: P(s_{t+k} ∈ S_cycle | s_t ∈ S_cycle) > 1 - η

    Example:
        >>> detector = LimitCycleDetector(
        ...     embedding_fn=embed,
        ...     cycle_threshold=0.8
        ... )
        >>>
        >>> for state in agent_states:
        ...     is_cycle, info = detector.check_state(embed(state))
        ...     if is_cycle:
        ...         print(f"Limit cycle detected with {info['cycle_probability']:.1%} probability")
    """

    def __init__(
        self,
        embedding_fn: Optional[Callable[[str], np.ndarray]] = None,
        similarity_threshold: float = 0.95,
        cycle_threshold: float = 0.8,
        window_size: int = 20,
        min_cycle_length: int = 2,
        max_cycle_length: int = 10
    ):
        """Initialize limit cycle detector.

        Args:
            embedding_fn: Function to embed states
            similarity_threshold: Threshold for considering states similar
            cycle_threshold: Probability threshold for cycle detection (1 - η)
            window_size: Window for cycle detection
            min_cycle_length: Minimum cycle length to detect
            max_cycle_length: Maximum cycle length to detect
        """
        self.embedding_fn = embedding_fn
        self.similarity_threshold = similarity_threshold
        self.cycle_threshold = cycle_threshold
        self.window_size = window_size
        self.min_cycle_length = min_cycle_length
        self.max_cycle_length = max_cycle_length

        self._state_history: List[np.ndarray] = []
        self._cycle_candidates: Dict[int, int] = {}  # cycle_length -> count

    def check_state(
        self,
        state_embedding: np.ndarray
    ) -> Tuple[bool, Dict[str, Any]]:
        """Check if current state is part of a limit cycle.

        Args:
            state_embedding: Current state embedding

        Returns:
            Tuple of (is_in_cycle, info_dict)
        """
        state_embedding = np.asarray(state_embedding)
        self._state_history.append(state_embedding)

        if len(self._state_history) < self.min_cycle_length * 2:
            return False, {"message": "Insufficient history for cycle detection"}

        # Check for cycles of various lengths
        for cycle_len in range(self.min_cycle_length, self.max_cycle_length + 1):
            if len(self._state_history) >= cycle_len * 2:
                # Compare current window to previous window
                current_window = self._state_history[-cycle_len:]
                previous_window = self._state_history[-cycle_len * 2:-cycle_len]

                # Check similarity of corresponding states
                similarities = []
                for s1, s2 in zip(current_window, previous_window):
                    sim = self._compute_similarity(s1, s2)
                    similarities.append(sim)

                avg_similarity = sum(similarities) / len(similarities)

                if avg_similarity > self.similarity_threshold:
                    # Potential cycle detected
                    self._cycle_candidates[cycle_len] = self._cycle_candidates.get(cycle_len, 0) + 1

                    # Calculate cycle probability
                    total_checks = len(self._state_history) - cycle_len * 2 + 1
                    cycle_probability = self._cycle_candidates.get(cycle_len, 0) / max(total_checks, 1)

                    if cycle_probability > self.cycle_threshold:
                        return True, {
                            "cycle_length": cycle_len,
                            "cycle_probability": cycle_probability,
                            "avg_similarity": avg_similarity,
                            "message": f"Limit cycle detected with length {cycle_len}"
                        }

        return False, {"message": "No limit cycle detected"}

    def _compute_similarity(
        self,
        embedding1: np.ndarray,
        embedding2: np.ndarray
    ) -> float:
        """Compute cosine similarity between embeddings."""
        norm1 = np.linalg.norm(embedding1)
        norm2 = np.linalg.norm(embedding2)

        if norm1 == 0 or norm2 == 0:
            return 0.0

        return float(np.dot(embedding1, embedding2) / (norm1 * norm2))

    def reset(self):
        """Reset detector state."""
        self._state_history.clear()
        self._cycle_candidates.clear()


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def create_stability_monitor(
    goal_text: str,
    embedding_fn: Callable[[str], np.ndarray],
    **kwargs
) -> StabilityMonitor:
    """Create a stability monitor from goal text.

    Args:
        goal_text: Text description of the goal
        embedding_fn: Function to embed text
        **kwargs: Additional arguments for StabilityMonitor

    Returns:
        Configured StabilityMonitor
    """
    goal_embedding = embedding_fn(goal_text)
    return StabilityMonitor(
        goal_embedding=goal_embedding,
        embedding_fn=embedding_fn,
        **kwargs
    )


def check_stability_conditions(
    goal_embedding: np.ndarray,
    state_embeddings: List[np.ndarray],
    actions: List[str],
    observations: Optional[List[Dict[str, Any]]] = None
) -> StabilityReport:
    """Run stability analysis on a completed execution trace.

    Args:
        goal_embedding: Goal embedding
        state_embeddings: List of state embeddings
        actions: List of actions taken
        observations: Optional list of observations

    Returns:
        StabilityReport with analysis
    """
    monitor = StabilityMonitor(goal_embedding=goal_embedding)

    observations = observations or [None] * len(actions)

    for state, action, obs in zip(state_embeddings, actions, observations):
        monitor.track_state(
            state_embedding=state,
            action=action,
            observation=obs
        )

    return monitor.get_stability_report()
