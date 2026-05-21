"""
Closed-Loop Control System for Agentic AI.

Implements the control-theoretic perspective from Section III-F of the paper:
    o_t -> sense -> s_t -> decide -> a_t -> act -> o_{t+1}

This module provides:
- Explicit sense-decide-act feedback cycles
- Stability analysis (oscillation, deadlock, divergence detection)
- Per-phase latency tracking
- Non-stationarity detection for policy drift
"""

from __future__ import annotations

import time
import hashlib
import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Callable
from collections import deque

import numpy as np

logger = logging.getLogger(__name__)


class ControlPhase(Enum):
    """Phases of the control loop (Equation 4 in paper)."""
    SENSE = "sense"      # o_t -> s_t
    DECIDE = "decide"    # s_t -> a_t
    ACT = "act"          # a_t -> o_{t+1}


class StabilityStatus(Enum):
    """Overall stability assessment."""
    STABLE = "stable"
    OSCILLATING = "oscillating"
    DEADLOCKED = "deadlocked"
    DIVERGING = "diverging"
    UNKNOWN = "unknown"


@dataclass
class PhaseMetrics:
    """Metrics for a single control loop phase."""
    phase: ControlPhase
    start_time: float
    end_time: float

    @property
    def latency_ms(self) -> float:
        """Latency in milliseconds."""
        return (self.end_time - self.start_time) * 1000


@dataclass
class CycleMetrics:
    """Metrics for a complete sense-decide-act cycle."""
    cycle_number: int
    timestamp: float

    # Phase timings
    sense_metrics: Optional[PhaseMetrics] = None
    decide_metrics: Optional[PhaseMetrics] = None
    act_metrics: Optional[PhaseMetrics] = None

    # State information
    observation_hash: Optional[str] = None
    state_hash: Optional[str] = None
    action_hash: Optional[str] = None

    # Goal tracking
    goal_error: float = 0.0  # 1 - sim(g_0, g_t) from Equation 9

    @property
    def total_latency_ms(self) -> float:
        """Total cycle latency in milliseconds."""
        total = 0.0
        if self.sense_metrics:
            total += self.sense_metrics.latency_ms
        if self.decide_metrics:
            total += self.decide_metrics.latency_ms
        if self.act_metrics:
            total += self.act_metrics.latency_ms
        return total

    @property
    def latency_breakdown(self) -> Dict[str, float]:
        """Breakdown of latency by phase."""
        return {
            "sense_ms": self.sense_metrics.latency_ms if self.sense_metrics else 0.0,
            "decide_ms": self.decide_metrics.latency_ms if self.decide_metrics else 0.0,
            "act_ms": self.act_metrics.latency_ms if self.act_metrics else 0.0,
            "total_ms": self.total_latency_ms
        }


@dataclass
class StabilityAnalysis:
    """
    Results of stability analysis per Section III-F-1.

    Paper states:
    > "Over-reactive agents respond to every observation change,
    > leading to oscillation. Under-reactive agents maintain
    > plans despite changing conditions, failing to adapt."
    """

    status: StabilityStatus = StabilityStatus.UNKNOWN

    # Oscillation detection
    is_oscillating: bool = False
    oscillation_period: Optional[int] = None
    oscillation_actions: List[str] = field(default_factory=list)

    # Deadlock detection
    is_deadlocked: bool = False
    deadlock_duration_cycles: int = 0

    # Divergence detection (slow drift from goal)
    is_diverging: bool = False
    divergence_rate: float = 0.0  # Rate of goal error increase

    # Reactivity assessment
    reactivity_score: float = 0.5  # 0=under-reactive, 1=over-reactive

    # Confidence in analysis
    confidence: float = 0.0
    analysis_window_size: int = 0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "status": self.status.value,
            "oscillation": {
                "detected": self.is_oscillating,
                "period": self.oscillation_period,
                "actions": self.oscillation_actions
            },
            "deadlock": {
                "detected": self.is_deadlocked,
                "duration_cycles": self.deadlock_duration_cycles
            },
            "divergence": {
                "detected": self.is_diverging,
                "rate": self.divergence_rate
            },
            "reactivity_score": self.reactivity_score,
            "confidence": self.confidence
        }


class StabilityAnalyzer:
    """
    Analyzes control loop for stability pathologies.

    Detects three pathologies from Section III-F-3:
    1. Oscillation - Agent alternates between fixed set of actions
    2. Deadlock - Circular dependencies preventing progress
    3. Slow divergence - Gradual drift from objectives
    """

    def __init__(
        self,
        window_size: int = 20,
        oscillation_threshold: int = 3,
        deadlock_threshold: int = 5,
        divergence_threshold: float = 0.1
    ):
        """
        Initialize stability analyzer.

        Args:
            window_size: Number of cycles to analyze
            oscillation_threshold: Min repetitions to detect oscillation
            deadlock_threshold: Min unchanged cycles to detect deadlock
            divergence_threshold: Goal error increase rate threshold
        """
        self.window_size = window_size
        self.oscillation_threshold = oscillation_threshold
        self.deadlock_threshold = deadlock_threshold
        self.divergence_threshold = divergence_threshold

    def analyze(self, cycle_history: List[CycleMetrics]) -> StabilityAnalysis:
        """
        Perform comprehensive stability analysis.

        Args:
            cycle_history: Recent cycle metrics to analyze

        Returns:
            StabilityAnalysis with detected pathologies
        """
        if len(cycle_history) < 3:
            return StabilityAnalysis(
                status=StabilityStatus.UNKNOWN,
                confidence=0.0,
                analysis_window_size=len(cycle_history)
            )

        # Use most recent window_size cycles
        window = cycle_history[-self.window_size:]

        analysis = StabilityAnalysis(
            analysis_window_size=len(window)
        )

        # Run detection algorithms
        self._detect_oscillation(window, analysis)
        self._detect_deadlock(window, analysis)
        self._detect_divergence(window, analysis)
        self._compute_reactivity(window, analysis)

        # Determine overall status
        analysis.status = self._determine_status(analysis)

        # Compute confidence based on window size
        analysis.confidence = min(1.0, len(window) / self.window_size)

        return analysis

    def _detect_oscillation(
        self,
        window: List[CycleMetrics],
        analysis: StabilityAnalysis
    ) -> None:
        """
        Detect action oscillation patterns.

        Pattern: A -> B -> A -> B -> ... (period 2)
        Or: A -> B -> C -> A -> B -> C -> ... (period 3)
        """
        if len(window) < 4:
            return

        action_sequence = [c.action_hash for c in window if c.action_hash]

        if len(action_sequence) < 4:
            return

        # Check for periods 2 through window_size/2
        for period in range(2, len(action_sequence) // 2 + 1):
            repetitions = 0

            for i in range(len(action_sequence) - period):
                if action_sequence[i] == action_sequence[i + period]:
                    repetitions += 1

            # Calculate what fraction are matching
            total_comparisons = len(action_sequence) - period
            match_ratio = repetitions / total_comparisons if total_comparisons > 0 else 0

            if match_ratio > 0.8 and repetitions >= self.oscillation_threshold:
                analysis.is_oscillating = True
                analysis.oscillation_period = period
                # Get the repeating action pattern
                analysis.oscillation_actions = action_sequence[:period]
                return

    def _detect_deadlock(
        self,
        window: List[CycleMetrics],
        analysis: StabilityAnalysis
    ) -> None:
        """
        Detect deadlock - state unchanged despite actions.

        Pattern: Actions execute but state hash remains identical.
        """
        if len(window) < self.deadlock_threshold:
            return

        state_hashes = [c.state_hash for c in window if c.state_hash]

        if len(state_hashes) < self.deadlock_threshold:
            return

        # Count consecutive identical states
        consecutive_same = 1
        max_consecutive = 1

        for i in range(1, len(state_hashes)):
            if state_hashes[i] == state_hashes[i-1]:
                consecutive_same += 1
                max_consecutive = max(max_consecutive, consecutive_same)
            else:
                consecutive_same = 1

        if max_consecutive >= self.deadlock_threshold:
            analysis.is_deadlocked = True
            analysis.deadlock_duration_cycles = max_consecutive

    def _detect_divergence(
        self,
        window: List[CycleMetrics],
        analysis: StabilityAnalysis
    ) -> None:
        """
        Detect slow divergence from goals.

        Pattern: Goal error (Equation 9) monotonically increasing.
        """
        if len(window) < 5:
            return

        goal_errors = [c.goal_error for c in window]

        if not goal_errors or all(e == 0 for e in goal_errors):
            return

        # Compute trend using linear regression
        x = np.arange(len(goal_errors))
        y = np.array(goal_errors)

        # Simple linear regression
        n = len(x)
        sum_x = np.sum(x)
        sum_x_sq = np.sum(x**2)

        if n * sum_x_sq - sum_x**2 == 0:
            return

        slope = (n * np.sum(x * y) - sum_x * np.sum(y)) / \
                (n * sum_x_sq - sum_x**2)

        analysis.divergence_rate = float(slope)

        if slope > self.divergence_threshold:
            analysis.is_diverging = True

    def _compute_reactivity(
        self,
        window: List[CycleMetrics],
        analysis: StabilityAnalysis
    ) -> None:
        """
        Compute reactivity score.

        High reactivity: Actions change frequently
        Low reactivity: Actions remain same despite observation changes

        Score: 0 = under-reactive, 1 = over-reactive
        """
        if len(window) < 3:
            analysis.reactivity_score = 0.5
            return

        observation_changes = 0
        action_changes = 0

        for i in range(1, len(window)):
            if window[i].observation_hash != window[i-1].observation_hash:
                observation_changes += 1
            if window[i].action_hash != window[i-1].action_hash:
                action_changes += 1

        if observation_changes == 0:
            # No observation changes, can't assess reactivity
            analysis.reactivity_score = 0.5
            return

        # Reactivity = action changes / observation changes
        # Clamped to [0, 1]
        analysis.reactivity_score = min(1.0, action_changes / observation_changes)

    def _determine_status(self, analysis: StabilityAnalysis) -> StabilityStatus:
        """Determine overall stability status from individual checks."""
        if analysis.is_deadlocked:
            return StabilityStatus.DEADLOCKED
        if analysis.is_oscillating:
            return StabilityStatus.OSCILLATING
        if analysis.is_diverging:
            return StabilityStatus.DIVERGING
        return StabilityStatus.STABLE


class ClosedLoopController:
    """
    Implements the closed-loop control perspective from Section III-F.

    Wraps an agent's SPAR methods in an explicit feedback loop with:
    - Per-phase latency tracking
    - Stability analysis
    - Goal drift monitoring
    - Non-stationarity detection

    Usage:
        ```python
        from sage.agents import ReActAgent
        from sage.core.control_loop import ClosedLoopController

        agent = ReActAgent(...)
        controller = ClosedLoopController(agent, goal="Complete the task")

        while not controller.should_terminate():
            metrics = controller.run_cycle(observation)

            # Check stability
            stability = controller.get_stability_analysis()
            if stability.status != StabilityStatus.STABLE:
                # Handle instability
                pass
        ```
    """

    def __init__(
        self,
        agent: Any,  # BaseAgent or compatible
        goal: str,
        embedding_fn: Optional[Callable[[str], np.ndarray]] = None,
        max_cycles: int = 100,
        stability_window: int = 20
    ):
        """
        Initialize closed-loop controller.

        Args:
            agent: Agent with sense(), plan()/select_action(), act() methods
            goal: The goal string for drift detection
            embedding_fn: Function to embed text (for goal drift).
                         If None, goal drift won't be computed.
            max_cycles: Maximum cycles before forced termination
            stability_window: Window size for stability analysis
        """
        self.agent = agent
        self.goal = goal
        self.embedding_fn = embedding_fn
        self.max_cycles = max_cycles

        # Embed original goal for drift detection
        self.goal_embedding: Optional[np.ndarray] = None
        if embedding_fn:
            self.goal_embedding = embedding_fn(goal)

        # State
        self.cycle_count = 0
        self.cycle_history: List[CycleMetrics] = []
        self.terminated = False

        # Analysis
        self.stability_analyzer = StabilityAnalyzer(window_size=stability_window)

        # Current cycle state
        self._current_observation: Any = None
        self._current_state: Any = None
        self._current_action: Any = None

    def run_cycle(self, observation: Any) -> CycleMetrics:
        """
        Execute one complete sense-decide-act cycle.

        This is the core feedback loop from Equation 4:
            o_t -> sense -> s_t -> decide -> a_t -> act -> o_{t+1}

        Args:
            observation: Raw observation from environment

        Returns:
            CycleMetrics for this cycle
        """
        self.cycle_count += 1
        cycle_start = time.time()

        metrics = CycleMetrics(
            cycle_number=self.cycle_count,
            timestamp=cycle_start
        )

        # ============ SENSE PHASE ============
        # o_t -> s_t
        sense_start = time.time()

        self._current_observation = observation
        metrics.observation_hash = self._hash_value(observation)

        # Call agent's sense method
        if hasattr(self.agent, 'sense'):
            self._current_state = self.agent.sense(observation)
        else:
            # Fallback: observation becomes state
            self._current_state = observation

        metrics.state_hash = self._hash_value(self._current_state)

        sense_end = time.time()
        metrics.sense_metrics = PhaseMetrics(
            phase=ControlPhase.SENSE,
            start_time=sense_start,
            end_time=sense_end
        )

        # ============ DECIDE PHASE ============
        # s_t -> a_t
        decide_start = time.time()

        # Call agent's planning/decision method
        if hasattr(self.agent, 'plan'):
            self._current_action = self.agent.plan(self._current_state)
        elif hasattr(self.agent, 'select_action'):
            self._current_action = self.agent.select_action(self._current_state)
        else:
            raise AttributeError("Agent must have 'plan' or 'select_action' method")

        metrics.action_hash = self._hash_value(self._current_action)

        decide_end = time.time()
        metrics.decide_metrics = PhaseMetrics(
            phase=ControlPhase.DECIDE,
            start_time=decide_start,
            end_time=decide_end
        )

        # Compute goal drift if embedding function available
        if self.embedding_fn and self.goal_embedding is not None:
            metrics.goal_error = self._compute_goal_drift()

        # ============ ACT PHASE ============
        # a_t -> o_{t+1}
        act_start = time.time()

        # Call agent's act method
        if hasattr(self.agent, 'act'):
            result = self.agent.act(self._current_action)
        elif hasattr(self.agent, 'execute'):
            result = self.agent.execute(self._current_action)
        else:
            raise AttributeError("Agent must have 'act' or 'execute' method")

        act_end = time.time()
        metrics.act_metrics = PhaseMetrics(
            phase=ControlPhase.ACT,
            start_time=act_start,
            end_time=act_end
        )

        # Store metrics
        self.cycle_history.append(metrics)

        # Log cycle completion
        logger.debug(
            f"Cycle {self.cycle_count}: "
            f"latency={metrics.total_latency_ms:.1f}ms, "
            f"goal_error={metrics.goal_error:.3f}"
        )

        return metrics

    def _compute_goal_drift(self) -> float:
        """
        Compute goal drift using Equation 9:
            Drift_t = 1 - sim(g_0, g_t)

        Returns:
            Goal drift score in [0, 1]. 0 = no drift, 1 = complete drift.
        """
        if self.embedding_fn is None or self.goal_embedding is None:
            return 0.0

        # Infer current goal from recent actions/reasoning
        # This is an approximation - in practice would analyze reasoning traces
        current_context = str(self._current_action)
        if hasattr(self._current_action, 'reasoning'):
            current_context = self._current_action.reasoning

        current_embedding = self.embedding_fn(current_context)

        # Cosine similarity
        norm_goal = np.linalg.norm(self.goal_embedding)
        norm_current = np.linalg.norm(current_embedding)

        if norm_goal == 0 or norm_current == 0:
            return 0.0

        similarity = np.dot(self.goal_embedding, current_embedding) / (norm_goal * norm_current)

        # Drift = 1 - similarity
        return float(1.0 - similarity)

    def _hash_value(self, value: Any) -> str:
        """Create hash of value for comparison."""
        return hashlib.md5(str(value).encode()).hexdigest()[:8]

    def get_stability_analysis(self) -> StabilityAnalysis:
        """
        Get current stability analysis.

        Returns:
            StabilityAnalysis with oscillation, deadlock, divergence detection
        """
        return self.stability_analyzer.analyze(self.cycle_history)

    def should_terminate(self) -> bool:
        """
        Check if controller should terminate.

        Termination conditions:
        1. Agent signaled completion
        2. Max cycles reached
        3. Critical instability detected
        """
        if self.terminated:
            return True

        if self.cycle_count >= self.max_cycles:
            logger.warning(f"Max cycles ({self.max_cycles}) reached")
            self.terminated = True
            return True

        # Check for critical instability
        stability = self.get_stability_analysis()
        if stability.status == StabilityStatus.DEADLOCKED:
            logger.warning("Deadlock detected, terminating")
            self.terminated = True
            return True

        # Check if agent signals completion
        if hasattr(self.agent, 'is_complete') and self.agent.is_complete():
            self.terminated = True
            return True

        return False

    def get_summary(self) -> Dict[str, Any]:
        """Get summary of control loop execution."""
        stability = self.get_stability_analysis()

        latencies = [c.total_latency_ms for c in self.cycle_history]
        goal_errors = [c.goal_error for c in self.cycle_history]

        return {
            "total_cycles": self.cycle_count,
            "terminated": self.terminated,
            "stability": stability.to_dict(),
            "latency": {
                "mean_ms": float(np.mean(latencies)) if latencies else 0,
                "max_ms": float(max(latencies)) if latencies else 0,
                "min_ms": float(min(latencies)) if latencies else 0,
                "std_ms": float(np.std(latencies)) if latencies else 0
            },
            "goal_drift": {
                "final": goal_errors[-1] if goal_errors else 0,
                "max": float(max(goal_errors)) if goal_errors else 0,
                "mean": float(np.mean(goal_errors)) if goal_errors else 0
            }
        }

    def reset(self) -> None:
        """Reset the controller state."""
        self.cycle_count = 0
        self.cycle_history = []
        self.terminated = False
        self._current_observation = None
        self._current_state = None
        self._current_action = None


# ============================================================
# Integration with existing BaseAgent
# ============================================================

def wrap_agent_with_control_loop(
    agent: Any,
    goal: str,
    embedding_fn: Optional[Callable] = None
) -> ClosedLoopController:
    """
    Convenience function to wrap an existing agent in a control loop.

    Args:
        agent: Agent with SPAR methods
        goal: Goal string
        embedding_fn: Optional embedding function for drift detection

    Returns:
        ClosedLoopController wrapping the agent
    """
    return ClosedLoopController(
        agent=agent,
        goal=goal,
        embedding_fn=embedding_fn
    )
