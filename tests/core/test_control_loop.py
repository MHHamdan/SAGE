"""Unit tests for the closed-loop control system (Section III-F).

Tests for:
1. Oscillation detection - Agent alternates between fixed actions
2. Deadlock detection - State unchanged despite actions
3. Divergence detection - Gradual drift from objectives
4. Reactivity assessment - Over/under-reactive behavior
"""

import sys
import os
from dataclasses import dataclass
from typing import Any, List, Optional, Dict, Callable
import types
import time
import hashlib
import logging
from dataclasses import dataclass as dc_dataclass, field
from enum import Enum
from collections import deque
import numpy as np

# Read and execute control_loop.py directly to avoid package __init__.py
control_loop_path = os.path.join(
    os.path.dirname(__file__), "..", "..", "src", "sage", "core", "control_loop.py"
)

# Create a module namespace with required imports
control_loop = types.ModuleType("control_loop")
control_loop.__dict__["__file__"] = control_loop_path

# Pre-populate with standard imports the module needs
control_loop.__dict__["time"] = time
control_loop.__dict__["hashlib"] = hashlib
control_loop.__dict__["logging"] = logging
control_loop.__dict__["dataclass"] = dc_dataclass
control_loop.__dict__["field"] = field
control_loop.__dict__["Enum"] = Enum
control_loop.__dict__["deque"] = deque
control_loop.__dict__["np"] = np
control_loop.__dict__["Any"] = Any
control_loop.__dict__["Dict"] = Dict
control_loop.__dict__["List"] = List
control_loop.__dict__["Optional"] = Optional
control_loop.__dict__["Callable"] = Callable

# Register the module before executing
sys.modules["control_loop"] = control_loop

# Execute the module code
with open(control_loop_path, "r") as f:
    code = f.read()
exec(compile(code, control_loop_path, "exec"), control_loop.__dict__)

# Extract classes
CycleMetrics = control_loop.__dict__["CycleMetrics"]
PhaseMetrics = control_loop.__dict__["PhaseMetrics"]
ControlPhase = control_loop.__dict__["ControlPhase"]
StabilityStatus = control_loop.__dict__["StabilityStatus"]
StabilityAnalysis = control_loop.__dict__["StabilityAnalysis"]
StabilityAnalyzer = control_loop.__dict__["StabilityAnalyzer"]
ClosedLoopController = control_loop.__dict__["ClosedLoopController"]


def create_cycle_metrics(
    cycle_number: int,
    action_hash: str,
    state_hash: str = "default_state",
    observation_hash: str = "default_obs",
    goal_error: float = 0.0,
) -> CycleMetrics:
    """Helper to create test CycleMetrics."""
    return CycleMetrics(
        cycle_number=cycle_number,
        timestamp=float(cycle_number),
        action_hash=action_hash,
        state_hash=state_hash,
        observation_hash=observation_hash,
        goal_error=goal_error,
    )


# =============================================================================
# OSCILLATION DETECTION TESTS (Section III-F-3)
# =============================================================================


def test_oscillation_detection_period_2():
    """Test detection of period-2 oscillation: A -> B -> A -> B -> ..."""
    analyzer = StabilityAnalyzer(window_size=20, oscillation_threshold=3)

    # Create A-B-A-B-A-B-A-B pattern (period 2)
    # Important: use different state_hash to avoid triggering deadlock detection
    history = []
    actions = ["action_A", "action_B"]
    for i in range(12):
        history.append(
            create_cycle_metrics(
                cycle_number=i,
                action_hash=actions[i % 2],
                state_hash=f"state_{i}",  # Different states to avoid deadlock
            )
        )

    result = analyzer.analyze(history)

    assert (
        result.is_oscillating
    ), f"Should detect period-2 oscillation. Got: {result.is_oscillating}"
    assert (
        result.oscillation_period == 2
    ), f"Period should be 2, got {result.oscillation_period}"
    assert result.status == StabilityStatus.OSCILLATING, f"Status: {result.status}"
    print("✓ test_oscillation_detection_period_2 passed")


def test_oscillation_detection_period_3():
    """Test detection of period-3 oscillation: A -> B -> C -> A -> B -> C -> ..."""
    analyzer = StabilityAnalyzer(window_size=20, oscillation_threshold=3)

    # Create A-B-C-A-B-C pattern (period 3)
    # Use different state_hash to avoid deadlock detection
    history = []
    actions = ["action_A", "action_B", "action_C"]
    for i in range(12):
        history.append(
            create_cycle_metrics(
                cycle_number=i, action_hash=actions[i % 3], state_hash=f"state_{i}"
            )
        )

    result = analyzer.analyze(history)

    assert result.is_oscillating, "Should detect period-3 oscillation"
    assert (
        result.oscillation_period == 3
    ), f"Period should be 3, got {result.oscillation_period}"
    print("✓ test_oscillation_detection_period_3 passed")


def test_no_oscillation_with_diverse_actions():
    """Test that diverse actions don't trigger oscillation."""
    analyzer = StabilityAnalyzer(window_size=20, oscillation_threshold=3)

    # Create diverse actions with no pattern
    history = []
    for i in range(10):
        history.append(
            create_cycle_metrics(cycle_number=i, action_hash=f"unique_action_{i}")
        )

    result = analyzer.analyze(history)

    assert (
        not result.is_oscillating
    ), "Should not detect oscillation with unique actions"
    print("✓ test_no_oscillation_with_diverse_actions passed")


def test_oscillation_requires_minimum_repetitions():
    """Test that oscillation needs minimum threshold repetitions."""
    analyzer = StabilityAnalyzer(
        window_size=20, oscillation_threshold=5  # High threshold
    )

    # Only 4 repetitions of A-B pattern
    history = []
    actions = ["action_A", "action_B"]
    for i in range(4):
        history.append(create_cycle_metrics(cycle_number=i, action_hash=actions[i % 2]))

    result = analyzer.analyze(history)

    # Should not detect since we don't meet threshold
    assert not result.is_oscillating, "Should not detect with insufficient repetitions"
    print("✓ test_oscillation_requires_minimum_repetitions passed")


# =============================================================================
# DEADLOCK DETECTION TESTS (Section III-F-3)
# =============================================================================


def test_deadlock_detection():
    """Test detection of deadlock - state unchanged despite actions."""
    analyzer = StabilityAnalyzer(window_size=20, deadlock_threshold=5)

    # Create history with same state but different actions
    history = []
    for i in range(10):
        history.append(
            create_cycle_metrics(
                cycle_number=i,
                action_hash=f"action_{i}",  # Different actions
                state_hash="stuck_state",  # Same state - deadlock!
            )
        )

    result = analyzer.analyze(history)

    assert result.is_deadlocked, "Should detect deadlock"
    assert result.deadlock_duration_cycles >= 5
    assert result.status == StabilityStatus.DEADLOCKED
    print("✓ test_deadlock_detection passed")


def test_no_deadlock_with_changing_states():
    """Test that changing states don't trigger deadlock."""
    analyzer = StabilityAnalyzer(window_size=20, deadlock_threshold=5)

    # Create history with changing states
    history = []
    for i in range(10):
        history.append(
            create_cycle_metrics(
                cycle_number=i,
                action_hash=f"action_{i}",
                state_hash=f"state_{i}",  # Different states
            )
        )

    result = analyzer.analyze(history)

    assert not result.is_deadlocked, "Should not detect deadlock with changing states"
    print("✓ test_no_deadlock_with_changing_states passed")


def test_deadlock_requires_consecutive_same_states():
    """Test that deadlock requires consecutive identical states."""
    analyzer = StabilityAnalyzer(window_size=20, deadlock_threshold=5)

    # Alternating states shouldn't trigger deadlock
    history = []
    for i in range(10):
        history.append(
            create_cycle_metrics(
                cycle_number=i,
                action_hash=f"action_{i}",
                state_hash=f"state_{i % 2}",  # Alternating states
            )
        )

    result = analyzer.analyze(history)

    assert not result.is_deadlocked, "Alternating states should not trigger deadlock"
    print("✓ test_deadlock_requires_consecutive_same_states passed")


def test_deadlock_takes_priority_over_oscillation():
    """Test that deadlock status takes priority over oscillation."""
    analyzer = StabilityAnalyzer(
        window_size=20, deadlock_threshold=5, oscillation_threshold=3
    )

    # Create both oscillation and deadlock conditions
    history = []
    actions = ["A", "B"]
    for i in range(10):
        history.append(
            create_cycle_metrics(
                cycle_number=i,
                action_hash=actions[i % 2],  # Oscillating actions
                state_hash="stuck_state",  # Deadlocked state
            )
        )

    result = analyzer.analyze(history)

    # Deadlock should take priority
    assert result.status == StabilityStatus.DEADLOCKED
    print("✓ test_deadlock_takes_priority_over_oscillation passed")


# =============================================================================
# DIVERGENCE DETECTION TESTS (Section III-F-3, Equation 9)
# =============================================================================


def test_divergence_detection():
    """Test detection of goal divergence - increasing goal error."""
    analyzer = StabilityAnalyzer(
        window_size=20, divergence_threshold=0.05  # Low threshold for testing
    )

    # Create history with monotonically increasing goal error
    history = []
    for i in range(10):
        history.append(
            create_cycle_metrics(
                cycle_number=i,
                action_hash=f"action_{i}",
                state_hash=f"state_{i}",
                goal_error=0.1 + i * 0.1,  # Increasing: 0.1, 0.2, 0.3, ...
            )
        )

    result = analyzer.analyze(history)

    assert result.is_diverging, "Should detect divergence with increasing goal error"
    assert result.divergence_rate > 0, "Divergence rate should be positive"
    assert result.status == StabilityStatus.DIVERGING
    print("✓ test_divergence_detection passed")


def test_no_divergence_with_stable_goal_error():
    """Test no divergence detected with stable goal error."""
    analyzer = StabilityAnalyzer(window_size=20, divergence_threshold=0.1)

    # Create history with stable goal error
    history = []
    for i in range(10):
        history.append(
            create_cycle_metrics(
                cycle_number=i,
                action_hash=f"action_{i}",
                state_hash=f"state_{i}",
                goal_error=0.2,  # Constant goal error
            )
        )

    result = analyzer.analyze(history)

    assert (
        not result.is_diverging
    ), "Should not detect divergence with stable goal error"
    print("✓ test_no_divergence_with_stable_goal_error passed")


def test_no_divergence_with_decreasing_goal_error():
    """Test no divergence when goal error is decreasing (converging)."""
    analyzer = StabilityAnalyzer(window_size=20, divergence_threshold=0.05)

    # Create history with decreasing goal error (converging to goal)
    history = []
    for i in range(10):
        history.append(
            create_cycle_metrics(
                cycle_number=i,
                action_hash=f"action_{i}",
                state_hash=f"state_{i}",
                goal_error=1.0 - i * 0.1,  # Decreasing: 1.0, 0.9, 0.8, ...
            )
        )

    result = analyzer.analyze(history)

    assert not result.is_diverging, "Should not detect divergence when converging"
    assert result.divergence_rate < 0, "Rate should be negative (converging)"
    print("✓ test_no_divergence_with_decreasing_goal_error passed")


def test_divergence_rate_computation():
    """Test that divergence rate is computed correctly."""
    analyzer = StabilityAnalyzer(window_size=20)

    # Linear increase: 0.0, 0.1, 0.2, ...
    history = []
    for i in range(10):
        history.append(
            create_cycle_metrics(
                cycle_number=i, action_hash=f"action_{i}", goal_error=i * 0.1
            )
        )

    result = analyzer.analyze(history)

    # Slope should be approximately 0.1
    assert (
        abs(result.divergence_rate - 0.1) < 0.02
    ), f"Expected rate ~0.1, got {result.divergence_rate}"
    print("✓ test_divergence_rate_computation passed")


# =============================================================================
# REACTIVITY ASSESSMENT TESTS
# =============================================================================


def test_high_reactivity_score():
    """Test high reactivity when actions change more than observations."""
    analyzer = StabilityAnalyzer(window_size=20)

    # Observations change rarely but actions change frequently
    history = []
    for i in range(10):
        history.append(
            create_cycle_metrics(
                cycle_number=i,
                action_hash=f"action_{i}",  # Every action different
                observation_hash=f"obs_{i // 5}",  # Observation changes every 5
            )
        )

    result = analyzer.analyze(history)

    # High reactivity = actions change more than observations warrant
    assert (
        result.reactivity_score > 0.5
    ), f"Reactivity should be high, got {result.reactivity_score}"
    print("✓ test_high_reactivity_score passed")


def test_low_reactivity_score():
    """Test low reactivity when actions don't change despite observation changes."""
    analyzer = StabilityAnalyzer(window_size=20)

    # Observations change frequently but actions stay same
    history = []
    for i in range(10):
        history.append(
            create_cycle_metrics(
                cycle_number=i,
                action_hash="same_action",  # Same action
                observation_hash=f"obs_{i}",  # Every observation different
            )
        )

    result = analyzer.analyze(history)

    # Low reactivity = actions don't respond to observations
    assert (
        result.reactivity_score < 0.5
    ), f"Reactivity should be low, got {result.reactivity_score}"
    print("✓ test_low_reactivity_score passed")


# =============================================================================
# STABILITY STATUS DETERMINATION TESTS
# =============================================================================


def test_stable_status_when_no_issues():
    """Test STABLE status when no pathologies detected."""
    analyzer = StabilityAnalyzer(window_size=20)

    # Create healthy history with no pathologies
    history = []
    for i in range(10):
        history.append(
            create_cycle_metrics(
                cycle_number=i,
                action_hash=f"action_{i}",
                state_hash=f"state_{i}",
                goal_error=0.1,  # Stable, low goal error
            )
        )

    result = analyzer.analyze(history)

    assert result.status == StabilityStatus.STABLE
    assert not result.is_oscillating
    assert not result.is_deadlocked
    assert not result.is_diverging
    print("✓ test_stable_status_when_no_issues passed")


def test_unknown_status_with_insufficient_history():
    """Test UNKNOWN status when insufficient history."""
    analyzer = StabilityAnalyzer(window_size=20)

    # Only 2 cycles - not enough for analysis
    history = [
        create_cycle_metrics(0, "action_0"),
        create_cycle_metrics(1, "action_1"),
    ]

    result = analyzer.analyze(history)

    assert result.status == StabilityStatus.UNKNOWN
    assert result.confidence == 0.0
    print("✓ test_unknown_status_with_insufficient_history passed")


def test_confidence_increases_with_window_size():
    """Test that confidence increases as window fills."""
    analyzer = StabilityAnalyzer(window_size=10)

    # 5 cycles = 50% confidence
    history = [create_cycle_metrics(i, f"action_{i}") for i in range(5)]
    result = analyzer.analyze(history)
    assert abs(result.confidence - 0.5) < 0.01

    # 10 cycles = 100% confidence
    history = [create_cycle_metrics(i, f"action_{i}") for i in range(10)]
    result = analyzer.analyze(history)
    assert abs(result.confidence - 1.0) < 0.01
    print("✓ test_confidence_increases_with_window_size passed")


# =============================================================================
# CYCLE METRICS TESTS
# =============================================================================


def test_cycle_metrics_latency_computation():
    """Test CycleMetrics latency computation."""
    import time as time_module

    metrics = CycleMetrics(cycle_number=1, timestamp=time_module.time())

    # Add phase metrics
    metrics.sense_metrics = PhaseMetrics(
        phase=ControlPhase.SENSE, start_time=0.0, end_time=0.1  # 100ms
    )
    metrics.decide_metrics = PhaseMetrics(
        phase=ControlPhase.DECIDE, start_time=0.1, end_time=0.3  # 200ms
    )
    metrics.act_metrics = PhaseMetrics(
        phase=ControlPhase.ACT, start_time=0.3, end_time=0.35  # 50ms
    )

    assert abs(metrics.sense_metrics.latency_ms - 100) < 0.1
    assert abs(metrics.decide_metrics.latency_ms - 200) < 0.1
    assert abs(metrics.act_metrics.latency_ms - 50) < 0.1
    assert abs(metrics.total_latency_ms - 350) < 0.1
    print("✓ test_cycle_metrics_latency_computation passed")


def test_cycle_metrics_latency_breakdown():
    """Test CycleMetrics latency breakdown."""
    metrics = CycleMetrics(cycle_number=1, timestamp=0.0)

    metrics.sense_metrics = PhaseMetrics(ControlPhase.SENSE, 0.0, 0.1)
    metrics.decide_metrics = PhaseMetrics(ControlPhase.DECIDE, 0.1, 0.3)
    metrics.act_metrics = PhaseMetrics(ControlPhase.ACT, 0.3, 0.35)

    breakdown = metrics.latency_breakdown

    assert "sense_ms" in breakdown
    assert "decide_ms" in breakdown
    assert "act_ms" in breakdown
    assert "total_ms" in breakdown
    print("✓ test_cycle_metrics_latency_breakdown passed")


# =============================================================================
# STABILITY ANALYSIS SERIALIZATION TESTS
# =============================================================================


def test_stability_analysis_to_dict():
    """Test StabilityAnalysis serialization."""
    analysis = StabilityAnalysis(
        status=StabilityStatus.OSCILLATING,
        is_oscillating=True,
        oscillation_period=2,
        oscillation_actions=["A", "B"],
        is_deadlocked=False,
        is_diverging=False,
        reactivity_score=0.7,
        confidence=0.9,
    )

    d = analysis.to_dict()

    assert d["status"] == "oscillating"
    assert d["oscillation"]["detected"] is True
    assert d["oscillation"]["period"] == 2
    assert d["deadlock"]["detected"] is False
    assert d["divergence"]["detected"] is False
    assert d["reactivity_score"] == 0.7
    assert d["confidence"] == 0.9
    print("✓ test_stability_analysis_to_dict passed")


# =============================================================================
# MOCK AGENT FOR CONTROLLER TESTS
# =============================================================================


class MockAgent:
    """Simple mock agent for testing ClosedLoopController."""

    def __init__(self, actions: List[str]):
        self.actions = actions
        self.action_index = 0
        self._complete = False

    def sense(self, observation: Any) -> Any:
        """Simple passthrough sense."""
        return {"observation": observation}

    def plan(self, state: Any) -> str:
        """Return next action from predefined list."""
        action = self.actions[self.action_index % len(self.actions)]
        self.action_index += 1
        return action

    def act(self, action: str) -> Any:
        """Execute action (no-op for testing)."""
        return {"result": f"executed_{action}"}

    def is_complete(self) -> bool:
        return self._complete

    def set_complete(self):
        self._complete = True


def test_closed_loop_controller_run_cycle():
    """Test ClosedLoopController.run_cycle()."""
    agent = MockAgent(["action_A", "action_B"])
    controller = ClosedLoopController(agent=agent, goal="test goal", max_cycles=100)

    metrics = controller.run_cycle("observation_1")

    assert metrics.cycle_number == 1
    assert metrics.observation_hash is not None
    assert metrics.state_hash is not None
    assert metrics.action_hash is not None
    assert metrics.sense_metrics is not None
    assert metrics.decide_metrics is not None
    assert metrics.act_metrics is not None
    print("✓ test_closed_loop_controller_run_cycle passed")


def test_closed_loop_controller_detects_oscillation():
    """Test that controller detects oscillation over multiple cycles."""
    # Agent that oscillates between two actions
    agent = MockAgent(["A", "B"])
    controller = ClosedLoopController(
        agent=agent, goal="test goal", max_cycles=100, stability_window=20
    )

    # Run multiple cycles to trigger oscillation detection
    for i in range(10):
        controller.run_cycle(f"obs_{i}")

    stability = controller.get_stability_analysis()

    assert stability.is_oscillating, "Should detect oscillation"
    assert stability.status == StabilityStatus.OSCILLATING
    print("✓ test_closed_loop_controller_detects_oscillation passed")


def test_closed_loop_controller_max_cycles_termination():
    """Test controller terminates at max_cycles."""
    agent = MockAgent(["action"])
    controller = ClosedLoopController(agent=agent, goal="test", max_cycles=5)

    # Run to max
    for i in range(5):
        controller.run_cycle(f"obs_{i}")

    assert controller.should_terminate(), "Should terminate at max cycles"
    assert controller.cycle_count == 5
    print("✓ test_closed_loop_controller_max_cycles_termination passed")


def test_closed_loop_controller_agent_completion():
    """Test controller terminates when agent signals completion."""
    agent = MockAgent(["action"])
    controller = ClosedLoopController(agent=agent, goal="test", max_cycles=100)

    controller.run_cycle("obs_1")
    assert not controller.should_terminate()

    agent.set_complete()
    assert controller.should_terminate()
    print("✓ test_closed_loop_controller_agent_completion passed")


def test_closed_loop_controller_summary():
    """Test controller summary generation."""
    agent = MockAgent(["action_A", "action_B"])
    controller = ClosedLoopController(agent=agent, goal="test", max_cycles=100)

    for i in range(5):
        controller.run_cycle(f"obs_{i}")

    summary = controller.get_summary()

    assert summary["total_cycles"] == 5
    assert "stability" in summary
    assert "latency" in summary
    assert "goal_drift" in summary
    assert "mean_ms" in summary["latency"]
    print("✓ test_closed_loop_controller_summary passed")


def test_closed_loop_controller_reset():
    """Test controller reset functionality."""
    agent = MockAgent(["action"])
    controller = ClosedLoopController(agent=agent, goal="test", max_cycles=100)

    for i in range(5):
        controller.run_cycle(f"obs_{i}")

    assert controller.cycle_count == 5
    assert len(controller.cycle_history) == 5

    controller.reset()

    assert controller.cycle_count == 0
    assert len(controller.cycle_history) == 0
    assert not controller.terminated
    print("✓ test_closed_loop_controller_reset passed")


# =============================================================================
# MAIN TEST RUNNER
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("Running Control Loop Unit Tests (Section III-F)")
    print("=" * 60)
    print()

    tests = [
        # Oscillation tests
        test_oscillation_detection_period_2,
        test_oscillation_detection_period_3,
        test_no_oscillation_with_diverse_actions,
        test_oscillation_requires_minimum_repetitions,
        # Deadlock tests
        test_deadlock_detection,
        test_no_deadlock_with_changing_states,
        test_deadlock_requires_consecutive_same_states,
        test_deadlock_takes_priority_over_oscillation,
        # Divergence tests
        test_divergence_detection,
        test_no_divergence_with_stable_goal_error,
        test_no_divergence_with_decreasing_goal_error,
        test_divergence_rate_computation,
        # Reactivity tests
        test_high_reactivity_score,
        test_low_reactivity_score,
        # Status tests
        test_stable_status_when_no_issues,
        test_unknown_status_with_insufficient_history,
        test_confidence_increases_with_window_size,
        # Metrics tests
        test_cycle_metrics_latency_computation,
        test_cycle_metrics_latency_breakdown,
        test_stability_analysis_to_dict,
        # Controller tests
        test_closed_loop_controller_run_cycle,
        test_closed_loop_controller_detects_oscillation,
        test_closed_loop_controller_max_cycles_termination,
        test_closed_loop_controller_agent_completion,
        test_closed_loop_controller_summary,
        test_closed_loop_controller_reset,
    ]

    passed = 0
    failed = 0

    for test_fn in tests:
        try:
            test_fn()
            passed += 1
        except AssertionError as e:
            print(f"✗ {test_fn.__name__} FAILED: {str(e)[:200]}")
            failed += 1
        except Exception as e:
            import traceback

            print(f"✗ {test_fn.__name__} ERROR: {e}")
            traceback.print_exc()
            failed += 1

    print()
    print("=" * 60)
    print(f"Results: {passed} passed, {failed} failed out of {len(tests)} tests")
    print("=" * 60)

    if failed > 0:
        sys.exit(1)
