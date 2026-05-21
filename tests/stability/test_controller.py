"""Unit tests for the controller policies.

Regression tests for determinism: each controller must produce identical
decisions given the same input sequence, regardless of call order.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "src"))

from sage.stability.controller import (
    MonitorSignals,
    InterventionDecision,
    NoControl,
    FixedScheduleController,
    ThresholdController,
    PredictiveController,
)


def _sig(drift=0.2, osc=0.1, fid=0.9, prog=0.7, turn=1, cost=0.0) -> MonitorSignals:
    return MonitorSignals(
        drift_score=drift,
        oscillation_score=osc,
        fidelity_score=fid,
        convergence_progress=prog,
        turn=turn,
        cost_so_far=cost,
    )


# ── NoControl ─────────────────────────────────────────────────────────────────


class TestNoControl:
    def test_always_noop(self):
        nc = NoControl()
        for turn in range(1, 60):
            d = nc.decide(_sig(drift=0.99, osc=0.99, fid=0.0, turn=turn))
            assert d.intervention is None

    def test_confidence_one(self):
        d = NoControl().decide(_sig())
        assert d.confidence == pytest.approx(1.0)

    def test_reset_noop(self):
        nc = NoControl()
        nc.reset()  # should not raise
        assert nc.decide(_sig()).intervention is None

    def test_deterministic(self):
        nc = NoControl()
        d1 = nc.decide(_sig(turn=5))
        nc.reset()
        d2 = nc.decide(_sig(turn=5))
        assert (d1.intervention is None) == (d2.intervention is None)


# ── FixedScheduleController ───────────────────────────────────────────────────


class TestFixedScheduleController:
    def test_fires_exactly_at_multiples_of_k(self):
        ctrl = FixedScheduleController(reanchor_every_k=10)
        for turn in range(1, 51):
            d = ctrl.decide(_sig(turn=turn))
            if turn % 10 == 0:
                assert (
                    d.intervention is not None
                ), f"Expected intervention at turn {turn}"
            else:
                assert d.intervention is None, f"Unexpected intervention at turn {turn}"

    def test_fires_at_k5(self):
        ctrl = FixedScheduleController(reanchor_every_k=5)
        for turn in [1, 2, 3, 4, 5, 6, 10, 15]:
            d = ctrl.decide(_sig(turn=turn))
            if turn % 5 == 0:
                assert d.intervention is not None
            else:
                assert d.intervention is None

    def test_intervention_is_goal_reanchor(self):
        ctrl = FixedScheduleController(reanchor_every_k=10)
        d = ctrl.decide(_sig(turn=10))
        assert d.intervention is not None
        assert d.intervention.name == "GoalReanchor"

    def test_deterministic(self):
        """Same input → same output after reset."""
        ctrl = FixedScheduleController(reanchor_every_k=10)
        decisions_1 = [
            ctrl.decide(_sig(turn=t)).intervention is not None for t in range(1, 21)
        ]
        ctrl.reset()
        decisions_2 = [
            ctrl.decide(_sig(turn=t)).intervention is not None for t in range(1, 21)
        ]
        assert decisions_1 == decisions_2


# ── ThresholdController ───────────────────────────────────────────────────────


class TestThresholdController:
    def test_fires_on_high_drift(self):
        ctrl = ThresholdController(drift_threshold=0.30, cooldown_turns=0)
        d = ctrl.decide(_sig(drift=0.50))
        assert d.intervention is not None
        assert d.intervention.name == "GoalReanchor"

    def test_no_fire_below_drift(self):
        ctrl = ThresholdController(drift_threshold=0.30, cooldown_turns=0)
        d = ctrl.decide(_sig(drift=0.10))
        assert d.intervention is None

    def test_fires_on_high_oscillation(self):
        ctrl = ThresholdController(oscillation_threshold=0.60, cooldown_turns=0)
        d = ctrl.decide(_sig(drift=0.10, osc=0.80))
        assert d.intervention is not None
        assert d.intervention.name == "ForceReplan"

    def test_fires_on_low_fidelity(self):
        ctrl = ThresholdController(fidelity_threshold=0.70, cooldown_turns=0)
        d = ctrl.decide(_sig(drift=0.10, osc=0.10, fid=0.30))
        assert d.intervention is not None
        assert d.intervention.name == "SchemaValidatedRetry"

    def test_cooldown_prevents_thrash(self):
        ctrl = ThresholdController(drift_threshold=0.30, cooldown_turns=3)
        d1 = ctrl.decide(_sig(drift=0.50, turn=1))
        assert d1.intervention is not None
        d2 = ctrl.decide(_sig(drift=0.50, turn=2))
        assert d2.intervention is None  # cooldown
        d3 = ctrl.decide(_sig(drift=0.50, turn=3))
        assert d3.intervention is None
        d4 = ctrl.decide(_sig(drift=0.50, turn=4))
        assert d4.intervention is not None  # cooldown expired

    def test_reset_clears_cooldown(self):
        ctrl = ThresholdController(drift_threshold=0.30, cooldown_turns=5)
        ctrl.decide(_sig(drift=0.50, turn=1))  # fires, sets cooldown
        ctrl.reset()
        d = ctrl.decide(_sig(drift=0.50, turn=2))
        assert d.intervention is not None, "Reset should clear cooldown"

    def test_drift_priority_over_oscillation(self):
        ctrl = ThresholdController(
            drift_threshold=0.30, oscillation_threshold=0.60, cooldown_turns=0
        )
        d = ctrl.decide(_sig(drift=0.50, osc=0.80))
        assert (
            d.intervention.name == "GoalReanchor"
        ), "Drift has priority over oscillation"

    def test_deterministic(self):
        """Identical signal sequences → identical decision sequences."""

        def _run(ctrl):
            signals = [
                _sig(drift=0.10, turn=1),
                _sig(drift=0.50, turn=2),
                _sig(drift=0.50, turn=3),
                _sig(drift=0.50, turn=4),
                _sig(drift=0.50, turn=5),
            ]
            return [ctrl.decide(s).intervention is not None for s in signals]

        ctrl_a = ThresholdController(drift_threshold=0.30, cooldown_turns=2)
        ctrl_b = ThresholdController(drift_threshold=0.30, cooldown_turns=2)
        assert _run(ctrl_a) == _run(ctrl_b)

    def test_confidence_proportional_to_excess(self):
        ctrl = ThresholdController(drift_threshold=0.30, cooldown_turns=0)
        d = ctrl.decide(_sig(drift=0.60))
        assert d.confidence > 0.0
        assert d.confidence <= 1.0


# ── PredictiveController ──────────────────────────────────────────────────────


class _MockPredictor:
    """Returns a fixed probability for testing."""

    def __init__(self, p: float = 0.0):
        self.p = p

    def predict_proba(self, signals, history):
        return self.p


class TestPredictiveController:
    def test_fires_above_threshold(self):
        ctrl = PredictiveController(
            predictor=_MockPredictor(p=0.8), fire_at_p=0.5, cooldown_turns=0
        )
        d = ctrl.decide(_sig(turn=1))
        assert d.intervention is not None

    def test_no_fire_below_threshold(self):
        ctrl = PredictiveController(
            predictor=_MockPredictor(p=0.2), fire_at_p=0.5, cooldown_turns=0
        )
        d = ctrl.decide(_sig(turn=1))
        assert d.intervention is None

    def test_cooldown_respected(self):
        ctrl = PredictiveController(
            predictor=_MockPredictor(p=0.9), fire_at_p=0.5, cooldown_turns=3
        )
        d1 = ctrl.decide(_sig(turn=1))
        assert d1.intervention is not None
        d2 = ctrl.decide(_sig(turn=2))
        assert d2.intervention is None
        d3 = ctrl.decide(_sig(turn=4))
        assert d3.intervention is not None

    def test_reset_clears_state(self):
        ctrl = PredictiveController(
            predictor=_MockPredictor(p=0.9), fire_at_p=0.5, cooldown_turns=5
        )
        ctrl.decide(_sig(turn=1))
        ctrl.reset()
        d = ctrl.decide(_sig(turn=2))
        assert d.intervention is not None, "Reset should clear cooldown and history"

    def test_chooses_reanchor_when_drift_dominant(self):
        ctrl = PredictiveController(
            predictor=_MockPredictor(p=0.9), fire_at_p=0.5, cooldown_turns=0
        )
        d = ctrl.decide(_sig(drift=0.8, osc=0.1, turn=1))
        assert d.intervention is not None
        assert d.intervention.name == "GoalReanchor"

    def test_chooses_replan_when_oscillation_dominant(self):
        ctrl = PredictiveController(
            predictor=_MockPredictor(p=0.9), fire_at_p=0.5, cooldown_turns=0
        )
        d = ctrl.decide(_sig(drift=0.1, osc=0.9, turn=1))
        assert d.intervention.name == "ForceReplan"

    def test_deterministic(self):
        """Same signals → same decisions from two independent instances."""
        sigs = [_sig(drift=0.3, turn=t) for t in range(1, 11)]
        ctrl_a = PredictiveController(
            predictor=_MockPredictor(p=0.7), fire_at_p=0.5, cooldown_turns=2
        )
        ctrl_b = PredictiveController(
            predictor=_MockPredictor(p=0.7), fire_at_p=0.5, cooldown_turns=2
        )
        da = [ctrl_a.decide(s).intervention is not None for s in sigs]
        db = [ctrl_b.decide(s).intervention is not None for s in sigs]
        assert da == db

    def test_confidence_equals_predicted_probability(self):
        ctrl = PredictiveController(
            predictor=_MockPredictor(p=0.75), fire_at_p=0.5, cooldown_turns=0
        )
        d = ctrl.decide(_sig(turn=1))
        assert d.confidence == pytest.approx(0.75)


# ── MonitorSignals dataclass ──────────────────────────────────────────────────


class TestMonitorSignals:
    def test_frozen(self):
        sig = _sig()
        with pytest.raises(Exception):  # FrozenInstanceError
            sig.drift_score = 0.99  # type: ignore

    def test_field_access(self):
        sig = _sig(drift=0.3, osc=0.5, fid=0.8, prog=0.6, turn=10, cost=1.5)
        assert sig.drift_score == pytest.approx(0.3)
        assert sig.turn == 10
        assert sig.cost_so_far == pytest.approx(1.5)
