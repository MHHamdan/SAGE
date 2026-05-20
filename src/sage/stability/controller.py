"""Adaptive Stability Controller — Controller core.

Defines MonitorSignals, InterventionDecision, the Controller protocol, and four
concrete policies:

  NoControl            — always NoOp (baseline)
  FixedScheduleController — re-anchors every k turns regardless of signals
  ThresholdController  — fires when a monitor crosses a tunable threshold
  PredictiveController — fires based on P(failure within k turns) from predictor

All policies are deterministic given fixed input sequences.
Cooldown (minimum turns between interventions) is supported on Threshold and
Predictive to prevent thrashing (see pitfall 3 in the design spec).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional, Protocol, runtime_checkable

from .interventions import (
    Intervention,
    GoalReanchor,
    ContextCompress,
    ForceReplan,
    SchemaValidatedRetry,
)

logger = logging.getLogger(__name__)


# ── Signal + decision types ────────────────────────────────────────────────────

@dataclass(frozen=True)
class MonitorSignals:
    """Compact summary of monitor state, consumed by all controllers."""
    drift_score: float           # [0, 1]; 0 = aligned with goal
    oscillation_score: float     # [0, 1]; fraction of recent action overlap
    fidelity_score: float        # [0, 1]; schema validation pass rate
    convergence_progress: float  # [0, 1]; best goal similarity seen so far
    turn: int
    cost_so_far: float


@dataclass(frozen=True)
class InterventionDecision:
    """Output of Controller.decide()."""
    intervention: Optional[Intervention]  # None = NoOp
    rationale: str
    confidence: float                     # [0, 1]


# ── Controller protocol ────────────────────────────────────────────────────────

@runtime_checkable
class Controller(Protocol):
    def decide(self, signals: MonitorSignals) -> InterventionDecision: ...
    def reset(self) -> None: ...


# ── Concrete policies ─────────────────────────────────────────────────────────

class NoControl:
    """Baseline: never intervenes.  Required for fair comparison."""

    name: str = "NoControl"

    def decide(self, signals: MonitorSignals) -> InterventionDecision:
        return InterventionDecision(
            intervention=None,
            rationale="no-control baseline",
            confidence=1.0,
        )

    def reset(self) -> None:
        pass


@dataclass
class FixedScheduleController:
    """Re-anchors goal every reanchor_every_k turns regardless of signals.

    Non-trivial baseline: tests whether *any* periodic intervention beats
    NoControl, isolating the value of adaptive timing vs. fixed timing.
    """
    name: str = field(default="FixedSchedule", init=False)
    reanchor_every_k: int = 10
    _intervention: GoalReanchor = field(default_factory=GoalReanchor, init=False, repr=False)

    def decide(self, signals: MonitorSignals) -> InterventionDecision:
        if signals.turn % self.reanchor_every_k == 0:
            return InterventionDecision(
                intervention=self._intervention,
                rationale=f"scheduled re-anchor at turn {signals.turn}",
                confidence=1.0,
            )
        return InterventionDecision(
            intervention=None,
            rationale="not a scheduled turn",
            confidence=1.0,
        )

    def reset(self) -> None:
        pass  # stateless beyond the schedule


@dataclass
class ThresholdController:
    """Fires when a monitor signal crosses a tunable threshold.

    Intervention selection priority:
      1. High drift → GoalReanchor  (most common case)
      2. High oscillation → ForceReplan  (stuck-in-cycle case)
      3. Low fidelity → SchemaValidatedRetry  (tool error case)

    Cooldown prevents thrashing: no intervention more often than every
    cooldown_turns turns.
    """
    name: str = field(default="ThresholdController", init=False)
    drift_threshold: float = 0.30
    oscillation_threshold: float = 0.60
    fidelity_threshold: float = 0.70
    cooldown_turns: int = 3

    _last_intervention_turn: int = field(default=-1000, init=False, repr=False)
    _reanchor: GoalReanchor = field(default_factory=GoalReanchor, init=False, repr=False)
    _replan: ForceReplan = field(default_factory=ForceReplan, init=False, repr=False)
    _retry: SchemaValidatedRetry = field(default_factory=SchemaValidatedRetry, init=False, repr=False)

    def _in_cooldown(self, turn: int) -> bool:
        return self._last_intervention_turn > -1000 and (turn - self._last_intervention_turn) < self.cooldown_turns

    def decide(self, signals: MonitorSignals) -> InterventionDecision:
        if self._in_cooldown(signals.turn):
            return InterventionDecision(
                intervention=None,
                rationale=f"cooldown (last at turn {self._last_intervention_turn})",
                confidence=0.9,
            )

        if signals.drift_score > self.drift_threshold:
            self._last_intervention_turn = signals.turn
            conf = min(1.0, signals.drift_score / max(self.drift_threshold, 1e-6))
            return InterventionDecision(
                intervention=self._reanchor,
                rationale=f"drift={signals.drift_score:.3f} > threshold={self.drift_threshold}",
                confidence=conf,
            )

        if signals.oscillation_score > self.oscillation_threshold:
            self._last_intervention_turn = signals.turn
            conf = min(1.0, signals.oscillation_score / max(self.oscillation_threshold, 1e-6))
            return InterventionDecision(
                intervention=self._replan,
                rationale=f"oscillation={signals.oscillation_score:.3f} > threshold={self.oscillation_threshold}",
                confidence=conf,
            )

        if signals.fidelity_score < self.fidelity_threshold:
            self._last_intervention_turn = signals.turn
            conf = min(1.0, (self.fidelity_threshold - signals.fidelity_score) / max(self.fidelity_threshold, 1e-6))
            return InterventionDecision(
                intervention=self._retry,
                rationale=f"fidelity={signals.fidelity_score:.3f} < threshold={self.fidelity_threshold}",
                confidence=conf,
            )

        return InterventionDecision(
            intervention=None,
            rationale="all signals within bounds",
            confidence=0.8,
        )

    def reset(self) -> None:
        self._last_intervention_turn = -1000


@dataclass
class PredictiveController:
    """Fires preemptively based on P(failure within lead_time_k turns).

    Uses a trained FailurePredictor.  Fires when probability exceeds
    fire_at_p.  Same cooldown mechanic as ThresholdController for fair
    comparison.
    """
    name: str = field(default="PredictiveController", init=False)
    predictor: object  # FailurePredictor — typed as object to avoid circular import
    lead_time_k: int = 5
    fire_at_p: float = 0.50
    cooldown_turns: int = 3

    _last_intervention_turn: int = field(default=-1000, init=False, repr=False)
    _signal_history: list = field(default_factory=list, init=False, repr=False)
    _reanchor: GoalReanchor = field(default_factory=GoalReanchor, init=False, repr=False)
    _replan: ForceReplan = field(default_factory=ForceReplan, init=False, repr=False)

    def _in_cooldown(self, turn: int) -> bool:
        return self._last_intervention_turn > -1000 and (turn - self._last_intervention_turn) < self.cooldown_turns

    def decide(self, signals: MonitorSignals) -> InterventionDecision:
        self._signal_history.append(signals)

        if self._in_cooldown(signals.turn):
            return InterventionDecision(
                intervention=None,
                rationale=f"cooldown (last at turn {self._last_intervention_turn})",
                confidence=0.9,
            )

        p_fail = self.predictor.predict_proba(signals, self._signal_history[:-1])

        if p_fail >= self.fire_at_p:
            self._last_intervention_turn = signals.turn
            # Choose intervention based on dominant signal
            if signals.drift_score >= signals.oscillation_score:
                chosen = self._reanchor
                action_desc = "GoalReanchor"
            else:
                chosen = self._replan
                action_desc = "ForceReplan"
            return InterventionDecision(
                intervention=chosen,
                rationale=f"P(fail|k={self.lead_time_k})={p_fail:.3f} >= {self.fire_at_p} → {action_desc}",
                confidence=p_fail,
            )

        return InterventionDecision(
            intervention=None,
            rationale=f"P(fail|k={self.lead_time_k})={p_fail:.3f} < {self.fire_at_p}",
            confidence=1.0 - p_fail,
        )

    def reset(self) -> None:
        self._last_intervention_turn = -1000
        self._signal_history.clear()


# ── Registry ──────────────────────────────────────────────────────────────────

CONTROLLER_NAMES = ["NoControl", "FixedSchedule", "ThresholdController", "PredictiveController"]
