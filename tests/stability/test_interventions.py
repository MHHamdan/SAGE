"""Unit tests for the intervention library.

Each test asserts pre/post state invariants from a mocked AgentState.
Covers: all 5 interventions, EscalationRequest, cost models, reversibility.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "src"))

from sage.stability.interventions import (
    AgentState,
    EscalationRequest,
    GoalReanchor,
    ContextCompress,
    ForceReplan,
    SchemaValidatedRetry,
    HumanEscalate,
)


def _unit(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    return v / n if n > 1e-12 else v


def _make_state(drift: float = 0.4) -> tuple[AgentState, np.ndarray]:
    """Return (state, goal_embedding) with specified goal-state angle."""
    rng = np.random.default_rng(0)
    goal = _unit(rng.standard_normal(64))
    # state is rotated 'drift' amount away from goal
    perp = _unit(rng.standard_normal(64) - np.dot(rng.standard_normal(64), goal) * goal)
    # cos(angle) = 1 - 2*drift  (drift in [0,1] maps from cos=1 to cos=-1)
    cos_a = 1.0 - 2.0 * drift
    state = _unit(cos_a * goal + np.sqrt(max(0, 1 - cos_a**2)) * perp)
    ag = AgentState(
        goal_embedding=goal.copy(),
        state_embedding=state.copy(),
        context_turns=[f"turn_{i}" for i in range(20)],
        turn=15,
        cost_so_far=0.3,
        plan=["plan_a", "plan_b"],
        last_tool_output={"result": "some_value"},
        intervention_count=0,
    )
    return ag, goal


def _cos_sim(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.clip(np.dot(_unit(a), _unit(b)), -1.0, 1.0))


class TestGoalReanchor:
    def test_state_moves_toward_goal(self):
        ag, goal = _make_state(drift=0.45)
        before_sim = _cos_sim(ag.state_embedding, goal)
        new_ag = GoalReanchor().apply(ag)
        after_sim = _cos_sim(new_ag.state_embedding, goal)
        assert after_sim > before_sim, "GoalReanchor must increase goal similarity"

    def test_state_is_unit_vector(self):
        ag, _ = _make_state()
        new_ag = GoalReanchor().apply(ag)
        assert abs(np.linalg.norm(new_ag.state_embedding) - 1.0) < 1e-6

    def test_intervention_count_incremented(self):
        ag, _ = _make_state()
        new_ag = GoalReanchor().apply(ag)
        assert new_ag.intervention_count == ag.intervention_count + 1

    def test_original_state_unchanged(self):
        ag, _ = _make_state()
        orig = ag.state_embedding.copy()
        GoalReanchor().apply(ag)
        np.testing.assert_array_equal(ag.state_embedding, orig)

    def test_cost_model(self):
        gr = GoalReanchor()
        assert gr.estimated_cost == pytest.approx(0.01)
        assert gr.reversible is True

    def test_zero_drift_no_op_effect(self):
        ag, goal = _make_state(drift=0.0)
        new_ag = GoalReanchor().apply(ag)
        before_sim = _cos_sim(ag.state_embedding, goal)
        after_sim = _cos_sim(new_ag.state_embedding, goal)
        assert after_sim >= before_sim - 1e-6


class TestContextCompress:
    def test_context_truncated(self):
        ag, _ = _make_state()
        keep_n = 5
        new_ag = ContextCompress(keep_recent_n=keep_n).apply(ag)
        assert len(new_ag.context_turns) == min(keep_n, len(ag.context_turns))

    def test_keeps_recent_turns(self):
        ag, _ = _make_state()
        keep_n = 5
        new_ag = ContextCompress(keep_recent_n=keep_n).apply(ag)
        assert new_ag.context_turns == ag.context_turns[-keep_n:]

    def test_state_nudged_toward_goal(self):
        ag, goal = _make_state(drift=0.45)
        before_sim = _cos_sim(ag.state_embedding, goal)
        new_ag = ContextCompress().apply(ag)
        after_sim = _cos_sim(new_ag.state_embedding, goal)
        assert after_sim >= before_sim - 1e-6

    def test_not_reversible(self):
        assert ContextCompress().reversible is False

    def test_cost_model(self):
        assert ContextCompress().estimated_cost == pytest.approx(0.05)


class TestForceReplan:
    def test_plan_replaced(self):
        ag, _ = _make_state()
        new_ag = ForceReplan().apply(ag)
        assert new_ag.plan != ag.plan
        assert len(new_ag.plan) > 0

    def test_strong_recovery(self):
        ag, goal = _make_state(drift=0.45)
        before_sim = _cos_sim(ag.state_embedding, goal)
        new_ag = ForceReplan().apply(ag)
        after_sim = _cos_sim(new_ag.state_embedding, goal)
        assert after_sim > before_sim, "ForceReplan must improve goal alignment"

    def test_stronger_recovery_than_reanchor(self):
        ag, goal = _make_state(drift=0.45)
        sim_reanchor = _cos_sim(GoalReanchor().apply(ag).state_embedding, goal)
        sim_replan = _cos_sim(ForceReplan().apply(ag).state_embedding, goal)
        assert sim_replan >= sim_reanchor, "ForceReplan has larger recovery_pull than GoalReanchor"

    def test_reversible(self):
        assert ForceReplan().reversible is True

    def test_cost_model(self):
        assert ForceReplan().estimated_cost == pytest.approx(0.06)


class TestSchemaValidatedRetry:
    def test_tool_output_updated(self):
        ag, _ = _make_state()
        new_ag = SchemaValidatedRetry().apply(ag)
        assert new_ag.last_tool_output is not None
        assert new_ag.last_tool_output.get("valid") is True

    def test_small_recovery(self):
        ag, goal = _make_state(drift=0.45)
        before_sim = _cos_sim(ag.state_embedding, goal)
        new_ag = SchemaValidatedRetry().apply(ag)
        after_sim = _cos_sim(new_ag.state_embedding, goal)
        assert after_sim >= before_sim - 1e-6

    def test_cost_model(self):
        sr = SchemaValidatedRetry()
        assert sr.estimated_cost == pytest.approx(0.01)
        assert sr.reversible is True


class TestHumanEscalate:
    def test_raises_escalation_request(self):
        ag, _ = _make_state()
        with pytest.raises(EscalationRequest):
            HumanEscalate().apply(ag)

    def test_zero_cost(self):
        he = HumanEscalate()
        assert he.estimated_cost == 0.0

    def test_name(self):
        assert HumanEscalate().name == "HumanEscalate"


class TestCostConvention:
    """Verify the double-counting convention stated in the module docstring."""

    def test_all_interventions_have_cost_attr(self):
        for cls in [GoalReanchor, ContextCompress, ForceReplan, SchemaValidatedRetry, HumanEscalate]:
            obj = cls()
            assert isinstance(obj.estimated_cost, float)
            assert obj.estimated_cost >= 0.0

    def test_all_interventions_have_name_attr(self):
        expected = {"GoalReanchor", "ContextCompress", "ForceReplan",
                    "SchemaValidatedRetry", "HumanEscalate"}
        found = {cls().name for cls in [GoalReanchor, ContextCompress, ForceReplan,
                                         SchemaValidatedRetry, HumanEscalate]}
        assert found == expected
