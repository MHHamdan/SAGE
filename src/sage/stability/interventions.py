"""Adaptive Stability Controller — Intervention library.

Five composable interventions.  Each is a pure function from AgentState →
AgentState; cost accounting lives here, not in the agent loop (to prevent
double-counting — see cost convention at bottom of file).

Conventions
-----------
* Interventions own their own cost; the agent loop must NOT re-charge the
  cost of applying an intervention.
* apply() is deterministic given the same AgentState.  Any stochasticity
  must be seeded externally via AgentState.rng_state.
* HumanEscalate is the only intervention that does not return an AgentState;
  it raises EscalationRequest instead.
"""

from __future__ import annotations

import copy
import hashlib
import logging
import time
from dataclasses import dataclass, field
from typing import Optional, Protocol, runtime_checkable

import numpy as np

logger = logging.getLogger(__name__)


# ── Agent state ────────────────────────────────────────────────────────────────

@dataclass
class AgentState:
    """Simulation state for one agent episode.

    numpy arrays are stored by reference; interventions must copy before
    modifying to preserve pure-function semantics.
    """
    goal_embedding: np.ndarray
    state_embedding: np.ndarray
    context_turns: list[str]
    turn: int
    cost_so_far: float
    plan: list[str]
    last_tool_output: Optional[dict]
    intervention_count: int = 0
    max_interventions: int = 10

    def replace(self, **kwargs) -> "AgentState":
        d = {
            "goal_embedding": self.goal_embedding.copy(),
            "state_embedding": self.state_embedding.copy(),
            "context_turns": list(self.context_turns),
            "turn": self.turn,
            "cost_so_far": self.cost_so_far,
            "plan": list(self.plan),
            "last_tool_output": copy.copy(self.last_tool_output),
            "intervention_count": self.intervention_count,
            "max_interventions": self.max_interventions,
        }
        d.update(kwargs)
        return AgentState(**d)


class EscalationRequest(Exception):
    """Raised by HumanEscalate; ends the agent loop."""


# ── Intervention protocol ──────────────────────────────────────────────────────

@runtime_checkable
class Intervention(Protocol):
    name: str
    estimated_cost: float
    reversible: bool

    def apply(self, state: AgentState) -> AgentState: ...


def _unit(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    return v / n if n > 1e-12 else v


def _log_intervention(name: str, pre_hash: str, post_hash: str, rationale: str) -> None:
    logger.debug(
        "Intervention %s applied | pre=%s post=%s | %s",
        name, pre_hash[:8], post_hash[:8], rationale,
    )


def _embed_hash(emb: np.ndarray) -> str:
    return hashlib.md5(emb.tobytes()).hexdigest()


# ── Concrete interventions ─────────────────────────────────────────────────────

@dataclass
class GoalReanchor:
    """Re-inject original goal embedding into context with recovery_pull fraction.

    Reversible: yes (the original state could be restored).
    Cost: one small LLM call ($0.01).
    """
    name: str = field(default="GoalReanchor", init=False)
    estimated_cost: float = field(default=0.01, init=False)
    reversible: bool = field(default=True, init=False)
    recovery_pull: float = 0.40

    def apply(self, state: AgentState) -> AgentState:
        pre = _embed_hash(state.state_embedding)
        new_emb = _unit(
            (1.0 - self.recovery_pull) * state.state_embedding
            + self.recovery_pull * state.goal_embedding
        )
        new_state = state.replace(
            state_embedding=new_emb,
            intervention_count=state.intervention_count + 1,
        )
        _log_intervention(self.name, pre, _embed_hash(new_emb), "goal re-anchor")
        return new_state


@dataclass
class ContextCompress:
    """Summarize turns older than keep_recent_n; truncates context window.

    Reversible: no (context tokens are discarded).
    Cost: one medium LLM summarization call ($0.05).

    Simulation effect: slight nudge toward goal (compressed context = less
    accumulated drift noise).
    """
    name: str = field(default="ContextCompress", init=False)
    estimated_cost: float = field(default=0.05, init=False)
    reversible: bool = field(default=False, init=False)
    keep_recent_n: int = 10
    compress_recovery: float = 0.12  # small nudge from discarding noisy context

    def apply(self, state: AgentState) -> AgentState:
        pre = _embed_hash(state.state_embedding)
        truncated = state.context_turns[-self.keep_recent_n:]
        # Small recovery from clearing old noisy context
        new_emb = _unit(
            (1.0 - self.compress_recovery) * state.state_embedding
            + self.compress_recovery * state.goal_embedding
        )
        new_state = state.replace(
            context_turns=truncated,
            state_embedding=new_emb,
            intervention_count=state.intervention_count + 1,
        )
        _log_intervention(self.name, pre, _embed_hash(new_emb), f"compressed to last {self.keep_recent_n} turns")
        return new_state


@dataclass
class ForceReplan:
    """Discard current plan; regenerate from current state.

    Reversible: yes (the old plan could be restored; new plan may diverge).
    Cost: one medium LLM call + plan generation ($0.06).

    Simulation effect: aggressive recovery toward goal (replanning resets
    direction).
    """
    name: str = field(default="ForceReplan", init=False)
    estimated_cost: float = field(default=0.06, init=False)
    reversible: bool = field(default=True, init=False)
    replan_recovery: float = 0.55

    def apply(self, state: AgentState) -> AgentState:
        pre = _embed_hash(state.state_embedding)
        new_emb = _unit(
            (1.0 - self.replan_recovery) * state.state_embedding
            + self.replan_recovery * state.goal_embedding
        )
        new_plan = [f"replan_step_{i}" for i in range(5)]
        new_state = state.replace(
            state_embedding=new_emb,
            plan=new_plan,
            intervention_count=state.intervention_count + 1,
        )
        _log_intervention(self.name, pre, _embed_hash(new_emb), "force replan")
        return new_state


@dataclass
class SchemaValidatedRetry:
    """Re-validate last tool output; retry with corrected args if invalid.

    Reversible: yes (original call can be replayed with corrected args).
    Cost: tool call cost × 2 ($0.01).

    Simulation effect: very small recovery (tool correction = minor drift fix).
    """
    name: str = field(default="SchemaValidatedRetry", init=False)
    estimated_cost: float = field(default=0.01, init=False)
    reversible: bool = field(default=True, init=False)
    retry_recovery: float = 0.05

    def apply(self, state: AgentState) -> AgentState:
        pre = _embed_hash(state.state_embedding)
        new_emb = _unit(
            (1.0 - self.retry_recovery) * state.state_embedding
            + self.retry_recovery * state.goal_embedding
        )
        new_tool_output = {"status": "retried", "valid": True}
        new_state = state.replace(
            state_embedding=new_emb,
            last_tool_output=new_tool_output,
            intervention_count=state.intervention_count + 1,
        )
        _log_intervention(self.name, pre, _embed_hash(new_emb), "schema-validated retry")
        return new_state


@dataclass
class HumanEscalate:
    """Raise EscalationRequest; ends the agent loop (handoff to human).

    Reversible: N/A (execution ends).
    Cost: $0 (handoff only; no LLM call).
    """
    name: str = field(default="HumanEscalate", init=False)
    estimated_cost: float = field(default=0.0, init=False)
    reversible: bool = field(default=False, init=False)

    def apply(self, state: AgentState) -> AgentState:
        logger.warning("HumanEscalate triggered at turn %d", state.turn)
        raise EscalationRequest(
            f"Human escalation requested at turn {state.turn} "
            f"(drift={state.cost_so_far:.3f})"
        )


# ── Registry ──────────────────────────────────────────────────────────────────

ALL_INTERVENTIONS: dict[str, type] = {
    "GoalReanchor": GoalReanchor,
    "ContextCompress": ContextCompress,
    "ForceReplan": ForceReplan,
    "SchemaValidatedRetry": SchemaValidatedRetry,
    "HumanEscalate": HumanEscalate,
}

# Cost convention: interventions own their cost.
# The agent loop charges base_cost_per_turn for each turn ONLY.
# When apply() is called, add intervention.estimated_cost to cumulative_cost
# BEFORE the next turn's base_cost is added.  Do not charge it again.
