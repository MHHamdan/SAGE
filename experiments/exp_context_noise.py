"""Experiment A3 — Context Noise / Goal Drift (Proposition 1 violation, G1).

CLI flags:
  --reanchor-intervals  list of re-anchoring intervals (default sweep)
  --trials              trials per interval (default 10)
  --reanchor-interval   single interval value (CLI harness mode; None = no re-anchoring)
  --seed                base random seed

Output: results/exp_a3.csv
Columns: reanchor_interval, trial, turn, drift_score, task_completed
"""

from __future__ import annotations

import argparse
import csv
import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
# NOTE: a previous version inserted "agentic_ai_toolkit/src" here, but no
# such directory exists in this repo — the package lives at src/sage/.
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

RESULTS_DIR = ROOT / "results"
RESULTS_DIR.mkdir(exist_ok=True)

EMBEDDING_DIM = 64  # lightweight embedding dimension for experiments


# ── embedding helpers ─────────────────────────────────────────────────────────

def _unit(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    return v / n if n > 1e-12 else v


def goal_drift_score(goal_emb: np.ndarray, state_emb: np.ndarray) -> float:
    """Cosine-distance-based drift: 0 = aligned, 1 = orthogonal."""
    g = _unit(goal_emb.astype(float))
    s = _unit(state_emb.astype(float))
    cosine_sim = float(np.clip(np.dot(g, s), -1.0, 1.0))
    return float((1.0 - cosine_sim) / 2.0)


def make_goal_embedding(seed: int) -> np.ndarray:
    rng = np.random.RandomState(seed)
    return _unit(rng.randn(EMBEDDING_DIM))


def drift_step(
    current: np.ndarray,
    goal: np.ndarray,
    drift_rate: float,
    rng: random.Random,
) -> np.ndarray:
    """Randomly perturb current state embedding away from goal."""
    noise_scale = drift_rate * rng.gauss(1.0, 0.2)
    noise = np.array([rng.gauss(0, noise_scale) for _ in range(EMBEDDING_DIM)])
    new_state = current + noise
    return _unit(new_state)


# ── research-agent simulation ─────────────────────────────────────────────────

DRIFT_RATE_BASE = 0.06   # per turn without re-anchoring
RECOVERY_RATE = 0.4      # how much re-anchoring pulls state back toward goal


def run_research_agent(
    reanchor_interval: int | None,
    rng: random.Random,
    total_turns: int = 50,
    goal_seed: int = 0,
) -> list[dict[str, float | bool | int]]:
    """Run research agent for total_turns, optionally re-anchoring at interval."""
    goal_emb = make_goal_embedding(goal_seed)
    np.random.seed(goal_seed)
    state_emb = _unit(goal_emb + 0.1 * np.random.randn(EMBEDDING_DIM))

    records: list[dict[str, float | bool | int]] = []

    for turn in range(1, total_turns + 1):
        # Re-anchor: re-inject original goal embedding into state
        if reanchor_interval is not None and turn % reanchor_interval == 0:
            state_emb = _unit(
                (1 - RECOVERY_RATE) * state_emb + RECOVERY_RATE * goal_emb
            )

        # Drift step
        state_emb = drift_step(state_emb, goal_emb, DRIFT_RATE_BASE, rng)

        drift = goal_drift_score(goal_emb, state_emb)
        records.append({"turn": turn, "drift_score": float(drift)})

    # Task is considered completed if drift at turn 50 < 0.35
    final_drift = records[-1]["drift_score"]
    task_completed = final_drift < 0.35
    for rec in records:
        rec["task_completed"] = task_completed

    return records


# ── experiment runner ─────────────────────────────────────────────────────────

def run_experiment(
    reanchor_intervals: list[int | None] | None = None,
    trials: int = 10,
    base_seed: int = 42,
    reanchor_interval_single: int | None = -1,  # sentinel meaning "not set"
) -> list[dict[str, Any]]:
    """Run Experiment A3 across re-anchor intervals and trials."""
    if reanchor_interval_single != -1:
        reanchor_intervals = [reanchor_interval_single]
    if reanchor_intervals is None:
        reanchor_intervals = [5, 10, 20, None]

    rows: list[dict[str, Any]] = []

    for interval in reanchor_intervals:
        for trial in range(trials):
            seed = base_seed + (interval or 0) * 100 + trial
            rng = random.Random(seed)
            goal_seed = base_seed + trial
            records = run_research_agent(
                reanchor_interval=interval,
                rng=rng,
                goal_seed=goal_seed,
            )
            for rec in records:
                rows.append({
                    "reanchor_interval": interval if interval is not None else "None",
                    "trial": trial,
                    "turn": rec["turn"],
                    "drift_score": round(rec["drift_score"], 6),
                    "task_completed": rec["task_completed"],
                })

    return rows


def write_csv(rows: list[dict[str, Any]]) -> None:
    path = RESULTS_DIR / "exp_a3.csv"
    fieldnames = ["reanchor_interval", "trial", "turn", "drift_score", "task_completed"]
    with path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)
    print(f"Wrote {path}")


def print_summary(rows: list[dict[str, Any]]) -> None:
    # Group by (reanchor_interval, trial) — report turn-50 stats
    final: dict[Any, list] = defaultdict(list)
    for r in rows:
        if r["turn"] == 50:
            final[r["reanchor_interval"]].append(r)

    print("\nExperiment A3 — Context Noise / Goal Drift (at turn 50):")
    print(f"  {'reanchor':>10s}  {'mean_drift':>11s}  {'completion_rate':>15s}")
    for interval in sorted(final.keys(), key=lambda x: x if x != "None" else 999):
        g = final[interval]
        mean_drift = float(np.mean([r["drift_score"] for r in g]))
        compl = sum(1 for r in g if r["task_completed"]) / len(g)
        print(f"  {str(interval):>10s}  {mean_drift:>11.4f}  {compl:>15.2%}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Experiment A3 — Context Noise / Goal Drift")
    parser.add_argument("--reanchor-intervals", nargs="+", type=int, default=None)
    parser.add_argument("--reanchor-interval", type=int, default=None,
                        help="Single interval value; omit for no re-anchoring")
    parser.add_argument("--no-reanchor", action="store_true",
                        help="Run with reanchor_interval=None")
    parser.add_argument("--trials", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    # Determine single-interval sentinel
    single = -1  # not set
    if args.no_reanchor:
        single = None
    elif args.reanchor_interval is not None:
        single = args.reanchor_interval

    print("Running Experiment A3 — context noise / goal drift …")
    rows = run_experiment(
        reanchor_intervals=args.reanchor_intervals,
        trials=args.trials,
        base_seed=args.seed,
        reanchor_interval_single=single,
    )
    write_csv(rows)
    print_summary(rows)
    print("Done.")


if __name__ == "__main__":
    main()
