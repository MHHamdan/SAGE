"""Experiment A2 — Progress Monotonicity (Proposition 1 violation, G1).

CLI flags:
  --stall-probs     list of stall probabilities (default sweep)
  --trials          trials per stall prob (default 20)
  --stall-prob      single stall probability (CLI harness mode)
  --seed            base random seed

Output: results/exp_a2.csv
Columns: stall_prob, trial, deadlock_detected, turns_to_detection, task_success
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
sys.path.insert(0, str(ROOT / "src"))  # allow direct import of sage
sys.path.insert(0, str(ROOT))

RESULTS_DIR = ROOT / "results"
RESULTS_DIR.mkdir(exist_ok=True)

# ── scheduling task definition ────────────────────────────────────────────────

SUBTASKS = [
    "parse_requirements",
    "resolve_dependencies",
    "allocate_resources",
    "schedule_slots",
    "validate_conflicts",
    "generate_calendar",
    "notify_participants",
    "commit_schedule",
]


def run_planning_agent(
    stall_prob: float,
    rng: random.Random,
    max_turns: int = 80,
) -> dict[str, Any]:
    """Simulate a planning agent on an 8-step scheduling task.

    At each subtask, with probability stall_prob the precondition is
    marked unmet, forcing a retry.  We track whether a deadlock is
    detected (same subtask retried ≥ 5 consecutive times).
    """
    subtask_idx = 0
    turn = 0
    consecutive_retries = 0
    deadlock_detected = False
    turns_to_detection = max_turns
    retry_history: list[str] = []

    while turn < max_turns and subtask_idx < len(SUBTASKS):
        current = SUBTASKS[subtask_idx]
        retry_history.append(current)
        turn += 1

        # Stall: precondition unmet
        if rng.random() < stall_prob:
            consecutive_retries += 1
            # Detection: ≥ 5 consecutive retries of the same subtask
            if consecutive_retries >= 5 and not deadlock_detected:
                deadlock_detected = True
                turns_to_detection = turn
            # Stay at same subtask
            continue

        # Subtask succeeds
        consecutive_retries = 0
        subtask_idx += 1

    task_success = subtask_idx >= len(SUBTASKS)
    return {
        "deadlock_detected": deadlock_detected,
        "turns_to_detection": turns_to_detection,
        "task_success": task_success,
    }


# ── experiment runner ─────────────────────────────────────────────────────────

def run_experiment(
    stall_probs: list[float] | None = None,
    trials: int = 20,
    base_seed: int = 42,
    stall_prob_single: float | None = None,
) -> list[dict[str, Any]]:
    """Run Experiment A2 across stall probabilities and trials."""
    if stall_prob_single is not None:
        stall_probs = [stall_prob_single]
    if stall_probs is None:
        stall_probs = [0.0, 0.25, 0.5]

    rows: list[dict[str, Any]] = []

    for sp in stall_probs:
        for trial in range(trials):
            seed = base_seed + int(sp * 10000) + trial
            rng = random.Random(seed)
            result = run_planning_agent(stall_prob=sp, rng=rng)
            rows.append({
                "stall_prob": sp,
                "trial": trial,
                "deadlock_detected": result["deadlock_detected"],
                "turns_to_detection": result["turns_to_detection"],
                "task_success": result["task_success"],
            })

    return rows


def write_csv(rows: list[dict[str, Any]]) -> None:
    path = RESULTS_DIR / "exp_a2.csv"
    fieldnames = ["stall_prob", "trial", "deadlock_detected", "turns_to_detection", "task_success"]
    with path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)
    print(f"Wrote {path}")


def print_summary(rows: list[dict[str, Any]]) -> None:
    grouped: dict[float, list] = defaultdict(list)
    for r in rows:
        grouped[r["stall_prob"]].append(r)

    print("\nExperiment A2 — Progress Monotonicity Summary:")
    print(f"  {'stall_prob':>10s}  {'deadlock_rate':>13s}  {'mean_turns_det':>15s}  {'task_sr':>8s}")
    for sp in sorted(grouped):
        g = grouped[sp]
        dr = sum(1 for r in g if r["deadlock_detected"]) / len(g)
        # Only average turns_to_detection for trials where deadlock was detected
        det_turns = [r["turns_to_detection"] for r in g if r["deadlock_detected"]]
        mtd = float(np.mean(det_turns)) if det_turns else float("nan")
        sr = sum(1 for r in g if r["task_success"]) / len(g)
        print(f"  {sp:>10.2f}  {dr:>13.2%}  {mtd:>15.1f}  {sr:>8.2%}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Experiment A2 — Progress Monotonicity")
    parser.add_argument("--stall-probs", nargs="+", type=float, default=None)
    parser.add_argument("--stall-prob", type=float, default=None)
    parser.add_argument("--trials", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    print("Running Experiment A2 — progress monotonicity …")
    rows = run_experiment(
        stall_probs=args.stall_probs,
        trials=args.trials,
        base_seed=args.seed,
        stall_prob_single=args.stall_prob,
    )
    write_csv(rows)
    print_summary(rows)
    print("Done.")


if __name__ == "__main__":
    main()
