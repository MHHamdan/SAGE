"""Experiment A1 — Observation Fidelity (Proposition 1 violation, G1).

CLI flags:
  --injection-rate  fraction of tool responses that are corrupted (default sweep)
  --trials          trials per injection rate (default 20)
  --seed            base random seed

Output: results/exp_a1.csv
Columns: injection_rate, trial, success, steps_to_failure, oscillation_detected
"""

from __future__ import annotations

import argparse
import csv
import random
import sys
import time
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

# ── synthetic Python file for the file-editing task ──────────────────────────

PYTHON_SOURCE = """\
def process_item(items):
    return [x * 2 for x in items]

def validate_input(data):
    return data is not None

def transform_output(result):
    return str(result)
"""

TARGET_FUNCTIONS = ["process_item", "validate_input", "transform_output"]


# ── minimal ReAct-style agent harness ────────────────────────────────────────

class FileEditingTask:
    """Simulates a file-editing task with optional observation corruption."""

    def __init__(self, injection_rate: float, rng: random.Random) -> None:
        self.injection_rate = injection_rate
        self.rng = rng
        self.source = PYTHON_SOURCE
        self.edits_applied: list[str] = []
        self.oscillation_actions: list[str] = []
        self._done = False

    def read_file(self) -> dict[str, Any]:
        """Tool: read file content. May be corrupted."""
        if self.rng.random() < self.injection_rate:
            # Inject corrupted observation (random JSON schema violation)
            return {
                "error": "corrupt",
                "content": self.rng.choice([
                    '{"__type__": "undefined"}',
                    "null\x00\xff",
                    "{malformed json",
                    "",
                ]),
            }
        return {"content": self.source, "error": None}

    def write_function(self, func_name: str, new_body: str) -> dict[str, Any]:
        """Tool: rewrite a function. May be corrupted."""
        if self.rng.random() < self.injection_rate:
            return {"error": "write_failed", "content": None}
        if func_name in TARGET_FUNCTIONS:
            self.edits_applied.append(func_name)
        return {"error": None, "written": func_name}

    def verify(self) -> dict[str, Any]:
        """Tool: run tests (always uncorrupted — final check)."""
        success = set(self.edits_applied) >= set(TARGET_FUNCTIONS)
        return {"tests_passed": success, "edited": list(self.edits_applied)}


def run_react_agent(
    task: FileEditingTask,
    max_turns: int = 30,
) -> dict[str, Any]:
    """Minimal ReAct agent loop for file-editing task."""
    PLAN = [
        ("read_file", {}),
        ("write_function", {"func_name": "process_item", "new_body": "return items or []"}),
        ("write_function", {"func_name": "validate_input", "new_body": "return bool(data)"}),
        ("write_function", {"func_name": "transform_output", "new_body": "return repr(result)"}),
        ("verify", {}),
    ]

    step = 0
    plan_idx = 0
    consecutive_errors = 0
    oscillation_window: list[str] = []
    oscillation_detected = False
    steps_to_failure = max_turns

    while step < max_turns and plan_idx < len(PLAN):
        action_name, action_args = PLAN[plan_idx]
        oscillation_window.append(action_name)

        # Detect oscillation: same action repeated ≥ 3 times in last 6 steps
        if len(oscillation_window) >= 6:
            recent = oscillation_window[-6:]
            if recent.count(action_name) >= 3:
                oscillation_detected = True

        # Execute action
        if action_name == "read_file":
            obs = task.read_file()
        elif action_name == "write_function":
            obs = task.write_function(**action_args)
        else:
            obs = task.verify()

        step += 1

        if obs.get("error"):
            consecutive_errors += 1
            # Agent retries same step up to 3 times, then gives up
            if consecutive_errors >= 3:
                steps_to_failure = step
                return {
                    "success": False,
                    "steps_to_failure": steps_to_failure,
                    "oscillation_detected": oscillation_detected,
                    "turns_used": step,
                }
            # Stay at same plan index (retry)
            continue
        else:
            consecutive_errors = 0
            plan_idx += 1

    verify_obs = task.verify()
    success = bool(verify_obs.get("tests_passed"))
    return {
        "success": success,
        "steps_to_failure": step if not success else max_turns,
        "oscillation_detected": oscillation_detected,
        "turns_used": step,
    }


# ── experiment runner ─────────────────────────────────────────────────────────

def run_experiment(
    injection_rates: list[float] | None = None,
    trials: int = 20,
    base_seed: int = 42,
    inject_obs_error: float | None = None,
) -> list[dict[str, Any]]:
    """Run Experiment A1 across injection rates and trials."""
    if inject_obs_error is not None:
        injection_rates = [inject_obs_error]
    if injection_rates is None:
        injection_rates = [0.0, 0.1, 0.2, 0.4]

    rows: list[dict[str, Any]] = []

    for rate in injection_rates:
        for trial in range(trials):
            seed = base_seed + int(rate * 1000) + trial
            rng = random.Random(seed)
            task = FileEditingTask(injection_rate=rate, rng=rng)
            result = run_react_agent(task)
            rows.append({
                "injection_rate": rate,
                "trial": trial,
                "success": result["success"],
                "steps_to_failure": result["steps_to_failure"],
                "oscillation_detected": result["oscillation_detected"],
            })

    return rows


def write_csv(rows: list[dict[str, Any]]) -> None:
    path = RESULTS_DIR / "exp_a1.csv"
    fieldnames = ["injection_rate", "trial", "success", "steps_to_failure", "oscillation_detected"]
    with path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)
    print(f"Wrote {path}")


def print_summary(rows: list[dict[str, Any]]) -> None:
    from collections import defaultdict
    grouped: dict[float, list] = defaultdict(list)
    for r in rows:
        grouped[r["injection_rate"]].append(r)

    print("\nExperiment A1 — Observation Fidelity Summary:")
    print(f"  {'injection_rate':>16s}  {'success_rate':>12s}  {'mean_steps_fail':>16s}  {'osc_rate':>9s}")
    for rate in sorted(grouped):
        g = grouped[rate]
        sr = sum(1 for r in g if r["success"]) / len(g)
        msf = float(np.mean([r["steps_to_failure"] for r in g]))
        osc = sum(1 for r in g if r["oscillation_detected"]) / len(g)
        print(f"  {rate:>16.2f}  {sr:>12.2%}  {msf:>16.1f}  {osc:>9.2%}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Experiment A1 — Observation Fidelity")
    parser.add_argument("--injection-rates", nargs="+", type=float, default=None)
    parser.add_argument("--inject-obs-error", type=float, default=None)
    parser.add_argument("--trials", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    print("Running Experiment A1 — observation fidelity …")
    rows = run_experiment(
        injection_rates=args.injection_rates,
        trials=args.trials,
        base_seed=args.seed,
        inject_obs_error=args.inject_obs_error,
    )
    write_csv(rows)
    print_summary(rows)
    print("Done.")


if __name__ == "__main__":
    main()
