"""E4 threshold-sensitivity surface (Supplementary Material H.8).

Re-runs the E4 THRESHOLD policy offline (fully simulated, seed-locked — no live
calls) over the grid:

    k = re-anchor interval  ∈ {5, 8, 10, 12, 15}   (mapped to cooldown_turns)
    B = oscillation bound   ∈ {3, 4, 5, 6, 7}       (max repeats of any single
                                                     action in the last window
                                                     before ForceReplan fires)

See AUDIT.md FLAG B: the ThresholdController has no native "re-anchor interval"
or integer "oscillation bound"; this is the faithful mapping onto its
`cooldown_turns` (re-anchor cadence) and a count-based oscillation trigger that
replaces the float `oscillation_threshold`. Drift/fidelity thresholds are held
at the E4 defaults (0.30 / 0.70). Axes are labelled with the concrete knob used.

Reports the completion-rate surface (5x5, 50 tasks x 3 seeds per cell) and how
far it moves from the committed E4 ThresholdController completion.

Output: results/e4_threshold_surface/{e4_threshold_surface.csv,
        e4_threshold_surface.pdf, manifest.json}
"""
from __future__ import annotations

import csv
import json
import sys
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from experiments.e4_closed_loop import (  # noqa: E402
    E4_CONFIG, _unit, goal_drift_score, make_goal_embedding, drift_step,
    _oscillation_score, _fidelity_score,
)
from sage.stability import MonitorSignals, AgentState, ThresholdController  # noqa: E402
from sage.stability.traces import _get_git_sha, _get_env_hash, now_iso  # noqa: E402

OUT = ROOT / "results" / "e4_threshold_surface"
OUT.mkdir(parents=True, exist_ok=True)

K_GRID = [5, 8, 10, 12, 15]      # re-anchor interval → cooldown_turns
B_GRID = [3, 4, 5, 6, 7]         # oscillation bound → oscillation_threshold = B/5
WINDOW = 10                      # matches _oscillation_score half-window (=5)


def b_to_threshold(B: int) -> float:
    """Map integer oscillation bound B to E4's float oscillation_threshold.

    _oscillation_score = overlap / 5 (half-window). B recurring actions => score
    B/5; the committed E4 config oscillation_threshold=0.60 corresponds to B=3
    (3-of-5 overlap), reconciling Table II's B=3. B in {6,7} => threshold>1.0,
    i.e., the oscillation trigger is effectively disabled (drift/fidelity only).
    """
    return B / 5.0


def run_cell(k: int, B: int, cfg: dict, base_seed: int) -> dict:
    """One (k,B) cell using the ACTUAL sage ThresholdController (faithful to E4)."""
    dim, total_turns = cfg["embedding_dim"], cfg["total_turns"]
    drift_rate, base_cost = cfg["drift_rate_base"], cfg["base_cost_per_turn"]
    max_int = cfg["max_interventions_per_task"]
    successes, costs, n_ints = [], [], []

    for seed_offset in range(cfg["n_seeds"]):
        seed = base_seed + seed_offset * 1000
        for task_idx in range(cfg["n_tasks"]):
            task_seed = base_seed + task_idx
            episode_seed = seed + task_idx * 13 + seed_offset * 97
            rng = np.random.default_rng(episode_seed)
            goal = make_goal_embedding(task_seed, dim)
            state = _unit(goal + 0.1 * rng.standard_normal(dim))
            ctl = ThresholdController(
                drift_threshold=cfg["drift_threshold"],
                oscillation_threshold=b_to_threshold(B),
                fidelity_threshold=cfg["fidelity_threshold"],
                cooldown_turns=k)
            ctl.reset()
            action_history: list[str] = []
            cumulative_cost = 0.0
            intervention_count = 0
            best_sim = float(np.clip(np.dot(_unit(goal), _unit(state)), -1.0, 1.0))
            final_drift = 0.0

            for turn in range(1, total_turns + 1):
                action = f"action_{int(rng.integers(0, 8))}"
                action_history.append(action)
                state = drift_step(state, goal, drift_rate, rng)
                drift = goal_drift_score(goal, state)
                osc = _oscillation_score(action_history)
                fid = _fidelity_score(rng)
                sim = 1.0 - drift * 2
                best_sim = max(best_sim, sim)
                signals = MonitorSignals(
                    drift_score=min(1.0, max(0.0, drift)),
                    oscillation_score=min(1.0, max(0.0, osc)),
                    fidelity_score=min(1.0, max(0.0, fid)),
                    convergence_progress=min(1.0, max(0.0, (best_sim + 1.0) / 2.0)),
                    turn=turn, cost_so_far=cumulative_cost)
                decision = ctl.decide(signals)
                intervention = decision.intervention
                intervention_cost = 0.0
                if intervention is not None and intervention_count < max_int:
                    ag = AgentState(goal_embedding=goal, state_embedding=state,
                                    context_turns=action_history.copy(), turn=turn,
                                    cost_so_far=cumulative_cost, plan=[],
                                    last_tool_output=None,
                                    intervention_count=intervention_count)
                    new_ag = intervention.apply(ag)
                    state = new_ag.state_embedding.copy()
                    intervention_cost = intervention.estimated_cost
                    intervention_count += 1
                cumulative_cost += base_cost + intervention_cost
                final_drift = drift

            successes.append(final_drift < cfg["completion_drift_threshold"])
            costs.append(cumulative_cost)
            n_ints.append(intervention_count)

    comp = float(np.mean(successes))
    return {"k": k, "B": B, "completion_rate": comp,
            "mean_cost": float(np.mean(costs)),
            "cnsr": comp / max(float(np.mean(costs)), 1e-9),
            "mean_interventions": float(np.mean(n_ints)),
            "n_eval": len(successes)}


def run(base_seed: int = 42) -> None:
    cfg = dict(E4_CONFIG)
    print(f"[0.5] sweeping {len(K_GRID)}x{len(B_GRID)} threshold surface "
          f"({cfg['n_tasks']}x{cfg['n_seeds']} eval/cell) …")
    rows = []
    surface = np.zeros((len(K_GRID), len(B_GRID)))
    for ik, k in enumerate(K_GRID):
        for ib, B in enumerate(B_GRID):
            r = run_cell(k, B, cfg, base_seed)
            rows.append(r)
            surface[ik, ib] = r["completion_rate"]
            print(f"    k={k:>2} B={B}  completion={r['completion_rate']:.3f} "
                  f"cost=${r['mean_cost']:.3f} interv={r['mean_interventions']:.1f}")

    with (OUT / "e4_threshold_surface.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)

    comp_min, comp_max = surface.min(), surface.max()
    comp_mean = float(surface.mean())

    # heatmap
    fig, ax = plt.subplots(figsize=(6.5, 5))
    im = ax.imshow(surface, cmap="viridis", aspect="auto", origin="lower")
    ax.set_xticks(range(len(B_GRID))); ax.set_xticklabels(B_GRID)
    ax.set_yticks(range(len(K_GRID))); ax.set_yticklabels(K_GRID)
    ax.set_xlabel("oscillation bound B (oscillation_threshold = B/5)")
    ax.set_ylabel("re-anchor interval k (cooldown turns)")
    ax.set_title("E4 THRESHOLD policy — completion-rate surface")
    for ik in range(len(K_GRID)):
        for ib in range(len(B_GRID)):
            ax.text(ib, ik, f"{surface[ik, ib]:.2f}", ha="center", va="center",
                    color="white" if surface[ik, ib] < (comp_min + comp_max) / 2 else "black",
                    fontsize=9)
    fig.colorbar(im, ax=ax, label="task completion rate")
    fig.tight_layout(); fig.savefig(OUT / "e4_threshold_surface.pdf", bbox_inches="tight")
    plt.close(fig)

    manifest = {
        "experiment": "0.5_e4_threshold_surface", "git_sha": _get_git_sha(),
        "env_hash": _get_env_hash(), "timestamp": now_iso(), "base_seed": base_seed,
        "k_grid": K_GRID, "B_grid": B_GRID, "window": WINDOW,
        "eval_per_cell": cfg["n_tasks"] * cfg["n_seeds"],
        "completion_min": round(float(comp_min), 4),
        "completion_max": round(float(comp_max), 4),
        "completion_mean": round(comp_mean, 4),
        "completion_range": round(float(comp_max - comp_min), 4),
        "config": {kk: cfg[kk] for kk in ("n_tasks", "n_seeds", "total_turns",
                                          "drift_threshold", "fidelity_threshold",
                                          "completion_drift_threshold")},
        "mapping_note": "Faithful reconstruction using the actual sage "
                        "ThresholdController. k -> cooldown_turns; "
                        "B -> oscillation_threshold = B/5 (E4 committed osc_thr=0.60 "
                        "<=> B=3). B in {6,7} => threshold>1 => oscillation trigger "
                        "disabled. The committed E4 THRESHOLD cell is (k=3, B=3), "
                        "which is below the swept k-grid; reported separately.",
        "as_run_committed_cell": {"cooldown_turns": 3, "oscillation_threshold": 0.60,
                                  "B_equiv": 3, "committed_completion": 0.8933},
    }
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"\n[0.5] completion surface: min={comp_min:.3f} max={comp_max:.3f} "
          f"mean={comp_mean:.3f} range={comp_max-comp_min:.3f}")
    print(f"      Wrote {OUT}/")


if __name__ == "__main__":
    run()
