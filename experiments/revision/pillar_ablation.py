"""Phase 2.1 — Per-pillar ablation (Supp B.7, main §VIII ABL \\todonum).

On the research/long-horizon family (E4 closed-loop machinery, 5 seeds), compares
Full-SAGE against four leave-one-out variants of the SAGE control loop:

  Full-SAGE   monitors on; cost-aware selection; taxonomy->intervention routing;
              interventions enforced.
  -Stabilize  monitors OFF -> controller receives null/nominal signals (blind);
              the predictor never sees the true drift, so it cannot fire.
  -Assess     no cost-aware selection -> the costlier intervention is chosen for
              the same pathology (drift: ForceReplan $0.06 instead of the
              cost-aware GoalReanchor $0.01).
  -Govern     no taxonomy->monitor mapping -> a generic, mis-routed intervention
              (SchemaValidatedRetry, recovery 0.05) is applied regardless of the
              detected pathology.
  -Enforce    interventions OFF = open-loop -> the controller decides but nothing
              is applied (equivalent to NoControl).

Reports completion + CNSR with bootstrap 95% CIs and the Δ-completion attributable
to each pillar (Full − variant). Deterministic; no live calls.

Output: results/pillar_ablation/{pillar_ablation.csv, pillar_ablation_table.tex,
        manifest.json}
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from experiments.e4_closed_loop import (  # noqa: E402
    E4_CONFIG, _unit, goal_drift_score, make_goal_embedding, drift_step,
    _oscillation_score, _fidelity_score, pretrain_predictor,
    bootstrap_proportion_ci, cohens_h,
)
from sage.stability import (  # noqa: E402
    MonitorSignals, AgentState, GoalReanchor, ForceReplan, ContextCompress,
    SchemaValidatedRetry,
)
from sage.stability.traces import _get_git_sha, _get_env_hash, now_iso  # noqa: E402

OUT = ROOT / "results" / "pillar_ablation"
OUT.mkdir(parents=True, exist_ok=True)

N_SEEDS = 5
FIRE_AT_P = 0.50
COOLDOWN = 3
LEAD_K = 5

VARIANTS = ["Full-SAGE", "-Stabilize", "-Assess", "-Govern", "-Enforce"]

# null signal fed to a blinded controller (monitors off)
NULL_SIGNAL = dict(drift_score=0.0, oscillation_score=0.0, fidelity_score=1.0,
                   convergence_progress=1.0)


def select_intervention(signals: MonitorSignals, govern: bool, assess: bool):
    """Taxonomy->intervention routing (Govern) + cost-aware choice (Assess)."""
    if not govern:
        return SchemaValidatedRetry()  # mis-routed generic (weak recovery 0.05)
    if signals.drift_score >= signals.oscillation_score:
        # drift pathology: cost-aware = cheap GoalReanchor; cost-blind = ForceReplan
        return GoalReanchor() if assess else ForceReplan()
    # oscillation pathology
    return ForceReplan() if assess else ContextCompress()


def run_episode(task_seed, episode_seed, cfg, predictor, variant) -> dict:
    stabilize = variant != "-Stabilize"
    assess = variant != "-Assess"
    govern = variant != "-Govern"
    enforce = variant != "-Enforce"

    dim, total_turns = cfg["embedding_dim"], cfg["total_turns"]
    drift_rate, base_cost = cfg["drift_rate_base"], cfg["base_cost_per_turn"]
    max_int = cfg["max_interventions_per_task"]

    rng = np.random.default_rng(episode_seed)
    goal = make_goal_embedding(task_seed, dim)
    state = _unit(goal + 0.1 * rng.standard_normal(dim))
    action_history: list[str] = []
    signal_history: list[MonitorSignals] = []
    cumulative_cost = 0.0
    intervention_count = 0
    last_int_turn = -1000
    best_sim = float(np.clip(np.dot(_unit(goal), _unit(state)), -1.0, 1.0))
    final_drift = 0.0

    for turn in range(1, total_turns + 1):
        action_history.append(f"action_{int(rng.integers(0, 8))}")
        state = drift_step(state, goal, drift_rate, rng)
        drift = goal_drift_score(goal, state)
        osc = _oscillation_score(action_history)
        fid = _fidelity_score(rng)
        sim = 1.0 - drift * 2
        best_sim = max(best_sim, sim)

        true_sig = MonitorSignals(
            drift_score=min(1.0, max(0.0, drift)),
            oscillation_score=min(1.0, max(0.0, osc)),
            fidelity_score=min(1.0, max(0.0, fid)),
            convergence_progress=min(1.0, max(0.0, (best_sim + 1.0) / 2.0)),
            turn=turn, cost_so_far=cumulative_cost)

        # Stabilize pillar: what the controller actually sees
        if stabilize:
            seen_sig = true_sig
        else:
            seen_sig = MonitorSignals(**NULL_SIGNAL, turn=turn, cost_so_far=cumulative_cost)

        in_cooldown = (last_int_turn > -1000 and (turn - last_int_turn) < COOLDOWN)
        p_fail = predictor.predict_proba(seen_sig, signal_history[:])
        fire = (p_fail >= FIRE_AT_P) and not in_cooldown
        signal_history.append(seen_sig)

        intervention_cost = 0.0
        if fire and intervention_count < max_int:
            interv = select_intervention(true_sig, govern, assess)
            if enforce:  # Enforce pillar: actually apply
                ag = AgentState(goal_embedding=goal, state_embedding=state,
                                context_turns=action_history.copy(), turn=turn,
                                cost_so_far=cumulative_cost, plan=[],
                                last_tool_output=None,
                                intervention_count=intervention_count)
                new_ag = interv.apply(ag)
                state = new_ag.state_embedding.copy()
                intervention_cost = interv.estimated_cost
            intervention_count += 1
            last_int_turn = turn

        cumulative_cost += base_cost + intervention_cost
        final_drift = drift

    success = final_drift < cfg["completion_drift_threshold"]
    return {"success": success, "total_cost": cumulative_cost,
            "intervention_count": intervention_count}


def bootstrap_cnsr_ci(successes, costs, n=10000, seed=42):
    rng = np.random.default_rng(seed)
    s = np.array(successes, float); c = np.array(costs, float)
    boots = []
    for _ in range(n):
        idx = rng.integers(0, len(s), len(s))
        sr = s[idx].mean(); mc = max(c[idx].mean(), 1e-9)
        boots.append(sr / mc)
    return float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))


def run(base_seed: int = 42) -> None:
    cfg = dict(E4_CONFIG)
    print("[2.1] pre-training predictor (deterministic) …")
    predictor = pretrain_predictor(cfg, base_seed)

    results = {v: {"succ": [], "cost": [], "ints": []} for v in VARIANTS}
    for variant in VARIANTS:
        print(f"[2.1] {variant} …")
        for so in range(N_SEEDS):
            seed = base_seed + so * 1000
            for task_idx in range(cfg["n_tasks"]):
                task_seed = base_seed + task_idx
                ep_seed = seed + task_idx * 13 + so * 97
                r = run_episode(task_seed, ep_seed, cfg, predictor, variant)
                results[variant]["succ"].append(r["success"])
                results[variant]["cost"].append(r["total_cost"])
                results[variant]["ints"].append(r["intervention_count"])

    rng = np.random.default_rng(base_seed)
    rows = []
    agg = {}
    for v in VARIANTS:
        succ = results[v]["succ"]; cost = results[v]["cost"]; ints = results[v]["ints"]
        cmean, clo, chi = bootstrap_proportion_ci(succ, cfg["n_bootstrap"], rng=rng)
        mc = float(np.mean(cost))
        cnsr = cmean / max(mc, 1e-9)
        cnsr_lo, cnsr_hi = bootstrap_cnsr_ci(succ, cost, seed=base_seed)
        agg[v] = dict(completion=cmean, clo=clo, chi=chi, mean_cost=mc, cnsr=cnsr,
                      cnsr_lo=cnsr_lo, cnsr_hi=cnsr_hi, mean_ints=float(np.mean(ints)))

    full = agg["Full-SAGE"]
    for v in VARIANTS:
        a = agg[v]
        d_comp = full["completion"] - a["completion"]
        rows.append({
            "variant": v, "n_eval": len(results[v]["succ"]),
            "completion": round(a["completion"], 4),
            "completion_ci_lo": round(a["clo"], 4), "completion_ci_hi": round(a["chi"], 4),
            "mean_cost": round(a["mean_cost"], 4),
            "cnsr": round(a["cnsr"], 4),
            "cnsr_ci_lo": round(a["cnsr_lo"], 4), "cnsr_ci_hi": round(a["cnsr_hi"], 4),
            "mean_interventions": round(a["mean_ints"], 2),
            "delta_completion_vs_full": round(d_comp, 4),
            "cohens_h_vs_full": round(cohens_h(full["completion"], a["completion"]), 4),
        })

    with (OUT / "pillar_ablation.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)

    # LaTeX
    tex = [
        r"\begin{table}[t]", r"\centering",
        r"\caption{Per-pillar leave-one-out ablation on the research family "
        r"(5 seeds, 50 tasks; bootstrap 95\% CIs). $\Delta$Compl.\ is Full-SAGE "
        r"minus the variant (drop attributable to the removed pillar).}",
        r"\label{tab:pillar_ablation}",
        r"\begin{tabular}{lcccc}", r"\toprule",
        r"Variant & Completion (95\% CI) & CNSR & Interv. & $\Delta$Compl. \\",
        r"\midrule",
    ]
    for r in rows:
        star = r"\textbf{" if r["variant"] == "Full-SAGE" else ""
        end = "}" if star else ""
        tex.append(
            f"{star}{r['variant']}{end} & {r['completion']:.1%} "
            f"[{r['completion_ci_lo']:.1%}, {r['completion_ci_hi']:.1%}] & "
            f"{r['cnsr']:.2f} & {r['mean_interventions']:.1f} & "
            f"{r['delta_completion_vs_full']:+.1%} \\\\".replace("%", r"\%"))
    tex += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    (OUT / "pillar_ablation_table.tex").write_text("\n".join(tex) + "\n")

    manifest = {
        "experiment": "2.1_pillar_ablation", "git_sha": _get_git_sha(),
        "env_hash": _get_env_hash(), "timestamp": now_iso(), "base_seed": base_seed,
        "n_seeds": N_SEEDS, "fire_at_p": FIRE_AT_P, "cooldown": COOLDOWN,
        "config": {k: cfg[k] for k in ("n_tasks", "total_turns", "embedding_dim",
                                       "drift_rate_base", "completion_drift_threshold",
                                       "base_cost_per_turn", "n_bootstrap")},
        "pillar_operationalization": {
            "Stabilize": "monitors off -> null signals to controller",
            "Assess": "cost-aware intervention selection (GoalReanchor $0.01 vs ForceReplan $0.06)",
            "Govern": "taxonomy->intervention routing; off -> SchemaValidatedRetry always",
            "Enforce": "apply interventions; off -> open-loop (NoControl)",
        },
        "results": {r["variant"]: r for r in rows},
    }
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2))

    print("\n[2.1] Per-pillar ablation:")
    print(f"  {'variant':>12s} {'completion':>12s} {'CNSR':>8s} {'interv':>8s} {'dCompl':>8s}")
    for r in rows:
        print(f"  {r['variant']:>12s} {r['completion']:>11.1%} "
              f"[{r['completion_ci_lo']:.2f},{r['completion_ci_hi']:.2f}] {r['cnsr']:>7.2f} "
              f"{r['mean_interventions']:>8.1f} {r['delta_completion_vs_full']:>+7.1%}")
    print(f"  Wrote {OUT}/")


if __name__ == "__main__":
    run()
