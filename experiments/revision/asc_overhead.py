"""Phase 0.4 — ASC computational overhead (Supp B.8, main §VII).

Instruments ASC.Decide() by replaying the E4 episodes deterministically
(seed-locked, identical to the committed traces — no live calls). Per decision
we time, separately:

  * monitor-signal computation  (drift / oscillation / fidelity / convergence
    from the state embedding — the work the monitors do each turn)
  * controller.decide()         (the ASC decision itself)
  * logistic-head eval          (scaler.transform + LogisticRegression.predict_proba)
  * feature extraction          (extract_features, 8-dim vector build)

Reports mean / p50 / p95 ms, predictive-head parameter memory, process peak RSS,
and added latency per turn as a fraction of simulated per-turn compute.

Output: results/asc_overhead/{asc_overhead.csv, manifest.json}
"""
from __future__ import annotations

import csv
import json
import resource
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from experiments.e4_closed_loop import (  # noqa: E402
    E4_CONFIG, _unit, goal_drift_score, make_goal_embedding, drift_step,
    _oscillation_score, _fidelity_score, pretrain_predictor,
)
from sage.stability import (  # noqa: E402
    MonitorSignals, NoControl, FixedScheduleController, ThresholdController,
    PredictiveController, AgentState,
)
from sage.stability.predictor import extract_features  # noqa: E402
from sage.stability.traces import _get_git_sha, _get_env_hash, now_iso  # noqa: E402

OUT = ROOT / "results" / "asc_overhead"
OUT.mkdir(parents=True, exist_ok=True)

perf = time.perf_counter


def predictive_head_memory_bytes(predictor) -> int:
    """Sum resident bytes of the trained logistic head + scaler parameters."""
    m = predictor._model
    s = predictor._scaler
    total = 0
    for arr in (getattr(m, "coef_", None), getattr(m, "intercept_", None),
                getattr(m, "classes_", None), getattr(s, "mean_", None),
                getattr(s, "scale_", None), getattr(s, "var_", None)):
        if arr is not None:
            total += int(np.asarray(arr).nbytes)
    return total


def run(base_seed: int = 42) -> None:
    cfg = dict(E4_CONFIG)
    dim = cfg["embedding_dim"]
    total_turns = cfg["total_turns"]
    drift_rate = cfg["drift_rate_base"]
    max_int = cfg["max_interventions_per_task"]
    base_cost = cfg["base_cost_per_turn"]

    print("[0.4] pre-training predictor (deterministic) …")
    predictor = pretrain_predictor(cfg, base_seed)
    head_bytes = predictive_head_memory_bytes(predictor)

    conditions = {
        "NoControl": lambda: NoControl(),
        "FixedSchedule": lambda: FixedScheduleController(reanchor_every_k=cfg["fixed_schedule_k"]),
        "ThresholdController": lambda: ThresholdController(
            drift_threshold=cfg["drift_threshold"],
            oscillation_threshold=cfg["oscillation_threshold"],
            fidelity_threshold=cfg["fidelity_threshold"],
            cooldown_turns=cfg["cooldown_turns"]),
        "PredictiveController": lambda: PredictiveController(
            predictor=predictor, lead_time_k=cfg["predictive_lead_time_k"],
            fire_at_p=cfg["predictive_fire_at_p"], cooldown_turns=cfg["cooldown_turns"]),
    }

    # per-condition accumulators of per-decision timings (seconds)
    timings: dict[str, dict[str, list]] = {
        c: {"signal": [], "decide": [], "feat": [], "logit": [], "turn": []}
        for c in conditions
    }

    for cond_name, factory in conditions.items():
        print(f"[0.4] replaying {cond_name} …")
        for seed_offset in range(cfg["n_seeds"]):
            seed = base_seed + seed_offset * 1000
            for task_idx in range(cfg["n_tasks"]):
                task_seed = base_seed + task_idx
                episode_seed = seed + task_idx * 13 + seed_offset * 97
                rng = np.random.default_rng(episode_seed)
                goal = make_goal_embedding(task_seed, dim)
                state = _unit(goal + 0.1 * rng.standard_normal(dim))
                controller = factory()
                controller.reset()
                action_history: list[str] = []
                signal_history: list[MonitorSignals] = []
                cumulative_cost = 0.0
                intervention_count = 0
                best_sim = float(np.clip(np.dot(_unit(goal), _unit(state)), -1.0, 1.0))

                for turn in range(1, total_turns + 1):
                    t_turn0 = perf()
                    action = f"action_{int(rng.integers(0, 8))}"
                    action_history.append(action)
                    state = drift_step(state, goal, drift_rate, rng)

                    # ---- monitor-signal computation (timed) ----
                    t0 = perf()
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
                    t_signal = perf() - t0

                    # ---- attribution micro-timings for the predictive head ----
                    t_feat = t_logit = 0.0
                    if cond_name == "PredictiveController":
                        tf0 = perf()
                        feats = extract_features(signals, list(signal_history))
                        t_feat = perf() - tf0
                        tl0 = perf()
                        fs = predictor._scaler.transform(feats.reshape(1, -1))
                        predictor._model.predict_proba(fs)
                        t_logit = perf() - tl0

                    # ---- controller.decide() (timed) ----
                    td0 = perf()
                    decision = controller.decide(signals)
                    t_decide = perf() - td0

                    signal_history.append(signals)

                    # apply intervention (kept for faithful turn-time accounting)
                    intervention_cost = 0.0
                    if decision.intervention is not None and intervention_count < max_int:
                        ag = AgentState(goal_embedding=goal, state_embedding=state,
                                        context_turns=action_history.copy(), turn=turn,
                                        cost_so_far=cumulative_cost, plan=[],
                                        last_tool_output=None,
                                        intervention_count=intervention_count)
                        try:
                            new_ag = decision.intervention.apply(ag)
                            state = new_ag.state_embedding.copy()
                            intervention_cost = decision.intervention.estimated_cost
                            intervention_count += 1
                        except Exception:
                            intervention_count += 1
                    cumulative_cost += base_cost + intervention_cost
                    t_turn = perf() - t_turn0

                    tc = timings[cond_name]
                    tc["signal"].append(t_signal)
                    tc["decide"].append(t_decide)
                    tc["feat"].append(t_feat)
                    tc["logit"].append(t_logit)
                    tc["turn"].append(t_turn)

    peak_rss_kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss  # KB on Linux

    # ---- aggregate ----
    def ms_stats(xs: list[float]) -> tuple[float, float, float]:
        a = np.asarray(xs) * 1e3
        return float(a.mean()), float(np.percentile(a, 50)), float(np.percentile(a, 95))

    rows = []
    for cond in conditions:
        tc = timings[cond]
        d_mean, d_p50, d_p95 = ms_stats(tc["decide"])
        s_mean, _, _ = ms_stats(tc["signal"])
        turn_mean = float(np.mean(tc["turn"]) * 1e3)
        feat_mean = float(np.mean(tc["feat"]) * 1e3) if any(tc["feat"]) else 0.0
        logit_mean = float(np.mean(tc["logit"]) * 1e3) if any(tc["logit"]) else 0.0
        added_pct = 100.0 * float(np.mean(tc["decide"])) / max(float(np.mean(tc["turn"])), 1e-12)
        rows.append({
            "controller": cond,
            "n_decisions": len(tc["decide"]),
            "decide_ms_mean": round(d_mean, 5),
            "decide_ms_p50": round(d_p50, 5),
            "decide_ms_p95": round(d_p95, 5),
            "signal_ms_mean": round(s_mean, 5),
            "feature_extract_ms_mean": round(feat_mean, 5),
            "logistic_head_ms_mean": round(logit_mean, 5),
            "turn_ms_mean": round(turn_mean, 5),
            "added_latency_pct_of_turn": round(added_pct, 3),
            "predictive_head_mem_bytes": head_bytes if cond == "PredictiveController" else 0,
            "process_peak_rss_kb": peak_rss_kb,
        })

    csv_path = OUT / "asc_overhead.csv"
    with csv_path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    manifest = {
        "experiment": "0.4_asc_overhead",
        "git_sha": _get_git_sha(), "env_hash": _get_env_hash(),
        "timestamp": now_iso(), "base_seed": base_seed,
        "config": {k: cfg[k] for k in ("n_tasks", "n_seeds", "total_turns",
                                       "embedding_dim", "predictive_lead_time_k")},
        "predictive_head_mem_bytes": head_bytes,
        "process_peak_rss_kb": peak_rss_kb,
        "note": "Timings via time.perf_counter over all replayed E4 decisions "
                "(4 controllers x 3 seeds x 50 tasks x 50 turns = 30000 each). "
                "No API calls; deterministic replay of committed E4 episodes.",
    }
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2))

    print("\n[0.4] ASC overhead (ms/decision):")
    print(f"  {'controller':>22s} {'mean':>8s} {'p50':>8s} {'p95':>8s} {'logit':>8s} {'feat':>8s} {'added%':>8s}")
    for r in rows:
        print(f"  {r['controller']:>22s} {r['decide_ms_mean']:>8.4f} {r['decide_ms_p50']:>8.4f} "
              f"{r['decide_ms_p95']:>8.4f} {r['logistic_head_ms_mean']:>8.4f} "
              f"{r['feature_extract_ms_mean']:>8.4f} {r['added_latency_pct_of_turn']:>7.2f}%")
    print(f"\n  Predictive-head parameter memory: {head_bytes} bytes ({head_bytes/1024:.2f} KB)")
    print(f"  Process peak RSS: {peak_rss_kb} KB ({peak_rss_kb/1024:.1f} MB)")
    print(f"  Wrote {csv_path}")


if __name__ == "__main__":
    run()
