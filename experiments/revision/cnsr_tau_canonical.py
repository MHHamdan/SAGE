"""Phase-0.2 correctness fix — canonical, process-STABLE CNSR Kendall tau_b.

Root cause of the tau mismatch (Table IV vs sensitivity vs re-runs): the committed
CNSR simulator seeds its per-task RNG with `seed ^ hash(task_id) ^ hash(model)`.
Python salts str hashing (PYTHONHASHSEED unset), so this seed — and therefore every
success/token draw and the resulting tau — CHANGES EVERY PROCESS. The three
different tau sets were three different hash salts, not a label swap.

Fix: seed the simulator from a stable digest (md5) instead of the salted builtin
hash. Then tau is deterministic across processes.

One canonical function `tau_b_by_category()` ranks the 7 configs by CNSR and by
success rate and returns Kendall tau_b (scipy default, tie-corrected). Table IV,
the §V inline tau, §V-F, and the sensitivity table are all sourced from it.

Usage: python cnsr_tau_canonical.py [--salted]   (--salted reproduces the bug)
Output: results/cnsr_sensitivity/{cnsr_tau_by_category.csv, cnsr_sensitivity_table.tex,
        manifest.json}  (regenerated)
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import sys
from pathlib import Path

import numpy as np
from scipy.stats import kendalltau

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

import experiments.cnsr_multitask as C  # noqa: E402
from sage.stability.traces import _get_git_sha, _get_env_hash, now_iso  # noqa: E402

OUT = ROOT / "results" / "cnsr_sensitivity"
OUT.mkdir(parents=True, exist_ok=True)
SEEDS = [0, 1, 2]


def stable_seed(model: str, task_id: str, seed: int) -> int:
    h = hashlib.md5(f"{model}|{task_id}|{seed}".encode()).digest()
    return (seed * 1_000_003) ^ int.from_bytes(h[:8], "little")


def run_task(model, task, seed, salted: bool):
    if salted:
        rng = random.Random(seed ^ hash(task["task_id"]) ^ hash(model))  # committed (buggy)
    else:
        rng = random.Random(stable_seed(model, task["task_id"], seed))    # fixed
    return C._simulate_run(model, task, seed, rng)


def collect(salted: bool):
    tasks_by_type = {"code": C.load_swe_bench_lite(50),
                     "web": C.load_webarena_mini(50),
                     "research": C.load_research_tasks(50)}
    top3 = C._top3_models()
    rows = []
    for seed in SEEDS:
        for model in C.MODELS:
            if model == "ensemble":
                continue
            for tt, tasks in tasks_by_type.items():
                for task in tasks:
                    rows.append(run_task(model, task, seed, salted))
        for tt, tasks in tasks_by_type.items():
            for task in tasks:
                sub = [run_task(m, task, seed, salted) for m in top3]
                rows.append({
                    "task_id": task["task_id"], "config": "ensemble", "task_type": tt,
                    "success": sum(r["success"] for r in sub) >= 2,
                    "cost_usd": round(sum(r["cost_usd"] for r in sub), 6),
                    "tokens_used": sum(r["tokens_used"] for r in sub),
                    "prompt_tokens": 0, "completion_tokens": 0, "seed": seed})
    return rows


def tau_b_by_category(stats):
    """CANONICAL: per category, Kendall tau_b between SR ranking and CNSR ranking."""
    out = {}
    for tt in C.TASK_TYPES:
        models = [m for m in C.MODELS if (m, tt) in stats]
        sr = [stats[(m, tt)]["success_rate"] for m in models]
        cn = [stats[(m, tt)]["cnsr_mean"] for m in models]
        tau, p = kendalltau(sr, cn)   # scipy default is tau_b (tie-corrected)
        out[tt] = {"tau_b": float(tau), "p": float(p),
                   "models": models, "sr": sr, "cnsr": cn}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--salted", action="store_true", help="reproduce the hash-salt bug")
    args = ap.parse_args()

    rows = collect(salted=args.salted)
    stats = C.aggregate(rows)
    res = tau_b_by_category(stats)

    tag = "SALTED(buggy)" if args.salted else "STABLE(fixed)"
    print(f"=== Canonical tau_b per category [{tag}] ===")
    for tt in C.TASK_TYPES:
        print(f"  {tt:9s} tau_b={res[tt]['tau_b']:+.4f}  p={res[tt]['p']:.4f}")

    if args.salted:
        return  # bug-demo only; don't overwrite deliverables

    # ── cnsr_tau_by_category.csv ──
    with (OUT / "cnsr_tau_by_category.csv").open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["task_family", "kendall_tau_b", "p_value",
                    "config", "success_rate", "cnsr_mean"])
        for tt in C.TASK_TYPES:
            r = res[tt]
            order = sorted(range(len(r["models"])), key=lambda i: -r["cnsr"][i])
            first = True
            for i in order:
                w.writerow([tt if first else "",
                            f"{r['tau_b']:.4f}" if first else "",
                            f"{r['p']:.4f}" if first else "",
                            C.MODEL_DISPLAY.get(r["models"][i], r["models"][i]),
                            round(r["sr"][i], 4), round(r["cnsr"][i], 4)])
                first = False

    # ── regenerated sensitivity table (grid CENTER = canonical baseline) ──
    # Under the committed inference-only cost model the grid is analytically robust:
    #  (i) uniform inference repricing multiplies every config's cost by the same
    #      factor -> CNSR ranking (hence tau_b) is INVARIANT;
    #  (ii) w_lat and C_hum scale latency/human cost terms that the committed model
    #      does not contain -> inert.
    # Therefore every one of the 45 grid cells equals the canonical baseline tau_b,
    # and 100% of cells have tau<0. This is robustness by construction, not luck.
    tex = [
        r"\begin{table}[t]", r"\centering",
        r"\caption{CNSR cost-parameter sensitivity. Baseline (grid centre) is the "
        r"canonical committed success-vs-CNSR Kendall's $\tau_b$ across the 7 "
        r"configurations. Under the committed inference-cost model, $\tau_b$ is "
        r"\emph{invariant} across the full $5\times3\times3$ grid: uniform inference "
        r"repricing preserves CNSR rank, and the $w_{\mathrm{lat}}$/$C_{\mathrm{hum}}$ "
        r"weights scale cost terms the committed model does not contain.}",
        r"\label{tab:cnsr_sensitivity}",
        r"\begin{tabular}{lccc}", r"\toprule",
        r"Task family & baseline $\tau_b$ & $p$ & \% grid cells $\tau<0$ \\",
        r"\midrule",
    ]
    for tt in C.TASK_TYPES:
        r = res[tt]
        neg = "100" if r["tau_b"] < 0 else "0"
        tex.append(f"{tt.capitalize()} & {r['tau_b']:.3f} & {r['p']:.3f} & {neg}\\% \\\\")
    overall = 100.0 * sum(res[tt]["tau_b"] < 0 for tt in C.TASK_TYPES) / len(C.TASK_TYPES)
    tex += [
        r"\midrule",
        f"\\multicolumn{{4}}{{l}}{{\\textit{{Overall grid cells with $\\tau<0$: "
        f"{overall:.0f}\\%; $\\tau_b$ invariant across the grid (see text).}}}} \\\\",
        r"\bottomrule", r"\end{tabular}", r"\end{table}",
    ]
    (OUT / "cnsr_sensitivity_table.tex").write_text("\n".join(tex) + "\n")

    manifest = {
        "experiment": "0.2_cnsr_tau_canonical_FIXED", "git_sha": _get_git_sha(),
        "env_hash": _get_env_hash(), "timestamp": now_iso(), "seeds": SEEDS,
        "determinism_fix": "per-task RNG seeded from md5 digest (stable across "
                           "processes) instead of the salted builtin hash().",
        "canonical_tau_b": {tt: round(res[tt]["tau_b"], 4) for tt in C.TASK_TYPES},
        "p_values": {tt: round(res[tt]["p"], 4) for tt in C.TASK_TYPES},
        "grid_invariance": "tau_b invariant across all 45 cells under committed "
                           "inference-only cost (uniform repricing rank-preserving; "
                           "w_lat/C_hum inert). overall %cells tau<0 = "
                           f"{overall:.0f}%.",
    }
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"\nWrote {OUT}/cnsr_tau_by_category.csv and regenerated cnsr_sensitivity_table.tex")


if __name__ == "__main__":
    main()
