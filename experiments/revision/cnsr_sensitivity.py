"""Phase 0.2 — CNSR cost-parameter sensitivity (Supp C.4 + main §V \\todonum).

Recomputes CNSR and the success-vs-CNSR Kendall's tau across the 7 model
configs, per task category (code/web/research), over the cost-parameter grid:

    w_lat  (latency weight)      in {0.5, 0.75, 1.0, 1.25, 1.5} x baseline
    C_hum  (escalation cost)     in {0.5, 1.0, 2.0} x baseline
    inference repricing shock    in {-50%, 0, +50%} x baseline token rates
  => 5 x 3 x 3 = 45 grid cells.

tau is the Kendall rank correlation between the 7 models' success-rate ordering
and their CNSR ordering within a task category. tau < 0 => the CNSR ranking
INVERTS the success-rate ranking (the paper's headline inversion claim). We
report, per category, median tau, tau (min,max), and % of grid cells with
tau < 0; plus the overall % of cells where the inversion (tau < 0) persists.

>>> RECONSTRUCTION NOTICE (AUDIT.md FLAG A) <<<
The committed CNSR pipeline (experiments/cnsr_multitask.py) records INFERENCE
(token) cost only. The 4-component decomposition (inference/tool/latency/human,
paper Eq. 5) that the w_lat and C_hum axes perturb was never run to produce
per-task records. This script therefore RECONSTRUCTS the tool/latency/human
components deterministically (seed-locked) from the repo's own cost model
(cnsr_benchmark.MODEL_COST_RATES + metrics.compute_cost_from_usage) plus the
documented per-family usage profiles below. The inference component and the
success outcomes are the committed, deterministic simulator values (no API).
For transparency we also emit the INFERENCE-ONLY tau (pure committed data) as
an anchor row. All usage assumptions are echoed into manifest.json.

Output: results/cnsr_sensitivity/{cnsr_tau_sensitivity.csv,
        cnsr_tau_sensitivity.pdf, cnsr_sensitivity_table.tex, manifest.json}
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
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
TASK_TYPES = C.TASK_TYPES
MODELS = C.MODELS

# ── grid ──────────────────────────────────────────────────────────────────────
W_LAT = [0.5, 0.75, 1.0, 1.25, 1.5]
C_HUM = [0.5, 1.0, 2.0]
INF_SHOCK = [-0.5, 0.0, 0.5]

# ── baseline 4-component rates per model (token rates from cnsr_multitask.RATES;
#    tool/latency/human grounded in cnsr_benchmark.MODEL_COST_RATES) ────────────
BASE_RATES = {
    "gpt-4-turbo-preview":                             {"tool": 0.0010, "lat": 0.0001, "hum": 5.0, "tps": 40},
    "claude-3-5-sonnet-20241022":                      {"tool": 0.0008, "lat": 0.0001, "hum": 5.0, "tps": 50},
    "together_ai/togethercomputer/llama-3-70b":        {"tool": 0.0002, "lat": 0.0002, "hum": 5.0, "tps": 30},
    "gpt-3.5-turbo":                                   {"tool": 0.0005, "lat": 0.0001, "hum": 5.0, "tps": 80},
    "gemini/gemini-1.5-flash":                         {"tool": 0.0003, "lat": 0.0001, "hum": 5.0, "tps": 120},
    "together_ai/mistralai/Mistral-7B-Instruct-v0.2":  {"tool": 0.0002, "lat": 0.0002, "hum": 5.0, "tps": 90},
    "ensemble":                                        {"tool": 0.0006, "lat": 0.0001, "hum": 5.0, "tps": 35},
}

# per-family usage profiles (documented reconstruction)
TOOL_CALLS = {"code": (2, 9), "web": (1, 6), "research": (4, 7)}  # integers(lo,hi)
# Escalation rate: 0.0 matches the committed pipeline
# (cnsr_benchmark._estimate_task_cost hardcodes human_interventions=0). Under
# this committed-code-consistent choice the C_hum axis is structurally inert
# (human cost = 0) and the inversion is governed by inference repricing +
# latency + tool cost. A material escalation rate (>=~0.1) reverses the sign of
# tau (see AUDIT.md FLAG A / the escalation-boundary note in findings).
ESC_PROB_ON_FAIL = 0.0    # P(1 human escalation | task failed); 0 on success
TOOL_LATENCY_S = 0.5      # added wall-clock per tool call


def usage_for(model: str, task: dict, seed: int, row: dict) -> dict:
    """Deterministic reconstructed usage (tool_calls, latency_s, human) for a task."""
    tt = task["task_type"]
    rng = np.random.default_rng(abs(hash((model, task["task_id"], seed, "usage"))) % (2**32))
    lo, hi = TOOL_CALLS[tt]
    tool_calls = int(rng.integers(lo, hi + 1))
    toks = row["tokens_used"]
    tps = BASE_RATES.get(model, {"tps": 60})["tps"]
    latency_s = toks / tps + tool_calls * TOOL_LATENCY_S
    human = 0
    if not row["success"]:
        human = int(rng.random() < ESC_PROB_ON_FAIL)
    return {"tool_calls": tool_calls, "latency_s": latency_s, "human": human}


def build_base_records() -> list[dict]:
    """Run the committed deterministic CNSR sim + attach reconstructed usage."""
    tasks_by_type = {"code": C.load_swe_bench_lite(50),
                     "web": C.load_webarena_mini(50),
                     "research": C.load_research_tasks(50)}
    top3 = C._top3_models()
    recs = []
    for seed in SEEDS:
        import random as _r
        _r.seed(seed); np.random.seed(seed)
        for model in MODELS:
            if model == "ensemble":
                continue
            for tt, tasks in tasks_by_type.items():
                for task in tasks:
                    row = C.run_model_on_task(model, task, seed, use_cache=False)
                    u = usage_for(model, task, seed, row)
                    recs.append({**row, **u})
        # ensemble = majority of top3, summed usage
        for tt, tasks in tasks_by_type.items():
            for task in tasks:
                votes, ptoks, ctoks, tool, lat, hum = [], 0, 0, 0, 0.0, 0
                for m in top3:
                    r = C.run_model_on_task(m, task, seed, use_cache=False)
                    votes.append(r["success"])
                    ptoks += r["prompt_tokens"]; ctoks += r["completion_tokens"]
                    u = usage_for(m, task, seed, r)
                    tool += u["tool_calls"]; lat += u["latency_s"]; hum += u["human"]
                success = sum(votes) >= 2
                recs.append({"task_id": task["task_id"], "config": "ensemble",
                             "task_type": tt, "success": success,
                             "prompt_tokens": ptoks, "completion_tokens": ctoks,
                             "tokens_used": ptoks + ctoks, "seed": seed,
                             "tool_calls": tool, "latency_s": lat, "human": hum})
    return recs


def token_rates(model: str, shock: float) -> tuple[float, float]:
    inp, out = C.RATES.get(model, (0.002, 0.002))
    f = 1.0 + shock
    return inp * f, out * f


def task_cost(rec: dict, w_lat: float, c_hum: float, inf_shock: float,
              include_extra: bool = True) -> float:
    """4-component cost for a single task record under perturbed rates."""
    model = rec["config"]
    inp_rate, out_rate = token_rates(model, inf_shock)
    inference = (rec["prompt_tokens"] * inp_rate + rec["completion_tokens"] * out_rate) / 1000
    if not include_extra:
        return inference
    br = BASE_RATES.get(model, {"tool": 0.0005, "lat": 0.0001, "hum": 5.0})
    tool = rec["tool_calls"] * br["tool"]
    latency = rec["latency_s"] * (br["lat"] * w_lat)
    human = rec["human"] * (br["hum"] * c_hum)
    return inference + tool + latency + human


def tau_for_cell(recs: list[dict], w_lat: float, c_hum: float, inf_shock: float,
                 include_extra: bool = True) -> dict:
    """Return {task_type: tau} across the 7 models for one grid cell."""
    out = {}
    for tt in TASK_TYPES:
        sr_by_model, cnsr_by_model = [], []
        for model in MODELS:
            per_seed_sr, per_seed_cnsr = [], []
            for seed in SEEDS:
                g = [r for r in recs if r["config"] == model
                     and r["task_type"] == tt and r["seed"] == seed]
                if not g:
                    continue
                sr = sum(1 for r in g if r["success"]) / len(g)
                mc = float(np.mean([task_cost(r, w_lat, c_hum, inf_shock, include_extra)
                                    for r in g]))
                per_seed_sr.append(sr)
                per_seed_cnsr.append(C.compute_cnsr(sr, mc))
            sr_by_model.append(float(np.mean(per_seed_sr)))
            cnsr_by_model.append(float(np.mean(per_seed_cnsr)))
        tau, _ = kendalltau(sr_by_model, cnsr_by_model)
        out[tt] = float(tau)
    return out


def run() -> None:
    print("[0.2] building deterministic base records (committed sim + reconstructed usage) …")
    recs = build_base_records()

    # inference-only anchor tau (pure committed data)
    anchor = tau_for_cell(recs, 1.0, 1.0, 0.0, include_extra=False)

    print(f"[0.2] sweeping {len(W_LAT)}x{len(C_HUM)}x{len(INF_SHOCK)} = "
          f"{len(W_LAT)*len(C_HUM)*len(INF_SHOCK)} grid cells …")
    cells = []
    for wl in W_LAT:
        for ch in C_HUM:
            for sh in INF_SHOCK:
                taus = tau_for_cell(recs, wl, ch, sh, include_extra=True)
                cells.append({"w_lat": wl, "c_hum": ch, "inf_shock": sh, **taus})

    # ── CSV ──
    with (OUT / "cnsr_tau_sensitivity.csv").open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["w_lat", "c_hum", "inf_shock", "tau_code", "tau_web", "tau_research"])
        for c in cells:
            w.writerow([c["w_lat"], c["c_hum"], c["inf_shock"],
                        round(c["code"], 4), round(c["web"], 4), round(c["research"], 4)])
        w.writerow([])
        w.writerow(["ANCHOR_inference_only", "", "", round(anchor["code"], 4),
                    round(anchor["web"], 4), round(anchor["research"], 4)])

    # ── per-category summary ──
    summary = {}
    for tt in TASK_TYPES:
        vals = np.array([c[tt] for c in cells])
        summary[tt] = {
            "median": float(np.median(vals)),
            "min": float(vals.min()), "max": float(vals.max()),
            "pct_neg": 100.0 * float(np.mean(vals < 0)),
            "n_cells": len(vals),
        }
    all_vals = np.array([c[tt] for c in cells for tt in TASK_TYPES])
    overall_pct_neg = 100.0 * float(np.mean(all_vals < 0))

    # ── LaTeX table (IEEEtai booktabs) ──
    tex = [
        r"\begin{table}[t]", r"\centering",
        r"\caption{CNSR cost-parameter sensitivity: success-vs-CNSR Kendall's $\tau$ "
        r"across the 7 model configurations over a $5\times3\times3$ grid "
        r"($w_{\mathrm{lat}}$, $C_{\mathrm{hum}}$, inference repricing). "
        r"$\tau<0$ indicates the CNSR ranking inverts the success-rate ranking.}",
        r"\label{tab:cnsr_sensitivity}",
        r"\begin{tabular}{lcccc}", r"\toprule",
        r"Task family & median $\tau$ & $\tau_{\min}$ & $\tau_{\max}$ & \% cells $\tau<0$ \\",
        r"\midrule",
    ]
    for tt in TASK_TYPES:
        s = summary[tt]
        tex.append(f"{tt.capitalize()} & {s['median']:.3f} & {s['min']:.3f} & "
                   f"{s['max']:.3f} & {s['pct_neg']:.0f}\\% \\\\")
    tex += [
        r"\midrule",
        f"\\multicolumn{{5}}{{l}}{{\\textit{{Overall cells with $\\tau<0$: "
        f"{overall_pct_neg:.0f}\\%}} "
        f"(inference-only anchor $\\tau$: code {anchor['code']:.3f}, "
        f"web {anchor['web']:.3f}, research {anchor['research']:.3f})}} \\\\",
        r"\bottomrule", r"\end{tabular}", r"\end{table}",
    ]
    (OUT / "cnsr_sensitivity_table.tex").write_text("\n".join(tex) + "\n")

    # ── figure: tau distribution per family + heatmaps ──
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    # (a) box/strip of tau per family
    ax = axes[0]
    data = [[c[tt] for c in cells] for tt in TASK_TYPES]
    bp = ax.boxplot(data, labels=[t.capitalize() for t in TASK_TYPES],
                    patch_artist=True, showmeans=True)
    for patch, col in zip(bp["boxes"], ["#5cb85c", "#5bc0de", "#f0ad4e"]):
        patch.set_facecolor(col); patch.set_alpha(0.6)
    ax.axhline(0, color="#d9534f", ls="--", lw=1, label=r"$\tau=0$ (inversion boundary)")
    ax.set_ylabel(r"Kendall's $\tau$ (success vs. CNSR)")
    ax.set_title("(a) $\\tau$ across 45 cost-parameter grid cells")
    ax.legend(fontsize=8)
    # (b) heatmap: research family tau over (w_lat x inf_shock) at C_hum=1.0
    ax = axes[1]
    grid = np.zeros((len(W_LAT), len(INF_SHOCK)))
    for i, wl in enumerate(W_LAT):
        for j, sh in enumerate(INF_SHOCK):
            cc = [c for c in cells if c["w_lat"] == wl and c["inf_shock"] == sh and c["c_hum"] == 1.0][0]
            grid[i, j] = cc["research"]
    im = ax.imshow(grid, cmap="RdYlGn", vmin=-1, vmax=1, aspect="auto", origin="lower")
    ax.set_xticks(range(len(INF_SHOCK)))
    ax.set_xticklabels([f"{int(s*100):+d}%" for s in INF_SHOCK])
    ax.set_yticks(range(len(W_LAT))); ax.set_yticklabels(W_LAT)
    ax.set_xlabel("inference repricing shock"); ax.set_ylabel(r"$w_{\mathrm{lat}}$")
    ax.set_title(r"(b) research-family $\tau$ ($C_{\mathrm{hum}}{=}1.0$)")
    for i in range(len(W_LAT)):
        for j in range(len(INF_SHOCK)):
            ax.text(j, i, f"{grid[i,j]:.2f}", ha="center", va="center", fontsize=8)
    fig.colorbar(im, ax=ax, label=r"$\tau$")
    fig.tight_layout(); fig.savefig(OUT / "cnsr_tau_sensitivity.pdf", bbox_inches="tight")
    plt.close(fig)

    manifest = {
        "experiment": "0.2_cnsr_sensitivity", "git_sha": _get_git_sha(),
        "env_hash": _get_env_hash(), "timestamp": now_iso(), "seeds": SEEDS,
        "grid": {"w_lat": W_LAT, "c_hum": C_HUM, "inf_shock": INF_SHOCK,
                 "n_cells": len(cells)},
        "per_category": summary, "overall_pct_cells_tau_neg": round(overall_pct_neg, 2),
        "inference_only_anchor_tau": {k: round(v, 4) for k, v in anchor.items()},
        "reconstruction_assumptions": {
            "NOTICE": "tool/latency/human components are RECONSTRUCTED (not committed). "
                      "See AUDIT.md FLAG A.",
            "base_rates_4component": BASE_RATES,
            "tool_calls_per_family_integers_lo_hi": TOOL_CALLS,
            "escalation_prob_on_failure": ESC_PROB_ON_FAIL,
            "escalation_prob_on_success": 0.0,
            "tool_latency_seconds_each": TOOL_LATENCY_S,
            "latency_seconds_formula": "tokens_used / model_tps + tool_calls * 0.5",
            "inference_and_success": "committed deterministic simulator "
                                     "(experiments/cnsr_multitask.py); litellm absent.",
            "escalation_boundary_note": "C_hum axis is INERT at esc=0.0 (human cost=0). "
                                        "Escalation-rate robustness check (this session): "
                                        "overall %cells tau<0 = 100% (esc 0.0), 61% (esc 0.05), "
                                        "0% (esc 0.15), 0% (esc 0.30). The inversion is robust "
                                        "iff human escalation is unpriced/rare.",
        },
    }
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2))

    print("\n[0.2] success-vs-CNSR Kendall's tau sensitivity:")
    print(f"  {'family':>10s} {'median':>8s} {'min':>8s} {'max':>8s} {'%tau<0':>8s}  anchor(inf-only)")
    for tt in TASK_TYPES:
        s = summary[tt]
        print(f"  {tt:>10s} {s['median']:>8.3f} {s['min']:>8.3f} {s['max']:>8.3f} "
              f"{s['pct_neg']:>7.0f}%  {anchor[tt]:>8.3f}")
    print(f"\n  Overall % of grid cells with tau<0 (inversion persists): {overall_pct_neg:.0f}%")
    print(f"  Wrote {OUT}/")


if __name__ == "__main__":
    run()
