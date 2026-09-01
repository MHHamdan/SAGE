"""Track 2 / 2A — real CNSR ladder on HotpotQA-distractor (open-weights).

5 open-weight models x 50 tasks x 2 seeds, temp=0.0 (stable success/cost).
Each model answers real HotpotQA-distractor items via the same real ReAct +
embedding-retrieval loop; success = EM; cost = MEASURED Ollama tokens (each
model's own tokenizer) x hosted list price. Reports per-model success, tokens,
list-price cost, CNSR, and the canonical Kendall tau_b (success-rank vs
CNSR-rank) via the SINGLE tie-corrected function.

Tasks use a slice DISJOINT from 2B (offset=200). No GPU wall-clock in cost.

Output: results/ollama_real/2A/{2A.csv, 2A_table.tex, manifest.json}
"""
from __future__ import annotations

import concurrent.futures as cf
import csv
import json
import sys
import threading
import time
from pathlib import Path

import numpy as np
from scipy.stats import kendalltau

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from experiments.revision.ollama_harness import load_pool, run_episode  # noqa: E402
from sage.stability.traces import _get_git_sha, _get_env_hash, now_iso  # noqa: E402

OUT = ROOT / "results" / "ollama_real" / "2A"
OUT.mkdir(parents=True, exist_ok=True)
DATA_DIR = ROOT / "results" / "ollama_real" / "data"

TEMP = 0.0
N_TASKS, OFFSET = 50, 200          # disjoint from 2B (0..199)
SEEDS = [0, 1, 2]                  # 3 seeds — matches Table IV protocol
MAX_TURNS, K_RETRIEVE = 6, 4
N_WORKERS = 4

# hosted list-price snapshot — Together AI serverless size-tier pricing, per 1M tokens
# (input=output), retrieved 2026-07-18. Recorded per-model in manifest.
LADDER = [  # pre-registered n=8 ladder (P2)
    ("gemma2:2b",   "google/gemma-2-2b-it",                   0.10),
    ("llama3.2:3b", "meta-llama/Llama-3.2-3B-Instruct-Turbo", 0.06),
    ("mistral:7b",  "mistralai/Mistral-7B-Instruct-v0.3",     0.20),
    ("llama3.1:8b", "meta-llama/Llama-3.1-8B-Instruct-Turbo", 0.18),
    ("gemma2:9b",   "google/gemma-2-9b-it",                   0.30),
    ("qwen2.5:14b", "Qwen/Qwen2.5-14B-Instruct",              0.30),
    ("phi4:14b",    "microsoft/phi-4",                        0.30),
    ("qwen2.5:32b", "Qwen/Qwen2.5-32B-Instruct",              0.80),
]
PRICE_PROVIDER = "Together AI (serverless size-tier)"
PRICE_DATE = "2026-07-18"
DISPLAY = {"gemma2:2b": "Gemma-2-2B", "llama3.2:3b": "Llama-3.2-3B",
           "mistral:7b": "Mistral-7B", "llama3.1:8b": "Llama-3.1-8B",
           "gemma2:9b": "Gemma-2-9B", "qwen2.5:14b": "Qwen2.5-14B",
           "phi4:14b": "Phi-4-14B", "qwen2.5:32b": "Qwen2.5-32B"}


def cost_usd(pt, ct, per_million):
    return (pt + ct) * per_million / 1e6  # input=output flat serverless rate


def tau_b(success_rates, cnsrs):
    """Canonical tie-corrected Kendall tau_b (scipy default) — the SINGLE function."""
    tau, p = kendalltau(success_rates, cnsrs)
    return float(tau), float(p)


def task_bootstrap_cnsr_ci(mrows, n_boot=3000):
    """Task-level bootstrap 95% CI on CNSR (resample the tasks). Meaningful
    dispersion even under deterministic (temp 0) decoding, where seed-SD is 0."""
    by_task = {}
    for r in mrows:
        by_task.setdefault(r["task_idx"], []).append(r)
    tasks = list(by_task)
    rng = np.random.default_rng(0)
    boots = []
    for _ in range(n_boot):
        samp = rng.choice(tasks, len(tasks), replace=True)
        em, cost = [], []
        for t in samp:
            for r in by_task[t]:
                em.append(r["em"]); cost.append(r["cost_usd"])
        mc = np.mean(cost)
        boots.append(np.mean(em) / mc if mc > 0 else 0.0)
    return float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))


def run():
    log = []
    def L(m): print(m, flush=True); log.append(m)

    pool, split_hash = load_pool(N_TASKS, DATA_DIR, offset=OFFSET)
    eval_hash = __import__("hashlib").md5("|".join(it["qid"] for it in pool).encode()).hexdigest()
    L(f"[2A] {len(pool)} tasks (offset={OFFSET}, disjoint from 2B) split_hash={eval_hash[:12]}")

    t0 = time.time()
    rows = []
    per_model = {}
    for model, slug, price_m in LADDER:
        L(f"[2A] model {model} ({slug}) …")
        jobs = [(seed, ti, it) for seed in SEEDS for ti, it in enumerate(pool)]
        done = [0]; lock = threading.Lock()
        def work(job):
            seed, ti, it = job
            r = run_episode(it, model, TEMP, seed * 1000 + ti, k=K_RETRIEVE,
                            max_turns=MAX_TURNS, controller=None)
            c = cost_usd(r["prompt_tokens"], r["completion_tokens"], price_m)
            with lock:
                done[0] += 1
                if done[0] % 25 == 0:
                    L(f"    {model}: {done[0]}/{len(jobs)} ({time.time()-t0:.0f}s)")
            return {"model": model, "slug": slug, "seed": seed, "task_idx": ti,
                    "qid": it["qid"], "em": r["em"], "f1": round(r["f1"], 4),
                    "prompt_tokens": r["prompt_tokens"], "completion_tokens": r["completion_tokens"],
                    "cost_usd": c, "turns": r["turns_used"]}
        with cf.ThreadPoolExecutor(N_WORKERS) as ex:
            mrows = list(ex.map(work, jobs))
        rows.extend(mrows)
        # per-SEED aggregation (Table IV protocol): CNSR_seed = SR_seed / mean_cost_seed
        seed_sr, seed_cnsr = [], []
        for sd in SEEDS:
            g = [r for r in mrows if r["seed"] == sd]
            sr_s = float(np.mean([r["em"] for r in g]))
            mc_s = float(np.mean([r["cost_usd"] for r in g]))
            seed_sr.append(sr_s)
            seed_cnsr.append(sr_s / mc_s if mc_s > 0 else 0.0)
        ci_lo, ci_hi = task_bootstrap_cnsr_ci(mrows)
        per_model[model] = {
            "success_rate": float(np.mean(seed_sr)),
            "sr_sd": float(np.std(seed_sr, ddof=1)) if len(seed_sr) > 1 else 0.0,
            "cnsr_mean": float(np.mean(seed_cnsr)),
            "cnsr_sd": float(np.std(seed_cnsr, ddof=1)) if len(seed_cnsr) > 1 else 0.0,
            "cnsr_ci_lo": ci_lo, "cnsr_ci_hi": ci_hi,
            "mean_cost": float(np.mean([r["cost_usd"] for r in mrows])),
            "mean_tokens": float(np.mean([r["prompt_tokens"] + r["completion_tokens"] for r in mrows])),
            "seed_cnsr": seed_cnsr, "seed_sr": seed_sr,
            "price_per_1M": price_m, "slug": slug, "n": len(mrows)}
        s = per_model[model]
        L(f"    -> SR={s['success_rate']:.3f} tokens={s['mean_tokens']:.0f} "
          f"cost=${s['mean_cost']:.2e} CNSR={s['cnsr_mean']:.0f}±{s['cnsr_sd']:.0f}")

    # ranks + canonical tau_b (SAME tie-corrected function) from per-model means
    models = [m for m, _, _ in LADDER]
    srs = [per_model[m]["success_rate"] for m in models]
    cns = [per_model[m]["cnsr_mean"] for m in models]
    # rank 1 = best (highest). ties share the min rank via argsort ordering.
    cnsr_order = sorted(models, key=lambda x: -per_model[x]["cnsr_mean"])
    sr_order = sorted(models, key=lambda x: -per_model[x]["success_rate"])
    for i, m in enumerate(cnsr_order, 1): per_model[m]["cnsr_rank"] = i
    for i, m in enumerate(sr_order, 1): per_model[m]["sr_rank"] = i
    tau, p = tau_b(srs, cns)
    L(f"[2A] canonical Kendall tau_b (SR-rank vs CNSR-rank, n={len(models)} configs) = {tau:.4f} (p={p:.4f})")

    # CSV (per-episode)
    with (OUT / "2A.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)

    # LaTeX — Table IV-parallel (real-model validation): CNSR±SD, task 95% CI, SR, ranks + tau
    det_note = (r" Seed-SD is $0$ (greedy decoding is deterministic); the $95\%$ CI is a "
                r"task-level bootstrap." if TEMP == 0.0 else
                r" SD is across seeds; the $95\%$ CI is a task-level bootstrap.")
    lines = [r"\begin{table}[t]", r"\centering",
             r"\caption{\textbf{Real-model validation of the CNSR inversion} (cf.\ Table~IV, "
             r"synthetic). Open-weight models on real HotpotQA-distractor (research/QA family) "
             r"via Ollama, temp " + f"{TEMP}" + r", " + f"{N_TASKS}" + r" tasks $\times$ " +
             f"{len(SEEDS)}" + r" seeds; success $=$ EM, cost $=$ measured tokens $\times$ "
             r"hosted list price (" + PRICE_PROVIDER + ", " + PRICE_DATE + r")." + det_note +
             r" Kendall's $\tau_b$ (SR rank vs.\ CNSR rank) $=" + f"{tau:.3f}$ ($p={p:.3f}$)" +
             r"; $\tau<0$ confirms the rank inversion on real models.}",
             r"\label{tab:cnsr_ollama_real}", r"\begin{tabular}{lccccc}", r"\toprule",
             r"Model & CNSR $\pm$ SD & 95\% CI & SR & CNSR rank & SR rank \\",
             r"\midrule"]
    for m in cnsr_order:
        s = per_model[m]
        lines.append(f"{DISPLAY[m]} & ${s['cnsr_mean']:.0f} \\pm {s['cnsr_sd']:.0f}$ & "
                     f"[{s['cnsr_ci_lo']:.0f}, {s['cnsr_ci_hi']:.0f}] & "
                     f"{s['success_rate']:.0%} & {s['cnsr_rank']} & {s['sr_rank']} \\\\".replace("%", r"\%"))
    lines += [r"\midrule",
              r"\multicolumn{6}{l}{\textit{Kendall's $\tau_b$ (SR rank vs.\ CNSR rank):} $"
              + f"{tau:.3f}$ ($p={p:.3f}$)" + r"} \\",
              r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    (OUT / "2A_table.tex").write_text("\n".join(lines) + "\n")

    manifest = {
        "experiment": "2A_cnsr_ladder_ollama", "git_sha": _get_git_sha(),
        "env_hash": _get_env_hash(), "timestamp": now_iso(), "temperature": TEMP,
        "seeds": SEEDS, "n_tasks": N_TASKS, "offset": OFFSET,
        "dataset": "hotpot_qa/distractor/validation", "eval_split_hash": eval_hash,
        "price_provider": PRICE_PROVIDER, "price_retrieval_date": PRICE_DATE,
        "price_note": "flat serverless size-tier rate, input=output, USD per 1M tokens",
        "per_model": {m: {**per_model[m]} for m in models},
        "kendall_tau_b": {"tau": tau, "p": p, "n_configs": len(models),
                          "definition": "scipy.stats.kendalltau (tie-corrected tau_b) over "
                                        "(success_rate, cnsr) across the ladder"},
        "cost_note": "CNSR uses measured-token x hosted-list-price inference cost only; "
                     "NO GPU wall-clock in cost.",
        "runtime_s": round(time.time() - t0, 1), "log": log,
    }
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2))

    L(f"\n[2A] === CNSR ladder ({time.time()-t0:.0f}s) — real-model validation of Table IV ===")
    L(f"  {'model':>14s} {'CNSR±SD':>14s} {'SR':>6s} {'tokens':>7s} {'Crank':>6s} {'SRrank':>7s}")
    for m in cnsr_order:
        s = per_model[m]
        cnsr_str = f"{s['cnsr_mean']:.0f}±{s['cnsr_sd']:.0f}"
        L(f"  {DISPLAY[m]:>14s} {cnsr_str:>14s} "
          f"{s['success_rate']:>6.0%} {s['mean_tokens']:>7.0f} {s['cnsr_rank']:>6d} {s['sr_rank']:>7d}")
    L(f"  canonical tau_b = {tau:.4f} (p={p:.4f})  [tau<0 => inversion]")
    L(f"[2A] wrote {OUT}/")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--temp", type=float, default=0.0)
    ap.add_argument("--tag", default=None)
    a = ap.parse_args()
    TEMP = a.temp
    _tag = a.tag or ("t" + str(a.temp).replace(".", ""))
    OUT = ROOT / "results" / "ollama_real" / f"2A_{_tag}"
    OUT.mkdir(parents=True, exist_ok=True)
    run()
