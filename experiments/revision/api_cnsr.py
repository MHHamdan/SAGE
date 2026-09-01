"""API-measured CNSR (Table IV companion) via OpenRouter — current-gen proprietary
+ open models, REAL calls, MEASURED tokens x REAL list price.

NB: the exact 2024 Table IV models (Claude-3.5-Sonnet, Gemini-1.5-Flash, Mistral-7B)
are retired on OpenRouter (mid-2026). This uses their current-generation successors,
clearly labelled. Generation goes to OpenRouter; retrieval/embeddings stay LOCAL
(Ollama nomic-embed) so only generation incurs cost. Same real ReAct HotpotQA-distractor
harness and the SAME 50 tasks (offset 200) as tab:cnsr_local for an apples-to-apples pair.

Output: results/api_real/TableIV_measured/{cnsr_api.csv, cnsr_api_table.tex, manifest.json}
"""
from __future__ import annotations

import concurrent.futures as cf
import csv
import json
import os
import sys
import threading
import time
from pathlib import Path

import numpy as np
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[2]
load_dotenv(dotenv_path=str(ROOT / ".env"))
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

import litellm  # noqa: E402
litellm.suppress_debug_info = True
from experiments.revision.ollama_harness import load_pool, run_episode  # noqa: E402
from experiments.revision.ollama_2a import tau_b, task_bootstrap_cnsr_ci  # noqa: E402
from sage.stability.traces import _get_git_sha, _get_env_hash, now_iso  # noqa: E402

OUT = ROOT / "results" / "api_real" / "TableIV_measured"
OUT.mkdir(parents=True, exist_ok=True)
DATA_DIR = ROOT / "results" / "ollama_real" / "data"

TEMP = 0.0
N_TASKS, OFFSET, SEEDS = 50, 200, [0]     # same 50 tasks as tab:cnsr_local; temp0 => 1 seed
MAX_TURNS, K_RETRIEVE, N_WORKERS = 6, 4, 4

# (openrouter_slug, display, price_in_per_1M, price_out_per_1M) — OpenRouter live prices 2026-07-20
LADDER = [
    ("openai/gpt-4-turbo",                "GPT-4-Turbo",       10.0,  30.0),
    ("anthropic/claude-sonnet-4.5",       "Claude-Sonnet-4.5",  3.0,  15.0),
    ("google/gemini-2.5-flash",           "Gemini-2.5-Flash",   0.30,  2.5),
    ("openai/gpt-4o-mini",                "GPT-4o-mini",        0.15,  0.60),
    ("meta-llama/llama-3.1-70b-instruct", "Llama-3.1-70B",      0.40,  0.40),
    ("meta-llama/llama-3.1-8b-instruct",  "Llama-3.1-8B",       0.05,  0.08),
    ("mistralai/mistral-nemo",            "Mistral-Nemo",       0.019, 0.030),
]
PRICE_PROVIDER = "OpenRouter (passthrough provider list price)"
PRICE_DATE = "2026-07-20"


def openrouter_generate(model, prompt, temperature, seed, max_tokens=400, system=None):
    """Matches ollama_generate's signature/return. Retries once on transient error."""
    messages = ([{"role": "system", "content": system}] if system else []) + \
               [{"role": "user", "content": prompt}]
    last = None
    for _ in range(2):
        try:
            r = litellm.completion(model="openrouter/" + model, messages=messages,
                                   temperature=temperature, max_tokens=max_tokens, timeout=90)
            u = r.usage
            return {"text": r.choices[0].message.content or "",
                    "prompt_tokens": int(u.prompt_tokens), "completion_tokens": int(u.completion_tokens)}
        except Exception as e:
            last = e; time.sleep(2)
    raise last


def cost_usd(pt, ct, pin, pout):
    return (pt * pin + ct * pout) / 1e6


def run():
    log = []
    def L(m): print(m, flush=True); log.append(m)

    pool, split_hash = load_pool(N_TASKS, DATA_DIR, offset=OFFSET)
    L(f"[API] {len(pool)} tasks (offset={OFFSET}, same as tab:cnsr_local) split={split_hash[:12]}")
    est = sum(150 * cost_usd(2500, 200, pin, pout) for _, _, pin, pout in LADDER)
    L(f"[API] PROJECTION: {len(LADDER)} models x {N_TASKS} tasks x ~3 calls; est ~${est:.1f} (+5.5% fee).")

    t0 = time.time(); rows = []; per_model = {}
    for slug, disp, pin, pout in LADDER:
        L(f"[API] {disp} ({slug}) …")
        jobs = [(seed, ti, it) for seed in SEEDS for ti, it in enumerate(pool)]
        done = [0]; lock = threading.Lock()
        def work(job):
            seed, ti, it = job
            r = run_episode(it, slug, TEMP, seed * 1000 + ti, k=K_RETRIEVE, max_turns=MAX_TURNS,
                            controller=None, generate=openrouter_generate)
            c = cost_usd(r["prompt_tokens"], r["completion_tokens"], pin, pout)
            with lock:
                done[0] += 1
                if done[0] % 20 == 0:
                    L(f"    {disp}: {done[0]}/{len(jobs)} (${sum(x['cost_usd'] for x in rows):.2f} so far, {time.time()-t0:.0f}s)")
            return {"model": slug, "display": disp, "seed": seed, "task_idx": ti, "qid": it["qid"],
                    "em": r["em"], "f1": round(r["f1"], 4), "prompt_tokens": r["prompt_tokens"],
                    "completion_tokens": r["completion_tokens"], "cost_usd": c, "turns": r["turns_used"]}
        with cf.ThreadPoolExecutor(N_WORKERS) as ex:
            mrows = list(ex.map(work, jobs))
        rows.extend(mrows)
        em = np.array([r["em"] for r in mrows], float)
        sr = float(em.mean()); mc = float(np.mean([r["cost_usd"] for r in mrows]))
        ci_lo, ci_hi = task_bootstrap_cnsr_ci(mrows)
        per_model[slug] = {"display": disp, "success_rate": sr, "mean_cost": mc,
                           "cnsr_mean": sr / mc if mc > 0 else 0.0, "cnsr_ci_lo": ci_lo, "cnsr_ci_hi": ci_hi,
                           "mean_tokens": float(np.mean([r["prompt_tokens"] + r["completion_tokens"] for r in mrows])),
                           "spend_usd": float(np.sum([r["cost_usd"] for r in mrows])),
                           "price_in_per_1M": pin, "price_out_per_1M": pout, "n": len(mrows)}
        s = per_model[slug]
        L(f"    -> SR={sr:.1%} CNSR={s['cnsr_mean']:.0f} tokens={s['mean_tokens']:.0f} spend=${s['spend_usd']:.2f}")

    models = [m for m, _, _, _ in LADDER]
    srs = [per_model[m]["success_rate"] for m in models]
    cns = [per_model[m]["cnsr_mean"] for m in models]
    cnsr_order = sorted(models, key=lambda x: -per_model[x]["cnsr_mean"])
    sr_order = sorted(models, key=lambda x: -per_model[x]["success_rate"])
    for i, m in enumerate(cnsr_order, 1): per_model[m]["cnsr_rank"] = i
    for i, m in enumerate(sr_order, 1): per_model[m]["sr_rank"] = i
    tau, p = tau_b(srs, cns)
    total_spend = sum(per_model[m]["spend_usd"] for m in models)
    L(f"[API] tau_b = {tau:.4f} (p={p:.4f}, n={len(models)});  TOTAL SPEND ${total_spend:.2f}")

    with (OUT / "cnsr_api.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)

    lines = [r"\begin{table}[t]", r"\centering",
             r"\caption{\textbf{API-measured CNSR} (current-generation proprietary + open "
             r"models via OpenRouter; companion to Table~IV and tab:cnsr\_local). Real calls, "
             r"measured tokens $\times$ live list price (" + PRICE_DATE + r"); same 50 "
             r"HotpotQA-distractor tasks and EM grading as the local table; temp 0.0. NB: the "
             r"2024 Table~IV models are retired, so successors are used. Kendall's $\tau_b$ "
             r"(SR rank vs.\ CNSR rank) $=" + f"{tau:.3f}$ ($p={p:.3f}$)" + r".}",
             r"\label{tab:cnsr_api}", r"\begin{tabular}{lccccc}", r"\toprule",
             r"Model & CNSR & 95\% CI & SR (EM) & CNSR rank & SR rank \\", r"\midrule"]
    for m in cnsr_order:
        s = per_model[m]
        lines.append(f"{s['display']} & {s['cnsr_mean']:.0f} & [{s['cnsr_ci_lo']:.0f}, {s['cnsr_ci_hi']:.0f}] & "
                     f"{s['success_rate']:.0%} & {s['cnsr_rank']} & {s['sr_rank']} \\\\".replace("%", r"\%"))
    lines += [r"\midrule",
              r"\multicolumn{6}{l}{\textit{Kendall's $\tau_b$:} $" + f"{tau:.3f}$ ($p={p:.3f}$); "
              f"total measured spend \\${total_spend:.2f}" + r"} \\",
              r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    (OUT / "cnsr_api_table.tex").write_text("\n".join(lines) + "\n")

    (OUT / "manifest.json").write_text(json.dumps({
        "experiment": "API_measured_cnsr_TableIV_companion", "git_sha": _get_git_sha(),
        "env_hash": _get_env_hash(), "timestamp": now_iso(), "temperature": TEMP, "seeds": SEEDS,
        "n_tasks": N_TASKS, "offset": OFFSET, "dataset": "hotpot_qa/distractor/validation",
        "eval_split_hash": split_hash, "price_provider": PRICE_PROVIDER, "price_date": PRICE_DATE,
        "note_models_retired": "2024 Table IV models retired on OpenRouter; current-gen successors used.",
        "embeddings": "local Ollama nomic-embed (no API cost)",
        "per_model": per_model, "kendall_tau_b": {"tau": tau, "p": p, "n_configs": len(models)},
        "total_measured_spend_usd": total_spend, "runtime_s": round(time.time() - t0, 1), "log": log,
    }, indent=2, default=str))

    L(f"\n[API] === CNSR (measured) — companion to Table IV ===")
    for m in cnsr_order:
        s = per_model[m]
        L(f"  {s['display']:>18s} CNSR={s['cnsr_mean']:>6.0f} SR={s['success_rate']:>4.0%} "
          f"Cr={s['cnsr_rank']} SRr={s['sr_rank']} spend=${s['spend_usd']:.2f}")
    L(f"  tau_b={tau:.3f} (p={p:.3f})  TOTAL SPEND ${total_spend:.2f}")
    L(f"[API] wrote {OUT}/")


if __name__ == "__main__":
    run()
