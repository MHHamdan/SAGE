"""P1 — controller on the long-horizon, TERMINAL-DERAILMENT regime (decisive test).

The committed E4 is a synthetic goal-drift simulation (no LLM). Its regime is:
long horizon + a pathology (drift) that is TERMINAL under NoControl and reversible
by intervention. HotpotQA-2B did not reproduce the effect because its forced-answer
step gave every derailed agent a free recovery, masking terminal failure.

P1 restores that regime on the real agent: long turn budget, NO forced answer (an
agent that never commits an answer fails terminally = derailment), strongest local
model (qwen2.5:32b), temp 0.6, 5 seeds. We test whether the controller RESCUES
derailed episodes, and we diagnose the base rate of terminal-vs-recovered pathology.
No tuning; if it is null here too, that is the finding.

Output: results/ollama_real/P1/{P1.csv, P1_table.tex, manifest.json, diagnosis.md}
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
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from experiments.revision.ollama_harness import load_pool, run_episode  # noqa: E402
from experiments.revision.ollama_2b import (  # noqa: E402
    features_at, NoControlC, ThresholdC, PredictiveC,
    bca_ci, cnsr_ci, mcnemar, cohens_h, OSC_THR,
)
from sage.stability.traces import _get_git_sha, _get_env_hash, now_iso  # noqa: E402

OUT = ROOT / "results" / "ollama_real" / "P1"
OUT.mkdir(parents=True, exist_ok=True)
DATA_DIR = ROOT / "results" / "ollama_real" / "data"

AGENT = "qwen2.5:32b"          # strongest local model
TEMP = 0.6
SEEDS = [0, 1, 2, 3, 4]
N_EVAL, N_TRAIN = 12, 12
OFFSET = 250                   # disjoint from 2B (0-199) and 2A (200-249)
MAX_TURNS, K_RETRIEVE = 8, 4   # long-horizon; NO forced answer
FORCE_FINAL = False            # <-- terminal derailment enabled
N_WORKERS = 4

PRICE = {"provider": "Together AI (serverless size-tier)",
         "slug": "Qwen/Qwen2.5-32B-Instruct", "per_1M_usd": 0.80,
         "retrieval_date": "2026-07-18"}


def token_cost(pt, ct):
    return (pt + ct) * PRICE["per_1M_usd"] / 1e6


def episode(item, seed, ti, controller):
    r = run_episode(item, AGENT, TEMP, seed * 1000 + ti, k=K_RETRIEVE,
                    max_turns=MAX_TURNS, controller=controller, force_final_answer=FORCE_FINAL)
    osc_max = max((s["oscillation_score"] for s in r["signal_trace"]), default=0.0)
    return {"em": r["em"], "f1": round(r["f1"], 4), "answered": int(r["answered"]),
            "derailed": int(r["derailed"]), "osc_max": round(osc_max, 3),
            "turns": r["turns_used"], "n_interventions": r["n_interventions"],
            "prompt_tokens": r["prompt_tokens"], "completion_tokens": r["completion_tokens"],
            "cost_usd": round(token_cost(r["prompt_tokens"], r["completion_tokens"]), 8)}


def train_predictor(train_items, t0, L):
    records = []
    def one(it, idx):
        r = run_episode(it, AGENT, TEMP, 700 + idx, k=K_RETRIEVE, max_turns=MAX_TURNS,
                        controller=None, force_final_answer=FORCE_FINAL)
        failed = int(r["em"] == 0)
        return [(it["qid"], features_at(r["signal_trace"], t), failed)
                for t in range(len(r["signal_trace"]))]
    with cf.ThreadPoolExecutor(N_WORKERS) as ex:
        for chunk in ex.map(lambda p: one(*p), [(it, i) for i, it in enumerate(train_items)]):
            records.extend(chunk)
    X = np.vstack([r[1] for r in records]); y = np.array([r[2] for r in records])
    if len(set(y)) < 2:
        class _Const:
            classes_ = np.array([0, 1])
            def __init__(self, p): self.p = p
            def predict_proba(self, Z): return np.tile([1 - self.p, self.p], (len(Z), 1))
        L(f"[P1] predictor single-class (pos_rate={y.mean():.2f}); constant head.")
        return StandardScaler().fit(X), _Const(float(y.mean())), float("nan")
    qids = sorted({r[0] for r in records}); q2i = {q: i for i, q in enumerate(qids)}
    groups = np.array([q2i[r[0]] for r in records])
    aucs = []
    for tr, te in StratifiedGroupKFold(n_splits=5).split(X, y, groups):
        if len(set(y[tr])) < 2:
            continue
        sc = StandardScaler().fit(X[tr]); m = LogisticRegression(class_weight="balanced", max_iter=1000)
        m.fit(sc.transform(X[tr]), y[tr])
        if len(set(y[te])) > 1:
            aucs.append(roc_auc_score(y[te], m.predict_proba(sc.transform(X[te]))[:, list(m.classes_).index(1)]))
    scaler = StandardScaler().fit(X)
    model = LogisticRegression(class_weight="balanced", max_iter=1000).fit(scaler.transform(X), y)
    L(f"[P1] predictor: {len(records)} turn-samples pos_rate={y.mean():.2f} CV-AUC={np.mean(aucs) if aucs else float('nan'):.3f}")
    return scaler, model, float(np.mean(aucs) if aucs else float("nan"))


POLICIES = ["NoControl", "Threshold", "Predictive"]


def run():
    log = []
    def L(m): print(m, flush=True); log.append(m)
    n_ep = len(POLICIES) * len(SEEDS) * N_EVAL + N_TRAIN
    L(f"[P1] PROJECTION: {n_ep} episodes ({AGENT}, ~{MAX_TURNS} turns, temp {TEMP}); "
      f"32B ~serial ~90s/episode => ~{n_ep*90/3600:.1f} h. $0 local (list-price accounting only).")

    pool, split_hash = load_pool(N_TRAIN + N_EVAL, DATA_DIR, offset=OFFSET)
    train_items, eval_items = pool[:N_TRAIN], pool[N_TRAIN:N_TRAIN + N_EVAL]
    assert not ({it["qid"] for it in train_items} & {it["qid"] for it in eval_items})
    eval_hash = __import__("hashlib").md5("|".join(it["qid"] for it in eval_items).encode()).hexdigest()
    L(f"[P1] split {len(train_items)} train / {len(eval_items)} eval (disjoint) eval_hash={eval_hash[:12]}")

    t0 = time.time()
    scaler, model, cv_auc = train_predictor(train_items, t0, L)

    ctl_factory = {"NoControl": lambda: NoControlC(),
                   "Threshold": lambda: ThresholdC(),
                   "Predictive": lambda: PredictiveC(scaler, model)}
    jobs = [(pol, seed, ti, it) for pol in POLICIES for seed in SEEDS
            for ti, it in enumerate(eval_items)]
    L(f"[P1] running {len(jobs)} eval episodes …")
    done = [0]; lock = threading.Lock()
    def work(job):
        pol, seed, ti, it = job
        row = {"policy": pol, "seed": seed, "task_idx": ti, "qid": it["qid"],
               **episode(it, seed, ti, ctl_factory[pol]())}
        with lock:
            done[0] += 1
            if done[0] % 20 == 0:
                L(f"    {done[0]}/{len(jobs)} ({time.time()-t0:.0f}s)")
        return row
    with cf.ThreadPoolExecutor(N_WORKERS) as ex:
        rows = list(ex.map(work, jobs))
    L(f"[P1] eval done in {time.time()-t0:.0f}s")

    with (OUT / "P1.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)

    # ── per-policy metrics ──
    nc = [r for r in rows if r["policy"] == "NoControl"]
    nc_key = {(r["seed"], r["task_idx"]): r["em"] for r in nc}
    summ = {}
    for pol in POLICIES:
        pr = [r for r in rows if r["policy"] == pol]
        em = [r["em"] for r in pr]; cost = [r["cost_usd"] for r in pr]
        comp = float(np.mean(em)); mc = float(np.mean(cost))
        clo, chi = bca_ci(em); cnlo, cnhi = cnsr_ci(em, cost)
        paired = {(r["seed"], r["task_idx"]): r["em"] for r in pr}
        common = sorted(set(nc_key) & set(paired))
        mp = mcnemar([nc_key[k] for k in common], [paired[k] for k in common]) if pol != "NoControl" else float("nan")
        summ[pol] = {"completion": comp, "clo": clo, "chi": chi, "mean_cost": mc,
                     "cnsr": comp / mc if mc > 0 else 0.0, "cnsr_lo": cnlo, "cnsr_hi": cnhi,
                     "interv": float(np.mean([r["n_interventions"] for r in pr])),
                     "derail_rate": float(np.mean([r["derailed"] for r in pr])),
                     "mcnemar_p": mp,
                     "cohens_h_vs_nc": cohens_h(comp, summ["NoControl"]["completion"]) if "NoControl" in summ else 0.0}

    # ── TERMINALITY diagnosis (NoControl) ──
    nc_em = np.array([r["em"] for r in nc]); nc_der = np.array([r["derailed"] for r in nc])
    nc_osc = np.array([r["osc_max"] for r in nc])
    fails = nc_em == 0
    patho = nc_osc > OSC_THR
    diag = {
        "base_completion": float(nc_em.mean()),
        "derail_rate": float(nc_der.mean()),
        "n_fail": int(fails.sum()),
        "frac_fail_derailed": float(nc_der[fails].mean()) if fails.sum() else 0.0,
        "frac_fail_wrong_answer": float((1 - nc_der[fails]).mean()) if fails.sum() else 0.0,
        "n_pathology_episodes": int(patho.sum()),
        "pathology_terminal_rate": float(nc_der[patho].mean()) if patho.sum() else 0.0,
        "pathology_recovered_rate": float((1 - nc_der[patho]).mean()) if patho.sum() else 0.0,
    }
    # does the controller rescue tasks that derailed under NoControl?
    derailed_keys = {(r["seed"], r["task_idx"]) for r in nc if r["derailed"]}
    rescue = {}
    for pol in ["Threshold", "Predictive"]:
        pr = [r for r in rows if r["policy"] == pol and (r["seed"], r["task_idx"]) in derailed_keys]
        rescue[pol] = {"n": len(pr), "completion_on_nc_derailed": float(np.mean([r["em"] for r in pr])) if pr else 0.0,
                       "still_derailed": float(np.mean([r["derailed"] for r in pr])) if pr else 0.0}

    _write_table(summ, cv_auc)
    _write_diagnosis(diag, rescue, summ, cv_auc)
    manifest = {
        "experiment": "P1_longhorizon_terminal_derailment", "git_sha": _get_git_sha(),
        "env_hash": _get_env_hash(), "timestamp": now_iso(), "agent": AGENT, "temp": TEMP,
        "seeds": SEEDS, "n_eval": N_EVAL, "n_train": N_TRAIN, "max_turns": MAX_TURNS,
        "force_final_answer": FORCE_FINAL, "offset": OFFSET, "eval_split_hash": eval_hash,
        "predictor_cv_auc": cv_auc, "price_snapshot": PRICE,
        "regime_note": "long-horizon, NO forced answer => derailment is terminal (the E4 "
                       "failure mode). Contrast 2B which forced a final answer.",
        "summary": summ, "terminality_diagnosis": diag, "controller_rescue": rescue,
        "runtime_s": round(time.time() - t0, 1), "log": log,
    }
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2, default=str))

    L("\n[P1] === completion by policy (long-horizon, terminal derailment) ===")
    for pol in POLICIES:
        s = summ[pol]
        mp = "—" if s["mcnemar_p"] != s["mcnemar_p"] else f"{s['mcnemar_p']:.3f}"
        L(f"  {pol:>11s} comp={s['completion']:.1%} [{s['clo']:.1%},{s['chi']:.1%}] "
          f"CNSR={s['cnsr']:.0f} interv={s['interv']:.1f} derail={s['derail_rate']:.1%} "
          f"McNemar={mp} h={s['cohens_h_vs_nc']:+.2f}")
    L("[P1] === terminality diagnosis (NoControl) ===")
    L(f"  base completion={diag['base_completion']:.1%}  derail rate={diag['derail_rate']:.1%}")
    L(f"  of failures: {diag['frac_fail_derailed']:.0%} derailed (terminal) vs "
      f"{diag['frac_fail_wrong_answer']:.0%} wrong-answer (capability)")
    L(f"  pathology episodes (osc>{OSC_THR}): {diag['n_pathology_episodes']}; "
      f"terminal={diag['pathology_terminal_rate']:.0%} recovered={diag['pathology_recovered_rate']:.0%}")
    for pol in ["Threshold", "Predictive"]:
        L(f"  rescue of NC-derailed by {pol}: completion={rescue[pol]['completion_on_nc_derailed']:.1%} "
          f"still_derailed={rescue[pol]['still_derailed']:.1%} (n={rescue[pol]['n']})")
    L(f"[P1] wrote {OUT}/")


def _write_table(summ, cv_auc):
    lines = [r"\begin{table}[t]", r"\centering",
             r"\caption{P1: controller on the long-horizon, terminal-derailment regime "
             r"(qwen2.5:32b, temp 0.6, 5 seeds, no forced answer; BCa 95\% CIs). "
             f"Predictor CV-AUC $={cv_auc:.2f}$. $p$: McNemar vs.\\ NoControl.}}",
             r"\label{tab:p1_longhorizon}", r"\begin{tabular}{lccccc}", r"\toprule",
             r"Policy & Completion (95\% CI) & CNSR & Interv. & Derail & McNemar $p$ \\",
             r"\midrule"]
    for pol in POLICIES:
        s = summ[pol]
        mp = "---" if s["mcnemar_p"] != s["mcnemar_p"] else f"{s['mcnemar_p']:.3f}"
        lines.append(f"{pol} & {s['completion']:.1%} [{s['clo']:.1%}, {s['chi']:.1%}] & "
                     f"{s['cnsr']:.0f} & {s['interv']:.1f} & {s['derail_rate']:.0%} & {mp} \\\\".replace("%", r"\%"))
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    (OUT / "P1_table.tex").write_text("\n".join(lines) + "\n")


def _write_diagnosis(diag, rescue, summ, cv_auc):
    md = [
        "# P1 — Long-horizon terminal-derailment diagnosis",
        "",
        f"Regime: qwen2.5:32b, temp 0.6, 5 seeds, max_turns={MAX_TURNS}, **no forced answer** "
        "(derailment is terminal). This restores the E4 failure mode that 2B's forced-answer masked.",
        "",
        "## Does the monitored pathology precede *unrecoverable* failure?",
        f"- NoControl base completion: **{diag['base_completion']:.1%}**; derail rate "
        f"{diag['derail_rate']:.1%}.",
        f"- Of NoControl failures ({diag['n_fail']}): **{diag['frac_fail_derailed']:.0%} are "
        f"derailments** (ran out of budget without answering = terminal) vs "
        f"**{diag['frac_fail_wrong_answer']:.0%} wrong-answers** (answered but incorrect = "
        f"capability-limited).",
        f"- Pathology episodes (oscillation > {OSC_THR}): {diag['n_pathology_episodes']}. Of these, "
        f"**{diag['pathology_terminal_rate']:.0%} are terminal** (derail) and "
        f"**{diag['pathology_recovered_rate']:.0%} self-recover** (still answer).",
        "",
        "## Does the controller rescue the derailed episodes?",
    ]
    for pol in ["Threshold", "Predictive"]:
        md.append(f"- On the (seed,task) episodes that derailed under NoControl, **{pol}** completion "
                  f"= {rescue[pol]['completion_on_nc_derailed']:.1%}, still-derailed "
                  f"{rescue[pol]['still_derailed']:.1%} (n={rescue[pol]['n']}).")
    md += [
        "",
        "## Verdict",
        f"NoControl {summ['NoControl']['completion']:.1%} vs Predictive "
        f"{summ['Predictive']['completion']:.1%} (McNemar "
        + ("n/a" if summ['Predictive']['mcnemar_p'] != summ['Predictive']['mcnemar_p'] else f"p={summ['Predictive']['mcnemar_p']:.3f}")
        + f", Cohen's h={summ['Predictive']['cohens_h_vs_nc']:+.2f}). "
        "The controller helps iff derailment is both common and rescuable; the numbers above "
        "show whether that holds. Reported straight, no tuning.",
    ]
    (OUT / "diagnosis.md").write_text("\n".join(md))


if __name__ == "__main__":
    run()
