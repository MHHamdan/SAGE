"""Track 2 / 2B — real E4/E4-T controller ablation on HotpotQA-distractor.

4 policies (NoControl / FixedSchedule / Threshold / Predictive) x 5 seeds x 50
eval tasks, agent=llama3.1:8b, temp=0.6. Predictor trained on 150 disjoint
training tasks (NoControl traces), task-stratified 5-fold CV on QUESTION ID +
assert_no_leakage. Everything real: EM success, embedding retrieval, monitor
signals from real outputs, measured Ollama tokens x hosted list price.

Reporting:
  (a) ALL 50 tasks  — the honest headline.
  (b) pathology subset — tasks where oscillation crosses its threshold >=1x under
      NoControl (where the controller can actually help).
Both with BCa 95% CIs, McNemar, Cohen's h. Fidelity reported as measured.

Output: results/ollama_real/2B/{2B.csv, 2B_table.tex, manifest.json, REPORT.md}
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
import scipy.stats
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from experiments.revision.ollama_harness import load_pool, run_episode  # noqa: E402
from sage.stability.traces import _get_git_sha, _get_env_hash, now_iso  # noqa: E402

OUT = ROOT / "results" / "ollama_real" / "2B"
OUT.mkdir(parents=True, exist_ok=True)
DATA_DIR = ROOT / "results" / "ollama_real" / "data"

AGENT = "llama3.1:8b"
TEMP = 0.6
N_TRAIN, N_EVAL = 150, 50
SEEDS = [0, 1, 2, 3, 4]
MAX_TURNS, K_RETRIEVE = 6, 4
OSC_THR, DRIFT_THR = 0.60, 0.50
FIRE_P, COOLDOWN, FIXED_K = 0.50, 2, 3
N_WORKERS = 4

# hosted list-price snapshot (recorded per manifest requirement)
PRICE = {"provider": "Together AI",
         "slug": "meta-llama/Llama-3.1-8B-Instruct-Turbo",
         "input_per_1k_usd": 0.00018, "output_per_1k_usd": 0.00018,
         "retrieval_date": "2026-07-18",
         "note": "public list price $0.18 / 1M tokens (input=output)"}


def token_cost(pt: int, ct: int) -> float:
    return (pt * PRICE["input_per_1k_usd"] + ct * PRICE["output_per_1k_usd"]) / 1000.0


# ── features from a real signal trace (8-dim, E5 order) ────────────────────────
def features_at(trace: list[dict], t: int) -> np.ndarray:
    cur = trace[t]
    prev = trace[t - 1] if t > 0 else cur
    maxdrift = max(s["drift_score"] for s in trace[:t + 1])
    return np.array([cur["drift_score"], cur["oscillation_score"], cur["fidelity_score"],
                     cur["convergence_progress"],
                     cur["drift_score"] - prev["drift_score"],
                     cur["oscillation_score"] - prev["oscillation_score"],
                     cur["fidelity_score"] - prev["fidelity_score"], maxdrift],
                    dtype=np.float64)


# ── controllers (real, stateful per episode) ──────────────────────────────────
class NoControlC:
    def __call__(self, signals, turn):
        return None


class FixedC:
    def __init__(self, k=FIXED_K):
        self.k = k
    def __call__(self, signals, turn):
        return "GoalReanchor" if turn % self.k == 0 else None


class ThresholdC:
    def __init__(self):
        self.last = -100
    def __call__(self, signals, turn):
        if turn - self.last < COOLDOWN:
            return None
        if signals["oscillation_score"] > OSC_THR:
            self.last = turn
            return "ForceReplan"
        if signals["drift_score"] > DRIFT_THR:
            self.last = turn
            return "GoalReanchor"
        return None


class PredictiveC:
    def __init__(self, scaler, model):
        self.scaler, self.model = scaler, model
        self.hist, self.last = [], -100
    def __call__(self, signals, turn):
        self.hist.append(signals)
        if turn - self.last < COOLDOWN:
            return None
        feat = features_at(self.hist, len(self.hist) - 1).reshape(1, -1)
        p = self.model.predict_proba(self.scaler.transform(feat))[0]
        pos = list(self.model.classes_).index(1) if 1 in self.model.classes_ else 0
        if float(p[pos]) >= FIRE_P:
            self.last = turn
            return "ForceReplan" if signals["oscillation_score"] >= signals["drift_score"] else "GoalReanchor"
        return None


POLICIES = {"NoControl": lambda pr: NoControlC(),
            "FixedSchedule": lambda pr: FixedC(),
            "Threshold": lambda pr: ThresholdC(),
            "Predictive": lambda pr: PredictiveC(*pr)}


# ── stats ──────────────────────────────────────────────────────────────────────
def bca_ci(x, stat=np.mean):
    x = np.asarray(x, float)
    if len(x) < 2 or np.allclose(x, x[0]):
        m = float(stat(x)); return m, m
    try:
        r = scipy.stats.bootstrap((x,), stat, method="BCa", confidence_level=0.95,
                                  n_resamples=5000, random_state=0)
        return float(r.confidence_interval.low), float(r.confidence_interval.high)
    except Exception:
        b = [float(stat(np.random.default_rng(i).choice(x, len(x)))) for i in range(2000)]
        return float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))


def cnsr_ci(em, cost):
    em, cost = np.asarray(em, float), np.asarray(cost, float)
    rng = np.random.default_rng(0); b = []
    for _ in range(4000):
        idx = rng.integers(0, len(em), len(em))
        mc = max(cost[idx].mean(), 1e-12); b.append(em[idx].mean() / mc)
    return float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))


def mcnemar(a, b):
    a, b = np.asarray(a, int), np.asarray(b, int)
    n01 = int(np.sum((a == 0) & (b == 1))); n10 = int(np.sum((a == 1) & (b == 0)))
    if n01 + n10 == 0:
        return 1.0
    chi2 = (abs(n01 - n10) - 1) ** 2 / (n01 + n10)
    return float(scipy.stats.chi2.sf(chi2, 1))


def cohens_h(p1, p2):
    return float(2 * np.arcsin(np.sqrt(np.clip(p1, 0, 1))) - 2 * np.arcsin(np.sqrt(np.clip(p2, 0, 1))))


# ── training the predictor on 150 disjoint tasks ──────────────────────────────
def train_predictor(train_items, log):
    records = []  # (qid, features, failed)
    def one(it, idx):
        r = run_episode(it, AGENT, TEMP, 100 + idx, k=K_RETRIEVE, max_turns=MAX_TURNS,
                        controller=None)
        failed = int(r["em"] == 0)
        return [(it["qid"], features_at(r["signal_trace"], t), failed)
                for t in range(len(r["signal_trace"]))]
    with cf.ThreadPoolExecutor(N_WORKERS) as ex:
        for chunk in ex.map(lambda p: one(*p), [(it, i) for i, it in enumerate(train_items)]):
            records.extend(chunk)
    X = np.vstack([r[1] for r in records]); y = np.array([r[2] for r in records])
    qids = sorted({r[0] for r in records}); q2i = {q: i for i, q in enumerate(qids)}
    groups = np.array([q2i[r[0]] for r in records])

    if len(set(y)) < 2:  # degenerate (all pass or all fail) — constant failure-rate head
        class _Const:
            classes_ = np.array([0, 1])
            def __init__(self, p): self.p = p
            def predict_proba(self, Z): return np.tile([1 - self.p, self.p], (len(Z), 1))
        log(f"[2B] WARNING: training labels single-class (pos_rate={y.mean():.2f}); "
            f"using constant failure-rate head.")
        return StandardScaler().fit(X), _Const(float(y.mean())), float("nan")

    # task-stratified 5-fold CV AUC on QUESTION ID
    aucs = []
    sgkf = StratifiedGroupKFold(n_splits=5)
    for tr, te in sgkf.split(X, y, groups):
        tr_q = {qids[g] for g in groups[tr]}; te_q = {qids[g] for g in groups[te]}
        assert not (tr_q & te_q), "LEAKAGE: shared qid across train/test"
        if len(set(y[tr])) < 2:
            continue
        sc = StandardScaler().fit(X[tr]); m = LogisticRegression(class_weight="balanced", max_iter=1000)
        m.fit(sc.transform(X[tr]), y[tr])
        if len(set(y[te])) > 1:
            pos = list(m.classes_).index(1)
            aucs.append(roc_auc_score(y[te], m.predict_proba(sc.transform(X[te]))[:, pos]))
    scaler = StandardScaler().fit(X)
    model = LogisticRegression(class_weight="balanced", max_iter=1000).fit(scaler.transform(X), y)
    log(f"[2B] predictor trained: {len(records)} turn-samples, pos_rate={y.mean():.2f}, "
        f"CV AUC={np.mean(aucs):.3f} (n_folds={len(aucs)}); "
        f"coef={dict(zip(['drift','osc','fid','conv','ddrift','dosc','dfid','maxdrift'], np.round(model.coef_[0],2)))}")
    return scaler, model, float(np.mean(aucs) if aucs else float('nan'))


# ── run all policies ──────────────────────────────────────────────────────────
def run(base_seed=0):
    log_lines = []
    def log(m):
        print(m, flush=True); log_lines.append(m)

    pool, split_hash = load_pool(N_TRAIN + N_EVAL, DATA_DIR)
    train_items, eval_items = pool[:N_TRAIN], pool[N_TRAIN:N_TRAIN + N_EVAL]
    assert not ({it["qid"] for it in train_items} & {it["qid"] for it in eval_items})
    eval_hash = __import__("hashlib").md5("|".join(it["qid"] for it in eval_items).encode()).hexdigest()
    log(f"[2B] split: {len(train_items)} train / {len(eval_items)} eval (disjoint qids); "
        f"eval_split_hash={eval_hash[:12]}")

    t0 = time.time()
    scaler, model, cv_auc = train_predictor(train_items, log)
    log(f"[2B] training done in {time.time()-t0:.0f}s")

    # build eval work items
    jobs = [(pol, seed, ti, it) for pol in POLICIES for seed in SEEDS
            for ti, it in enumerate(eval_items)]
    log(f"[2B] running {len(jobs)} eval episodes ({len(POLICIES)} policies x {len(SEEDS)} seeds x {len(eval_items)} tasks) …")
    rows = []
    done = [0]; lock = threading.Lock()

    def work(job):
        pol, seed, ti, it = job
        ctl = POLICIES[pol]((scaler, model))
        r = run_episode(it, AGENT, TEMP, seed * 1000 + ti, k=K_RETRIEVE,
                        max_turns=MAX_TURNS, controller=ctl)
        osc_max = max((s["oscillation_score"] for s in r["signal_trace"]), default=0.0)
        fid_min = min((s["fidelity_score"] for s in r["signal_trace"]), default=1.0)
        row = {"policy": pol, "seed": seed, "task_idx": ti, "qid": it["qid"],
               "em": r["em"], "f1": round(r["f1"], 4),
               "prompt_tokens": r["prompt_tokens"], "completion_tokens": r["completion_tokens"],
               "cost_usd": round(token_cost(r["prompt_tokens"], r["completion_tokens"]), 8),
               "turns": r["turns_used"], "n_interventions": r["n_interventions"],
               "osc_max": round(osc_max, 3), "fid_min": round(fid_min, 3)}
        with lock:
            done[0] += 1
            if done[0] % 50 == 0:
                log(f"    {done[0]}/{len(jobs)} episodes ({time.time()-t0:.0f}s)")
        return row

    with cf.ThreadPoolExecutor(N_WORKERS) as ex:
        rows = list(ex.map(work, jobs))
    log(f"[2B] all episodes done in {time.time()-t0:.0f}s")

    with (OUT / "2B.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)

    # pathology subset: tasks where NoControl oscillation crossed threshold >=1x (any seed)
    patho_tasks = {r["task_idx"] for r in rows
                   if r["policy"] == "NoControl" and r["osc_max"] > OSC_THR}
    log(f"[2B] pathology subset: {len(patho_tasks)}/{len(eval_items)} tasks "
        f"(oscillation>{OSC_THR} under NoControl)")

    # fidelity as measured
    all_fid = [s for r in rows for s in [r["fid_min"]]]
    log(f"[2B] fidelity as measured: min over episodes={min(all_fid):.3f}, "
        f"mean fid_min={np.mean(all_fid):.3f} "
        f"({'INERT (schema always valid)' if np.allclose(all_fid, 1.0) else 'variable'})")

    def summarize(subset_tasks):
        out = {}
        nc = [r for r in rows if r["policy"] == "NoControl" and r["task_idx"] in subset_tasks]
        nc_key = {(r["seed"], r["task_idx"]): r["em"] for r in nc}
        for pol in POLICIES:
            pr = [r for r in rows if r["policy"] == pol and r["task_idx"] in subset_tasks]
            em = [r["em"] for r in pr]; cost = [r["cost_usd"] for r in pr]
            comp = float(np.mean(em)); mc = float(np.mean(cost))
            clo, chi = bca_ci(em)
            cn = comp / mc if mc > 0 else 0.0
            cnlo, cnhi = cnsr_ci(em, cost)
            paired_p = {(r["seed"], r["task_idx"]): r["em"] for r in pr}
            common = sorted(set(nc_key) & set(paired_p))
            mp = mcnemar([nc_key[k] for k in common], [paired_p[k] for k in common]) if pol != "NoControl" else float('nan')
            h = cohens_h(comp, float(np.mean(em)) if pol == "NoControl" else np.mean([nc_key[k] for k in common]))
            out[pol] = {"n": len(pr), "completion": comp, "clo": clo, "chi": chi,
                        "mean_cost": mc, "cnsr": cn, "cnsr_lo": cnlo, "cnsr_hi": cnhi,
                        "interv": float(np.mean([r["n_interventions"] for r in pr])),
                        "mcnemar_p": mp,
                        "cohens_h_vs_nc": (cohens_h(comp, out["NoControl"]["completion"]) if "NoControl" in out else 0.0)}
        return out

    all_tasks = set(range(len(eval_items)))
    S_all = summarize(all_tasks)
    S_pat = summarize(patho_tasks) if patho_tasks else {}

    _write_table(S_all, S_pat, len(patho_tasks), len(eval_items), cv_auc)
    _write_manifest(split_hash, eval_hash, cv_auc, patho_tasks, eval_items, S_all, S_pat, all_fid, log_lines)
    _print_summary(S_all, S_pat, log)
    log(f"[2B] wrote {OUT}/")


def _fmt(s):
    return (f"{s['completion']:.1%} [{s['clo']:.1%},{s['chi']:.1%}]", f"{s['cnsr']:.3g}",
            f"{s['interv']:.1f}", ("—" if s['mcnemar_p']!=s['mcnemar_p'] else f"{s['mcnemar_p']:.3f}"))


def _write_table(S_all, S_pat, n_pat, n_tot, cv_auc):
    lines = [r"\begin{table}[t]", r"\centering",
             r"\caption{E4-T on real HotpotQA-distractor (llama3.1:8b, temp 0.6, 5 seeds; "
             r"BCa 95\% CIs). (a) all 50 tasks; (b) pathology subset "
             f"({n_pat}/{n_tot} tasks with oscillation$>${OSC_THR} under NoControl). "
             r"$p$: McNemar vs NoControl.}",
             r"\label{tab:e4t_ollama}", r"\begin{tabular}{lcccc}", r"\toprule",
             r"Policy & Completion (95\% CI) & CNSR & Interv. & McNemar $p$ \\",
             r"\midrule", r"\multicolumn{5}{l}{\textit{(a) All tasks}} \\"]
    for pol in POLICIES:
        c, cn, iv, p = _fmt(S_all[pol]); lines.append(f"{pol} & {c} & {cn} & {iv} & {p} \\\\".replace("%", r"\%"))
    if S_pat:
        lines += [r"\midrule", r"\multicolumn{5}{l}{\textit{(b) Pathology subset}} \\"]
        for pol in POLICIES:
            c, cn, iv, p = _fmt(S_pat[pol]); lines.append(f"{pol} & {c} & {cn} & {iv} & {p} \\\\".replace("%", r"\%"))
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    (OUT / "2B_table.tex").write_text("\n".join(lines) + "\n")


def _write_manifest(split_hash, eval_hash, cv_auc, patho, eval_items, S_all, S_pat, all_fid, log_lines):
    m = {"experiment": "2B_e4t_hotpotqa_ollama", "git_sha": _get_git_sha(),
         "env_hash": _get_env_hash(), "timestamp": now_iso(),
         "agent_model": AGENT, "temperature": TEMP, "seeds": SEEDS,
         "n_train": N_TRAIN, "n_eval": N_EVAL, "max_turns": MAX_TURNS, "k_retrieve": K_RETRIEVE,
         "controller_params": {"osc_thr": OSC_THR, "drift_thr": DRIFT_THR, "fire_p": FIRE_P,
                               "cooldown": COOLDOWN, "fixed_k": FIXED_K},
         "predictor_cv_auc": cv_auc,
         "dataset": "hotpot_qa/distractor/validation", "pool_split_hash": split_hash,
         "eval_split_hash": eval_hash,
         "n_pathology_tasks": len(patho), "pathology_task_idx": sorted(patho),
         "price_snapshot": PRICE,
         "fidelity_measured": {"min": float(min(all_fid)), "mean_fid_min": float(np.mean(all_fid)),
                               "inert": bool(np.allclose(all_fid, 1.0))},
         "summary_all": {k: {kk: (None if isinstance(vv, float) and vv != vv else vv)
                             for kk, vv in v.items()} for k, v in S_all.items()},
         "summary_pathology": {k: {kk: (None if isinstance(vv, float) and vv != vv else vv)
                                   for kk, vv in v.items()} for k, v in S_pat.items()},
         "cost_note": "CNSR uses inference token cost only (measured Ollama tokens x hosted "
                      "list price); NO GPU wall-clock in cost.",
         "log": log_lines}
    (OUT / "manifest.json").write_text(json.dumps(m, indent=2, default=str))


def _print_summary(S_all, S_pat, log):
    def mp(s):
        return "—" if s["mcnemar_p"] != s["mcnemar_p"] else f"{s['mcnemar_p']:.3f}"
    def line(pol, s):
        log(f"  {pol:>13s} {s['completion']:>7.1%} [{s['clo']:.1%},{s['chi']:.1%}] "
            f"CNSR={s['cnsr']:.3g} interv={s['interv']:.1f} McNemar={mp(s)} h={s['cohens_h_vs_nc']:+.2f}")
    log("\n[2B] === ALL TASKS ===")
    for pol, s in S_all.items():
        line(pol, s)
    if S_pat:
        log("[2B] === PATHOLOGY SUBSET ===")
        for pol, s in S_pat.items():
            line(pol, s)


if __name__ == "__main__":
    run()
