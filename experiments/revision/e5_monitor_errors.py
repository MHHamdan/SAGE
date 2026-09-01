"""Phase 0.3 — E5 monitor error analysis at k=5 (Supp B.9).

Regenerates the E5 combined-predictor out-of-fold (OOF) predictions
deterministically (seed=42, same task-stratified 5-fold protocol as the
committed E5 run), then produces:

  * confusion matrix (TP/FP/FN/TN) at the deployed operating threshold p>=0.50
    (= PredictiveController.fire_at_p), plus the Youden-J optimal threshold;
  * precision-recall curve + average precision;
  * qualitative breakdown of the 20 highest-confidence false positives and the
    20 highest-confidence false negatives, each clustered by the driving monitor
    signal (drift / oscillation / fidelity) and the failure mechanism it maps to.

Output: results/e5_monitor_errors/{confusion_k5.csv, pr_curve.pdf,
        fp_fn_breakdown.md, manifest.json}
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
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, precision_recall_curve,
                             roc_curve, roc_auc_score)
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from experiments.e5_predictive_validation import E5_CONFIG, generate_task_traces  # noqa: E402
from sage.stability.predictor import build_training_data, assert_no_leakage, FEATURE_NAMES  # noqa: E402
from sage.stability.traces import _get_git_sha, _get_env_hash, now_iso  # noqa: E402

OUT = ROOT / "results" / "e5_monitor_errors"
OUT.mkdir(parents=True, exist_ok=True)

K = 5
OP_THRESHOLD = 0.50  # deployed operating point (PredictiveController.fire_at_p)


def driving_signal(feat: np.ndarray) -> str:
    """Return the dominant monitor signal for a feature row.

    feat = [drift, osc, fid, conv, ddrift, dosc, dfid, maxdrift].
    Severity: drift=drift_score, oscillation=osc_score, fidelity=(1-fid_score).
    """
    sev = {"drift": float(feat[0]),
           "oscillation": float(feat[1]),
           "fidelity": float(1.0 - feat[2])}
    return max(sev, key=sev.get)


MECHANISM = {
    "drift": "goal drift — state embedding rotated away from goal (A3 mechanism)",
    "oscillation": "action cycling — agent stuck repeating a small action set",
    "fidelity": "schema/tool-output infidelity — malformed or invalid tool responses",
}


def run(base_seed: int = 42) -> None:
    cfg = dict(E5_CONFIG)
    print(f"[0.3] generating E5 traces (seed={base_seed}) …")
    records = generate_task_traces(cfg["n_violation_tasks"], cfg["n_control_tasks"],
                                   cfg, base_seed)

    X, y, groups = build_training_data(records, K, cfg["total_turns"])
    # parallel metadata aligned to records / rows of X
    meta = [(r.task_id, r.turn) for r in records]
    unique_task_ids = list({r.task_id for r in records})
    task_to_int = {t: i for i, t in enumerate(unique_task_ids)}
    int_to_task = {v: k for k, v in task_to_int.items()}

    sgkf = StratifiedGroupKFold(n_splits=cfg["n_cv_folds"])
    oof_score = np.full(len(y), np.nan)
    for train_idx, test_idx in sgkf.split(X, y, groups):
        train_tasks = {int_to_task[g] for g in groups[train_idx]}
        test_tasks = {int_to_task[g] for g in groups[test_idx]}
        assert_no_leakage(train_tasks, test_tasks)
        scaler = StandardScaler()
        Xtr = scaler.fit_transform(X[train_idx])
        Xte = scaler.transform(X[test_idx])
        model = LogisticRegression(class_weight="balanced", max_iter=1000, random_state=42)
        model.fit(Xtr, y[train_idx])
        pos = list(model.classes_).index(1)
        oof_score[test_idx] = model.predict_proba(Xte)[:, pos]

    mask = ~np.isnan(oof_score)
    y_true = y[mask].astype(int)
    scores = oof_score[mask]
    Xm = X[mask]
    meta_m = [meta[i] for i in np.where(mask)[0]]

    auc = roc_auc_score(y_true, scores)
    ap = average_precision_score(y_true, scores)

    # ---- confusion matrix at operating threshold ----
    def confusion(thr: float):
        pred = (scores >= thr).astype(int)
        tp = int(np.sum((pred == 1) & (y_true == 1)))
        fp = int(np.sum((pred == 1) & (y_true == 0)))
        fn = int(np.sum((pred == 0) & (y_true == 1)))
        tn = int(np.sum((pred == 0) & (y_true == 0)))
        return tp, fp, fn, tn

    tp, fp, fn, tn = confusion(OP_THRESHOLD)
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    spec = tn / (tn + fp) if (tn + fp) else 0.0

    # Youden-J optimal threshold (from ROC)
    fpr, tpr, roc_thr = roc_curve(y_true, scores)
    j = tpr - fpr
    j_thr = float(roc_thr[int(np.argmax(j))])
    tpj, fpj, fnj, tnj = confusion(j_thr)

    # ---- write confusion CSV ----
    with (OUT / "confusion_k5.csv").open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["threshold_name", "threshold", "TP", "FP", "FN", "TN",
                    "precision", "recall", "specificity", "f1", "auc", "ap",
                    "n_pos", "n_neg", "n_total"])
        w.writerow(["operating_p0.50", OP_THRESHOLD, tp, fp, fn, tn,
                    round(prec, 4), round(rec, 4), round(spec, 4), round(f1, 4),
                    round(auc, 4), round(ap, 4), int(y_true.sum()),
                    int((y_true == 0).sum()), len(y_true)])
        pj = tpj / (tpj + fpj) if (tpj + fpj) else 0.0
        rj = tpj / (tpj + fnj) if (tpj + fnj) else 0.0
        f1j = 2 * pj * rj / (pj + rj) if (pj + rj) else 0.0
        sj = tnj / (tnj + fpj) if (tnj + fpj) else 0.0
        w.writerow([f"youden_j_{j_thr:.3f}", round(j_thr, 4), tpj, fpj, fnj, tnj,
                    round(pj, 4), round(rj, 4), round(sj, 4), round(f1j, 4),
                    round(auc, 4), round(ap, 4), int(y_true.sum()),
                    int((y_true == 0).sum()), len(y_true)])

    # ---- PR curve ----
    p_arr, r_arr, _ = precision_recall_curve(y_true, scores)
    baseline = float(y_true.mean())
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(r_arr, p_arr, color="#5cb85c", lw=2, label=f"combined (AP={ap:.3f})")
    ax.axhline(baseline, color="gray", ls="--", lw=0.9,
               label=f"prevalence baseline ({baseline:.3f})")
    ax.scatter([rec], [prec], color="#d9534f", zorder=5,
               label=f"operating point p=0.50 (P={prec:.2f}, R={rec:.2f})")
    ax.set_xlabel("Recall"); ax.set_ylabel("Precision")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_title(f"E5 Precision–Recall (combined monitor, k={K})")
    ax.legend(fontsize=8, loc="upper right")
    fig.tight_layout(); fig.savefig(OUT / "pr_curve.pdf", bbox_inches="tight")
    plt.close(fig)

    # ---- FP / FN breakdown ----
    pred = (scores >= OP_THRESHOLD).astype(int)
    fp_idx = np.where((pred == 1) & (y_true == 0))[0]
    fn_idx = np.where((pred == 0) & (y_true == 1))[0]
    # highest-confidence FP = highest score; highest-confidence FN = lowest score
    fp_top = fp_idx[np.argsort(-scores[fp_idx])][:20]
    fn_top = fn_idx[np.argsort(scores[fn_idx])][:20]

    def cluster(idxs):
        from collections import Counter
        c = Counter(driving_signal(Xm[i]) for i in idxs)
        return c

    fp_clusters = cluster(fp_top)
    fn_clusters = cluster(fn_top)
    # cluster over ALL errors too (not just top-20) for context
    fp_all_clusters = cluster(fp_idx)
    fn_all_clusters = cluster(fn_idx)

    def fmt_rows(idxs):
        lines = []
        for rank, i in enumerate(idxs, 1):
            tid, turn = meta_m[i]
            sig = driving_signal(Xm[i])
            f = Xm[i]
            lines.append(
                f"| {rank} | `{tid}` | {turn} | {scores[i]:.3f} | {sig} | "
                f"{f[0]:.3f} | {f[1]:.3f} | {f[2]:.3f} | {f[7]:.3f} |")
        return lines

    md = [
        "# E5 Monitor Error Analysis — k=5 (Supp B.9)",
        "",
        f"- Deterministic regeneration, `seed={base_seed}`, task-stratified 5-fold CV "
        f"(`StratifiedGroupKFold`, no leakage). git `{_get_git_sha()}`.",
        f"- Pooled OOF predictions: **{len(y_true)}** samples "
        f"({int(y_true.sum())} positives / {int((y_true==0).sum())} negatives, "
        f"prevalence {baseline:.3f}).",
        f"- Combined-model AUC={auc:.3f}, AP={ap:.3f}.",
        "",
        "## Confusion matrix @ operating threshold p ≥ 0.50 (deployed fire_at_p)",
        "",
        "|        | pred fail | pred ok |",
        "|--------|-----------|---------|",
        f"| **actual fail** | TP={tp} | FN={fn} |",
        f"| **actual ok**   | FP={fp} | TN={tn} |",
        "",
        f"Precision={prec:.3f} · Recall={rec:.3f} · Specificity={spec:.3f} · F1={f1:.3f}",
        "",
        f"Youden-J optimal threshold = {j_thr:.3f} → TP={tpj}, FP={fpj}, FN={fnj}, TN={tnj} "
        f"(P={pj:.3f}, R={rj:.3f}).",
        "",
        "## False positives — driving-signal clusters",
        f"- Top-20 FP: " + ", ".join(f"{k}={v}" for k, v in fp_clusters.most_common()),
        f"- All {len(fp_idx)} FP: " + ", ".join(f"{k}={v}" for k, v in fp_all_clusters.most_common()),
        "",
        "### 20 highest-confidence false positives",
        "| # | task | turn | score | driving signal | drift | osc | fid | maxdrift |",
        "|---|------|------|-------|----------------|-------|-----|-----|----------|",
        *fmt_rows(fp_top),
        "",
        "## False negatives — driving-signal clusters",
        f"- Top-20 FN: " + ", ".join(f"{k}={v}" for k, v in fn_clusters.most_common()),
        f"- All {len(fn_idx)} FN: " + ", ".join(f"{k}={v}" for k, v in fn_all_clusters.most_common()),
        "",
        "### 20 highest-confidence false negatives (lowest predicted P among true failures)",
        "| # | task | turn | score | driving signal | drift | osc | fid | maxdrift |",
        "|---|------|------|-------|----------------|-------|-----|-----|----------|",
        *fmt_rows(fn_top),
        "",
        "## Failure-mechanism mapping",
        *[f"- **{k}** → {MECHANISM[k]}" for k in ("drift", "oscillation", "fidelity")],
    ]
    (OUT / "fp_fn_breakdown.md").write_text("\n".join(md))

    manifest = {
        "experiment": "0.3_e5_monitor_errors", "git_sha": _get_git_sha(),
        "env_hash": _get_env_hash(), "timestamp": now_iso(), "base_seed": base_seed,
        "k": K, "operating_threshold": OP_THRESHOLD, "youden_j_threshold": round(j_thr, 4),
        "config": {kk: cfg[kk] for kk in ("n_violation_tasks", "n_control_tasks",
                                          "total_turns", "n_cv_folds",
                                          "drift_rate_violation", "completion_drift_threshold")},
        "auc": round(float(auc), 4), "ap": round(float(ap), 4),
        "confusion_op": {"TP": tp, "FP": fp, "FN": fn, "TN": tn},
    }
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2))

    print(f"[0.3] AUC={auc:.3f} AP={ap:.3f} | @p0.50: TP={tp} FP={fp} FN={fn} TN={tn} "
          f"P={prec:.3f} R={rec:.3f}")
    print(f"      FP clusters(top20)={dict(fp_clusters)}  FN clusters(top20)={dict(fn_clusters)}")
    print(f"      Wrote {OUT}/")


if __name__ == "__main__":
    run()
