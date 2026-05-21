"""Predictive monitor head — k-step-ahead failure predictor.

Trains a logistic regression (+ optional MLP ablation) over monitor features
to estimate P(failure within next k turns) given current MonitorSignals.

Anti-leakage guarantee: split is ALWAYS by task_id.  See assert_no_leakage().

Feature vector (8 dims per turn):
  [drift, oscillation, fidelity, convergence_progress,
   Δdrift, Δoscillation, Δfidelity,
   max_drift_so_far]
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from sklearn.calibration import calibration_curve
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

from .controller import MonitorSignals

logger = logging.getLogger(__name__)

FEATURE_NAMES = [
    "drift_score",
    "oscillation_score",
    "fidelity_score",
    "convergence_progress",
    "delta_drift",
    "delta_oscillation",
    "delta_fidelity",
    "max_drift_so_far",
]


# ── Feature extraction ─────────────────────────────────────────────────────────

def extract_features(
    signals: MonitorSignals,
    history: list[MonitorSignals],
) -> np.ndarray:
    """Extract 8-dimensional feature vector from current + delta signals."""
    if history:
        prev = history[-1]
        delta_drift = signals.drift_score - prev.drift_score
        delta_osc = signals.oscillation_score - prev.oscillation_score
        delta_fid = signals.fidelity_score - prev.fidelity_score
        all_drifts = [s.drift_score for s in history] + [signals.drift_score]
        max_drift = max(all_drifts)
    else:
        delta_drift = 0.0
        delta_osc = 0.0
        delta_fid = 0.0
        max_drift = signals.drift_score

    return np.array([
        signals.drift_score,
        signals.oscillation_score,
        signals.fidelity_score,
        signals.convergence_progress,
        delta_drift,
        delta_osc,
        delta_fid,
        max_drift,
    ], dtype=np.float64)


# ── Training data construction ─────────────────────────────────────────────────

@dataclass
class TraceRecord:
    """Per-turn record for predictor training."""
    task_id: str
    turn: int
    features: np.ndarray
    task_failed: bool  # ultimate outcome of the episode


def build_training_data(
    traces: list[TraceRecord],
    k: int,
    total_turns: int = 50,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build (X, y, groups) for k-step-ahead failure prediction.

    y_t = 1 iff task_failed AND turn t is within k turns of turn total_turns
          (i.e., turn >= total_turns - k + 1).
    groups contains task_id for group-stratified CV split.

    Returns
    -------
    X : (n_samples, n_features)
    y : (n_samples,) binary
    groups : (n_samples,) — task_id encoded as int for sklearn GroupKFold
    """
    X, y, groups = [], [], []
    unique_tasks = sorted({r.task_id for r in traces})
    task_to_int = {t: i for i, t in enumerate(unique_tasks)}

    failure_onset = total_turns - k + 1  # first turn that gets label 1

    for rec in traces:
        label = 1 if (rec.task_failed and rec.turn >= failure_onset) else 0
        X.append(rec.features)
        y.append(label)
        groups.append(task_to_int[rec.task_id])

    return (
        np.array(X, dtype=np.float64),
        np.array(y, dtype=np.int32),
        np.array(groups, dtype=np.int32),
    )


def assert_no_leakage(
    train_task_ids: set[str],
    test_task_ids: set[str],
) -> None:
    overlap = train_task_ids & test_task_ids
    assert len(overlap) == 0, (
        f"Data leakage: {len(overlap)} task IDs appear in both train and test: "
        f"{sorted(overlap)[:5]}"
    )


# ── Predictor class ────────────────────────────────────────────────────────────

class FailurePredictor:
    """k-step-ahead failure predictor (logistic regression by default).

    Parameters
    ----------
    k : int
        Lead time in turns.
    model_type : str
        "logistic" (default, interpretable) or "mlp" (nonlinear ablation).
    random_state : int
        Seed for reproducibility.
    """

    def __init__(
        self,
        k: int = 5,
        model_type: str = "logistic",
        random_state: int = 42,
    ) -> None:
        self.k = k
        self.model_type = model_type
        self.random_state = random_state

        self._scaler = StandardScaler()
        self._model = self._make_model()
        self._fitted = False
        self._total_turns: int = 50

    def _make_model(self):
        if self.model_type == "logistic":
            return LogisticRegression(
                class_weight="balanced",
                max_iter=1000,
                random_state=self.random_state,
                solver="lbfgs",
            )
        elif self.model_type == "mlp":
            return MLPClassifier(
                hidden_layer_sizes=(32,),
                max_iter=500,
                random_state=self.random_state,
                early_stopping=True,
                validation_fraction=0.1,
            )
        else:
            raise ValueError(f"Unknown model_type: {self.model_type}")

    def fit(
        self,
        traces: list[TraceRecord],
        total_turns: int = 50,
    ) -> "FailurePredictor":
        self._total_turns = total_turns
        X, y, _ = build_training_data(traces, self.k, total_turns)
        X_scaled = self._scaler.fit_transform(X)
        self._model.fit(X_scaled, y)
        self._fitted = True
        logger.info(
            "FailurePredictor fitted: k=%d, n_samples=%d, pos_rate=%.2f%%",
            self.k, len(y), 100 * y.mean(),
        )
        return self

    def predict_proba(
        self,
        signals: MonitorSignals,
        history: list[MonitorSignals],
    ) -> float:
        """Return P(failure within next k turns)."""
        if not self._fitted:
            return 0.0
        feats = extract_features(signals, history).reshape(1, -1)
        feats_scaled = self._scaler.transform(feats)
        proba = self._model.predict_proba(feats_scaled)[0]
        # index 1 = positive class (failure)
        classes = list(self._model.classes_)
        pos_idx = classes.index(1) if 1 in classes else -1
        return float(proba[pos_idx]) if pos_idx >= 0 else 0.0

    def feature_importance(self) -> dict[str, float]:
        """Return standardized coefficients (logistic) or permutation importance (MLP)."""
        if not self._fitted:
            return {f: 0.0 for f in FEATURE_NAMES}
        if self.model_type == "logistic":
            coefs = self._model.coef_[0]
            return {name: float(c) for name, c in zip(FEATURE_NAMES, coefs)}
        else:
            return {name: 0.0 for name in FEATURE_NAMES}  # MLP: not directly available

    def cross_validate(
        self,
        traces: list[TraceRecord],
        n_splits: int = 5,
        total_turns: int = 50,
        n_bootstrap: int = 1000,
        rng: Optional[np.random.Generator] = None,
    ) -> dict:
        """5-fold CV with task-level stratification.  Returns metrics dict."""
        if rng is None:
            rng = np.random.default_rng(self.random_state)

        X, y, groups = build_training_data(traces, self.k, total_turns)

        # Verify no-leakage property is possible
        unique_tasks = {r.task_id for r in traces}
        logger.info("CV over %d tasks, %d samples", len(unique_tasks), len(X))

        sgkf = StratifiedGroupKFold(n_splits=n_splits)
        fold_aucs, fold_aps, fold_briers = [], [], []
        fold_train_tasks, fold_test_tasks = [], []

        for fold, (train_idx, test_idx) in enumerate(sgkf.split(X, y, groups)):
            X_tr, X_te = X[train_idx], X[test_idx]
            y_tr, y_te = y[train_idx], y[test_idx]

            # Collect task IDs for leakage check
            train_tasks = {traces[i].task_id for i in train_idx}
            test_tasks = {traces[i].task_id for i in test_idx}
            fold_train_tasks.append(train_tasks)
            fold_test_tasks.append(test_tasks)
            assert_no_leakage(train_tasks, test_tasks)

            scaler = StandardScaler()
            model = self._make_model()
            X_tr_s = scaler.fit_transform(X_tr)
            X_te_s = scaler.transform(X_te)
            model.fit(X_tr_s, y_tr)

            if len(np.unique(y_te)) < 2:
                logger.warning("Fold %d has only one class in test; skipping AUC", fold)
                continue

            proba = model.predict_proba(X_te_s)
            classes = list(model.classes_)
            pos_idx = classes.index(1) if 1 in classes else -1
            if pos_idx < 0:
                continue
            scores = proba[:, pos_idx]
            fold_aucs.append(roc_auc_score(y_te, scores))
            fold_aps.append(average_precision_score(y_te, scores))
            fold_briers.append(brier_score_loss(y_te, scores))

        def _bootstrap_mean_ci(vals: list[float]) -> tuple[float, float, float]:
            arr = np.array(vals)
            mn = float(arr.mean())
            boots = [rng.choice(arr, size=len(arr), replace=True).mean()
                     for _ in range(n_bootstrap)]
            lo, hi = float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))
            return mn, lo, hi

        auc_mean, auc_lo, auc_hi = _bootstrap_mean_ci(fold_aucs)
        ap_mean, ap_lo, ap_hi = _bootstrap_mean_ci(fold_aps)
        brier_mean, brier_lo, brier_hi = _bootstrap_mean_ci(fold_briers)

        return {
            "k": self.k,
            "n_folds": len(fold_aucs),
            "auc_mean": auc_mean,
            "auc_ci_lo": auc_lo,
            "auc_ci_hi": auc_hi,
            "ap_mean": ap_mean,
            "ap_ci_lo": ap_lo,
            "ap_ci_hi": ap_hi,
            "brier_mean": brier_mean,
            "brier_ci_lo": brier_lo,
            "brier_ci_hi": brier_hi,
            "fold_aucs": fold_aucs,
        }
