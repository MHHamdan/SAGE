"""Unit tests for the FailurePredictor.

Critical tests:
  1. Anti-leakage split: train and test task IDs must never overlap.
  2. AUC > chance on a simple synthetic dataset.
  3. Feature extraction produces correct shape.
  4. predict_proba returns float in [0, 1].
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "src"))

from sage.stability.controller import MonitorSignals
from sage.stability.predictor import (
    FailurePredictor,
    TraceRecord,
    extract_features,
    build_training_data,
    assert_no_leakage,
    FEATURE_NAMES,
)


def _sig(drift=0.2, osc=0.1, fid=0.9, prog=0.7, turn=1, cost=0.0) -> MonitorSignals:
    return MonitorSignals(
        drift_score=drift,
        oscillation_score=osc,
        fidelity_score=fid,
        convergence_progress=prog,
        turn=turn,
        cost_so_far=cost,
    )


def _make_synthetic_records(
    n_tasks: int = 60,
    total_turns: int = 20,
    fail_rate: float = 0.5,
    seed: int = 42,
) -> list[TraceRecord]:
    """Synthetic traces where failing tasks have monotonically increasing drift."""
    rng = np.random.default_rng(seed)
    records = []
    for i in range(n_tasks):
        task_id = f"task_{i:04d}"
        task_failed = rng.random() < fail_rate
        signal_history: list[MonitorSignals] = []
        for turn in range(1, total_turns + 1):
            if task_failed:
                drift = min(1.0, 0.05 * turn + rng.uniform(0, 0.05))
            else:
                drift = max(0.0, 0.10 + rng.uniform(-0.02, 0.02))
            sig = _sig(drift=drift, osc=drift * 0.5, fid=max(0, 1 - drift),
                       prog=max(0, 1 - drift), turn=turn)
            feats = extract_features(sig, list(signal_history))
            records.append(TraceRecord(
                task_id=task_id,
                turn=turn,
                features=feats,
                task_failed=task_failed,
            ))
            signal_history.append(sig)
    return records


# ── Feature extraction ─────────────────────────────────────────────────────────

class TestExtractFeatures:
    def test_shape(self):
        feats = extract_features(_sig(), history=[])
        assert feats.shape == (len(FEATURE_NAMES),)

    def test_zero_deltas_no_history(self):
        feats = extract_features(_sig(drift=0.3), history=[])
        assert feats[4] == pytest.approx(0.0)  # delta_drift
        assert feats[5] == pytest.approx(0.0)  # delta_osc

    def test_deltas_computed_correctly(self):
        prev = _sig(drift=0.2, osc=0.1, fid=0.9, turn=1)
        curr = _sig(drift=0.4, osc=0.3, fid=0.7, turn=2)
        feats = extract_features(curr, [prev])
        assert feats[4] == pytest.approx(0.4 - 0.2)
        assert feats[5] == pytest.approx(0.3 - 0.1)

    def test_max_drift_accumulates(self):
        history = [_sig(drift=0.5, turn=1), _sig(drift=0.3, turn=2)]
        curr = _sig(drift=0.4, turn=3)
        feats = extract_features(curr, history)
        assert feats[7] == pytest.approx(0.5)  # max_drift_so_far

    def test_feature_names_count(self):
        assert len(FEATURE_NAMES) == 8


# ── build_training_data ────────────────────────────────────────────────────────

class TestBuildTrainingData:
    def test_output_shapes(self):
        records = _make_synthetic_records(n_tasks=10, total_turns=10)
        X, y, groups = build_training_data(records, k=3, total_turns=10)
        assert X.shape == (len(records), len(FEATURE_NAMES))
        assert y.shape == (len(records),)
        assert groups.shape == (len(records),)

    def test_positive_labels_only_at_end_for_failed_tasks(self):
        records = _make_synthetic_records(n_tasks=10, total_turns=20, fail_rate=1.0, seed=1)
        k = 5
        total_turns = 20
        X, y, groups = build_training_data(records, k=k, total_turns=total_turns)
        for rec, label in zip(records, y):
            if label == 1:
                assert rec.task_failed, "Positive label only on failed tasks"
                assert rec.turn >= total_turns - k + 1, "Positive only in last k turns"
            else:
                assert not rec.task_failed or rec.turn < total_turns - k + 1

    def test_no_positives_for_successful_tasks(self):
        records = _make_synthetic_records(n_tasks=10, total_turns=20, fail_rate=0.0, seed=2)
        X, y, _ = build_training_data(records, k=5, total_turns=20)
        assert y.sum() == 0, "No positives when all tasks succeed"


# ── assert_no_leakage ─────────────────────────────────────────────────────────

class TestAntiLeakage:
    def test_no_leakage_passes_for_disjoint(self):
        train = {"task_001", "task_002", "task_003"}
        test = {"task_004", "task_005"}
        assert_no_leakage(train, test)  # should not raise

    def test_leakage_raises(self):
        train = {"task_001", "task_002"}
        test = {"task_002", "task_003"}
        with pytest.raises(AssertionError, match="leakage"):
            assert_no_leakage(train, test)

    def test_empty_sets_pass(self):
        assert_no_leakage(set(), set())


# ── FailurePredictor ──────────────────────────────────────────────────────────

class TestFailurePredictor:
    def test_fit_predict(self):
        records = _make_synthetic_records(n_tasks=40, total_turns=20, seed=0)
        pred = FailurePredictor(k=5, random_state=0)
        pred.fit(records, total_turns=20)
        sig = _sig(drift=0.8, osc=0.5, fid=0.2, turn=18)
        p = pred.predict_proba(sig, [])
        assert 0.0 <= p <= 1.0

    def test_no_fit_returns_zero(self):
        pred = FailurePredictor(k=5)
        p = pred.predict_proba(_sig(), [])
        assert p == pytest.approx(0.0)

    def test_auc_above_chance_on_separable_data(self):
        """Predictor should achieve AUC > 0.6 on clearly separable synthetic data."""
        records = _make_synthetic_records(n_tasks=80, total_turns=20, seed=42)
        pred = FailurePredictor(k=5, random_state=42)
        cv_result = pred.cross_validate(records, n_splits=3, total_turns=20, n_bootstrap=100)
        assert cv_result["auc_mean"] > 0.6, (
            f"Expected AUC > 0.6 on separable data, got {cv_result['auc_mean']:.3f}"
        )

    def test_cv_no_leakage_asserted(self):
        """cross_validate must not raise leakage error on valid data."""
        records = _make_synthetic_records(n_tasks=30, total_turns=10, seed=7)
        pred = FailurePredictor(k=3, random_state=7)
        pred.cross_validate(records, n_splits=3, total_turns=10, n_bootstrap=20)

    def test_feature_importance_returns_dict(self):
        records = _make_synthetic_records(n_tasks=20, total_turns=10, seed=3)
        pred = FailurePredictor(k=3, random_state=3)
        pred.fit(records, total_turns=10)
        fi = pred.feature_importance()
        assert set(fi.keys()) == set(FEATURE_NAMES)

    def test_high_drift_higher_failure_prob(self):
        """High-drift signal should yield higher failure probability than low-drift."""
        records = _make_synthetic_records(n_tasks=60, total_turns=20, seed=5)
        pred = FailurePredictor(k=5, random_state=5)
        pred.fit(records, total_turns=20)
        p_high = pred.predict_proba(_sig(drift=0.9, osc=0.7, fid=0.1, turn=18), [])
        p_low = pred.predict_proba(_sig(drift=0.05, osc=0.0, fid=0.98, turn=5), [])
        assert p_high > p_low, "High-drift late-turn signal should predict higher failure"

    def test_mlp_variant_works(self):
        records = _make_synthetic_records(n_tasks=40, total_turns=10, seed=9)
        pred = FailurePredictor(k=3, model_type="mlp", random_state=9)
        pred.fit(records, total_turns=10)
        p = pred.predict_proba(_sig(drift=0.5, turn=8), [])
        assert 0.0 <= p <= 1.0

    def test_cv_returns_correct_k(self):
        records = _make_synthetic_records(n_tasks=20, total_turns=10, seed=2)
        pred = FailurePredictor(k=3, random_state=2)
        result = pred.cross_validate(records, n_splits=3, total_turns=10, n_bootstrap=20)
        assert result["k"] == 3
