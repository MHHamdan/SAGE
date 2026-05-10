"""Smoke tests for E4 and E5 experiments.

Each test runs the experiment on a 5-task miniature subset and verifies that:
  - Output files are created
  - JSONL traces have the correct schema
  - Summary CSV has the expected columns
  - MANIFEST.json is present

These tests must run in <60 seconds total on typical hardware.
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))


# ── E4 smoke test ──────────────────────────────────────────────────────────────

class TestE4Smoke:
    @pytest.fixture(scope="class")
    def e4_results(self, tmp_path_factory):
        """Run E4 on 5 tasks × 1 seed.  Override RESULTS_DIR to tmp."""
        import experiments.e4_closed_loop as e4

        tmp_dir = tmp_path_factory.mktemp("e4")
        orig_dir = e4.RESULTS_DIR

        # Patch results dir
        e4.RESULTS_DIR = tmp_dir
        (tmp_dir / "raw_traces").mkdir(exist_ok=True)
        (tmp_dir / "figures").mkdir(exist_ok=True)

        cfg = dict(e4.E4_CONFIG)
        cfg["n_tasks"] = 5
        cfg["n_seeds"] = 1
        cfg["n_pretrain_tasks"] = 10
        cfg["n_bootstrap"] = 100

        try:
            condition_results = e4.run_experiment(base_seed=99, cfg=cfg)
        finally:
            e4.RESULTS_DIR = orig_dir
            # Restore subdirs for other tests
            import experiments.e4_closed_loop as e4_fresh
            e4_fresh.RESULTS_DIR = orig_dir

        return condition_results, tmp_dir

    def test_four_conditions_returned(self, e4_results):
        cond_results, _ = e4_results
        assert set(cond_results.keys()) == {
            "NoControl", "FixedSchedule", "ThresholdController", "PredictiveController"
        }

    def test_completion_rates_in_range(self, e4_results):
        cond_results, _ = e4_results
        for cond, res in cond_results.items():
            assert 0.0 <= res["completion_mean"] <= 1.0, f"{cond} completion out of range"
            assert res["completion_ci_lo"] <= res["completion_mean"] <= res["completion_ci_hi"]

    def test_summary_csv_created(self, e4_results):
        _, tmp_dir = e4_results
        summary = tmp_dir / "summary.csv"
        assert summary.exists(), "summary.csv not created"
        with summary.open() as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        assert len(rows) > 0
        expected_cols = {"task_id", "controller", "seed", "success", "total_cost",
                          "turns_used", "intervention_count", "final_drift", "cnsr"}
        assert expected_cols.issubset(set(rows[0].keys()))

    def test_manifest_created(self, e4_results):
        _, tmp_dir = e4_results
        manifest = tmp_dir / "MANIFEST.json"
        assert manifest.exists(), "MANIFEST.json not created"
        data = json.loads(manifest.read_text())
        assert "seed" in data
        assert "config" in data

    def test_report_md_created(self, e4_results):
        _, tmp_dir = e4_results
        report = tmp_dir / "REPORT.md"
        assert report.exists(), "REPORT.md not created"
        content = report.read_text()
        assert "Completion Rate" in content
        assert "H4.1" in content

    def test_trace_files_created(self, e4_results):
        _, tmp_dir = e4_results
        traces_dir = tmp_dir / "raw_traces"
        jsonl_files = list(traces_dir.glob("*.jsonl"))
        assert len(jsonl_files) > 0, "No trace JSONL files created"

    def test_trace_schema_valid(self, e4_results):
        _, tmp_dir = e4_results
        traces_dir = tmp_dir / "raw_traces"
        jsonl_files = list(traces_dir.glob("*.jsonl"))
        assert jsonl_files, "No traces to validate"
        first_file = sorted(jsonl_files)[0]
        from agentic_toolkit.stability.traces import TraceEvent
        with first_file.open() as f:
            for i, line in enumerate(f):
                if i >= 3:
                    break
                event = TraceEvent.model_validate_json(line.strip())
                assert event.task_id is not None
                assert 1 <= event.turn <= 50

    def test_no_control_has_zero_interventions(self, e4_results):
        cond_results, _ = e4_results
        # NoControl must never apply interventions
        nc = cond_results["NoControl"]
        assert nc["mean_interventions"] == pytest.approx(0.0)

    def test_cnsr_nonnegative(self, e4_results):
        cond_results, _ = e4_results
        for cond, res in cond_results.items():
            assert res["cnsr"] >= 0.0, f"{cond} has negative CNSR"


# ── E5 smoke test ──────────────────────────────────────────────────────────────

class TestE5Smoke:
    @pytest.fixture(scope="class")
    def e5_results(self, tmp_path_factory):
        """Run E5 on 20 violation + 10 control tasks."""
        import experiments.e5_predictive_validation as e5

        tmp_dir = tmp_path_factory.mktemp("e5")
        orig_dir = e5.RESULTS_DIR

        e5.RESULTS_DIR = tmp_dir
        (tmp_dir / "predictors").mkdir(exist_ok=True)
        (tmp_dir / "figures").mkdir(exist_ok=True)

        cfg = dict(e5.E5_CONFIG)
        cfg["n_violation_tasks"] = 20
        cfg["n_control_tasks"] = 10
        cfg["n_bootstrap"] = 50
        cfg["n_cv_folds"] = 3

        try:
            results = e5.run_experiment(base_seed=77, cfg=cfg)
        finally:
            e5.RESULTS_DIR = orig_dir

        return results, tmp_dir

    def test_all_predictor_variants_present(self, e5_results):
        results, _ = e5_results
        expected = {"drift_only", "oscillation_only", "fidelity_only", "combined", "mlp"}
        assert expected.issubset(set(results.keys()))

    def test_all_lead_times_present(self, e5_results):
        results, _ = e5_results
        for name, res_k in results.items():
            for k in [3, 5, 10]:
                assert k in res_k, f"Predictor {name} missing lead_time k={k}"

    def test_auc_in_range(self, e5_results):
        results, _ = e5_results
        for name, res_k in results.items():
            for k, rd in res_k.items():
                auc = rd.get("auc", 0.5)
                assert 0.0 <= auc <= 1.0, f"{name} k={k}: AUC={auc} out of range"

    def test_no_leakage_in_cv(self, e5_results):
        """Verify that no task_id appears in both train and test across folds."""
        import experiments.e5_predictive_validation as e5
        from agentic_toolkit.stability.predictor import (
            build_training_data, assert_no_leakage,
        )
        from sklearn.model_selection import StratifiedGroupKFold

        cfg = dict(e5.E5_CONFIG)
        cfg["n_violation_tasks"] = 15
        cfg["n_control_tasks"] = 10
        records = e5.generate_task_traces(15, 10, cfg, base_seed=99)
        X, y, groups = build_training_data(records, k=3, total_turns=50)
        unique_task_ids = list({r.task_id for r in records})
        task_to_int = {t: i for i, t in enumerate(unique_task_ids)}
        int_to_task = {v: k for k, v in task_to_int.items()}

        sgkf = StratifiedGroupKFold(n_splits=3)
        for train_idx, test_idx in sgkf.split(X, y, groups):
            train_tasks = {int_to_task[groups[i]] for i in train_idx}
            test_tasks = {int_to_task[groups[i]] for i in test_idx}
            assert_no_leakage(train_tasks, test_tasks)

    def test_summary_csv_exists(self, e5_results):
        _, tmp_dir = e5_results
        summary = tmp_dir / "summary.csv"
        assert summary.exists()
        with summary.open() as f:
            rows = list(csv.DictReader(f))
        assert len(rows) > 0
        assert {"predictor", "k", "auc"}.issubset(set(rows[0].keys()))

    def test_report_md_exists(self, e5_results):
        _, tmp_dir = e5_results
        report = tmp_dir / "REPORT.md"
        assert report.exists()
        content = report.read_text()
        assert "H5.1" in content
        assert "combined" in content

    def test_manifest_exists(self, e5_results):
        _, tmp_dir = e5_results
        assert (tmp_dir / "MANIFEST.json").exists()

    def test_pickled_predictors_exist(self, e5_results):
        _, tmp_dir = e5_results
        pkl_files = list((tmp_dir / "predictors").glob("*.pkl"))
        assert len(pkl_files) > 0

    def test_combined_auc_gte_single_signal(self, e5_results):
        """H5.2: combined should be at least as good as any single signal (at k=5)."""
        results, _ = e5_results
        k = 5
        single_aucs = [
            results["drift_only"].get(k, {}).get("auc", 0.5),
            results["oscillation_only"].get(k, {}).get("auc", 0.5),
            results["fidelity_only"].get(k, {}).get("auc", 0.5),
        ]
        combined_auc = results["combined"].get(k, {}).get("auc", 0.5)
        best_single = max(single_aucs)
        # Allow 0.05 tolerance for small-sample variance in smoke test
        assert combined_auc >= best_single - 0.05, (
            f"Combined AUC {combined_auc:.3f} much worse than best single {best_single:.3f}"
        )
