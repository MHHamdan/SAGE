"""Evaluation metrics shim.

Exports compute_cnsr and related helpers from sage.evaluation.metrics.
Supports two modes:
  1. Package installed (pip install -e .): direct import works.
  2. Uninstalled (repo clone only): loads the source file directly via importlib.

Usage:
    from eval.metrics import compute_cnsr
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


def _load_direct() -> object:
    """Load sage.evaluation.metrics directly from source file."""
    src = (
        Path(__file__).resolve().parent.parent
        / "src" / "sage" / "evaluation" / "metrics.py"
    )
    mod_name = "sage.evaluation.metrics"
    if mod_name in sys.modules:
        return sys.modules[mod_name]
    spec = importlib.util.spec_from_file_location(mod_name, src)
    mod = importlib.util.module_from_spec(spec)  # type: ignore[arg-type]
    sys.modules[mod_name] = mod
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    return mod


try:
    from sage.evaluation.metrics import (  # type: ignore[import]
        compute_cnsr,
        compute_cnsr_from_results,
        compute_cost_from_usage,
        TaskCostBreakdown,
        TaskResult,
        MetricsCollector,
        AggregatedMetrics,
    )
except (ImportError, ModuleNotFoundError):
    _m = _load_direct()
    compute_cnsr = _m.compute_cnsr  # type: ignore[attr-defined]
    compute_cnsr_from_results = _m.compute_cnsr_from_results  # type: ignore[attr-defined]
    compute_cost_from_usage = _m.compute_cost_from_usage  # type: ignore[attr-defined]
    TaskCostBreakdown = _m.TaskCostBreakdown  # type: ignore[attr-defined]
    TaskResult = _m.TaskResult  # type: ignore[attr-defined]
    MetricsCollector = _m.MetricsCollector  # type: ignore[attr-defined]
    AggregatedMetrics = _m.AggregatedMetrics  # type: ignore[attr-defined]

__all__ = [
    "compute_cnsr",
    "compute_cnsr_from_results",
    "compute_cost_from_usage",
    "TaskCostBreakdown",
    "TaskResult",
    "MetricsCollector",
    "AggregatedMetrics",
]
