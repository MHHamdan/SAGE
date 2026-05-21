"""Experiment runner for reproducible evaluations.

This module provides infrastructure for running reproducible experiments
with configurable models, planners, and evaluation settings.
"""

import os
import yaml
import json
import logging
from typing import Optional, Dict, Any, List, Callable, Type
from dataclasses import dataclass, field
from pathlib import Path
from datetime import datetime
import uuid

from .seeding import set_global_seed, get_reproducibility_info, derive_seed
from .cost import CostTracker, CostSummary
from .logging import JSONLLogger

logger = logging.getLogger(__name__)


@dataclass
class ExperimentConfig:
    """Configuration for an experiment."""
    name: str = "experiment"
    seed: int = 42
    num_runs: int = 5
    output_dir: str = "results"

    # LLM settings
    llm_provider: str = "ollama"
    llm_model: str = "llama3.1:8b"
    llm_base_url: str = "http://localhost:11434"
    llm_temperature: float = 0.1
    llm_max_tokens: int = 4096

    # Planning settings
    planner_type: str = "hybrid"
    max_plan_steps: int = 10
    replan_threshold: float = 0.3

    # Evaluation settings
    eval_window_size: int = 50
    bootstrap_samples: int = 1000

    # Feature flags
    enable_memory: bool = True
    enable_guardrails: bool = True
    enable_cost_tracking: bool = True

    @classmethod
    def from_yaml(cls, filepath: str) -> "ExperimentConfig":
        """Load config from YAML file.

        Args:
            filepath: Path to YAML file

        Returns:
            ExperimentConfig instance
        """
        with open(filepath, "r") as f:
            data = yaml.safe_load(f)

        # Flatten nested config
        flat = {}
        for section, values in data.items():
            if isinstance(values, dict):
                for key, value in values.items():
                    flat_key = f"{section}_{key}" if section != "experiment" else key
                    flat[flat_key] = value
            else:
                flat[section] = values

        # Map to config fields
        config_map = {
            "name": flat.get("name", "experiment"),
            "seed": flat.get("seed", 42),
            "num_runs": flat.get("num_runs", 5),
            "output_dir": flat.get("output_dir", "results"),
            "llm_provider": flat.get("llm_provider", "ollama"),
            "llm_model": flat.get("llm_model", "llama3.1:8b"),
            "llm_base_url": flat.get("llm_base_url", "http://localhost:11434"),
            "llm_temperature": flat.get("llm_temperature", 0.1),
            "llm_max_tokens": flat.get("llm_max_tokens", 4096),
            "planner_type": flat.get("planner_type", "hybrid"),
            "max_plan_steps": flat.get("max_plan_steps", 10),
            "replan_threshold": flat.get("replan_threshold", 0.3),
            "eval_window_size": flat.get("eval_window_size", 50),
            "bootstrap_samples": flat.get("bootstrap_samples", 1000),
            "enable_memory": flat.get("enable_memory", True),
            "enable_guardrails": flat.get("enable_guardrails", True),
            "enable_cost_tracking": flat.get("enable_cost_tracking", True),
        }

        return cls(**config_map)

    def to_yaml(self, filepath: str):
        """Save config to YAML file.

        Args:
            filepath: Path to save to
        """
        data = {
            "experiment": {
                "name": self.name,
                "seed": self.seed,
                "num_runs": self.num_runs,
                "output_dir": self.output_dir,
            },
            "llm": {
                "provider": self.llm_provider,
                "model": self.llm_model,
                "base_url": self.llm_base_url,
                "temperature": self.llm_temperature,
                "max_tokens": self.llm_max_tokens,
            },
            "planning": {
                "type": self.planner_type,
                "max_steps": self.max_plan_steps,
                "replan_threshold": self.replan_threshold,
            },
            "evaluation": {
                "window_size": self.eval_window_size,
                "bootstrap_samples": self.bootstrap_samples,
            },
            "features": {
                "memory": self.enable_memory,
                "guardrails": self.enable_guardrails,
                "cost_tracking": self.enable_cost_tracking,
            },
        }

        with open(filepath, "w") as f:
            yaml.dump(data, f, default_flow_style=False)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "seed": self.seed,
            "num_runs": self.num_runs,
            "output_dir": self.output_dir,
            "llm": {
                "provider": self.llm_provider,
                "model": self.llm_model,
                "temperature": self.llm_temperature,
            },
            "planner": self.planner_type,
            "features": {
                "memory": self.enable_memory,
                "guardrails": self.enable_guardrails,
            },
        }


@dataclass
class RunResult:
    """Result from a single experiment run."""
    run_id: str
    seed: int
    success: bool
    metrics: Dict[str, float] = field(default_factory=dict)
    cost_summary: Optional[CostSummary] = None
    error: Optional[str] = None
    duration_ms: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "run_id": self.run_id,
            "seed": self.seed,
            "success": self.success,
            "metrics": self.metrics,
            "cost_summary": self.cost_summary.to_dict() if self.cost_summary else None,
            "error": self.error,
            "duration_ms": self.duration_ms,
            "metadata": self.metadata,
        }


@dataclass
class ExperimentResult:
    """Result from a complete experiment."""
    experiment_id: str
    config: ExperimentConfig
    runs: List[RunResult] = field(default_factory=list)
    start_time: str = field(default_factory=lambda: datetime.now().isoformat())
    end_time: Optional[str] = None
    reproducibility_info: Dict[str, Any] = field(default_factory=dict)

    @property
    def num_successful(self) -> int:
        """Number of successful runs."""
        return sum(1 for r in self.runs if r.success)

    @property
    def success_rate(self) -> float:
        """Overall success rate."""
        if not self.runs:
            return 0.0
        return self.num_successful / len(self.runs)

    def aggregate_metrics(self) -> Dict[str, Dict[str, float]]:
        """Aggregate metrics across runs.

        Returns:
            Dictionary with mean, std, min, max for each metric
        """
        if not self.runs:
            return {}

        # Collect all metric values
        metric_values: Dict[str, List[float]] = {}
        for run in self.runs:
            for key, value in run.metrics.items():
                if key not in metric_values:
                    metric_values[key] = []
                metric_values[key].append(value)

        # Compute statistics
        aggregated = {}
        for key, values in metric_values.items():
            n = len(values)
            mean = sum(values) / n
            variance = sum((v - mean) ** 2 for v in values) / n if n > 1 else 0
            std = variance ** 0.5

            aggregated[key] = {
                "mean": mean,
                "std": std,
                "min": min(values),
                "max": max(values),
                "n": n,
            }

        return aggregated

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "experiment_id": self.experiment_id,
            "config": self.config.to_dict(),
            "num_runs": len(self.runs),
            "num_successful": self.num_successful,
            "success_rate": self.success_rate,
            "runs": [r.to_dict() for r in self.runs],
            "aggregated_metrics": self.aggregate_metrics(),
            "start_time": self.start_time,
            "end_time": self.end_time,
            "reproducibility_info": self.reproducibility_info,
        }

    def save(self, filepath: str):
        """Save results to JSON file.

        Args:
            filepath: Path to save to
        """
        with open(filepath, "w") as f:
            json.dump(self.to_dict(), f, indent=2)


class ExperimentRunner:
    """Runner for reproducible experiments.

    Handles seeding, logging, cost tracking, and result aggregation.

    Example:
        >>> config = ExperimentConfig(name="test", num_runs=3)
        >>> runner = ExperimentRunner(config)
        >>>
        >>> def run_task(run_id, seed, **kwargs):
        ...     # Do task work
        ...     return {"success": True, "accuracy": 0.85}
        >>>
        >>> results = runner.run(run_task)
        >>> print(f"Success rate: {results.success_rate:.2%}")
    """

    def __init__(
        self,
        config: ExperimentConfig,
        logger_class: Type[JSONLLogger] = JSONLLogger,
    ):
        """Initialize the runner.

        Args:
            config: Experiment configuration
            logger_class: Logger class to use
        """
        self.config = config
        self.experiment_id = f"{config.name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.logger_class = logger_class

        # Create output directory
        self.output_dir = Path(config.output_dir) / self.experiment_id
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Save config
        config.to_yaml(self.output_dir / "config.yaml")

        logger.info(f"Experiment runner initialized: {self.experiment_id}")

    def run(
        self,
        task_fn: Callable[[str, int], Dict[str, Any]],
        setup_fn: Optional[Callable[[], None]] = None,
        teardown_fn: Optional[Callable[[], None]] = None,
    ) -> ExperimentResult:
        """Run the experiment.

        Args:
            task_fn: Function that takes (run_id, seed) and returns results dict
            setup_fn: Optional setup function called before each run
            teardown_fn: Optional teardown function called after each run

        Returns:
            ExperimentResult with all runs
        """
        result = ExperimentResult(
            experiment_id=self.experiment_id,
            config=self.config,
            reproducibility_info=get_reproducibility_info(),
        )

        with self.logger_class(
            self.experiment_id,
            output_dir=str(self.output_dir / "logs"),
        ) as exp_logger:

            for run_idx in range(self.config.num_runs):
                run_id = f"run_{run_idx:03d}"
                run_seed = derive_seed(self.config.seed, run_id)

                logger.info(f"Starting {run_id} with seed {run_seed}")
                exp_logger.set_run_id(run_id)

                # Set seed
                set_global_seed(run_seed)

                # Setup
                if setup_fn:
                    setup_fn()

                # Create cost tracker
                cost_tracker = CostTracker(model=self.config.llm_model)

                # Run task
                import time
                start_time = time.time()
                run_result = RunResult(run_id=run_id, seed=run_seed, success=False)

                try:
                    task_result = task_fn(run_id, run_seed)

                    run_result.success = task_result.get("success", True)
                    run_result.metrics = {
                        k: v for k, v in task_result.items()
                        if k != "success" and isinstance(v, (int, float))
                    }
                    run_result.metadata = {
                        k: v for k, v in task_result.items()
                        if k != "success" and not isinstance(v, (int, float))
                    }

                except Exception as e:
                    logger.error(f"Run {run_id} failed: {e}")
                    run_result.error = str(e)
                    exp_logger.log_error(e, f"Run {run_id} failed")

                run_result.duration_ms = (time.time() - start_time) * 1000
                run_result.cost_summary = cost_tracker.get_summary()

                # Log evaluation
                exp_logger.log_evaluation(run_result.metrics, run_id)

                # Teardown
                if teardown_fn:
                    teardown_fn()

                result.runs.append(run_result)
                logger.info(f"Completed {run_id}: success={run_result.success}")

        result.end_time = datetime.now().isoformat()

        # Save results
        result.save(self.output_dir / "results.json")

        return result


def load_config(config_path: str) -> ExperimentConfig:
    """Load experiment configuration from file.

    Args:
        config_path: Path to YAML config file

    Returns:
        ExperimentConfig instance
    """
    return ExperimentConfig.from_yaml(config_path)


def run_experiment(
    config_path: str,
    task_fn: Callable[[str, int], Dict[str, Any]],
) -> ExperimentResult:
    """Convenience function to run an experiment from config file.

    Args:
        config_path: Path to config YAML
        task_fn: Task function

    Returns:
        ExperimentResult
    """
    config = load_config(config_path)
    runner = ExperimentRunner(config)
    return runner.run(task_fn)
