"""Evaluation harness for reproducible agent benchmarking."""

import logging
import time
import json
from typing import Optional, List, Dict, Any, Callable
from dataclasses import dataclass, field
from pathlib import Path
from datetime import datetime

from .benchmarks import (
    Benchmark,
    BenchmarkTask,
    BenchmarkResult,
    TaskDifficulty,
)
from .metrics import MetricsCollector, AggregatedMetrics

logger = logging.getLogger(__name__)


@dataclass
class EvaluationConfig:
    """Configuration for evaluation runs.

    Attributes:
        seed: Random seed for reproducibility
        num_runs: Number of evaluation runs
        max_parallel: Maximum parallel tasks
        timeout_per_task: Timeout per task in seconds
        save_results: Whether to save results to file
        output_dir: Directory for output files
        verbose: Verbose logging
        benchmarks: List of benchmark names to run
        difficulty_filter: Filter tasks by difficulty
        task_limit: Maximum tasks per benchmark
    """
    seed: int = 42
    num_runs: int = 1
    max_parallel: int = 1
    timeout_per_task: float = 300.0
    save_results: bool = True
    output_dir: str = "results"
    verbose: bool = False
    benchmarks: List[str] = field(default_factory=lambda: ["all"])
    difficulty_filter: Optional[str] = None
    task_limit: Optional[int] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "seed": self.seed,
            "num_runs": self.num_runs,
            "max_parallel": self.max_parallel,
            "timeout_per_task": self.timeout_per_task,
            "save_results": self.save_results,
            "output_dir": self.output_dir,
            "verbose": self.verbose,
            "benchmarks": self.benchmarks,
            "difficulty_filter": self.difficulty_filter,
            "task_limit": self.task_limit,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "EvaluationConfig":
        """Create from dictionary."""
        return cls(**data)

    @classmethod
    def from_yaml(cls, path: str) -> "EvaluationConfig":
        """Load from YAML file."""
        import yaml
        with open(path) as f:
            data = yaml.safe_load(f)
        return cls.from_dict(data.get("evaluation", data))


@dataclass
class EvaluationResult:
    """Complete evaluation result.

    Attributes:
        config: Evaluation configuration
        benchmark_results: Results per benchmark
        aggregate_metrics: Aggregated metrics
        start_time: Evaluation start time
        end_time: Evaluation end time
        metadata: Additional metadata
    """
    config: EvaluationConfig
    benchmark_results: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    aggregate_metrics: Dict[str, Any] = field(default_factory=dict)
    start_time: str = ""
    end_time: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "config": self.config.to_dict(),
            "benchmark_results": self.benchmark_results,
            "aggregate_metrics": self.aggregate_metrics,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "metadata": self.metadata,
        }

    def save(self, path: str):
        """Save results to JSON file."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2, default=str)


class EvaluationHarness:
    """Harness for running reproducible evaluations.

    Example:
        >>> harness = EvaluationHarness(config=EvaluationConfig(seed=42))
        >>> harness.register_benchmark(ReasoningBenchmark())
        >>> harness.register_agent(my_agent)
        >>> result = harness.run()
        >>> print(f"CNSR: {result.aggregate_metrics['cnsr']:.2f}")
    """

    def __init__(
        self,
        config: Optional[EvaluationConfig] = None,
    ):
        """Initialize the harness.

        Args:
            config: Evaluation configuration
        """
        self.config = config or EvaluationConfig()
        self._benchmarks: Dict[str, Benchmark] = {}
        self._agent_fn: Optional[Callable] = None
        self._metrics = MetricsCollector()
        self._aggregated = AggregatedMetrics()

    def register_benchmark(self, benchmark: Benchmark):
        """Register a benchmark.

        Args:
            benchmark: Benchmark to register
        """
        self._benchmarks[benchmark.name] = benchmark
        logger.info(f"Registered benchmark: {benchmark.name}")

    def register_agent(self, agent_fn: Callable):
        """Register the agent to evaluate.

        Args:
            agent_fn: Function that takes BenchmarkTask and returns output
        """
        self._agent_fn = agent_fn
        logger.info("Registered agent for evaluation")

    def run(self) -> EvaluationResult:
        """Run the full evaluation.

        Returns:
            EvaluationResult with all metrics
        """
        if not self._agent_fn:
            raise ValueError("No agent registered. Call register_agent first.")

        if not self._benchmarks:
            raise ValueError("No benchmarks registered. Call register_benchmark first.")

        # Set seed for reproducibility
        self._set_seed(self.config.seed)

        result = EvaluationResult(
            config=self.config,
            start_time=datetime.utcnow().isoformat(),
        )

        # Determine which benchmarks to run
        benchmarks_to_run = self._select_benchmarks()

        # Run each benchmark
        for name, benchmark in benchmarks_to_run.items():
            logger.info(f"Running benchmark: {name}")
            benchmark_result = self._run_benchmark(benchmark)
            result.benchmark_results[name] = benchmark_result

        # Compute aggregate metrics
        result.aggregate_metrics = self._compute_aggregate_metrics(result)
        result.end_time = datetime.utcnow().isoformat()

        # Save if configured
        if self.config.save_results:
            output_path = Path(self.config.output_dir) / f"eval_{result.start_time.replace(':', '-')}.json"
            result.save(str(output_path))
            logger.info(f"Results saved to: {output_path}")

        return result

    def _select_benchmarks(self) -> Dict[str, Benchmark]:
        """Select benchmarks based on config."""
        if "all" in self.config.benchmarks:
            return self._benchmarks

        return {
            name: bench
            for name, bench in self._benchmarks.items()
            if name in self.config.benchmarks
        }

    def _run_benchmark(self, benchmark: Benchmark) -> Dict[str, Any]:
        """Run a single benchmark.

        Args:
            benchmark: Benchmark to run

        Returns:
            Benchmark results
        """
        # Get difficulty filter
        difficulty = None
        if self.config.difficulty_filter:
            difficulty = TaskDifficulty(self.config.difficulty_filter)

        # Get tasks
        tasks = benchmark.get_tasks(
            difficulty=difficulty,
            limit=self.config.task_limit,
        )

        task_results = []

        for task in tasks:
            result = self._run_task(task, benchmark)
            task_results.append(result.to_dict())
            benchmark.record_result(result)

        return {
            "summary": benchmark.get_summary(),
            "task_results": task_results,
        }

    def _run_task(
        self,
        task: BenchmarkTask,
        benchmark: Benchmark,
    ) -> BenchmarkResult:
        """Run a single task.

        Args:
            task: Task to run
            benchmark: Parent benchmark

        Returns:
            BenchmarkResult
        """
        start_time = time.time()
        timeout = min(task.timeout_seconds, self.config.timeout_per_task)

        try:
            # Run agent
            output = self._run_with_timeout(
                lambda: self._agent_fn(task),
                timeout,
            )

            duration = time.time() - start_time

            # Evaluate output
            success = benchmark.evaluate_output(task, output)

            result = BenchmarkResult(
                task_id=task.task_id,
                success=success,
                output=output,
                steps_taken=getattr(output, 'steps', 0) if hasattr(output, 'steps') else 0,
                duration_seconds=duration,
                cost_usd=self._estimate_cost(duration),
            )

        except TimeoutError:
            result = BenchmarkResult(
                task_id=task.task_id,
                success=False,
                error="Timeout",
                duration_seconds=timeout,
            )

        except Exception as e:
            result = BenchmarkResult(
                task_id=task.task_id,
                success=False,
                error=str(e),
                duration_seconds=time.time() - start_time,
            )
            logger.error(f"Task {task.task_id} failed: {e}")

        # Record metrics
        self._metrics.record("success", 1.0 if result.success else 0.0)
        self._metrics.record("duration", result.duration_seconds)
        self._metrics.record("cost", result.cost_usd)

        return result

    def _run_with_timeout(
        self,
        fn: Callable,
        timeout: float,
    ) -> Any:
        """Run function with timeout.

        Args:
            fn: Function to run
            timeout: Timeout in seconds

        Returns:
            Function result

        Raises:
            TimeoutError: If timeout exceeded
        """
        import threading

        result = {"value": None, "error": None}

        def target():
            try:
                result["value"] = fn()
            except Exception as e:
                result["error"] = e

        thread = threading.Thread(target=target)
        thread.start()
        thread.join(timeout=timeout)

        if thread.is_alive():
            raise TimeoutError(f"Execution timed out after {timeout}s")

        if result["error"]:
            raise result["error"]

        return result["value"]

    def _estimate_cost(self, duration: float) -> float:
        """Estimate cost based on duration.

        For Ollama (local), cost is essentially 0.
        This can be overridden for API-based models.

        Args:
            duration: Execution duration in seconds

        Returns:
            Estimated cost in USD
        """
        # Default: Ollama is free (local inference)
        return 0.0

    def _compute_aggregate_metrics(
        self,
        result: EvaluationResult,
    ) -> Dict[str, Any]:
        """Compute aggregate metrics across all benchmarks.

        Args:
            result: Evaluation result

        Returns:
            Aggregate metrics
        """
        total_tasks = 0
        total_successes = 0
        total_cost = 0.0
        total_duration = 0.0

        for bench_result in result.benchmark_results.values():
            summary = bench_result.get("summary", {})
            total_tasks += summary.get("total_tasks", 0)
            total_successes += summary.get("successes", 0)
            total_cost += summary.get("total_cost_usd", 0)
            total_duration += summary.get("total_duration_seconds", 0)

        success_rate = total_successes / total_tasks if total_tasks > 0 else 0
        mean_cost = total_cost / total_tasks if total_tasks > 0 else 0

        return {
            "total_tasks": total_tasks,
            "total_successes": total_successes,
            "success_rate": success_rate,
            "total_cost_usd": total_cost,
            "mean_cost_usd": mean_cost,
            "total_duration_seconds": total_duration,
            "mean_duration_seconds": total_duration / total_tasks if total_tasks else 0,
            "cnsr": success_rate / mean_cost if mean_cost > 0 else float('inf'),
        }

    def _set_seed(self, seed: int):
        """Set random seed for reproducibility.

        Args:
            seed: Random seed
        """
        import random
        random.seed(seed)

        try:
            import numpy as np
            np.random.seed(seed)
        except ImportError:
            pass

        try:
            import torch
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
        except ImportError:
            pass


def run_evaluation(
    agent_fn: Callable,
    config: Optional[EvaluationConfig] = None,
    benchmarks: Optional[List[Benchmark]] = None,
) -> EvaluationResult:
    """Convenience function to run evaluation.

    Args:
        agent_fn: Agent function to evaluate
        config: Evaluation configuration
        benchmarks: List of benchmarks (uses default suite if None)

    Returns:
        EvaluationResult
    """
    from .benchmarks import create_benchmark_suite

    harness = EvaluationHarness(config=config)
    harness.register_agent(agent_fn)

    # Register benchmarks
    benchmark_list = benchmarks or create_benchmark_suite()
    for benchmark in benchmark_list:
        harness.register_benchmark(benchmark)

    return harness.run()
