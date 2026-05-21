"""
CNSR Benchmark Module for Empirical Validation

Provides tools for empirically validating the Cost-Normalized Success Rate (CNSR)
metric across benchmarks, including:
- Pareto frontier analysis
- Ranking divergence from success-rate-only rankings
- Sensitivity analysis for cost aggregation methods

This module addresses IEEE TAI Review Issue M1: Lack of Empirical Validation for CNSR Metric.

Reference: Section XI-C of the paper
    CNSR = Success_Rate / Mean_Cost (Equation 6)

Example:
    >>> from sage.evaluation.cnsr_benchmark import CNSRBenchmark
    >>>
    >>> benchmark = CNSRBenchmark(seed=42)
    >>> results = benchmark.run_evaluation(
    ...     agents=[agent1, agent2, agent3],
    ...     benchmark_name="AgentBench",
    ...     n_samples=50
    ... )
    >>>
    >>> # Pareto analysis
    >>> pareto = benchmark.compute_pareto_frontier(results)
    >>> print(f"Dominated configurations: {pareto.dominated_count}")
    >>>
    >>> # Ranking divergence
    >>> divergence = benchmark.ranking_divergence(results)
    >>> print(f"Ranking inversions: {divergence.inversion_count}")
"""

from __future__ import annotations

import logging
import time
import statistics
from enum import Enum
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Callable, Tuple, Protocol
from abc import ABC, abstractmethod

import numpy as np

from .metrics import (
    TaskCostBreakdown,
    TaskResult,
    compute_cnsr_from_results,
    compute_cost_from_usage,
    DEFAULT_COST_RATES,
)

logger = logging.getLogger(__name__)


# =============================================================================
# COST RATES FOR DIFFERENT MODELS
# =============================================================================

MODEL_COST_RATES: Dict[str, Dict[str, float]] = {
    # OpenAI models (as of December 2024)
    "gpt-4-turbo": {
        "token_input_per_1k": 0.01,
        "token_output_per_1k": 0.03,
        "tool_call_cost": 0.001,
        "latency_per_second": 0.0001,
        "human_intervention_cost": 5.0,
    },
    "gpt-3.5-turbo": {
        "token_input_per_1k": 0.0005,
        "token_output_per_1k": 0.0015,
        "tool_call_cost": 0.0005,
        "latency_per_second": 0.0001,
        "human_intervention_cost": 5.0,
    },
    # Anthropic models (as of December 2024)
    "claude-3-opus": {
        "token_input_per_1k": 0.015,
        "token_output_per_1k": 0.075,
        "tool_call_cost": 0.001,
        "latency_per_second": 0.0001,
        "human_intervention_cost": 5.0,
    },
    "claude-3-sonnet": {
        "token_input_per_1k": 0.003,
        "token_output_per_1k": 0.015,
        "tool_call_cost": 0.0008,
        "latency_per_second": 0.0001,
        "human_intervention_cost": 5.0,
    },
    # Local models (Ollama - zero API cost)
    "llama-3-70b": {
        "token_input_per_1k": 0.0,
        "token_output_per_1k": 0.0,
        "tool_call_cost": 0.0002,  # Compute cost estimate
        "latency_per_second": 0.0002,  # Higher latency cost for local
        "human_intervention_cost": 5.0,
    },
    "llama-3-8b": {
        "token_input_per_1k": 0.0,
        "token_output_per_1k": 0.0,
        "tool_call_cost": 0.0001,
        "latency_per_second": 0.0001,
        "human_intervention_cost": 5.0,
    },
    # Ensemble (weighted average - uses routing)
    "ensemble": {
        "token_input_per_1k": 0.002,  # Blended rate
        "token_output_per_1k": 0.006,
        "tool_call_cost": 0.0006,
        "latency_per_second": 0.0001,
        "human_intervention_cost": 5.0,
    },
}


# =============================================================================
# PROTOCOLS AND BASE CLASSES
# =============================================================================


class AgentProtocol(Protocol):
    """Protocol for agents that can be evaluated."""

    def run(self, query: str, **kwargs) -> str:
        """Run the agent on a query."""
        ...

    @property
    def name(self) -> str:
        """Agent name/identifier."""
        ...


@dataclass
class BenchmarkConfig:
    """Configuration for a benchmark evaluation.

    Attributes:
        name: Benchmark name (e.g., "AgentBench", "SWE-Bench")
        n_samples: Number of samples to evaluate
        timeout_seconds: Maximum time per task
        cost_rates: Cost rates to use for this benchmark
    """

    name: str
    n_samples: int = 100
    timeout_seconds: float = 300.0
    cost_rates: Optional[Dict[str, float]] = None

    def get_cost_rates(self) -> Dict[str, float]:
        """Get cost rates, using defaults if not specified."""
        return self.cost_rates or DEFAULT_COST_RATES


@dataclass
class AgentEvaluationResult:
    """Results for a single agent on a benchmark.

    Attributes:
        agent_name: Name of the evaluated agent
        benchmark_name: Name of the benchmark
        task_results: Individual task results
        success_rate: Overall success rate (0-1)
        mean_cost: Mean cost per task (USD)
        median_cost: Median cost per task (USD)
        p75_cost: 75th percentile cost (USD)
        cost_variance: Variance in costs
        cnsr: Cost-Normalized Success Rate
        total_duration: Total evaluation time (seconds)
        metadata: Additional evaluation metadata
    """

    agent_name: str
    benchmark_name: str
    task_results: List[TaskResult]
    success_rate: float
    mean_cost: float
    median_cost: float
    p75_cost: float
    cost_variance: float
    cnsr: float
    total_duration: float
    metadata: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_task_results(
        cls,
        agent_name: str,
        benchmark_name: str,
        task_results: List[TaskResult],
        total_duration: float,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> "AgentEvaluationResult":
        """Create from task results with computed statistics."""
        if not task_results:
            return cls(
                agent_name=agent_name,
                benchmark_name=benchmark_name,
                task_results=[],
                success_rate=0.0,
                mean_cost=0.0,
                median_cost=0.0,
                p75_cost=0.0,
                cost_variance=0.0,
                cnsr=0.0,
                total_duration=total_duration,
                metadata=metadata or {},
            )

        # Compute statistics
        successes = sum(1 for r in task_results if r.success)
        success_rate = successes / len(task_results)

        costs = [r.cost.total_cost for r in task_results]
        mean_cost = statistics.mean(costs)
        median_cost = statistics.median(costs)
        sorted_costs = sorted(costs)
        p75_idx = int(len(sorted_costs) * 0.75)
        p75_cost = sorted_costs[p75_idx] if sorted_costs else 0.0
        cost_variance = statistics.variance(costs) if len(costs) > 1 else 0.0

        # CNSR
        cnsr = success_rate / mean_cost if mean_cost > 0 else float("inf")

        return cls(
            agent_name=agent_name,
            benchmark_name=benchmark_name,
            task_results=task_results,
            success_rate=success_rate,
            mean_cost=mean_cost,
            median_cost=median_cost,
            p75_cost=p75_cost,
            cost_variance=cost_variance,
            cnsr=cnsr,
            total_duration=total_duration,
            metadata=metadata or {},
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "agent_name": self.agent_name,
            "benchmark_name": self.benchmark_name,
            "success_rate": self.success_rate,
            "mean_cost": self.mean_cost,
            "median_cost": self.median_cost,
            "p75_cost": self.p75_cost,
            "cost_variance": self.cost_variance,
            "cnsr": self.cnsr,
            "total_duration": self.total_duration,
            "num_tasks": len(self.task_results),
            "num_successes": sum(1 for r in self.task_results if r.success),
            "metadata": self.metadata,
        }


@dataclass
class CNSRResults:
    """Aggregated CNSR benchmark results across agents and benchmarks.

    Attributes:
        results: List of per-agent evaluation results
        benchmark_configs: Configurations used for benchmarks
        seed: Random seed used for reproducibility
    """

    results: List[AgentEvaluationResult]
    benchmark_configs: List[BenchmarkConfig]
    seed: int

    def get_results_by_benchmark(
        self, benchmark_name: str
    ) -> List[AgentEvaluationResult]:
        """Get results filtered by benchmark name."""
        return [r for r in self.results if r.benchmark_name == benchmark_name]

    def get_results_by_agent(self, agent_name: str) -> List[AgentEvaluationResult]:
        """Get results filtered by agent name."""
        return [r for r in self.results if r.agent_name == agent_name]

    def to_dataframe_dict(self) -> List[Dict[str, Any]]:
        """Convert to list of dictionaries suitable for DataFrame creation."""
        return [r.to_dict() for r in self.results]


# =============================================================================
# PARETO ANALYSIS
# =============================================================================


@dataclass
class ParetoPoint:
    """A point in the Pareto analysis.

    Attributes:
        agent_name: Agent name
        benchmark_name: Benchmark name
        success_rate: Success rate (higher is better)
        mean_cost: Mean cost (lower is better)
        cnsr: CNSR score
        is_dominated: Whether this point is dominated by others
        dominates: List of agents this point dominates
    """

    agent_name: str
    benchmark_name: str
    success_rate: float
    mean_cost: float
    cnsr: float
    is_dominated: bool = False
    dominates: List[str] = field(default_factory=list)


@dataclass
class ParetoAnalysis:
    """Results of Pareto frontier analysis.

    Attributes:
        points: All points in the analysis
        frontier_points: Points on the Pareto frontier (non-dominated)
        dominated_points: Points below the frontier (dominated)
        dominated_count: Number of dominated configurations
        dominated_percentage: Percentage of dominated configurations
    """

    points: List[ParetoPoint]
    frontier_points: List[ParetoPoint]
    dominated_points: List[ParetoPoint]
    dominated_count: int
    dominated_percentage: float

    def get_frontier_agents(self) -> List[str]:
        """Get names of agents on the Pareto frontier."""
        return [p.agent_name for p in self.frontier_points]

    def get_dominated_agents(self) -> List[str]:
        """Get names of dominated agents."""
        return [p.agent_name for p in self.dominated_points]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "total_points": len(self.points),
            "frontier_count": len(self.frontier_points),
            "dominated_count": self.dominated_count,
            "dominated_percentage": self.dominated_percentage,
            "frontier_agents": self.get_frontier_agents(),
            "dominated_agents": self.get_dominated_agents(),
        }


# =============================================================================
# RANKING DIVERGENCE ANALYSIS
# =============================================================================


class RankingMethod(Enum):
    """Methods for ranking agents."""

    SUCCESS_RATE = "success_rate"
    MEAN_COST = "mean_cost"
    CNSR = "cnsr"
    MEDIAN_COST_CNSR = "median_cost_cnsr"
    P75_COST_CNSR = "p75_cost_cnsr"


@dataclass
class RankingDivergence:
    """Single ranking divergence between two methods.

    Attributes:
        agent_a: First agent in the comparison
        agent_b: Second agent in the comparison
        method1_ranking: Ranking by first method (1 = best)
        method2_ranking: Ranking by second method
        is_inversion: Whether rankings are inverted
    """

    agent_a: str
    agent_b: str
    method1_ranking: Tuple[int, int]  # (rank_a, rank_b) by method 1
    method2_ranking: Tuple[int, int]  # (rank_a, rank_b) by method 2
    is_inversion: bool


@dataclass
class DivergenceReport:
    """Report of ranking divergences between methods.

    Attributes:
        method1: First ranking method
        method2: Second ranking method
        divergences: List of individual divergences
        inversion_count: Number of pairwise ranking inversions
        total_pairs: Total number of pairwise comparisons
        inversion_rate: Rate of inversions (0-1)
        kendall_tau: Kendall's tau correlation coefficient (-1 to 1)
    """

    method1: RankingMethod
    method2: RankingMethod
    divergences: List[RankingDivergence]
    inversion_count: int
    total_pairs: int
    inversion_rate: float
    kendall_tau: float

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "method1": self.method1.value,
            "method2": self.method2.value,
            "inversion_count": self.inversion_count,
            "total_pairs": self.total_pairs,
            "inversion_rate": self.inversion_rate,
            "kendall_tau": self.kendall_tau,
        }


# =============================================================================
# SENSITIVITY ANALYSIS
# =============================================================================


@dataclass
class SensitivityResult:
    """Result of sensitivity analysis for a single agent.

    Attributes:
        agent_name: Agent name
        cnsr_mean: CNSR using mean cost
        cnsr_median: CNSR using median cost
        cnsr_p75: CNSR using 75th percentile cost
        ranking_stable: Whether ranking is stable across aggregations
    """

    agent_name: str
    cnsr_mean: float
    cnsr_median: float
    cnsr_p75: float
    ranking_stable: bool


@dataclass
class SensitivityReport:
    """Report of CNSR sensitivity to cost aggregation method.

    Attributes:
        results: Per-agent sensitivity results
        ranking_change_count: Number of ranking changes across methods
        total_comparisons: Total pairwise comparisons
        stability_rate: Rate of stable rankings (0-1)
        cv_threshold: Coefficient of variation threshold used
        high_variance_agents: Agents with high cost variance
    """

    results: List[SensitivityResult]
    ranking_change_count: int
    total_comparisons: int
    stability_rate: float
    cv_threshold: float
    high_variance_agents: List[str]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "ranking_change_count": self.ranking_change_count,
            "total_comparisons": self.total_comparisons,
            "stability_rate": self.stability_rate,
            "cv_threshold": self.cv_threshold,
            "high_variance_agents": self.high_variance_agents,
        }


# =============================================================================
# CNSR BENCHMARK CLASS
# =============================================================================


class CNSRBenchmark:
    """
    Empirical validation of Cost-Normalized Success Rate metric.

    Runs agent configurations across benchmarks and computes:
    - Success rates
    - Cost breakdowns (inference, tools, latency)
    - CNSR scores
    - Pareto frontier analysis
    - Ranking divergence from success-rate-only

    Implements Section XI-C-3 of the paper: Empirical Validation of CNSR.

    Example:
        >>> benchmark = CNSRBenchmark(seed=42)
        >>>
        >>> # Define agents to evaluate
        >>> agents = [gpt4_agent, claude_agent, llama_agent]
        >>>
        >>> # Run evaluation
        >>> results = benchmark.run_evaluation(
        ...     agents=agents,
        ...     benchmark_configs=[
        ...         BenchmarkConfig("AgentBench", n_samples=50),
        ...         BenchmarkConfig("SWE-Bench", n_samples=100),
        ...     ]
        ... )
        >>>
        >>> # Pareto analysis
        >>> pareto = benchmark.compute_pareto_frontier(results)
        >>> print(f"Frontier agents: {pareto.get_frontier_agents()}")
        >>>
        >>> # Ranking divergence
        >>> divergence = benchmark.ranking_divergence(results)
        >>> print(f"Inversions: {divergence.inversion_count}/{divergence.total_pairs}")
    """

    def __init__(
        self,
        seed: int = 42,
        cost_rates: Optional[Dict[str, Dict[str, float]]] = None,
        task_loader: Optional[Callable[[str, int], List[Dict[str, Any]]]] = None,
    ):
        """Initialize CNSR benchmark.

        Args:
            seed: Random seed for reproducibility
            cost_rates: Model-specific cost rates (uses MODEL_COST_RATES if None)
            task_loader: Optional custom task loader function
        """
        self.seed = seed
        self.cost_rates = cost_rates or MODEL_COST_RATES
        self.task_loader = task_loader
        self._rng = np.random.default_rng(seed)

        logger.info(f"CNSRBenchmark initialized with seed={seed}")

    def run_evaluation(
        self,
        agents: List[Any],
        benchmark_configs: Optional[List[BenchmarkConfig]] = None,
        benchmark_name: Optional[str] = None,
        n_samples: int = 100,
    ) -> CNSRResults:
        """Run all agents on all benchmarks, tracking costs.

        Args:
            agents: List of agents to evaluate (must have name and run method)
            benchmark_configs: List of benchmark configurations
            benchmark_name: Single benchmark name (alternative to configs)
            n_samples: Number of samples if using benchmark_name

        Returns:
            CNSRResults with all evaluation data
        """
        # Build configs
        if benchmark_configs is None:
            if benchmark_name is None:
                benchmark_name = "default"
            benchmark_configs = [BenchmarkConfig(benchmark_name, n_samples)]

        all_results: List[AgentEvaluationResult] = []

        for config in benchmark_configs:
            logger.info(f"Evaluating on {config.name} with {config.n_samples} samples")

            for agent in agents:
                agent_name = getattr(agent, "name", str(agent))
                logger.info(f"  Evaluating agent: {agent_name}")

                result = self._evaluate_agent_on_benchmark(agent=agent, config=config)
                all_results.append(result)

        return CNSRResults(
            results=all_results, benchmark_configs=benchmark_configs, seed=self.seed
        )

    def _evaluate_agent_on_benchmark(
        self, agent: Any, config: BenchmarkConfig
    ) -> AgentEvaluationResult:
        """Evaluate a single agent on a single benchmark.

        Args:
            agent: Agent to evaluate
            config: Benchmark configuration

        Returns:
            AgentEvaluationResult with all metrics
        """
        agent_name = getattr(agent, "name", str(agent))

        # Get cost rates for this agent
        rates = self.cost_rates.get(agent_name.lower(), DEFAULT_COST_RATES)

        # Load tasks
        tasks = self._load_tasks(config.name, config.n_samples)

        task_results: List[TaskResult] = []
        start_time = time.time()

        for i, task in enumerate(tasks):
            task_start = time.time()

            try:
                # Run agent on task
                if hasattr(agent, "run"):
                    output = agent.run(task.get("query", str(task)))
                else:
                    output = str(agent)

                # Evaluate success
                success = self._evaluate_task_success(task, output)

                # Estimate costs
                task_duration = time.time() - task_start
                cost = self._estimate_task_cost(
                    output=output, duration=task_duration, rates=rates
                )

                task_results.append(
                    TaskResult(
                        task_id=f"{config.name}_{i}",
                        success=success,
                        cost=cost,
                        duration_seconds=task_duration,
                        steps_taken=1,  # Can be enhanced
                        metadata={"task": task},
                    )
                )

            except Exception as e:
                logger.warning(f"Task {i} failed with error: {e}")
                task_results.append(
                    TaskResult(
                        task_id=f"{config.name}_{i}",
                        success=False,
                        cost=TaskCostBreakdown(),
                        duration_seconds=time.time() - task_start,
                        error=str(e),
                    )
                )

        total_duration = time.time() - start_time

        return AgentEvaluationResult.from_task_results(
            agent_name=agent_name,
            benchmark_name=config.name,
            task_results=task_results,
            total_duration=total_duration,
            metadata={"config": config.name, "rates": rates},
        )

    def _load_tasks(self, benchmark_name: str, n_samples: int) -> List[Dict[str, Any]]:
        """Load tasks for a benchmark.

        Args:
            benchmark_name: Name of the benchmark
            n_samples: Number of samples to load

        Returns:
            List of task dictionaries
        """
        if self.task_loader:
            return self.task_loader(benchmark_name, n_samples)

        # Default: generate synthetic tasks
        return [
            {"id": i, "query": f"Task {i} for {benchmark_name}"}
            for i in range(n_samples)
        ]

    def _evaluate_task_success(self, task: Dict[str, Any], output: str) -> bool:
        """Evaluate whether task was successful.

        Args:
            task: Task definition
            output: Agent output

        Returns:
            True if successful
        """
        # Default: random success based on seed for reproducibility
        expected = task.get("expected_output")
        if expected is not None:
            return output.strip().lower() == str(expected).strip().lower()

        # Random success for demo
        return self._rng.random() > 0.3

    def _estimate_task_cost(
        self, output: str, duration: float, rates: Dict[str, float]
    ) -> TaskCostBreakdown:
        """Estimate task cost based on output and duration.

        Args:
            output: Agent output
            duration: Task duration in seconds
            rates: Cost rates to use

        Returns:
            TaskCostBreakdown with estimated costs
        """
        # Estimate tokens (rough approximation)
        input_tokens = 500 + self._rng.integers(0, 500)
        output_tokens = len(output.split()) * 1.5 + self._rng.integers(0, 100)
        tool_calls = self._rng.integers(1, 5)

        return compute_cost_from_usage(
            input_tokens=int(input_tokens),
            output_tokens=int(output_tokens),
            tool_calls=int(tool_calls),
            latency_seconds=duration,
            human_interventions=0,
            rates=rates,
        )

    def compute_pareto_frontier(
        self, results: CNSRResults, benchmark_name: Optional[str] = None
    ) -> ParetoAnalysis:
        """Identify dominated configurations using Pareto analysis.

        A configuration is dominated if another configuration achieves
        both higher success rate AND lower cost.

        Args:
            results: CNSR benchmark results
            benchmark_name: Optional filter for specific benchmark

        Returns:
            ParetoAnalysis with frontier and dominated points
        """
        # Filter results if benchmark specified
        if benchmark_name:
            filtered_results = results.get_results_by_benchmark(benchmark_name)
        else:
            filtered_results = results.results

        if not filtered_results:
            return ParetoAnalysis(
                points=[],
                frontier_points=[],
                dominated_points=[],
                dominated_count=0,
                dominated_percentage=0.0,
            )

        # Create points
        points: List[ParetoPoint] = []
        for r in filtered_results:
            points.append(
                ParetoPoint(
                    agent_name=r.agent_name,
                    benchmark_name=r.benchmark_name,
                    success_rate=r.success_rate,
                    mean_cost=r.mean_cost,
                    cnsr=r.cnsr,
                )
            )

        # Determine dominance
        for i, p1 in enumerate(points):
            for j, p2 in enumerate(points):
                if i == j:
                    continue

                # p2 dominates p1 if p2 has >= success rate AND <= cost
                # with at least one strict inequality
                if (
                    p2.success_rate >= p1.success_rate
                    and p2.mean_cost <= p1.mean_cost
                    and (
                        p2.success_rate > p1.success_rate or p2.mean_cost < p1.mean_cost
                    )
                ):
                    p1.is_dominated = True
                    if p1.agent_name not in p2.dominates:
                        p2.dominates.append(p1.agent_name)

        frontier_points = [p for p in points if not p.is_dominated]
        dominated_points = [p for p in points if p.is_dominated]

        return ParetoAnalysis(
            points=points,
            frontier_points=frontier_points,
            dominated_points=dominated_points,
            dominated_count=len(dominated_points),
            dominated_percentage=(
                len(dominated_points) / len(points) * 100 if points else 0.0
            ),
        )

    def ranking_divergence(
        self,
        results: CNSRResults,
        method1: RankingMethod = RankingMethod.SUCCESS_RATE,
        method2: RankingMethod = RankingMethod.CNSR,
        benchmark_name: Optional[str] = None,
    ) -> DivergenceReport:
        """Compare CNSR rankings vs success-rate rankings.

        Calculates the number of pairwise ranking inversions between
        two ranking methods, demonstrating when CNSR provides different
        insights than raw success rate.

        Args:
            results: CNSR benchmark results
            method1: First ranking method
            method2: Second ranking method
            benchmark_name: Optional filter for specific benchmark

        Returns:
            DivergenceReport with inversion statistics
        """
        # Filter results
        if benchmark_name:
            filtered_results = results.get_results_by_benchmark(benchmark_name)
        else:
            # Use first benchmark's results for consistent comparison
            if results.benchmark_configs:
                filtered_results = results.get_results_by_benchmark(
                    results.benchmark_configs[0].name
                )
            else:
                filtered_results = results.results

        if len(filtered_results) < 2:
            return DivergenceReport(
                method1=method1,
                method2=method2,
                divergences=[],
                inversion_count=0,
                total_pairs=0,
                inversion_rate=0.0,
                kendall_tau=1.0,
            )

        # Get rankings by each method
        def get_metric(r: AgentEvaluationResult, method: RankingMethod) -> float:
            if method == RankingMethod.SUCCESS_RATE:
                return r.success_rate
            elif method == RankingMethod.MEAN_COST:
                return -r.mean_cost  # Negate so higher = better
            elif method == RankingMethod.CNSR:
                return r.cnsr
            elif method == RankingMethod.MEDIAN_COST_CNSR:
                return (
                    r.success_rate / r.median_cost
                    if r.median_cost > 0
                    else float("inf")
                )
            elif method == RankingMethod.P75_COST_CNSR:
                return r.success_rate / r.p75_cost if r.p75_cost > 0 else float("inf")
            return 0.0

        # Sort by each method
        sorted1 = sorted(
            filtered_results, key=lambda r: get_metric(r, method1), reverse=True
        )
        sorted2 = sorted(
            filtered_results, key=lambda r: get_metric(r, method2), reverse=True
        )

        # Create rank mappings
        rank1 = {r.agent_name: i for i, r in enumerate(sorted1)}
        rank2 = {r.agent_name: i for i, r in enumerate(sorted2)}

        # Count inversions
        divergences: List[RankingDivergence] = []
        inversion_count = 0
        agents = list(rank1.keys())

        for i in range(len(agents)):
            for j in range(i + 1, len(agents)):
                a, b = agents[i], agents[j]

                r1_a, r1_b = rank1[a], rank1[b]
                r2_a, r2_b = rank2[a], rank2[b]

                # Check if ordering is inverted
                order1 = r1_a < r1_b  # True if a ranked higher by method1
                order2 = r2_a < r2_b  # True if a ranked higher by method2

                is_inversion = order1 != order2
                if is_inversion:
                    inversion_count += 1

                divergences.append(
                    RankingDivergence(
                        agent_a=a,
                        agent_b=b,
                        method1_ranking=(r1_a, r1_b),
                        method2_ranking=(r2_a, r2_b),
                        is_inversion=is_inversion,
                    )
                )

        total_pairs = len(divergences)
        inversion_rate = inversion_count / total_pairs if total_pairs > 0 else 0.0

        # Kendall's tau: (concordant - discordant) / total
        concordant = total_pairs - inversion_count
        kendall_tau = (
            (concordant - inversion_count) / total_pairs if total_pairs > 0 else 1.0
        )

        return DivergenceReport(
            method1=method1,
            method2=method2,
            divergences=divergences,
            inversion_count=inversion_count,
            total_pairs=total_pairs,
            inversion_rate=inversion_rate,
            kendall_tau=kendall_tau,
        )

    def sensitivity_analysis(
        self,
        results: CNSRResults,
        aggregations: Optional[List[str]] = None,
        cv_threshold: float = 1.5,
        benchmark_name: Optional[str] = None,
    ) -> SensitivityReport:
        """Test CNSR robustness to cost aggregation method.

        Computes CNSR using different cost aggregation methods (mean, median,
        p75) and checks if rankings remain stable.

        Args:
            results: CNSR benchmark results
            aggregations: List of aggregation methods to test
            cv_threshold: Coefficient of variation threshold for flagging
            benchmark_name: Optional filter for specific benchmark

        Returns:
            SensitivityReport with stability analysis
        """
        if aggregations is None:
            aggregations = ["mean", "median", "p75"]

        # Filter results
        if benchmark_name:
            filtered_results = results.get_results_by_benchmark(benchmark_name)
        else:
            filtered_results = results.results

        if not filtered_results:
            return SensitivityReport(
                results=[],
                ranking_change_count=0,
                total_comparisons=0,
                stability_rate=1.0,
                cv_threshold=cv_threshold,
                high_variance_agents=[],
            )

        sensitivity_results: List[SensitivityResult] = []
        high_variance_agents: List[str] = []

        for r in filtered_results:
            # Compute CNSR with different aggregations
            cnsr_mean = r.cnsr  # Already computed with mean
            cnsr_median = (
                r.success_rate / r.median_cost if r.median_cost > 0 else float("inf")
            )
            cnsr_p75 = r.success_rate / r.p75_cost if r.p75_cost > 0 else float("inf")

            # Check ranking stability (simplified: all CNSR values similar order)
            values = [cnsr_mean, cnsr_median, cnsr_p75]
            finite_values = [v for v in values if v != float("inf")]

            if len(finite_values) >= 2:
                mean_cnsr = statistics.mean(finite_values)
                std_cnsr = (
                    statistics.stdev(finite_values) if len(finite_values) > 1 else 0
                )
                cv = std_cnsr / mean_cnsr if mean_cnsr > 0 else 0
                ranking_stable = cv < 0.5  # Less than 50% variation
            else:
                ranking_stable = True

            # Check cost variance
            if r.mean_cost > 0:
                cost_cv = np.sqrt(r.cost_variance) / r.mean_cost
                if cost_cv > cv_threshold:
                    high_variance_agents.append(r.agent_name)

            sensitivity_results.append(
                SensitivityResult(
                    agent_name=r.agent_name,
                    cnsr_mean=cnsr_mean,
                    cnsr_median=cnsr_median,
                    cnsr_p75=cnsr_p75,
                    ranking_stable=ranking_stable,
                )
            )

        # Count ranking changes across aggregation methods
        ranking_change_count = 0
        total_comparisons = 0

        for i in range(len(sensitivity_results)):
            for j in range(i + 1, len(sensitivity_results)):
                r1, r2 = sensitivity_results[i], sensitivity_results[j]

                # Compare rankings across methods
                for agg1 in ["mean", "median", "p75"]:
                    for agg2 in ["mean", "median", "p75"]:
                        if agg1 >= agg2:
                            continue

                        cnsr1_1 = getattr(r1, f"cnsr_{agg1}")
                        cnsr1_2 = getattr(r2, f"cnsr_{agg1}")
                        cnsr2_1 = getattr(r1, f"cnsr_{agg2}")
                        cnsr2_2 = getattr(r2, f"cnsr_{agg2}")

                        order1 = cnsr1_1 > cnsr1_2
                        order2 = cnsr2_1 > cnsr2_2

                        total_comparisons += 1
                        if order1 != order2:
                            ranking_change_count += 1

        stability_rate = (
            1 - (ranking_change_count / total_comparisons)
            if total_comparisons > 0
            else 1.0
        )

        return SensitivityReport(
            results=sensitivity_results,
            ranking_change_count=ranking_change_count,
            total_comparisons=total_comparisons,
            stability_rate=stability_rate,
            cv_threshold=cv_threshold,
            high_variance_agents=high_variance_agents,
        )

    def generate_validation_report(
        self, results: CNSRResults, output_format: str = "dict"
    ) -> Dict[str, Any]:
        """Generate comprehensive validation report.

        Args:
            results: CNSR benchmark results
            output_format: Output format ("dict" or "markdown")

        Returns:
            Validation report as dictionary
        """
        # Pareto analysis for each benchmark
        pareto_by_benchmark = {}
        for config in results.benchmark_configs:
            pareto = self.compute_pareto_frontier(results, config.name)
            pareto_by_benchmark[config.name] = pareto.to_dict()

        # Overall ranking divergence
        divergence = self.ranking_divergence(results)

        # Sensitivity analysis
        sensitivity = self.sensitivity_analysis(results)

        report = {
            "summary": {
                "total_agents": len(set(r.agent_name for r in results.results)),
                "total_benchmarks": len(results.benchmark_configs),
                "total_evaluations": len(results.results),
                "seed": results.seed,
            },
            "results_table": results.to_dataframe_dict(),
            "pareto_analysis": pareto_by_benchmark,
            "ranking_divergence": divergence.to_dict(),
            "sensitivity_analysis": sensitivity.to_dict(),
        }

        if output_format == "markdown":
            return self._format_as_markdown(report)

        return report

    def _format_as_markdown(self, report: Dict[str, Any]) -> str:
        """Format report as markdown string."""
        lines = [
            "# CNSR Validation Report",
            "",
            "## Summary",
            f"- Total agents: {report['summary']['total_agents']}",
            f"- Total benchmarks: {report['summary']['total_benchmarks']}",
            f"- Random seed: {report['summary']['seed']}",
            "",
            "## Results",
            "",
            "| Agent | Benchmark | Success Rate | Mean Cost | CNSR |",
            "|-------|-----------|--------------|-----------|------|",
        ]

        for r in report["results_table"]:
            lines.append(
                f"| {r['agent_name']} | {r['benchmark_name']} | "
                f"{r['success_rate']:.1%} | ${r['mean_cost']:.2f} | {r['cnsr']:.2f} |"
            )

        lines.extend(
            [
                "",
                "## Ranking Divergence",
                f"- Inversion rate: {report['ranking_divergence']['inversion_rate']:.1%}",
                f"- Kendall's tau: {report['ranking_divergence']['kendall_tau']:.3f}",
                "",
                "## Sensitivity Analysis",
                f"- Stability rate: {report['sensitivity_analysis']['stability_rate']:.1%}",
            ]
        )

        return "\n".join(lines)


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================


def create_cnsr_benchmark(
    seed: int = 42, use_default_costs: bool = True
) -> CNSRBenchmark:
    """Create a CNSR benchmark with default configuration.

    Args:
        seed: Random seed for reproducibility
        use_default_costs: Whether to use default model cost rates

    Returns:
        Configured CNSRBenchmark instance
    """
    cost_rates = MODEL_COST_RATES if use_default_costs else None
    return CNSRBenchmark(seed=seed, cost_rates=cost_rates)


def quick_cnsr_comparison(
    agents: List[Any], n_samples: int = 50, seed: int = 42
) -> Dict[str, float]:
    """Quick CNSR comparison between agents.

    Args:
        agents: List of agents to compare
        n_samples: Number of samples per agent
        seed: Random seed

    Returns:
        Dictionary mapping agent names to CNSR scores
    """
    benchmark = create_cnsr_benchmark(seed=seed)
    results = benchmark.run_evaluation(agents=agents, n_samples=n_samples)

    return {r.agent_name: r.cnsr for r in results.results}
