"""Tests for CNSR Benchmark module."""

import pytest
import numpy as np
from unittest.mock import Mock

from sage.evaluation.cnsr_benchmark import (
    CNSRBenchmark,
    BenchmarkConfig,
    AgentEvaluationResult,
    CNSRResults,
    ParetoAnalysis,
    DivergenceReport,
    SensitivityReport,
    RankingMethod,
    MODEL_COST_RATES,
    create_cnsr_benchmark,
    quick_cnsr_comparison,
)
from sage.evaluation.metrics import TaskResult, TaskCostBreakdown


class TestCNSRBenchmark:
    """Tests for CNSRBenchmark class."""

    def test_initialization(self):
        """Test benchmark initialization."""
        benchmark = CNSRBenchmark(seed=42)
        assert benchmark.seed == 42
        assert benchmark.cost_rates is not None

    def test_initialization_with_custom_costs(self):
        """Test initialization with custom cost rates."""
        custom_rates = {"test-model": {"token_input_per_1k": 0.01}}
        benchmark = CNSRBenchmark(seed=42, cost_rates=custom_rates)
        assert benchmark.cost_rates == custom_rates

    def test_run_evaluation_basic(self):
        """Test basic evaluation run."""
        benchmark = CNSRBenchmark(seed=42)

        # Create mock agents
        agent1 = Mock()
        agent1.name = "Agent1"
        agent1.run = Mock(return_value="response")

        agent2 = Mock()
        agent2.name = "Agent2"
        agent2.run = Mock(return_value="response")

        # Run evaluation
        results = benchmark.run_evaluation(
            agents=[agent1, agent2], benchmark_name="test", n_samples=10
        )

        assert isinstance(results, CNSRResults)
        assert len(results.results) == 2
        assert results.seed == 42

    def test_pareto_frontier_computation(self):
        """Test Pareto frontier analysis."""
        benchmark = CNSRBenchmark(seed=42)

        # Create mock results
        results = CNSRResults(
            results=[
                AgentEvaluationResult(
                    agent_name="Agent1",
                    benchmark_name="test",
                    task_results=[],
                    success_rate=0.8,
                    mean_cost=1.0,
                    median_cost=1.0,
                    p75_cost=1.2,
                    cost_variance=0.1,
                    cnsr=0.8,
                    total_duration=10.0,
                ),
                AgentEvaluationResult(
                    agent_name="Agent2",
                    benchmark_name="test",
                    task_results=[],
                    success_rate=0.6,
                    mean_cost=2.0,  # Dominated by Agent1
                    median_cost=2.0,
                    p75_cost=2.5,
                    cost_variance=0.2,
                    cnsr=0.3,
                    total_duration=10.0,
                ),
            ],
            benchmark_configs=[BenchmarkConfig("test", n_samples=10)],
            seed=42,
        )

        pareto = benchmark.compute_pareto_frontier(results)

        assert isinstance(pareto, ParetoAnalysis)
        assert len(pareto.points) == 2
        assert pareto.dominated_count >= 1  # Agent2 should be dominated

    def test_ranking_divergence(self):
        """Test ranking divergence analysis."""
        benchmark = CNSRBenchmark(seed=42)

        # Create results with inverted rankings
        results = CNSRResults(
            results=[
                AgentEvaluationResult(
                    agent_name="HighSR-HighCost",
                    benchmark_name="test",
                    task_results=[],
                    success_rate=0.9,
                    mean_cost=5.0,
                    median_cost=5.0,
                    p75_cost=6.0,
                    cost_variance=0.5,
                    cnsr=0.18,  # Low CNSR despite high SR
                    total_duration=10.0,
                ),
                AgentEvaluationResult(
                    agent_name="LowSR-LowCost",
                    benchmark_name="test",
                    task_results=[],
                    success_rate=0.5,
                    mean_cost=0.1,
                    median_cost=0.1,
                    p75_cost=0.15,
                    cost_variance=0.01,
                    cnsr=5.0,  # High CNSR despite low SR
                    total_duration=10.0,
                ),
            ],
            benchmark_configs=[BenchmarkConfig("test", n_samples=10)],
            seed=42,
        )

        divergence = benchmark.ranking_divergence(
            results, method1=RankingMethod.SUCCESS_RATE, method2=RankingMethod.CNSR
        )

        assert isinstance(divergence, DivergenceReport)
        assert divergence.inversion_count > 0  # Rankings should be inverted

    def test_sensitivity_analysis(self):
        """Test sensitivity analysis."""
        benchmark = CNSRBenchmark(seed=42)

        # Create task results with cost variance
        task_results = [
            TaskResult(
                task_id=f"task_{i}",
                success=i % 3 != 0,
                cost=TaskCostBreakdown(inference_cost=0.1 * (i + 1), tool_cost=0.01),
            )
            for i in range(20)
        ]

        results = CNSRResults(
            results=[
                AgentEvaluationResult.from_task_results(
                    agent_name="Agent1",
                    benchmark_name="test",
                    task_results=task_results,
                    total_duration=10.0,
                ),
            ],
            benchmark_configs=[BenchmarkConfig("test", n_samples=20)],
            seed=42,
        )

        sensitivity = benchmark.sensitivity_analysis(results)

        assert isinstance(sensitivity, SensitivityReport)
        assert len(sensitivity.results) == 1

    def test_generate_validation_report(self):
        """Test report generation."""
        benchmark = CNSRBenchmark(seed=42)

        mock_agent = Mock()
        mock_agent.name = "TestAgent"
        mock_agent.run = Mock(return_value="response")

        results = benchmark.run_evaluation(agents=[mock_agent], n_samples=5)

        report = benchmark.generate_validation_report(results)

        assert "summary" in report
        assert "results_table" in report
        assert "pareto_analysis" in report


class TestAgentEvaluationResult:
    """Tests for AgentEvaluationResult."""

    def test_from_task_results(self):
        """Test creation from task results."""
        task_results = [
            TaskResult(
                task_id="1", success=True, cost=TaskCostBreakdown(inference_cost=0.1)
            ),
            TaskResult(
                task_id="2", success=True, cost=TaskCostBreakdown(inference_cost=0.2)
            ),
            TaskResult(
                task_id="3", success=False, cost=TaskCostBreakdown(inference_cost=0.15)
            ),
        ]

        result = AgentEvaluationResult.from_task_results(
            agent_name="Test",
            benchmark_name="test",
            task_results=task_results,
            total_duration=10.0,
        )

        assert result.success_rate == pytest.approx(2 / 3)
        assert result.mean_cost == pytest.approx(0.15)

    def test_empty_results(self):
        """Test with empty results."""
        result = AgentEvaluationResult.from_task_results(
            agent_name="Test",
            benchmark_name="test",
            task_results=[],
            total_duration=0.0,
        )

        assert result.success_rate == 0.0
        assert result.cnsr == 0.0

    def test_to_dict(self):
        """Test serialization."""
        result = AgentEvaluationResult(
            agent_name="Test",
            benchmark_name="test",
            task_results=[],
            success_rate=0.8,
            mean_cost=1.0,
            median_cost=1.0,
            p75_cost=1.2,
            cost_variance=0.1,
            cnsr=0.8,
            total_duration=10.0,
        )

        d = result.to_dict()
        assert d["agent_name"] == "Test"
        assert d["success_rate"] == 0.8
        assert d["cnsr"] == 0.8


class TestConvenienceFunctions:
    """Tests for convenience functions."""

    def test_create_cnsr_benchmark(self):
        """Test benchmark factory function."""
        benchmark = create_cnsr_benchmark(seed=123)
        assert isinstance(benchmark, CNSRBenchmark)
        assert benchmark.seed == 123

    def test_quick_cnsr_comparison(self):
        """Test quick comparison function."""
        agent1 = Mock()
        agent1.name = "A"
        agent1.run = Mock(return_value="response")

        agent2 = Mock()
        agent2.name = "B"
        agent2.run = Mock(return_value="response")

        results = quick_cnsr_comparison([agent1, agent2], n_samples=5, seed=42)

        assert "A" in results
        assert "B" in results


class TestBenchmarkConfig:
    """Tests for BenchmarkConfig."""

    def test_default_config(self):
        """Test default configuration."""
        config = BenchmarkConfig("test")
        assert config.name == "test"
        assert config.n_samples == 100
        assert config.timeout_seconds == 300.0

    def test_custom_config(self):
        """Test custom configuration."""
        config = BenchmarkConfig(name="custom", n_samples=50, timeout_seconds=60.0)
        assert config.n_samples == 50
        assert config.timeout_seconds == 60.0

    def test_get_cost_rates(self):
        """Test cost rate retrieval."""
        config = BenchmarkConfig("test")
        rates = config.get_cost_rates()
        assert "tool_call_cost" in rates


class TestModelCostRates:
    """Tests for model cost rates."""

    def test_all_models_have_required_fields(self):
        """Test all models have required cost fields."""
        required_fields = [
            "token_input_per_1k",
            "token_output_per_1k",
            "tool_call_cost",
        ]

        for model, rates in MODEL_COST_RATES.items():
            for field in required_fields:
                assert field in rates, f"Model {model} missing {field}"

    def test_cost_rates_non_negative(self):
        """Test all cost rates are non-negative."""
        for model, rates in MODEL_COST_RATES.items():
            for field, value in rates.items():
                assert value >= 0, f"Model {model} has negative {field}"
