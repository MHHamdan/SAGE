"""Unit tests for the 4-component cost model (Section XI-C).

Tests verify that CNSR computation properly integrates all cost components:
    C_total = C_inference + C_tools + C_latency + C_human (Equation 5)
    CNSR = Success_Rate / Mean_Cost (Equation 6)
"""

import pytest
import math

from sage.evaluation.metrics import (
    TaskCostBreakdown,
    TaskResult,
    compute_cnsr_from_results,
    compute_cost_from_usage,
    DEFAULT_COST_RATES,
    compute_cnsr,
)


class TestTaskCostBreakdown:
    """Tests for TaskCostBreakdown dataclass."""

    def test_default_values(self):
        """Test default cost breakdown is all zeros."""
        cost = TaskCostBreakdown()
        assert cost.inference_cost == 0.0
        assert cost.tool_cost == 0.0
        assert cost.latency_cost == 0.0
        assert cost.human_cost == 0.0
        assert cost.total_cost == 0.0

    def test_total_cost_computation(self):
        """Test total cost is sum of all 4 components (Equation 5)."""
        cost = TaskCostBreakdown(
            inference_cost=0.10,  # C_inference
            tool_cost=0.05,  # C_tools
            latency_cost=0.02,  # C_latency
            human_cost=5.00,  # C_human
        )
        expected_total = 0.10 + 0.05 + 0.02 + 5.00
        assert abs(cost.total_cost - expected_total) < 1e-10

    def test_to_dict(self):
        """Test conversion to dictionary."""
        cost = TaskCostBreakdown(
            inference_cost=0.10, tool_cost=0.05, latency_cost=0.02, human_cost=5.00
        )
        d = cost.to_dict()
        assert d["inference_cost"] == 0.10
        assert d["tool_cost"] == 0.05
        assert d["latency_cost"] == 0.02
        assert d["human_cost"] == 5.00
        assert "total_cost" in d

    def test_addition(self):
        """Test adding two cost breakdowns."""
        cost1 = TaskCostBreakdown(
            inference_cost=0.10, tool_cost=0.05, latency_cost=0.02, human_cost=5.00
        )
        cost2 = TaskCostBreakdown(
            inference_cost=0.20, tool_cost=0.10, latency_cost=0.03, human_cost=0.00
        )
        total = cost1 + cost2
        assert total.inference_cost == 0.30
        assert total.tool_cost == 0.15
        assert total.latency_cost == 0.05
        assert total.human_cost == 5.00

    def test_all_four_components_contribute(self):
        """Critical: Verify all 4 components contribute to total cost."""
        # Each component alone
        inference_only = TaskCostBreakdown(inference_cost=1.0)
        tool_only = TaskCostBreakdown(tool_cost=1.0)
        latency_only = TaskCostBreakdown(latency_cost=1.0)
        human_only = TaskCostBreakdown(human_cost=1.0)

        # Each should contribute to total
        assert inference_only.total_cost == 1.0, "Inference cost not included"
        assert tool_only.total_cost == 1.0, "Tool cost not included"
        assert latency_only.total_cost == 1.0, "Latency cost not included"
        assert human_only.total_cost == 1.0, "Human cost not included"

        # Combined should be sum of all
        combined = inference_only + tool_only + latency_only + human_only
        assert combined.total_cost == 4.0, "Not all components sum correctly"


class TestTaskResult:
    """Tests for TaskResult dataclass."""

    def test_basic_creation(self):
        """Test creating a task result."""
        result = TaskResult(
            task_id="task_001",
            success=True,
            cost=TaskCostBreakdown(inference_cost=0.10),
            duration_seconds=5.0,
            steps_taken=3,
        )
        assert result.task_id == "task_001"
        assert result.success is True
        assert result.cost.inference_cost == 0.10
        assert result.duration_seconds == 5.0
        assert result.steps_taken == 3

    def test_to_dict(self):
        """Test conversion to dictionary."""
        result = TaskResult(
            task_id="task_001",
            success=True,
            cost=TaskCostBreakdown(inference_cost=0.10, tool_cost=0.05),
        )
        d = result.to_dict()
        assert d["task_id"] == "task_001"
        assert d["success"] is True
        assert "cost" in d
        assert d["cost"]["inference_cost"] == 0.10
        assert d["cost"]["tool_cost"] == 0.05


class TestComputeCostFromUsage:
    """Tests for compute_cost_from_usage helper function."""

    def test_default_rates(self):
        """Test with default cost rates."""
        cost = compute_cost_from_usage(
            input_tokens=1000,
            output_tokens=500,
            tool_calls=5,
            latency_seconds=10.0,
            human_interventions=0,
        )
        # Default rates: token=0, tool=0.001, latency=0.0001, human=5.0
        assert cost.inference_cost == 0.0  # 0 token cost for local
        assert cost.tool_cost == 0.005  # 5 * 0.001
        assert cost.latency_cost == 0.001  # 10 * 0.0001
        assert cost.human_cost == 0.0  # 0 interventions

    def test_custom_rates(self):
        """Test with custom cost rates (e.g., API pricing)."""
        custom_rates = {
            "token_input_per_1k": 0.01,  # $0.01 per 1K input
            "token_output_per_1k": 0.03,  # $0.03 per 1K output
            "tool_call_cost": 0.01,  # $0.01 per tool call
            "latency_per_second": 0.001,  # $0.001 per second
            "human_intervention_cost": 10.0,  # $10 per intervention
        }
        cost = compute_cost_from_usage(
            input_tokens=1000,
            output_tokens=1000,
            tool_calls=10,
            latency_seconds=60.0,
            human_interventions=1,
            rates=custom_rates,
        )
        assert cost.inference_cost == 0.04  # (1 * 0.01) + (1 * 0.03)
        assert cost.tool_cost == 0.10  # 10 * 0.01
        assert cost.latency_cost == 0.06  # 60 * 0.001
        assert cost.human_cost == 10.0  # 1 * 10.0

    def test_human_intervention_dominates(self):
        """Test that human intervention significantly increases cost."""
        cost_no_human = compute_cost_from_usage(
            tool_calls=10, latency_seconds=60.0, human_interventions=0
        )
        cost_with_human = compute_cost_from_usage(
            tool_calls=10, latency_seconds=60.0, human_interventions=1
        )
        # Human intervention should add $5 (default rate)
        assert cost_with_human.total_cost - cost_no_human.total_cost == 5.0


class TestComputeCNSRFromResults:
    """Tests for compute_cnsr_from_results with full cost model."""

    def test_empty_results(self):
        """Test with no results."""
        metrics = compute_cnsr_from_results([])
        assert metrics["cnsr"] == 0.0
        assert metrics["success_rate"] == 0.0
        assert metrics["total_tasks"] == 0

    def test_basic_cnsr_computation(self):
        """Test basic CNSR computation (Equation 6)."""
        results = [
            TaskResult(
                task_id="task_1",
                success=True,
                cost=TaskCostBreakdown(inference_cost=0.50),
            ),
            TaskResult(
                task_id="task_2",
                success=True,
                cost=TaskCostBreakdown(inference_cost=0.50),
            ),
        ]
        metrics = compute_cnsr_from_results(results)
        # 100% success, $0.50 mean cost -> CNSR = 1.0 / 0.50 = 2.0
        assert metrics["success_rate"] == 1.0
        assert metrics["mean_total_cost"] == 0.50
        assert metrics["cnsr"] == 2.0

    def test_all_cost_components_in_cnsr(self):
        """Critical: Verify CNSR includes all 4 cost components."""
        results = [
            TaskResult(
                task_id="task_1",
                success=True,
                cost=TaskCostBreakdown(
                    inference_cost=0.10,
                    tool_cost=0.05,
                    latency_cost=0.02,
                    human_cost=0.03,
                ),
            ),
        ]
        metrics = compute_cnsr_from_results(results)

        # Total cost should be sum of all 4 components
        expected_total = 0.10 + 0.05 + 0.02 + 0.03
        assert abs(metrics["mean_total_cost"] - expected_total) < 1e-10

        # CNSR should use this total
        expected_cnsr = 1.0 / expected_total
        assert abs(metrics["cnsr"] - expected_cnsr) < 1e-10

    def test_cost_breakdown_in_results(self):
        """Test that cost breakdown is included in results."""
        results = [
            TaskResult(
                task_id="task_1",
                success=True,
                cost=TaskCostBreakdown(
                    inference_cost=0.10,
                    tool_cost=0.05,
                    latency_cost=0.02,
                    human_cost=5.00,
                ),
            ),
            TaskResult(
                task_id="task_2",
                success=False,
                cost=TaskCostBreakdown(
                    inference_cost=0.20,
                    tool_cost=0.10,
                    latency_cost=0.03,
                    human_cost=0.00,
                ),
            ),
        ]
        metrics = compute_cnsr_from_results(results)

        breakdown = metrics["cost_breakdown"]
        assert "mean_inference_cost" in breakdown
        assert "mean_tool_cost" in breakdown
        assert "mean_latency_cost" in breakdown
        assert "mean_human_cost" in breakdown

        # Check mean values
        assert breakdown["mean_inference_cost"] == 0.15  # (0.10 + 0.20) / 2
        assert breakdown["mean_tool_cost"] == 0.075  # (0.05 + 0.10) / 2
        assert breakdown["mean_latency_cost"] == 0.025  # (0.02 + 0.03) / 2
        assert breakdown["mean_human_cost"] == 2.50  # (5.00 + 0.00) / 2

    def test_zero_cost_with_success(self):
        """Test CNSR with zero cost and successes returns inf."""
        results = [
            TaskResult(
                task_id="task_1", success=True, cost=TaskCostBreakdown()  # All zeros
            ),
        ]
        metrics = compute_cnsr_from_results(results)
        assert metrics["cnsr"] == float("inf")

    def test_zero_cost_with_no_success(self):
        """Test CNSR with zero cost and no successes returns 0."""
        results = [
            TaskResult(
                task_id="task_1", success=False, cost=TaskCostBreakdown()  # All zeros
            ),
        ]
        metrics = compute_cnsr_from_results(results)
        assert metrics["cnsr"] == 0.0

    def test_mixed_success_rates(self):
        """Test CNSR with mixed success rates."""
        results = [
            TaskResult(
                task_id="t1", success=True, cost=TaskCostBreakdown(inference_cost=1.0)
            ),
            TaskResult(
                task_id="t2", success=True, cost=TaskCostBreakdown(inference_cost=1.0)
            ),
            TaskResult(
                task_id="t3", success=False, cost=TaskCostBreakdown(inference_cost=1.0)
            ),
            TaskResult(
                task_id="t4", success=False, cost=TaskCostBreakdown(inference_cost=1.0)
            ),
        ]
        metrics = compute_cnsr_from_results(results)
        # 50% success, $1.0 mean cost -> CNSR = 0.5 / 1.0 = 0.5
        assert metrics["success_rate"] == 0.5
        assert metrics["mean_total_cost"] == 1.0
        assert metrics["cnsr"] == 0.5


class TestCNSRIntegration:
    """Integration tests for CNSR with the full cost model."""

    def test_paper_example_system_a(self):
        """Test paper example: System A - 80% success at $0.50/task."""
        # System A: 80 successes, 20 failures, $0.50 per task
        results = []
        for i in range(80):
            results.append(
                TaskResult(
                    task_id=f"a_{i}",
                    success=True,
                    cost=TaskCostBreakdown(inference_cost=0.50),
                )
            )
        for i in range(20):
            results.append(
                TaskResult(
                    task_id=f"a_fail_{i}",
                    success=False,
                    cost=TaskCostBreakdown(inference_cost=0.50),
                )
            )

        metrics = compute_cnsr_from_results(results)
        # CNSR = 0.80 / 0.50 = 1.6
        assert metrics["success_rate"] == 0.80
        assert metrics["mean_total_cost"] == 0.50
        assert abs(metrics["cnsr"] - 1.6) < 1e-10

    def test_paper_example_system_b(self):
        """Test paper example: System B - 90% success at $2.00/task."""
        # System B: 90 successes, 10 failures, $2.00 per task
        results = []
        for i in range(90):
            results.append(
                TaskResult(
                    task_id=f"b_{i}",
                    success=True,
                    cost=TaskCostBreakdown(inference_cost=2.00),
                )
            )
        for i in range(10):
            results.append(
                TaskResult(
                    task_id=f"b_fail_{i}",
                    success=False,
                    cost=TaskCostBreakdown(inference_cost=2.00),
                )
            )

        metrics = compute_cnsr_from_results(results)
        # CNSR = 0.90 / 2.00 = 0.45
        assert metrics["success_rate"] == 0.90
        assert metrics["mean_total_cost"] == 2.00
        assert abs(metrics["cnsr"] - 0.45) < 1e-10

    def test_system_a_better_than_b_by_cnsr(self):
        """System A (lower success, lower cost) beats System B by CNSR."""
        # System A: 80% @ $0.50 -> CNSR 1.6
        results_a = [
            TaskResult(
                task_id=f"a_{i}",
                success=(i < 80),
                cost=TaskCostBreakdown(inference_cost=0.50),
            )
            for i in range(100)
        ]
        # System B: 90% @ $2.00 -> CNSR 0.45
        results_b = [
            TaskResult(
                task_id=f"b_{i}",
                success=(i < 90),
                cost=TaskCostBreakdown(inference_cost=2.00),
            )
            for i in range(100)
        ]

        cnsr_a = compute_cnsr_from_results(results_a)["cnsr"]
        cnsr_b = compute_cnsr_from_results(results_b)["cnsr"]

        assert (
            cnsr_a > cnsr_b
        ), "System A should have higher CNSR despite lower success rate"

    def test_human_intervention_impact_on_cnsr(self):
        """Test that human intervention significantly impacts CNSR."""
        # System without human intervention
        results_auto = [
            TaskResult(
                task_id="auto_1",
                success=True,
                cost=TaskCostBreakdown(
                    inference_cost=0.10,
                    tool_cost=0.05,
                    latency_cost=0.01,
                    human_cost=0.0,
                ),
            )
        ]

        # System with human intervention
        results_human = [
            TaskResult(
                task_id="human_1",
                success=True,
                cost=TaskCostBreakdown(
                    inference_cost=0.10,
                    tool_cost=0.05,
                    latency_cost=0.01,
                    human_cost=5.0,  # One human intervention
                ),
            )
        ]

        cnsr_auto = compute_cnsr_from_results(results_auto)["cnsr"]
        cnsr_human = compute_cnsr_from_results(results_human)["cnsr"]

        # CNSR with human intervention should be much lower
        assert (
            cnsr_human < cnsr_auto / 10
        ), "Human intervention should dramatically reduce CNSR"


class TestBackwardsCompatibility:
    """Test backwards compatibility with simple compute_cnsr function."""

    def test_simple_cnsr_matches_full_model(self):
        """Test that simple CNSR matches full model when only inference cost."""
        success_rate = 0.80
        mean_cost = 0.50

        # Simple function
        simple_cnsr = compute_cnsr(success_rate, mean_cost)

        # Full model with only inference cost
        results = [
            TaskResult(
                task_id=f"t_{i}",
                success=(i < 80),
                cost=TaskCostBreakdown(inference_cost=0.50),
            )
            for i in range(100)
        ]
        full_cnsr = compute_cnsr_from_results(results)["cnsr"]

        assert abs(simple_cnsr - full_cnsr) < 1e-10


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
