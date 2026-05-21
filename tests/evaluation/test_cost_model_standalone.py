"""Standalone unit tests for the 4-component cost model (Section XI-C).

This test can run without full package dependencies.
"""

import importlib.util
import os
import sys

# Direct import from the metrics.py file without package init
metrics_path = os.path.join(
    os.path.dirname(__file__), "..", "..", "src", "sage", "evaluation", "metrics.py"
)
spec = importlib.util.spec_from_file_location("metrics", metrics_path)
metrics = importlib.util.module_from_spec(spec)
spec.loader.exec_module(metrics)

# Extract classes and functions
TaskCostBreakdown = metrics.TaskCostBreakdown
TaskResult = metrics.TaskResult
compute_cnsr_from_results = metrics.compute_cnsr_from_results
compute_cost_from_usage = metrics.compute_cost_from_usage
DEFAULT_COST_RATES = metrics.DEFAULT_COST_RATES
compute_cnsr = metrics.compute_cnsr


def test_task_cost_breakdown_default_values():
    """Test default cost breakdown is all zeros."""
    cost = TaskCostBreakdown()
    assert cost.inference_cost == 0.0
    assert cost.tool_cost == 0.0
    assert cost.latency_cost == 0.0
    assert cost.human_cost == 0.0
    assert cost.total_cost == 0.0
    print("✓ test_task_cost_breakdown_default_values passed")


def test_total_cost_computation():
    """Test total cost is sum of all 4 components (Equation 5)."""
    cost = TaskCostBreakdown(
        inference_cost=0.10,  # C_inference
        tool_cost=0.05,  # C_tools
        latency_cost=0.02,  # C_latency
        human_cost=5.00,  # C_human
    )
    expected_total = 0.10 + 0.05 + 0.02 + 5.00
    assert abs(cost.total_cost - expected_total) < 1e-10
    print("✓ test_total_cost_computation passed")


def test_all_four_components_contribute():
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
    print("✓ test_all_four_components_contribute passed")


def test_cost_addition():
    """Test adding two cost breakdowns."""
    cost1 = TaskCostBreakdown(
        inference_cost=0.10, tool_cost=0.05, latency_cost=0.02, human_cost=5.00
    )
    cost2 = TaskCostBreakdown(
        inference_cost=0.20, tool_cost=0.10, latency_cost=0.03, human_cost=0.00
    )
    total = cost1 + cost2
    # Use tolerance for floating point
    assert abs(total.inference_cost - 0.30) < 1e-10
    assert abs(total.tool_cost - 0.15) < 1e-10
    assert abs(total.latency_cost - 0.05) < 1e-10
    assert abs(total.human_cost - 5.00) < 1e-10
    print("✓ test_cost_addition passed")


def test_compute_cost_from_usage():
    """Test compute_cost_from_usage helper function."""
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
    print("✓ test_compute_cost_from_usage passed")


def test_compute_cnsr_from_results_empty():
    """Test with no results."""
    metrics = compute_cnsr_from_results([])
    assert metrics["cnsr"] == 0.0
    assert metrics["success_rate"] == 0.0
    assert metrics["total_tasks"] == 0
    print("✓ test_compute_cnsr_from_results_empty passed")


def test_basic_cnsr_computation():
    """Test basic CNSR computation (Equation 6)."""
    results = [
        TaskResult(
            task_id="task_1", success=True, cost=TaskCostBreakdown(inference_cost=0.50)
        ),
        TaskResult(
            task_id="task_2", success=True, cost=TaskCostBreakdown(inference_cost=0.50)
        ),
    ]
    metrics = compute_cnsr_from_results(results)
    # 100% success, $0.50 mean cost -> CNSR = 1.0 / 0.50 = 2.0
    assert metrics["success_rate"] == 1.0
    assert metrics["mean_total_cost"] == 0.50
    assert metrics["cnsr"] == 2.0
    print("✓ test_basic_cnsr_computation passed")


def test_all_cost_components_in_cnsr():
    """Critical: Verify CNSR includes all 4 cost components."""
    results = [
        TaskResult(
            task_id="task_1",
            success=True,
            cost=TaskCostBreakdown(
                inference_cost=0.10, tool_cost=0.05, latency_cost=0.02, human_cost=0.03
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
    print("✓ test_all_cost_components_in_cnsr passed")


def test_cost_breakdown_in_results():
    """Test that cost breakdown is included in results."""
    results = [
        TaskResult(
            task_id="task_1",
            success=True,
            cost=TaskCostBreakdown(
                inference_cost=0.10, tool_cost=0.05, latency_cost=0.02, human_cost=5.00
            ),
        ),
        TaskResult(
            task_id="task_2",
            success=False,
            cost=TaskCostBreakdown(
                inference_cost=0.20, tool_cost=0.10, latency_cost=0.03, human_cost=0.00
            ),
        ),
    ]
    metrics = compute_cnsr_from_results(results)

    breakdown = metrics["cost_breakdown"]
    assert "mean_inference_cost" in breakdown
    assert "mean_tool_cost" in breakdown
    assert "mean_latency_cost" in breakdown
    assert "mean_human_cost" in breakdown

    # Check mean values with tolerance for floating point
    assert abs(breakdown["mean_inference_cost"] - 0.15) < 1e-10  # (0.10 + 0.20) / 2
    assert abs(breakdown["mean_tool_cost"] - 0.075) < 1e-10  # (0.05 + 0.10) / 2
    assert abs(breakdown["mean_latency_cost"] - 0.025) < 1e-10  # (0.02 + 0.03) / 2
    assert abs(breakdown["mean_human_cost"] - 2.50) < 1e-10  # (5.00 + 0.00) / 2
    print("✓ test_cost_breakdown_in_results passed")


def test_paper_example_system_a():
    """Test paper example: System A - 80% success at $0.50/task."""
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
    print("✓ test_paper_example_system_a passed")


def test_paper_example_system_b():
    """Test paper example: System B - 90% success at $2.00/task."""
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
    print("✓ test_paper_example_system_b passed")


def test_system_a_better_than_b_by_cnsr():
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
    print("✓ test_system_a_better_than_b_by_cnsr passed")


def test_human_intervention_impact_on_cnsr():
    """Test that human intervention significantly impacts CNSR."""
    # System without human intervention
    results_auto = [
        TaskResult(
            task_id="auto_1",
            success=True,
            cost=TaskCostBreakdown(
                inference_cost=0.10, tool_cost=0.05, latency_cost=0.01, human_cost=0.0
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
    print("✓ test_human_intervention_impact_on_cnsr passed")


def test_backwards_compatibility():
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
    print("✓ test_backwards_compatibility passed")


if __name__ == "__main__":
    print("=" * 60)
    print("Running Cost Model Unit Tests (Section XI-C)")
    print("=" * 60)
    print()

    tests = [
        test_task_cost_breakdown_default_values,
        test_total_cost_computation,
        test_all_four_components_contribute,
        test_cost_addition,
        test_compute_cost_from_usage,
        test_compute_cnsr_from_results_empty,
        test_basic_cnsr_computation,
        test_all_cost_components_in_cnsr,
        test_cost_breakdown_in_results,
        test_paper_example_system_a,
        test_paper_example_system_b,
        test_system_a_better_than_b_by_cnsr,
        test_human_intervention_impact_on_cnsr,
        test_backwards_compatibility,
    ]

    passed = 0
    failed = 0

    for test_fn in tests:
        try:
            test_fn()
            passed += 1
        except AssertionError as e:
            print(f"✗ {test_fn.__name__} FAILED: {e}")
            failed += 1
        except Exception as e:
            print(f"✗ {test_fn.__name__} ERROR: {e}")
            failed += 1

    print()
    print("=" * 60)
    print(f"Results: {passed} passed, {failed} failed out of {len(tests)} tests")
    print("=" * 60)

    if failed > 0:
        sys.exit(1)
