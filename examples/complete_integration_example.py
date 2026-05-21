#!/usr/bin/env python3
"""
Complete Integration Example for the Agentic AI Toolkit

This example demonstrates how all components of the toolkit work together:
1. Cost Model (Section XI-C) - 4-component cost tracking
2. CNSR Metric (Equation 6) - Cost-Normalized Success Rate
3. Control Loop (Section III-F) - Sense-Decide-Act with stability analysis
4. Autonomy Validation (Section IV-A) - 4 criteria testing
5. Failure Taxonomy (Section XV) - 10 pathology classes
6. Benchmark Evaluation - Comprehensive testing suite

Usage:
    python examples/complete_integration_example.py

Note:
    This example uses direct module imports to avoid package dependencies.
"""

import sys
import os
import types

# Get the src directory path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.join(SCRIPT_DIR, '..', 'src', 'sage', 'evaluation')


def load_module_from_source(module_name: str, file_path: str, package_name: str = None):
    """
    Load a module by reading and executing its source code.
    This handles relative imports by setting up the module namespace correctly.
    """
    with open(file_path, 'r') as f:
        source = f.read()

    # Create module
    module = types.ModuleType(module_name)
    module.__file__ = file_path

    if package_name:
        module.__package__ = package_name

    # Register module before exec so relative imports can find it
    sys.modules[module_name] = module

    # Execute in module's namespace
    exec(compile(source, file_path, 'exec'), module.__dict__)

    return module


# Set up the package structure for relative imports
pkg_name = 'sage.evaluation'

# Create parent packages if needed
if 'sage' not in sys.modules:
    agentic_pkg = types.ModuleType('sage')
    agentic_pkg.__path__ = [os.path.join(SCRIPT_DIR, '..', 'src', 'sage')]
    sys.modules['sage'] = agentic_pkg

if pkg_name not in sys.modules:
    eval_pkg = types.ModuleType(pkg_name)
    eval_pkg.__path__ = [SRC_DIR]
    sys.modules[pkg_name] = eval_pkg

# Load modules in dependency order
metrics_module = load_module_from_source(
    f'{pkg_name}.metrics',
    os.path.join(SRC_DIR, 'metrics.py'),
    pkg_name
)

failure_module = load_module_from_source(
    f'{pkg_name}.failure_taxonomy',
    os.path.join(SRC_DIR, 'failure_taxonomy.py'),
    pkg_name
)

autonomy_module = load_module_from_source(
    f'{pkg_name}.autonomy_validator',
    os.path.join(SRC_DIR, 'autonomy_validator.py'),
    pkg_name
)

pathology_benchmarks = load_module_from_source(
    f'{pkg_name}.pathology_benchmarks',
    os.path.join(SRC_DIR, 'pathology_benchmarks.py'),
    pkg_name
)

autonomy_benchmarks = load_module_from_source(
    f'{pkg_name}.autonomy_benchmarks',
    os.path.join(SRC_DIR, 'autonomy_benchmarks.py'),
    pkg_name
)


# =============================================================================
# 1. COST MODEL DEMONSTRATION (Section XI-C)
# =============================================================================

def demonstrate_cost_model():
    """
    Demonstrate the 4-component cost model from Equation 5:
        C_total = C_inference + C_tools + C_latency + C_human
    """
    print("=" * 60)
    print("1. COST MODEL DEMONSTRATION (Section XI-C)")
    print("=" * 60)

    TaskCostBreakdown = metrics_module.TaskCostBreakdown
    TaskResult = metrics_module.TaskResult
    compute_cnsr_from_results = metrics_module.compute_cnsr_from_results
    compute_cost_from_usage = metrics_module.compute_cost_from_usage

    # Example: Two different systems
    print("\nComparing two systems on the same 100 tasks:")

    # System A: Low cost, moderate success
    system_a_results = []
    for i in range(100):
        success = i < 80  # 80% success rate
        cost = TaskCostBreakdown(
            inference_cost=0.10,   # $0.10 per task
            tool_cost=0.05,
            latency_cost=0.02,
            human_cost=0.0         # No human intervention
        )
        system_a_results.append(TaskResult(
            task_id=f"a_{i}",
            success=success,
            cost=cost
        ))

    # System B: Higher cost, higher success
    system_b_results = []
    for i in range(100):
        success = i < 90  # 90% success rate
        cost = TaskCostBreakdown(
            inference_cost=0.50,   # $0.50 per task (more expensive model)
            tool_cost=0.20,
            latency_cost=0.10,
            human_cost=0.0
        )
        system_b_results.append(TaskResult(
            task_id=f"b_{i}",
            success=success,
            cost=cost
        ))

    # Compute CNSR
    metrics_a = compute_cnsr_from_results(system_a_results)
    metrics_b = compute_cnsr_from_results(system_b_results)

    print(f"\nSystem A:")
    print(f"  Success Rate: {metrics_a['success_rate']:.0%}")
    print(f"  Mean Cost: ${metrics_a['mean_total_cost']:.2f}")
    print(f"  CNSR: {metrics_a['cnsr']:.2f}")

    print(f"\nSystem B:")
    print(f"  Success Rate: {metrics_b['success_rate']:.0%}")
    print(f"  Mean Cost: ${metrics_b['mean_total_cost']:.2f}")
    print(f"  CNSR: {metrics_b['cnsr']:.2f}")

    winner = "A" if metrics_a['cnsr'] > metrics_b['cnsr'] else "B"
    print(f"\n=> System {winner} is more cost-effective by CNSR!")

    # Demonstrate compute_cost_from_usage
    print("\n--- Usage-based cost calculation ---")
    usage_cost = compute_cost_from_usage(
        input_tokens=5000,
        output_tokens=1000,
        tool_calls=10,
        latency_seconds=30.0,
        human_interventions=0
    )
    print(f"Cost from usage: ${usage_cost.total_cost:.4f}")
    print(f"  Breakdown: inference=${usage_cost.inference_cost:.4f}, "
          f"tools=${usage_cost.tool_cost:.4f}, "
          f"latency=${usage_cost.latency_cost:.4f}")

    return metrics_a, metrics_b


# =============================================================================
# 2. AUTONOMY VALIDATION DEMONSTRATION (Section IV-A)
# =============================================================================

def demonstrate_autonomy_validation():
    """
    Demonstrate autonomy validation for the 4 minimum criteria:
    1. Action Selection Freedom
    2. Goal-Directed Persistence
    3. Dynamic Termination
    4. Error Recovery
    """
    print("\n" + "=" * 60)
    print("2. AUTONOMY VALIDATION (Section IV-A)")
    print("=" * 60)

    AutonomyValidator = autonomy_module.AutonomyValidator
    AutonomyThresholds = autonomy_module.AutonomyThresholds
    GenuineAgent = autonomy_module.GenuineAgent
    ScriptedAgent = autonomy_module.ScriptedAgent
    FragileAgent = autonomy_module.FragileAgent

    # Create validator with custom thresholds
    thresholds = AutonomyThresholds(
        action_variation_ratio=0.5,
        persistence_obstacle_ratio=0.5,
        persistence_strategy_changes=1,
        persistence_progress=0.3,
        recovery_attempt_ratio=0.5,
        recovery_strategy_diversity=2
    )

    validator = AutonomyValidator(seed=42, thresholds=thresholds)

    # Test three different agent types
    agents = [
        ("GenuineAgent (should pass all 4)", GenuineAgent()),
        ("ScriptedAgent (fails action selection)", ScriptedAgent()),
        ("FragileAgent (fails persistence/recovery)", FragileAgent())
    ]

    for name, agent in agents:
        print(f"\n--- Testing {name} ---")
        result = validator.validate_all(agent)

        print(f"  Autonomy Level: {result.level.name} ({result.criteria_met}/4 criteria)")
        print(f"  Is Genuine Agent: {result.is_genuine_agent}")

        for criterion, test_result in result.test_results.items():
            status = "PASS" if test_result.passed else "FAIL"
            print(f"    {criterion.value}: {status}")

    return result


# =============================================================================
# 3. FAILURE TAXONOMY DEMONSTRATION (Section XV)
# =============================================================================

def demonstrate_failure_taxonomy():
    """
    Demonstrate the 10 failure pathology classes from Section XV.
    """
    print("\n" + "=" * 60)
    print("3. FAILURE TAXONOMY (Section XV)")
    print("=" * 60)

    FailurePathology = failure_module.FailurePathology
    MITIGATION_STRATEGIES = failure_module.MITIGATION_STRATEGIES

    print("\n10 Failure Pathology Classes:")
    for i, pathology in enumerate(FailurePathology, 1):
        print(f"  {i}. {pathology.value}")

    print("\nMitigation Strategies (Table IX excerpt):")
    for pathology, strategies in list(MITIGATION_STRATEGIES.items())[:3]:
        print(f"\n  {pathology.value}:")
        for i, strategy in enumerate(strategies, 1):
            print(f"    {i}. {strategy}")


# =============================================================================
# 4. BENCHMARK STATISTICS
# =============================================================================

def demonstrate_benchmarks():
    """
    Show statistics about available benchmarks.
    """
    print("\n" + "=" * 60)
    print("4. BENCHMARK STATISTICS")
    print("=" * 60)

    get_benchmark_statistics = pathology_benchmarks.get_benchmark_statistics
    get_autonomy_benchmark_statistics = autonomy_benchmarks.get_autonomy_benchmark_statistics

    # Pathology benchmarks
    path_stats = get_benchmark_statistics()
    print(f"\nPathology Benchmarks:")
    print(f"  Total Tasks: {path_stats['total_tasks']}")
    print(f"  Pathologies Covered: {path_stats['pathologies_covered']}")
    print(f"  Difficulty Distribution: {path_stats['difficulty_distribution']}")

    # Autonomy benchmarks
    auto_stats = get_autonomy_benchmark_statistics()
    print(f"\nAutonomy Benchmarks:")
    print(f"  Total Tasks: {auto_stats['total_tasks']}")
    print(f"  Criteria Covered: {auto_stats['criteria_covered']}")
    print(f"  Difficulty Distribution: {auto_stats['difficulty_distribution']}")


# =============================================================================
# 5. LONG-HORIZON EVALUATION
# =============================================================================

def demonstrate_long_horizon():
    """
    Demonstrate long-horizon evaluation metrics.
    """
    print("\n" + "=" * 60)
    print("5. LONG-HORIZON EVALUATION")
    print("=" * 60)

    class LongHorizonEvaluator:
        """Evaluator for tracking long-horizon agent performance."""

        def __init__(self, window_size: int = 50):
            self.window_size = window_size
            self.results = []
            self.costs = []
            self.incidents = []

        def record_task(self, success: bool, cost: float, incident_type: str = None):
            """Record a task result."""
            self.results.append(1 if success else 0)
            self.costs.append(cost)
            if incident_type:
                self.incidents.append(incident_type)

        def get_metrics(self):
            """Get comprehensive metrics."""
            if not self.results:
                return {}

            success_rate = sum(self.results) / len(self.results)
            mean_cost = sum(self.costs) / len(self.costs)

            # Rolling window success
            rolling = []
            for i in range(len(self.results)):
                start = max(0, i - self.window_size + 1)
                window = self.results[start:i+1]
                rolling.append(sum(window) / len(window))

            # Cost variance
            if len(self.costs) > 1:
                mean = sum(self.costs) / len(self.costs)
                variance = sum((c - mean) ** 2 for c in self.costs) / len(self.costs)
            else:
                variance = 0.0

            return {
                'success_rate': success_rate,
                'mean_cost': mean_cost,
                'cnsr': success_rate / mean_cost if mean_cost > 0 else 0,
                'incident_rate': len(self.incidents) / len(self.results),
                'cost_variance': variance,
                'rolling_success': rolling
            }

    # Simulate 100 tasks with degrading performance
    evaluator = LongHorizonEvaluator(window_size=10)

    print("\nSimulating 100 tasks with gradual performance degradation...")

    for i in range(100):
        # Success rate degrades over time (simulating goal drift)
        success_prob = 0.9 - (i / 200)  # 90% -> 40%
        success = i % (int(1 / success_prob) + 1) != 0

        cost = 0.10 + (i * 0.001)  # Slightly increasing costs

        incident = None
        if i > 70 and i % 10 == 0:
            incident = "violation"

        evaluator.record_task(success=success, cost=cost, incident_type=incident)

    metrics = evaluator.get_metrics()

    print(f"\nResults over 100 tasks:")
    print(f"  Overall Success Rate: {metrics['success_rate']:.1%}")
    print(f"  Mean Cost: ${metrics['mean_cost']:.3f}")
    print(f"  CNSR: {metrics['cnsr']:.2f}")
    print(f"  Incident Rate: {metrics['incident_rate']:.2f}")
    print(f"  Cost Variance: {metrics['cost_variance']:.6f}")

    # Show rolling success trend
    rolling = metrics['rolling_success']
    print(f"\n  Performance Trend (rolling 10-task window):")
    print(f"    Tasks 1-10: {rolling[9]:.0%}")
    print(f"    Tasks 45-55: {rolling[54]:.0%}")
    print(f"    Tasks 90-100: {rolling[-1]:.0%}")

    if rolling[-1] < rolling[9]:
        print(f"\n  WARNING: Performance degradation detected!")


# =============================================================================
# MAIN INTEGRATION
# =============================================================================

def main():
    """Run complete integration demonstration."""
    print("\n" + "=" * 60)
    print("AGENTIC AI TOOLKIT - COMPLETE INTEGRATION EXAMPLE")
    print("=" * 60)
    print("\nThis example demonstrates all major components of the toolkit")
    print("as described in the research paper.\n")

    try:
        # 1. Cost Model
        demonstrate_cost_model()

        # 2. Autonomy Validation
        demonstrate_autonomy_validation()

        # 3. Failure Taxonomy
        demonstrate_failure_taxonomy()

        # 4. Benchmarks
        demonstrate_benchmarks()

        # 5. Long-Horizon Evaluation
        demonstrate_long_horizon()

        print("\n" + "=" * 60)
        print("INTEGRATION EXAMPLE COMPLETE")
        print("=" * 60)
        print("\nAll components demonstrated successfully.")
        print("\nFor more detailed usage, see:")
        print("  - docs/PAPER_MAPPING.md for paper section references")
        print("  - tests/ for unit test examples")
        print("  - examples/ for additional usage patterns")

    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
