#!/usr/bin/env python
"""
Example: Evaluation Framework

This example demonstrates the evaluation tools available in the
Agentic AI Toolkit for measuring agent performance, cost efficiency,
and long-horizon behavior.
"""

import random
from sage.evaluation import (
    calculate_cnsr,
    evaluate_agent,
    rolling_window_success,
    goal_drift_score,
    IncidentTracker,
    CostTrajectory,
    LongHorizonEvaluator,
)


def demo_basic_metrics():
    """Demonstrate basic evaluation metrics."""
    print("\n" + "=" * 60)
    print("BASIC METRICS DEMO")
    print("=" * 60)

    # Scenario: Compare two agent systems
    print("\nComparing two agent systems on the same task set:\n")

    # System A: High success, low cost
    system_a = evaluate_agent(successes=80, total_tasks=100, total_cost=40.0)
    print("System A (Efficient):")
    print(f"  Success Rate: {system_a.success_rate:.1%}")
    print(f"  Mean Cost: ${system_a.mean_cost:.2f}")
    print(f"  CNSR: {system_a.cnsr:.2f}")

    # System B: Very high success, but expensive
    system_b = evaluate_agent(successes=95, total_tasks=100, total_cost=200.0)
    print("\nSystem B (Expensive):")
    print(f"  Success Rate: {system_b.success_rate:.1%}")
    print(f"  Mean Cost: ${system_b.mean_cost:.2f}")
    print(f"  CNSR: {system_b.cnsr:.2f}")

    # Analysis
    print("\nAnalysis:")
    print(f"  System B has {system_b.success_rate - system_a.success_rate:.1%} "
          f"higher success rate")
    print(f"  But System A has {system_a.cnsr / system_b.cnsr:.1f}x better CNSR")
    print(f"  System A is more cost-effective for budget-constrained deployments")


def demo_rolling_window():
    """Demonstrate rolling window success tracking."""
    print("\n" + "=" * 60)
    print("ROLLING WINDOW SUCCESS DEMO")
    print("=" * 60)

    # Simulate agent performance over time
    # Good start, then degradation, then recovery
    results = (
        [True] * 8 + [False] * 2 +  # 80% start
        [True] * 5 + [False] * 5 +  # 50% middle (degradation)
        [True] * 7 + [False] * 3    # 70% end (recovery)
    )

    window_size = 10
    rolling = rolling_window_success(results, window_size=window_size)

    print(f"\nSimulated {len(results)} tasks with window_size={window_size}")
    print("\nPerformance over time:")

    # Show key points
    checkpoints = [0, 9, 14, 19, 24, 29]
    for i in checkpoints:
        if i < len(rolling):
            print(f"  Task {i+1:2d}: {rolling[i]:.1%} rolling success")

    # Detect degradation
    print("\nDegradation detection:")
    min_success = min(rolling)
    min_idx = rolling.index(min_success)
    print(f"  Lowest point: {min_success:.1%} at task {min_idx + 1}")

    if min_success < 0.6:
        print("  WARNING: Performance dropped below 60% threshold")


def demo_goal_drift():
    """Demonstrate goal drift detection."""
    print("\n" + "=" * 60)
    print("GOAL DRIFT DETECTION DEMO")
    print("=" * 60)

    # Simulate embeddings (in practice, use real embedding models)
    # Using simple mock vectors for demonstration

    print("\nScenario: Monitoring agent objective alignment over time\n")

    # Original goal embedding (mock)
    original_goal = [0.9, 0.1, 0.2, 0.1, 0.3]  # "Summarize document"

    # Goals at different time points
    time_points = [
        ("t=0 (start)", [0.9, 0.1, 0.2, 0.1, 0.3]),     # Same as original
        ("t=1", [0.85, 0.15, 0.2, 0.1, 0.35]),          # Slight drift
        ("t=2", [0.7, 0.3, 0.25, 0.15, 0.4]),           # More drift
        ("t=3", [0.4, 0.6, 0.3, 0.2, 0.5]),             # Significant drift
        ("t=4 (drifted)", [0.2, 0.8, 0.4, 0.3, 0.6]),   # Major drift
    ]

    print("Goal drift scores over time:")
    for label, embedding in time_points:
        drift = goal_drift_score(original_goal, embedding)
        bar = "#" * int(drift * 20)
        print(f"  {label:15s}: {drift:.3f} {bar}")

    # Threshold check
    drift_threshold = 0.3
    print(f"\nDrift threshold: {drift_threshold}")
    for label, embedding in time_points:
        drift = goal_drift_score(original_goal, embedding)
        if drift > drift_threshold:
            print(f"  ALERT: {label} exceeds threshold!")


def demo_incident_tracking():
    """Demonstrate incident tracking."""
    print("\n" + "=" * 60)
    print("INCIDENT TRACKING DEMO")
    print("=" * 60)

    tracker = IncidentTracker()

    # Simulate incidents during agent operation
    incidents = [
        ("guardrail", "Content filter activated"),
        ("human_intervention", "User requested clarification"),
        ("guardrail", "Tool call rate limit"),
        ("violation", "Attempted unauthorized action"),
        ("termination", "Max iterations exceeded"),
        ("human_intervention", "User corrected output"),
    ]

    print("\nRecording incidents during 100 task execution:\n")
    for incident_type, description in incidents:
        tracker.record_incident(incident_type)
        print(f"  [{incident_type}] {description}")

    # Report
    total_tasks = 100
    print(f"\nIncident Summary ({total_tasks} tasks):")
    print(f"  Human interventions: {tracker.human_interventions}")
    print(f"  Guardrail activations: {tracker.guardrail_activations}")
    print(f"  Constraint violations: {tracker.constraint_violations}")
    print(f"  Unexpected terminations: {tracker.unexpected_terminations}")
    print(f"  Total incidents: {tracker.total_incidents}")
    print(f"  Incident rate: {tracker.incident_rate(total_tasks):.2%}")


def demo_cost_trajectory():
    """Demonstrate cost trajectory analysis."""
    print("\n" + "=" * 60)
    print("COST TRAJECTORY DEMO")
    print("=" * 60)

    # Simulate cost and success data
    random.seed(42)
    n_tasks = 20

    costs = [round(random.uniform(0.1, 0.5), 3) for _ in range(n_tasks)]
    successes = [random.random() > 0.3 for _ in range(n_tasks)]

    trajectory = CostTrajectory(costs=costs, successes=successes)

    print(f"\nSimulated {n_tasks} tasks:")
    print(f"  Total cost: ${sum(costs):.2f}")
    print(f"  Total successes: {sum(successes)}")

    # Cost per success over time
    cost_per_success = trajectory.cost_per_success()
    print("\nCost per successful task trajectory:")
    for i in [0, 4, 9, 14, 19]:
        if i < len(cost_per_success):
            cps = cost_per_success[i]
            if cps == float("inf"):
                print(f"  Task {i+1}: ∞ (no successes yet)")
            else:
                print(f"  Task {i+1}: ${cps:.3f}")

    # Variance
    variance = trajectory.cost_variance()
    print(f"\nCost variance: {variance:.4f}")
    if variance > 0.01:
        print("  (High variance indicates unpredictable costs)")
    else:
        print("  (Low variance indicates predictable costs)")


def demo_long_horizon_evaluator():
    """Demonstrate comprehensive long-horizon evaluation."""
    print("\n" + "=" * 60)
    print("LONG-HORIZON EVALUATION DEMO")
    print("=" * 60)

    random.seed(42)

    # Create evaluator
    evaluator = LongHorizonEvaluator(window_size=20)

    # Simulate 100 tasks with varying performance
    print("\nSimulating 100 agent tasks...")

    for i in range(100):
        # Vary success probability over time
        if i < 30:
            success_prob = 0.85  # Strong start
        elif i < 60:
            success_prob = 0.65  # Degradation
        else:
            success_prob = 0.75  # Partial recovery

        success = random.random() < success_prob
        cost = round(random.uniform(0.2, 0.6), 3)

        # Occasional incidents
        incident = None
        if random.random() < 0.05:
            incident = random.choice(["guardrail", "human_intervention"])

        evaluator.record_task(success=success, cost=cost, incident_type=incident)

    # Get comprehensive metrics
    metrics = evaluator.get_metrics()

    print("\nComprehensive Metrics:")
    print(f"  Total tasks: {metrics['total_tasks']}")
    print(f"  Overall success rate: {metrics['success_rate']:.1%}")
    print(f"  Mean cost per task: ${metrics['mean_cost']:.3f}")
    print(f"  CNSR: {metrics['cnsr']:.2f}")
    print(f"  Cost variance: {metrics['cost_variance']:.4f}")
    print(f"  Incident rate: {metrics['incident_rate']:.2%}")

    # Rolling success trend
    rolling = metrics["rolling_success"]
    print(f"\nPerformance trend (window={evaluator.window_size}):")
    print(f"  Start (task 30): {rolling[29]:.1%}")
    print(f"  Middle (task 60): {rolling[59]:.1%}")
    print(f"  End (task 100): {rolling[-1]:.1%}")

    # Incident breakdown
    incidents = metrics["incidents"]
    print("\nIncident breakdown:")
    for key, value in incidents.items():
        if value > 0:
            print(f"  {key}: {value}")


def main():
    """Run all evaluation demos."""
    print("Agentic AI Toolkit - Evaluation Framework Demo")
    print("=" * 60)

    demo_basic_metrics()
    demo_rolling_window()
    demo_goal_drift()
    demo_incident_tracking()
    demo_cost_trajectory()
    demo_long_horizon_evaluator()

    print("\n" + "=" * 60)
    print("Demo complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
