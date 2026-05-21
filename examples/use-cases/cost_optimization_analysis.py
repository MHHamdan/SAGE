#!/usr/bin/env python3
"""
Cost Optimization Analysis Example

Demonstrates LLM cost analysis, model comparison, and optimization
strategies using the 4-component cost model and CNSR metric.

Uses LOCAL Ollama models for REAL inference with actual token tracking.
Compares different model sizes to find optimal cost/performance tradeoffs.

Components used:
- TaskCostBreakdown: 4-component cost tracking
- compute_cnsr_from_results(): Cost-Normalized Success Rate
- Model comparison and routing optimization

Usage:
    python examples/use-cases/cost_optimization_analysis.py

Requires Ollama running locally: ollama serve
"""

import sys
import os
import types
import random
import time
from dataclasses import dataclass
from typing import List, Dict

# Setup paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.join(SCRIPT_DIR, '..', '..', 'src', 'sage', 'evaluation')

# Setup package structure for proper dataclass support
pkg_name = 'sage.evaluation'

if 'sage' not in sys.modules:
    agentic_pkg = types.ModuleType('sage')
    agentic_pkg.__path__ = [os.path.join(SCRIPT_DIR, '..', '..', 'src', 'sage')]
    sys.modules['sage'] = agentic_pkg

if pkg_name not in sys.modules:
    eval_pkg = types.ModuleType(pkg_name)
    eval_pkg.__path__ = [SRC_DIR]
    sys.modules[pkg_name] = eval_pkg


def load_module(name, filename):
    """Load module with proper package context."""
    module_name = f'{pkg_name}.{name}'
    path = os.path.join(SRC_DIR, filename)

    with open(path) as f:
        source = f.read()

    module = types.ModuleType(module_name)
    module.__file__ = path
    module.__package__ = pkg_name
    sys.modules[module_name] = module
    exec(compile(source, path, 'exec'), module.__dict__)
    return module


# Load metrics module
metrics = load_module('metrics', 'metrics.py')

# Import Ollama helper
from ollama_helper import (
    OllamaClient, check_ollama_ready, list_models,
    estimate_cost, TOKEN_RATES, check_ollama_available
)


# =============================================================================
# Model Profiles with Real Ollama Models
# =============================================================================

@dataclass
class ModelProfile:
    """Configuration for an Ollama model."""
    name: str
    model_id: str
    token_rate: float  # Estimated $ per token (for comparison)
    description: str


# Define available model tiers
MODEL_TIERS = {
    "small": ModelProfile(
        name="Small (gemma2:2b)",
        model_id="gemma2:2b",
        token_rate=0.00001,
        description="Fast, lightweight, good for simple tasks"
    ),
    "medium": ModelProfile(
        name="Medium (mistral:latest)",
        model_id="mistral:latest",
        token_rate=0.00003,
        description="Balanced performance and cost"
    ),
    "large": ModelProfile(
        name="Large (llama3.1:8b)",
        model_id="llama3.1:8b",
        token_rate=0.00005,
        description="High quality, more expensive"
    ),
}

# Test prompts for evaluation
TEST_PROMPTS = [
    ("What is 2 + 2?", "simple"),
    ("Explain photosynthesis in one sentence.", "medium"),
    ("What are the main causes of climate change?", "medium"),
    ("Write a haiku about programming.", "creative"),
    ("Summarize the key principles of machine learning.", "complex"),
]


# =============================================================================
# Evaluation Functions
# =============================================================================

def evaluate_model(model_id: str, num_tasks: int = 5) -> dict:
    """
    Evaluate a single model with real inference.

    Returns metrics including success rate, token usage, latency, and CNSR.
    """
    TaskCostBreakdown = metrics.TaskCostBreakdown
    TaskResult = metrics.TaskResult
    compute_cnsr = metrics.compute_cnsr_from_results

    client = OllamaClient(model=model_id, timeout=90)
    results = []
    total_tokens = 0
    total_latency = 0.0

    for i in range(num_tasks):
        prompt, complexity = TEST_PROMPTS[i % len(TEST_PROMPTS)]

        response = client.generate(
            prompt=prompt,
            temperature=0.7,
            max_tokens=150
        )

        if response.success:
            # Determine success based on response quality
            success = len(response.text.strip()) > 10
            tokens = response.total_tokens
            latency = response.latency_seconds
        else:
            success = False
            tokens = 0
            latency = response.latency_seconds

        total_tokens += tokens
        total_latency += latency

        # Calculate costs
        token_rate = TOKEN_RATES.get(model_id, 0.00003)
        cost = TaskCostBreakdown(
            inference_cost=tokens * token_rate,
            tool_cost=0.005,  # Minimal tool cost
            latency_cost=latency * 0.001,
            human_cost=0.0
        )

        results.append(TaskResult(f"task_{i}", success, cost))

    # Compute CNSR
    cnsr_metrics = compute_cnsr(results)

    return {
        "model_id": model_id,
        "num_tasks": num_tasks,
        "success_rate": cnsr_metrics["success_rate"],
        "cnsr": cnsr_metrics["cnsr"],
        "mean_cost": cnsr_metrics["mean_total_cost"],
        "total_tokens": total_tokens,
        "avg_latency": total_latency / num_tasks,
        "results": results
    }


def compare_models(available_models: List[str], num_tasks: int = 5):
    """Compare all available models on cost efficiency."""
    print("\n" + "=" * 70)
    print("MODEL COMPARISON (Real Inference)")
    print("=" * 70)

    analyses = {}

    for tier_name, profile in MODEL_TIERS.items():
        if profile.model_id in available_models:
            print(f"\nEvaluating {profile.name}...")
            print(f"  Running {num_tasks} tasks...")

            # Check if model is ready
            ready, error = check_ollama_ready(profile.model_id)
            if not ready:
                print(f"  SKIPPED: {error}")
                continue

            analysis = evaluate_model(profile.model_id, num_tasks)
            analyses[tier_name] = {
                "profile": profile,
                **analysis
            }
            print(f"  Done! Tokens: {analysis['total_tokens']}, CNSR: {analysis['cnsr']:.2f}")
        else:
            print(f"\nSkipping {profile.name} (not installed)")

    if not analyses:
        print("\nNo models available for comparison.")
        return {}

    # Print comparison table
    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)

    print(f"\n{'Model':<25} {'Success':<10} {'Cost':<12} {'CNSR':<10} {'Tokens':<10} {'Latency'}")
    print("-" * 80)

    for tier_name, a in analyses.items():
        print(f"{a['profile'].name:<25} {a['success_rate']:.1%}      "
              f"${a['mean_cost']:.4f}      {a['cnsr']:.2f}       "
              f"{a['total_tokens']:<10} {a['avg_latency']:.2f}s")

    # Find best
    if analyses:
        best = max(analyses.items(), key=lambda x: x[1]['cnsr'])
        print(f"\nBest by CNSR: {best[1]['profile'].name} ({best[1]['cnsr']:.2f})")

    return analyses


def analyze_cost_breakdown(analyses: dict):
    """Detailed cost breakdown for each model."""
    if not analyses:
        return

    print("\n" + "=" * 70)
    print("COST BREAKDOWN (4-Component Model)")
    print("=" * 70)
    print("\nC_total = C_inference + C_tools + C_latency + C_human (Equation 5)")

    for tier_name, a in analyses.items():
        results = a.get("results", [])
        if not results:
            continue

        total_cost = sum(r.cost.total_cost for r in results)
        total_inference = sum(r.cost.inference_cost for r in results)
        total_tools = sum(r.cost.tool_cost for r in results)
        total_latency = sum(r.cost.latency_cost for r in results)
        total_human = sum(r.cost.human_cost for r in results)

        print(f"\n{a['profile'].name}")
        print("-" * 40)

        if total_cost > 0:
            print(f"  inference: {total_inference/total_cost*100:>5.1f}%  ${total_inference:.6f}")
            print(f"  tools:     {total_tools/total_cost*100:>5.1f}%  ${total_tools:.6f}")
            print(f"  latency:   {total_latency/total_cost*100:>5.1f}%  ${total_latency:.6f}")
            print(f"  human:     {total_human/total_cost*100:>5.1f}%  ${total_human:.6f}")
            print(f"\n  Total: ${total_cost:.6f}")


def generate_recommendations(analyses: dict):
    """Generate optimization recommendations."""
    if not analyses:
        return None

    print("\n" + "=" * 70)
    print("RECOMMENDATIONS")
    print("=" * 70)

    best_cnsr = max(analyses.items(), key=lambda x: x[1]['cnsr'])
    cheapest = min(analyses.items(), key=lambda x: x[1]['mean_cost'])
    most_accurate = max(analyses.items(), key=lambda x: x[1]['success_rate'])

    print("\n## Key Findings\n")
    print(f"1. Best ROI (CNSR): {best_cnsr[1]['profile'].name}")
    print(f"   CNSR: {best_cnsr[1]['cnsr']:.2f}")
    print(f"2. Lowest Cost: {cheapest[1]['profile'].name}")
    print(f"   Cost: ${cheapest[1]['mean_cost']:.4f}/task")
    print(f"3. Highest Accuracy: {most_accurate[1]['profile'].name}")
    print(f"   Success: {most_accurate[1]['success_rate']:.1%}")

    print("\n## Optimization Strategies\n")
    print("1. TIERED ROUTING:")
    print("   - Simple queries -> Small model (gemma2:2b)")
    print("   - Standard queries -> Medium model (mistral)")
    print("   - Complex queries -> Large model (llama3.1:8b)")

    print("\n2. PROMPT OPTIMIZATION:")
    print("   - Shorter prompts reduce token costs")
    print("   - Clear instructions improve success rate")

    print("\n3. CACHING:")
    print("   - Cache common responses")
    print("   - Use embeddings for similarity matching")

    return best_cnsr[0]


def forecast_costs(model_id: str, daily_tasks: int = 100, days: int = 30):
    """Forecast costs for deployment period."""
    print("\n" + "=" * 70)
    print("COST FORECAST")
    print("=" * 70)

    token_rate = TOKEN_RATES.get(model_id, 0.00003)

    # Estimate tokens per task (based on typical usage)
    est_tokens_per_task = 300  # prompt + completion

    total_tasks = daily_tasks * days
    total_tokens = total_tasks * est_tokens_per_task
    total_cost = total_tokens * token_rate

    print(f"\nModel: {model_id}")
    print(f"Period: {days} days at {daily_tasks} tasks/day")
    print(f"Total tasks: {total_tasks:,}")

    print(f"\n## Estimated Costs\n")
    print(f"  Tokens: {total_tokens:,}")
    print(f"  Token cost: ${total_cost:.2f}")
    print(f"  Daily: ${total_cost/days:.2f}")
    print(f"  Per task: ${total_cost/total_tasks:.6f}")

    print(f"\n## Budget Recommendation\n")
    buffer = 1.3  # 30% buffer
    print(f"  Base estimate: ${total_cost:.2f}")
    print(f"  With 30% buffer: ${total_cost * buffer:.2f}")

    return total_cost * buffer


# =============================================================================
# Main
# =============================================================================

def main():
    """Run cost optimization analysis with real models."""
    print("=" * 70)
    print("LLM COST OPTIMIZATION ANALYSIS")
    print("Using Local Ollama Models with Real Token Tracking")
    print("=" * 70)

    # Check Ollama availability
    if not check_ollama_available():
        print("\nERROR: Ollama server not running.")
        print("Start with: ollama serve")
        return None

    available = list_models()
    print(f"\nAvailable models: {len(available)}")
    for m in available[:5]:
        print(f"  - {m}")
    if len(available) > 5:
        print(f"  ... and {len(available) - 5} more")

    # Check which tier models are available
    available_tiers = []
    for tier_name, profile in MODEL_TIERS.items():
        if profile.model_id in available:
            available_tiers.append(tier_name)
            print(f"\n[OK] {profile.name} available")
        else:
            print(f"\n[--] {profile.name} not installed")

    if not available_tiers:
        print("\nNo tier models available. Install with:")
        print("  ollama pull gemma2:2b")
        print("  ollama pull mistral:latest")
        return None

    # Compare available models (reduced tasks due to GPU constraints)
    num_tasks = 3  # Reduced for faster demo
    print(f"\nRunning comparison with {num_tasks} tasks per model...")

    analyses = compare_models(available, num_tasks=num_tasks)

    if analyses:
        # Cost breakdown
        analyze_cost_breakdown(analyses)

        # Recommendations
        best_model = generate_recommendations(analyses)

        # Forecast
        if best_model and best_model in analyses:
            model_id = analyses[best_model]["model_id"]
            budget = forecast_costs(model_id, daily_tasks=100, days=30)

        # Summary
        print("\n" + "=" * 70)
        print("SUMMARY")
        print("=" * 70)

        total_tokens = sum(a.get("total_tokens", 0) for a in analyses.values())
        print(f"\nTotal tokens used in analysis: {total_tokens:,}")
        print(f"Estimated analysis cost: ${estimate_cost(total_tokens, 'gemma2:2b'):.4f}")

        print("\nKey Takeaways:")
        print("  1. Smaller models often have better CNSR for simple tasks")
        print("  2. Use tiered routing for cost optimization")
        print("  3. Monitor CNSR continuously in production")

    return analyses


if __name__ == "__main__":
    main()
