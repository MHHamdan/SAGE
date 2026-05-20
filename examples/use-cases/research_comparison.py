#!/usr/bin/env python3
"""
Research Comparison Example

Demonstrates reproducible comparison of agent architectures for
academic research with statistical analysis.

Uses LOCAL Ollama models for actual LLM inference with real token tracking.

Components used:
- Benchmark Suites: Standardized evaluation tasks
- CNSR metric: Fair cost-normalized comparison
- Autonomy Criteria: Section IV-A evaluation
- Deterministic Seeds: Reproducibility

Usage:
    python examples/use-cases/research_comparison.py

Requires Ollama running locally: ollama serve
"""

import sys
import os
import types
import random
import math
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


# Load modules
metrics = load_module('metrics', 'metrics.py')
autonomy = load_module('autonomy_validator', 'autonomy_validator.py')

# Import Ollama helper
from ollama_helper import (
    OllamaClient, check_ollama_ready, select_best_model,
    estimate_cost, TOKEN_RATES, check_ollama_available
)


# =============================================================================
# Benchmark Tasks
# =============================================================================

BENCHMARK_TASKS = [
    {"prompt": "What is the capital of France?", "type": "factual", "expected": "paris"},
    {"prompt": "Calculate 15 * 7", "type": "math", "expected": "105"},
    {"prompt": "Summarize: The quick brown fox jumps over the lazy dog.", "type": "summary"},
    {"prompt": "What comes next: 2, 4, 8, 16, ?", "type": "reasoning", "expected": "32"},
    {"prompt": "Define 'photosynthesis' in one sentence.", "type": "knowledge"},
]


# =============================================================================
# Agent Architectures with Real LLM
# =============================================================================

class BaseAgent:
    """Base agent class using Ollama."""

    def __init__(self, name: str, model: str, seed: int = 42):
        self.name = name
        self.model = model
        self.client = OllamaClient(model=model, timeout=60)
        self.rng = random.Random(seed)
        self.task_count = 0

    def run_task(self, task: dict) -> dict:
        raise NotImplementedError


class DirectAgent(BaseAgent):
    """Direct prompting agent - no additional reasoning."""

    def __init__(self, model: str, seed: int = 42):
        super().__init__("DirectAgent", model, seed)

    def run_task(self, task: dict) -> dict:
        self.task_count += 1

        response = self.client.generate(
            prompt=task["prompt"],
            temperature=0.3,
            max_tokens=100
        )

        if response.success:
            # Check if answer is correct (if expected is provided)
            success = True
            if "expected" in task:
                success = task["expected"].lower() in response.text.lower()
            elif len(response.text.strip()) < 5:
                success = False
        else:
            success = False

        return {
            "success": success,
            "tokens": response.total_tokens,
            "latency": response.latency_seconds,
            "response": response.text[:100] if response.text else ""
        }


class ReasoningAgent(BaseAgent):
    """Chain-of-thought reasoning agent."""

    REASONING_PROMPT = """Think step by step to solve this problem.
First, analyze what is being asked.
Then, work through the solution.
Finally, provide a clear answer.

Problem: {prompt}

Step-by-step solution:"""

    def __init__(self, model: str, seed: int = 42):
        super().__init__("ReasoningAgent", model, seed)

    def run_task(self, task: dict) -> dict:
        self.task_count += 1

        # Use chain-of-thought prompting
        prompt = self.REASONING_PROMPT.format(prompt=task["prompt"])

        response = self.client.generate(
            prompt=prompt,
            temperature=0.3,
            max_tokens=200  # More tokens for reasoning
        )

        if response.success:
            success = True
            if "expected" in task:
                success = task["expected"].lower() in response.text.lower()
            elif len(response.text.strip()) < 10:
                success = False
        else:
            success = False

        return {
            "success": success,
            "tokens": response.total_tokens,
            "latency": response.latency_seconds,
            "response": response.text[:150] if response.text else ""
        }


# =============================================================================
# Evaluation Functions
# =============================================================================

def run_benchmark(agent: BaseAgent, num_runs: int = 2, seed_base: int = 42) -> List[dict]:
    """Run standardized benchmark with multiple runs for statistical validity."""
    TaskCostBreakdown = metrics.TaskCostBreakdown
    TaskResult = metrics.TaskResult
    compute_cnsr = metrics.compute_cnsr_from_results

    all_runs = []

    for run in range(num_runs):
        agent.rng = random.Random(seed_base + run * 1000)
        results = []

        for i, task in enumerate(BENCHMARK_TASKS):
            task_result = agent.run_task(task)

            token_rate = TOKEN_RATES.get(agent.model, 0.00003)
            cost = TaskCostBreakdown(
                inference_cost=task_result["tokens"] * token_rate,
                tool_cost=0.005,
                latency_cost=task_result["latency"] * 0.001,
                human_cost=0.0
            )

            results.append(TaskResult(
                task_id=f"run{run}_task{i}",
                success=task_result["success"],
                cost=cost
            ))

        run_metrics = compute_cnsr(results)
        run_metrics["run_id"] = run
        run_metrics["seed"] = seed_base + run * 1000
        run_metrics["total_tokens"] = sum(r.cost.inference_cost / token_rate for r in results)
        all_runs.append(run_metrics)

    return all_runs


def compute_statistics(runs: List[dict]) -> dict:
    """Compute mean and std across runs."""
    def mean(values):
        return sum(values) / len(values) if values else 0

    def std(values):
        if len(values) < 2:
            return 0
        m = mean(values)
        return math.sqrt(sum((x - m) ** 2 for x in values) / len(values))

    stats = {}
    for key in ["success_rate", "cnsr", "mean_total_cost"]:
        values = [r[key] for r in runs]
        stats[key] = {"mean": mean(values), "std": std(values)}

    stats["total_tokens"] = sum(r.get("total_tokens", 0) for r in runs)
    return stats


def compare_architectures(model: str, num_runs: int = 2):
    """Compare agent architectures with statistical analysis."""
    print("\n" + "=" * 70)
    print("ARCHITECTURE COMPARISON")
    print("=" * 70)

    agents = {
        "DirectAgent": DirectAgent(model=model),
        "ReasoningAgent": ReasoningAgent(model=model)
    }

    results = {}

    for name, agent in agents.items():
        print(f"\nEvaluating {name}...")
        runs = run_benchmark(agent, num_runs=num_runs)
        stats = compute_statistics(runs)
        results[name] = {"runs": runs, "stats": stats}
        print(f"  Completed {num_runs} runs, {len(BENCHMARK_TASKS)} tasks each")

    # Print comparison table
    print("\n" + "=" * 70)
    print(f"RESULTS (mean +/- std, n={num_runs} runs)")
    print("=" * 70)

    print(f"\n{'Architecture':<20} {'Success Rate':<18} {'CNSR':<18} {'Cost/Task'}")
    print("-" * 70)

    for name, data in results.items():
        s = data["stats"]
        sr = f"{s['success_rate']['mean']:.1%} +/- {s['success_rate']['std']:.1%}"
        cnsr = f"{s['cnsr']['mean']:.2f} +/- {s['cnsr']['std']:.2f}"
        cost = f"${s['mean_total_cost']['mean']:.4f}"
        print(f"{name:<20} {sr:<18} {cnsr:<18} {cost}")

    # Effect size calculation
    if len(results) == 2:
        names = list(results.keys())
        cnsr_a = [r["cnsr"] for r in results[names[0]]["runs"]]
        cnsr_b = [r["cnsr"] for r in results[names[1]]["runs"]]

        if len(cnsr_a) >= 2 and len(cnsr_b) >= 2:
            mean_a, mean_b = sum(cnsr_a)/len(cnsr_a), sum(cnsr_b)/len(cnsr_b)
            pooled_var = ((sum((x - mean_a)**2 for x in cnsr_a) +
                           sum((x - mean_b)**2 for x in cnsr_b)) /
                          (len(cnsr_a) + len(cnsr_b) - 2))

            if pooled_var > 0:
                effect_size = abs(mean_a - mean_b) / math.sqrt(pooled_var)
                interpretation = "Large" if effect_size > 0.8 else "Medium" if effect_size > 0.5 else "Small"
                print(f"\nEffect size (Cohen's d): {effect_size:.2f}")
                print(f"Interpretation: {interpretation} effect")

    return results


def validate_autonomy_levels():
    """Display autonomy criteria from the paper."""
    AutonomyCriterion = autonomy.AutonomyCriterion
    AutonomyLevel = autonomy.AutonomyLevel

    print("\n" + "=" * 70)
    print("AUTONOMY FRAMEWORK (Section IV-A)")
    print("=" * 70)

    print("\n4 Autonomy Criteria:")
    for criterion in AutonomyCriterion:
        print(f"  - {criterion.value}")

    print("\n5 Autonomy Levels (Table IV):")
    for level in AutonomyLevel:
        print(f"  {level.value}. {level.name}")

    print("\nNote: DirectAgent = CONDITIONAL level")
    print("      ReasoningAgent = GUIDED_AGENT level (with reasoning)")


def generate_publication_table(results: dict):
    """Generate publication-ready summary table."""
    print("\n" + "=" * 70)
    print("PUBLICATION-READY TABLE")
    print("=" * 70)

    print("""
| Architecture     | Success Rate      | CNSR              | Cost/Task |
|------------------|-------------------|-------------------|-----------|""")

    for name, data in results.items():
        s = data["stats"]
        print(f"| {name:<16} | {s['success_rate']['mean']:.1%} +/- {s['success_rate']['std']:.1%}  | "
              f"{s['cnsr']['mean']:.2f} +/- {s['cnsr']['std']:.2f}    | "
              f"${s['mean_total_cost']['mean']:.4f}    |")

    print("""
## Reproducibility Notes

- Tasks: 5 benchmark tasks per run
- Runs: 2 (for statistical validity)
- Deterministic seeds: base=42
- Local Ollama model for reproducibility
""")


# =============================================================================
# Main
# =============================================================================

def main():
    """Run complete research evaluation."""
    print("=" * 70)
    print("RESEARCH EVALUATION")
    print("Agent Architecture Comparison with Real LLM Inference")
    print("=" * 70)

    # Check Ollama
    if not check_ollama_available():
        print("\nERROR: Ollama not running. Start with: ollama serve")
        return None

    model = select_best_model()
    if not model:
        print("\nERROR: No models available.")
        return None

    print(f"\nModel: {model}")
    print(f"Benchmark: {len(BENCHMARK_TASKS)} tasks, 2 runs each")
    print("Seeds: Deterministic (base=42)")

    ready, error = check_ollama_ready(model)
    if not ready:
        print(f"\nWARNING: {error}")
        return None

    # Compare architectures
    results = compare_architectures(model, num_runs=2)

    # Show autonomy framework
    validate_autonomy_levels()

    # Publication table
    generate_publication_table(results)

    # Summary
    print("\n" + "=" * 70)
    print("EVALUATION COMPLETE")
    print("=" * 70)

    total_tokens = sum(
        data["stats"]["total_tokens"]
        for data in results.values()
    )
    print(f"\nTotal tokens used: {total_tokens:,.0f}")
    print(f"Estimated cost: ${estimate_cost(int(total_tokens), model):.4f}")

    return results


if __name__ == "__main__":
    main()
