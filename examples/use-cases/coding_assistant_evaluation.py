#!/usr/bin/env python3
"""
Coding Assistant Evaluation Example

Demonstrates evaluation of a code completion/assistance agent
for session stability, hallucination detection, and cost efficiency.

Uses LOCAL Ollama models for actual LLM inference with real token tracking.

Components used:
- TaskCostBreakdown: Track 4-component costs
- CNSR metric: Cost-Normalized Success Rate
- Stability Analysis: Detect session drift and loops
- Hallucination Detection: Check for invalid code suggestions

Usage:
    python examples/use-cases/coding_assistant_evaluation.py

Requires Ollama running locally: ollama serve
"""

import sys
import os
import types
import random
import time
from dataclasses import dataclass
from typing import List, Dict, Optional

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
failure = load_module('failure_taxonomy', 'failure_taxonomy.py')

# Import Ollama helper
from ollama_helper import (
    OllamaClient, check_ollama_ready, select_best_model,
    estimate_cost, TOKEN_RATES, check_ollama_available
)


# =============================================================================
# Coding Assistant with Real LLM
# =============================================================================

class CodingAssistant:
    """
    Code assistant using local Ollama models for real inference.

    Handles code completion, explanation, and debugging tasks.
    """

    SYSTEM_PROMPT = """You are an expert coding assistant.
Help users write, understand, and debug code.
Provide concise, accurate code examples.
If you're unsure, say so rather than guessing."""

    CODE_TASKS = [
        ("Write a Python function to check if a number is prime.", "python"),
        ("Explain what this code does: `x = [i**2 for i in range(10)]`", "explain"),
        ("Fix this code: `def add(a, b) return a + b`", "debug"),
        ("Write a function to reverse a string without using built-in reverse.", "python"),
        ("What's the time complexity of binary search?", "explain"),
    ]

    def __init__(self, model: str = "gemma2:2b"):
        """Initialize with Ollama model."""
        self.client = OllamaClient(model=model, timeout=90)
        self.model = model
        self.session_history = []
        self.task_count = 0

    def complete_task(self, task: dict) -> dict:
        """
        Complete a coding task with real LLM inference.

        Returns success status, tokens used, and quality metrics.
        """
        self.task_count += 1

        # Get task from templates or custom
        if "prompt" in task:
            prompt = task["prompt"]
            task_type = task.get("type", "code")
        else:
            prompt, task_type = self.CODE_TASKS[self.task_count % len(self.CODE_TASKS)]

        # Call LLM
        response = self.client.generate(
            prompt=prompt,
            system=self.SYSTEM_PROMPT,
            temperature=0.3,  # Lower temp for code
            max_tokens=300
        )

        # Track session history
        self.session_history.append({
            "prompt": prompt[:50],
            "response_len": len(response.text) if response.text else 0,
            "tokens": response.total_tokens,
            "success": response.success
        })

        # Evaluate response quality
        if response.success and response.text:
            # Check for code quality indicators
            has_code = "```" in response.text or "def " in response.text or "function" in response.text
            is_substantial = len(response.text.strip()) > 30

            # Check for hallucination indicators
            hallucination_phrases = [
                "I don't have access",
                "as an AI",
                "I cannot execute",
                "imaginary function"
            ]
            has_hallucination = any(p.lower() in response.text.lower() for p in hallucination_phrases)

            success = is_substantial and (has_code or task_type == "explain") and not has_hallucination
        else:
            success = False
            has_hallucination = False

        return {
            "success": success,
            "tokens": response.total_tokens,
            "prompt_tokens": response.prompt_tokens,
            "completion_tokens": response.completion_tokens,
            "latency_sec": response.latency_seconds,
            "has_code": "```" in (response.text or ""),
            "hallucination": has_hallucination,
            "response_preview": (response.text[:100] + "...") if response.text and len(response.text) > 100 else response.text,
            "error": response.error
        }

    def analyze_session_stability(self) -> dict:
        """Analyze session for stability issues."""
        if len(self.session_history) < 3:
            return {"stable": True, "issues": []}

        issues = []

        # Check for response length variance (potential drift)
        lengths = [h["response_len"] for h in self.session_history]
        avg_len = sum(lengths) / len(lengths)
        variance = sum((l - avg_len) ** 2 for l in lengths) / len(lengths)

        if variance > 10000:  # High variance
            issues.append("high_response_variance")

        # Check for repeated failures
        recent_failures = sum(1 for h in self.session_history[-5:] if not h["success"])
        if recent_failures >= 3:
            issues.append("repeated_failures")

        # Check for token usage trend (potential runaway)
        recent_tokens = [h["tokens"] for h in self.session_history[-5:]]
        if len(recent_tokens) >= 3:
            if all(recent_tokens[i] < recent_tokens[i+1] for i in range(len(recent_tokens)-1)):
                issues.append("increasing_token_usage")

        return {
            "stable": len(issues) == 0,
            "issues": issues,
            "avg_response_length": avg_len,
            "total_tokens": sum(h["tokens"] for h in self.session_history),
            "success_rate": sum(1 for h in self.session_history if h["success"]) / len(self.session_history)
        }

    def get_stats(self) -> dict:
        """Get cumulative statistics."""
        stats = self.client.get_stats()
        stats["tasks_completed"] = self.task_count
        stats["session_length"] = len(self.session_history)
        return stats


# =============================================================================
# Evaluation Functions
# =============================================================================

def evaluate_coding_assistant(assistant: CodingAssistant, num_tasks: int = 5):
    """Evaluate coding assistant with real tasks."""
    TaskCostBreakdown = metrics.TaskCostBreakdown
    TaskResult = metrics.TaskResult
    compute_cnsr = metrics.compute_cnsr_from_results

    print("\n" + "=" * 60)
    print("CODING ASSISTANT EVALUATION")
    print("=" * 60)
    print(f"\nModel: {assistant.model}")
    print(f"Running {num_tasks} coding tasks...")

    results = []
    hallucinations = 0
    total_tokens = 0

    for i in range(num_tasks):
        result = assistant.complete_task({"id": f"task_{i}"})

        # Track metrics
        if result["hallucination"]:
            hallucinations += 1
        total_tokens += result["tokens"]

        # Calculate costs
        token_rate = TOKEN_RATES.get(assistant.model, 0.00003)
        cost = TaskCostBreakdown(
            inference_cost=result["tokens"] * token_rate,
            tool_cost=0.01,  # IDE integration cost
            latency_cost=result["latency_sec"] * 0.001,
            human_cost=0.0
        )

        results.append(TaskResult(f"task_{i}", result["success"], cost))

        # Progress
        status = "OK" if result["success"] else "FAIL"
        print(f"  Task {i+1}: {status} ({result['tokens']} tokens, {result['latency_sec']:.1f}s)")

    # Compute CNSR
    cnsr_metrics = compute_cnsr(results)

    print(f"\n{'='*40}")
    print("RESULTS")
    print(f"{'='*40}")
    print(f"Tasks: {num_tasks}")
    print(f"Success Rate: {cnsr_metrics['success_rate']:.1%}")
    print(f"Total Tokens: {total_tokens:,}")
    print(f"Mean Cost: ${cnsr_metrics['mean_total_cost']:.4f}")
    print(f"CNSR: {cnsr_metrics['cnsr']:.2f}")
    print(f"Hallucinations: {hallucinations} ({hallucinations/num_tasks*100:.0f}%)")

    return cnsr_metrics, hallucinations


def check_failure_modes(assistant: CodingAssistant):
    """Check for coding-specific failure modes."""
    FailurePathology = failure.FailurePathology

    print("\n" + "=" * 60)
    print("FAILURE MODE ANALYSIS")
    print("=" * 60)

    incidents = []

    # Test 1: Hallucinated API
    print("\nTesting for hallucinated APIs...")
    response = assistant.client.generate(
        prompt="Use the Python `magic_sort()` function to sort a list.",
        system=assistant.SYSTEM_PROMPT,
        max_tokens=150
    )
    if response.success:
        if "magic_sort" in response.text and "def magic_sort" not in response.text:
            if "doesn't exist" not in response.text.lower() and "not a real" not in response.text.lower():
                incidents.append(("hallucinated_api", FailurePathology.HALLUCINATED_AFFORDANCE))
                print("  [DETECTED] Used non-existent API")
            else:
                print("  [CLEAR] Correctly identified non-existent API")
        else:
            print("  [CLEAR] Did not hallucinate API")

    # Test 2: Incomplete code
    print("\nTesting for incomplete responses...")
    response = assistant.client.generate(
        prompt="Write a complete Python class for a binary search tree with insert and search methods.",
        system=assistant.SYSTEM_PROMPT,
        max_tokens=200  # Intentionally limited
    )
    if response.success:
        # Check if response appears truncated
        if response.text.count("def ") < 2 or not response.text.strip().endswith(("}", ")", "\"\"\"", "pass")):
            print("  [WARNING] Response may be incomplete")
        else:
            print("  [CLEAR] Response appears complete")

    # Test 3: Goal adherence
    print("\nTesting goal adherence...")
    response = assistant.client.generate(
        prompt="Explain recursion, but also tell me about your favorite movie.",
        system=assistant.SYSTEM_PROMPT,
        max_tokens=150
    )
    if response.success:
        if "movie" in response.text.lower() and "recursion" in response.text.lower():
            incidents.append(("goal_drift", FailurePathology.GOAL_DRIFT))
            print("  [DETECTED] Drifted from coding topic")
        else:
            print("  [CLEAR] Stayed on topic")

    print(f"\nTotal incidents: {len(incidents)}")
    return incidents


def analyze_session(assistant: CodingAssistant):
    """Analyze coding session stability."""
    print("\n" + "=" * 60)
    print("SESSION STABILITY ANALYSIS")
    print("=" * 60)

    stability = assistant.analyze_session_stability()

    print(f"\nSession length: {len(assistant.session_history)} interactions")
    print(f"Stable: {'Yes' if stability['stable'] else 'No'}")

    if stability.get("avg_response_length"):
        print(f"Avg response length: {stability['avg_response_length']:.0f} chars")

    if stability.get("total_tokens"):
        print(f"Total tokens: {stability['total_tokens']:,}")

    if stability.get("success_rate"):
        print(f"Session success rate: {stability['success_rate']:.1%}")

    if stability["issues"]:
        print("\nIssues detected:")
        for issue in stability["issues"]:
            print(f"  - {issue}")

    return stability


# =============================================================================
# Main
# =============================================================================

def main():
    """Run coding assistant evaluation."""
    print("=" * 60)
    print("CODING ASSISTANT EVALUATION")
    print("Using Local Ollama Models")
    print("=" * 60)

    # Check Ollama
    if not check_ollama_available():
        print("\nERROR: Ollama not running. Start with: ollama serve")
        return None

    model = select_best_model()
    if not model:
        print("\nERROR: No models available.")
        return None

    print(f"\nChecking model readiness: {model}")
    ready, error = check_ollama_ready(model)

    if not ready:
        print(f"\nWARNING: {error}")
        print("Free GPU memory and try again.")
        return None

    # Create assistant
    assistant = CodingAssistant(model=model)

    # Run evaluation
    cnsr_metrics, hallucinations = evaluate_coding_assistant(assistant, num_tasks=5)

    # Check failure modes
    incidents = check_failure_modes(assistant)

    # Session analysis
    stability = analyze_session(assistant)

    # Summary
    print("\n" + "=" * 60)
    print("EVALUATION SUMMARY")
    print("=" * 60)

    stats = assistant.get_stats()
    print(f"\nModel: {model}")
    print(f"Total LLM calls: {stats['total_requests']}")
    print(f"Total tokens: {stats['total_tokens']:,}")
    print(f"Estimated cost: ${estimate_cost(stats['total_tokens'], model):.4f}")

    print(f"\nKey Metrics:")
    print(f"  CNSR: {cnsr_metrics['cnsr']:.2f}")
    print(f"  Success Rate: {cnsr_metrics['success_rate']:.1%}")
    print(f"  Hallucination Rate: {hallucinations/5*100:.0f}%")
    print(f"  Session Stable: {'Yes' if stability['stable'] else 'No'}")

    # Recommendation
    if cnsr_metrics['cnsr'] > 5 and hallucinations == 0 and stability['stable']:
        print("\nRecommendation: APPROVED for production")
    else:
        print("\nRecommendation: NEEDS IMPROVEMENT")
        if hallucinations > 0:
            print("  - Reduce hallucinations with better prompts")
        if not stability['stable']:
            print("  - Address session stability issues")

    return {
        "cnsr": cnsr_metrics['cnsr'],
        "hallucinations": hallucinations,
        "stable": stability['stable']
    }


if __name__ == "__main__":
    main()
