#!/usr/bin/env python3
"""
Multi-Agent System Evaluation Example

Demonstrates evaluation of multi-agent systems for coordination,
communication patterns, and distributed cost analysis.

Uses LOCAL Ollama models for actual LLM inference with real token tracking.

Components used:
- Cost tracking: Distributed cost analysis across agents
- Communication analysis: Message patterns between agents
- CNSR metric: System-level cost efficiency

Usage:
    python examples/use-cases/multi_agent_evaluation.py

Requires Ollama running locally: ollama serve
"""

import sys
import os
import types
import random
from dataclasses import dataclass, field
from typing import List, Dict, Set, Optional
from collections import defaultdict

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

# Import Ollama helper
from ollama_helper import (
    OllamaClient, check_ollama_ready, select_best_model,
    estimate_cost, TOKEN_RATES, check_ollama_available
)


# =============================================================================
# Multi-Agent System with Real LLM
# =============================================================================

@dataclass
class Message:
    """Inter-agent message."""
    sender: str
    receiver: str
    content: str
    tokens: int = 0


class BaseAgent:
    """Base agent in multi-agent system."""

    def __init__(self, agent_id: str, role: str, model: str):
        self.agent_id = agent_id
        self.role = role
        self.model = model
        self.client = OllamaClient(model=model, timeout=60)
        self.inbox: List[Message] = []
        self.outbox: List[Message] = []
        self.total_tokens = 0

    def process_message(self, message: Message) -> Optional[Message]:
        """Process incoming message and optionally respond."""
        self.inbox.append(message)
        return None

    def get_stats(self) -> dict:
        return {
            "agent_id": self.agent_id,
            "role": self.role,
            "messages_received": len(self.inbox),
            "messages_sent": len(self.outbox),
            "total_tokens": self.total_tokens
        }


class CoordinatorAgent(BaseAgent):
    """Coordinator that delegates tasks to workers."""

    SYSTEM_PROMPT = """You are a task coordinator.
Break down complex tasks and delegate subtasks to workers.
Be concise and specific in your delegations."""

    def __init__(self, model: str, workers: List[str]):
        super().__init__("coordinator", "coordinator", model)
        self.workers = workers

    def delegate_task(self, task: str) -> List[Message]:
        """Delegate a task to workers."""
        # Generate delegation plan
        prompt = f"""Task to delegate: {task}

Workers available: {', '.join(self.workers)}

For each worker, provide a one-line subtask assignment.
Format: worker_name: subtask description"""

        response = self.client.generate(
            prompt=prompt,
            system=self.SYSTEM_PROMPT,
            temperature=0.3,
            max_tokens=150
        )

        self.total_tokens += response.total_tokens

        messages = []
        if response.success:
            # Parse response and create messages
            for worker in self.workers:
                # Simple assignment
                subtask = f"Subtask from coordinator: Process '{task}' - your role: {worker}"
                msg = Message(
                    sender=self.agent_id,
                    receiver=worker,
                    content=subtask,
                    tokens=response.total_tokens // len(self.workers)
                )
                messages.append(msg)
                self.outbox.append(msg)

        return messages


class WorkerAgent(BaseAgent):
    """Worker agent that executes subtasks."""

    SYSTEM_PROMPT = """You are a task worker.
Execute assigned subtasks efficiently and report results concisely."""

    def __init__(self, agent_id: str, specialty: str, model: str):
        super().__init__(agent_id, f"worker_{specialty}", model)
        self.specialty = specialty

    def process_message(self, message: Message) -> Message:
        """Process task message and return result."""
        self.inbox.append(message)

        # Execute task with LLM
        prompt = f"""Task assignment: {message.content}

Your specialty: {self.specialty}

Provide a brief result (1-2 sentences)."""

        response = self.client.generate(
            prompt=prompt,
            system=self.SYSTEM_PROMPT,
            temperature=0.3,
            max_tokens=100
        )

        self.total_tokens += response.total_tokens

        # Create response message
        result_content = response.text[:100] if response.success else "Task failed"
        result_msg = Message(
            sender=self.agent_id,
            receiver=message.sender,
            content=f"Result: {result_content}",
            tokens=response.total_tokens
        )
        self.outbox.append(result_msg)

        return result_msg


# =============================================================================
# Evaluation Functions
# =============================================================================

def evaluate_multi_agent_task(coordinator: CoordinatorAgent, workers: List[WorkerAgent], task: str):
    """Evaluate a single multi-agent task execution."""
    print(f"\n  Task: {task[:50]}...")

    # Phase 1: Coordination
    delegation_messages = coordinator.delegate_task(task)
    print(f"  Delegated to {len(delegation_messages)} workers")

    # Phase 2: Worker execution
    responses = []
    for msg in delegation_messages:
        for worker in workers:
            if worker.agent_id == msg.receiver:
                response = worker.process_message(msg)
                responses.append(response)
                break

    print(f"  Received {len(responses)} responses")

    # Collect metrics
    total_tokens = coordinator.total_tokens + sum(w.total_tokens for w in workers)
    success = len(responses) == len(workers) and all(r.content for r in responses)

    return {
        "success": success,
        "total_tokens": total_tokens,
        "messages": len(delegation_messages) + len(responses),
        "coordinator_tokens": coordinator.total_tokens,
        "worker_tokens": {w.agent_id: w.total_tokens for w in workers}
    }


def analyze_communication_patterns(coordinator: CoordinatorAgent, workers: List[WorkerAgent]):
    """Analyze communication patterns in the system."""
    print("\n" + "=" * 60)
    print("COMMUNICATION ANALYSIS")
    print("=" * 60)

    # Count messages
    total_messages = len(coordinator.outbox) + sum(len(w.outbox) for w in workers)
    total_tokens = coordinator.total_tokens + sum(w.total_tokens for w in workers)

    print(f"\nTotal messages: {total_messages}")
    print(f"Total tokens: {total_tokens:,}")

    # Message distribution
    print("\nMessage distribution:")
    print(f"  Coordinator sent: {len(coordinator.outbox)}")
    for worker in workers:
        print(f"  {worker.agent_id} sent: {len(worker.outbox)}")

    # Token distribution
    print("\nToken distribution:")
    coord_pct = coordinator.total_tokens / total_tokens * 100 if total_tokens > 0 else 0
    print(f"  Coordinator: {coordinator.total_tokens:,} ({coord_pct:.1f}%)")

    for worker in workers:
        worker_pct = worker.total_tokens / total_tokens * 100 if total_tokens > 0 else 0
        print(f"  {worker.agent_id}: {worker.total_tokens:,} ({worker_pct:.1f}%)")

    return {
        "total_messages": total_messages,
        "total_tokens": total_tokens,
        "coordinator_ratio": coord_pct
    }


def evaluate_distributed_costs(coordinator: CoordinatorAgent, workers: List[WorkerAgent], num_tasks: int = 3):
    """Track costs across multi-agent system with real inference."""
    TaskCostBreakdown = metrics.TaskCostBreakdown
    TaskResult = metrics.TaskResult
    compute_cnsr = metrics.compute_cnsr_from_results

    print("\n" + "=" * 60)
    print("DISTRIBUTED COST ANALYSIS")
    print("=" * 60)

    # Test tasks
    test_tasks = [
        "Research the latest trends in renewable energy",
        "Summarize key points about machine learning",
        "Analyze the benefits of remote work",
    ]

    results = []
    agent_costs = defaultdict(float)

    for i in range(min(num_tasks, len(test_tasks))):
        print(f"\nTask {i+1}/{num_tasks}")

        # Reset token counters
        initial_coord = coordinator.total_tokens
        initial_workers = {w.agent_id: w.total_tokens for w in workers}

        # Execute task
        task_result = evaluate_multi_agent_task(coordinator, workers, test_tasks[i])

        # Calculate incremental costs
        coord_tokens = coordinator.total_tokens - initial_coord
        token_rate = TOKEN_RATES.get(coordinator.model, 0.00003)

        coord_cost = coord_tokens * token_rate
        agent_costs["coordinator"] += coord_cost

        worker_cost = 0
        for worker in workers:
            w_tokens = worker.total_tokens - initial_workers[worker.agent_id]
            w_cost = w_tokens * token_rate
            agent_costs[worker.agent_id] += w_cost
            worker_cost += w_cost

        # Communication cost
        comm_cost = task_result["messages"] * 0.001

        cost = TaskCostBreakdown(
            inference_cost=coord_cost + worker_cost,
            tool_cost=comm_cost,
            latency_cost=0.005,  # Fixed latency
            human_cost=0.0
        )

        results.append(TaskResult(f"task_{i}", task_result["success"], cost))

    # Compute CNSR
    cnsr_metrics = compute_cnsr(results)

    print(f"\n{'='*40}")
    print("RESULTS")
    print(f"{'='*40}")
    print(f"Tasks: {len(results)}")
    print(f"Success rate: {cnsr_metrics['success_rate']:.1%}")
    print(f"Mean cost: ${cnsr_metrics['mean_total_cost']:.4f}")
    print(f"CNSR: {cnsr_metrics['cnsr']:.2f}")

    print("\nCost by agent:")
    total_agent_cost = sum(agent_costs.values())
    for agent_id, cost in sorted(agent_costs.items()):
        pct = cost / total_agent_cost * 100 if total_agent_cost > 0 else 0
        print(f"  {agent_id}: ${cost:.4f} ({pct:.1f}%)")

    return cnsr_metrics, agent_costs


# =============================================================================
# Main
# =============================================================================

def main():
    """Run multi-agent system evaluation."""
    print("=" * 60)
    print("MULTI-AGENT SYSTEM EVALUATION")
    print("Coordinator-Worker Architecture with Real LLM")
    print("=" * 60)

    # Check Ollama
    if not check_ollama_available():
        print("\nERROR: Ollama not running. Start with: ollama serve")
        return None

    model = select_best_model()
    if not model:
        print("\nERROR: No models available.")
        return None

    print(f"\nModel: {model}")

    ready, error = check_ollama_ready(model)
    if not ready:
        print(f"\nWARNING: {error}")
        return None

    # Create multi-agent system
    worker_ids = ["researcher", "analyst", "writer"]
    workers = [WorkerAgent(wid, wid, model) for wid in worker_ids]
    coordinator = CoordinatorAgent(model, worker_ids)

    print(f"\nSystem: 1 coordinator + {len(workers)} workers")
    for w in workers:
        print(f"  - {w.agent_id} ({w.specialty})")

    # Evaluate
    cnsr_metrics, agent_costs = evaluate_distributed_costs(coordinator, workers, num_tasks=3)

    # Communication analysis
    comm_stats = analyze_communication_patterns(coordinator, workers)

    # Summary
    print("\n" + "=" * 60)
    print("EVALUATION SUMMARY")
    print("=" * 60)

    total_tokens = coordinator.total_tokens + sum(w.total_tokens for w in workers)
    print(f"\nTotal tokens: {total_tokens:,}")
    print(f"Estimated cost: ${estimate_cost(total_tokens, model):.4f}")

    print(f"\nKey Metrics:")
    print(f"  CNSR: {cnsr_metrics['cnsr']:.2f}")
    print(f"  Success Rate: {cnsr_metrics['success_rate']:.1%}")
    print(f"  Coordinator overhead: {comm_stats['coordinator_ratio']:.1f}%")

    return {
        "cnsr": cnsr_metrics['cnsr'],
        "total_tokens": total_tokens,
        "agent_costs": dict(agent_costs)
    }


if __name__ == "__main__":
    main()
