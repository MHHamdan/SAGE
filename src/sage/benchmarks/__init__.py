"""
Benchmark Adapters Module

Provides adapters for standard agent benchmarks:
- AgentBench: OS interaction tasks
- SWE-Bench: Software engineering tasks
- HotpotQA: Multi-hop reasoning tasks

These adapters integrate with the CNSR benchmark framework for
empirical validation of the Cost-Normalized Success Rate metric.

Example:
    >>> from sage.benchmarks import AgentBenchAdapter, SWEBenchAdapter
    >>>
    >>> # Load and run AgentBench tasks
    >>> agentbench = AgentBenchAdapter()
    >>> tasks = agentbench.load_tasks(subset="os", n=50)
    >>> for task in tasks:
    ...     result = agent.run(task.query)
    ...     score = agentbench.evaluate(agent, task)
"""

from .agentbench_adapter import (
    AgentBenchAdapter,
    AgentBenchResult,
    AgentBenchSubset,
    AgentBenchTask,
)
from .base_adapter import (
    BenchmarkAdapter,
    BenchmarkResult,
    BenchmarkTask,
)
from .hotpotqa_adapter import (
    HotpotQAAdapter,
    HotpotQAResult,
    HotpotQATask,
    HotpotQAType,
)
from .swebench_adapter import (
    SWEBenchAdapter,
    SWEBenchDifficulty,
    SWEBenchResult,
    SWEBenchTask,
)

__all__ = [
    # Base
    "BenchmarkAdapter",
    "BenchmarkTask",
    "BenchmarkResult",
    # AgentBench
    "AgentBenchAdapter",
    "AgentBenchTask",
    "AgentBenchResult",
    "AgentBenchSubset",
    # SWE-Bench
    "SWEBenchAdapter",
    "SWEBenchTask",
    "SWEBenchResult",
    "SWEBenchDifficulty",
    # HotpotQA
    "HotpotQAAdapter",
    "HotpotQATask",
    "HotpotQAResult",
    "HotpotQAType",
]
