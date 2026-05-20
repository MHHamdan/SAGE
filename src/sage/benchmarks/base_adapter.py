"""
Base Benchmark Adapter

Provides the base interface for benchmark adapters that integrate
with the CNSR validation framework.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Generic, TypeVar
from enum import Enum

logger = logging.getLogger(__name__)


@dataclass
class BenchmarkTask:
    """Base class for benchmark tasks.

    Attributes:
        task_id: Unique task identifier
        query: Task query/prompt
        expected_output: Expected output (for evaluation)
        metadata: Additional task metadata
    """
    task_id: str
    query: str
    expected_output: Optional[Any] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "task_id": self.task_id,
            "query": self.query,
            "expected_output": self.expected_output,
            "metadata": self.metadata
        }


@dataclass
class BenchmarkResult:
    """Base class for benchmark results.

    Attributes:
        task_id: Task identifier
        success: Whether task was successful
        output: Agent output
        score: Numeric score (0-1)
        error: Error message if failed
        metadata: Additional result metadata
    """
    task_id: str
    success: bool
    output: Optional[str] = None
    score: float = 0.0
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "task_id": self.task_id,
            "success": self.success,
            "output": self.output,
            "score": self.score,
            "error": self.error,
            "metadata": self.metadata
        }


T = TypeVar('T', bound=BenchmarkTask)
R = TypeVar('R', bound=BenchmarkResult)


class BenchmarkAdapter(ABC, Generic[T, R]):
    """Abstract base class for benchmark adapters.

    Benchmark adapters load tasks from standard benchmarks and
    provide evaluation methods for agent outputs.

    Example:
        >>> adapter = MyBenchmarkAdapter()
        >>> tasks = adapter.load_tasks(n=100)
        >>> for task in tasks:
        ...     output = agent.run(task.query)
        ...     result = adapter.evaluate(task, output)
        ...     print(f"Task {task.task_id}: {result.score}")
    """

    def __init__(self, data_path: Optional[str] = None, seed: int = 42):
        """Initialize benchmark adapter.

        Args:
            data_path: Path to benchmark data (optional)
            seed: Random seed for reproducibility
        """
        self.data_path = data_path
        self.seed = seed
        self._tasks: List[T] = []
        self._loaded = False

    @property
    @abstractmethod
    def name(self) -> str:
        """Benchmark name."""
        pass

    @property
    @abstractmethod
    def description(self) -> str:
        """Benchmark description."""
        pass

    @abstractmethod
    def load_tasks(self, n: Optional[int] = None, **kwargs) -> List[T]:
        """Load benchmark tasks.

        Args:
            n: Number of tasks to load (None for all)
            **kwargs: Additional loading options

        Returns:
            List of benchmark tasks
        """
        pass

    @abstractmethod
    def evaluate(self, task: T, output: str) -> R:
        """Evaluate agent output for a task.

        Args:
            task: The task
            output: Agent output

        Returns:
            Benchmark result
        """
        pass

    def get_statistics(self, results: List[R]) -> Dict[str, Any]:
        """Get aggregate statistics for results.

        Args:
            results: List of benchmark results

        Returns:
            Dictionary with statistics
        """
        if not results:
            return {
                "total": 0,
                "successes": 0,
                "success_rate": 0.0,
                "mean_score": 0.0
            }

        total = len(results)
        successes = sum(1 for r in results if r.success)
        mean_score = sum(r.score for r in results) / total

        return {
            "total": total,
            "successes": successes,
            "success_rate": successes / total,
            "mean_score": mean_score
        }
