"""Standard benchmarks for agent evaluation."""

import logging
import time
import json
from typing import Optional, List, Dict, Any, Callable
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
from enum import Enum

logger = logging.getLogger(__name__)


class TaskDifficulty(Enum):
    """Task difficulty levels."""
    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"
    EXPERT = "expert"


class TaskCategory(Enum):
    """Task categories."""
    REASONING = "reasoning"
    TOOL_USE = "tool_use"
    PLANNING = "planning"
    MULTI_AGENT = "multi_agent"
    MEMORY = "memory"
    CODING = "coding"


@dataclass
class BenchmarkTask:
    """A single benchmark task.

    Attributes:
        task_id: Unique task identifier
        name: Task name
        description: Task description
        category: Task category
        difficulty: Task difficulty
        input_data: Input for the task
        expected_output: Expected output (for evaluation)
        max_steps: Maximum steps allowed
        timeout_seconds: Maximum time allowed
        metadata: Additional task metadata
    """
    task_id: str
    name: str
    description: str
    category: TaskCategory
    difficulty: TaskDifficulty
    input_data: Dict[str, Any]
    expected_output: Any = None
    max_steps: int = 100
    timeout_seconds: float = 300.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "task_id": self.task_id,
            "name": self.name,
            "description": self.description,
            "category": self.category.value,
            "difficulty": self.difficulty.value,
            "input_data": self.input_data,
            "expected_output": self.expected_output,
            "max_steps": self.max_steps,
            "timeout_seconds": self.timeout_seconds,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "BenchmarkTask":
        """Create from dictionary."""
        return cls(
            task_id=data["task_id"],
            name=data["name"],
            description=data["description"],
            category=TaskCategory(data["category"]),
            difficulty=TaskDifficulty(data["difficulty"]),
            input_data=data["input_data"],
            expected_output=data.get("expected_output"),
            max_steps=data.get("max_steps", 100),
            timeout_seconds=data.get("timeout_seconds", 300.0),
            metadata=data.get("metadata", {}),
        )


@dataclass
class BenchmarkResult:
    """Result of running a benchmark task.

    Attributes:
        task_id: Task identifier
        success: Whether task was successful
        output: Agent output
        steps_taken: Number of steps taken
        duration_seconds: Time taken
        cost_usd: Estimated cost in USD
        error: Error message if failed
        metrics: Additional metrics
    """
    task_id: str
    success: bool
    output: Any = None
    steps_taken: int = 0
    duration_seconds: float = 0.0
    cost_usd: float = 0.0
    error: Optional[str] = None
    metrics: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "task_id": self.task_id,
            "success": self.success,
            "output": self.output,
            "steps_taken": self.steps_taken,
            "duration_seconds": self.duration_seconds,
            "cost_usd": self.cost_usd,
            "error": self.error,
            "metrics": self.metrics,
        }


class Benchmark(ABC):
    """Base class for benchmarks.

    Example:
        >>> benchmark = ReasoningBenchmark()
        >>> for task in benchmark.get_tasks():
        ...     result = agent.run(task)
        ...     benchmark.record_result(result)
        >>> summary = benchmark.get_summary()
    """

    def __init__(
        self,
        name: str,
        description: str = "",
        version: str = "1.0.0",
    ):
        """Initialize benchmark.

        Args:
            name: Benchmark name
            description: Benchmark description
            version: Benchmark version
        """
        self.name = name
        self.description = description
        self.version = version
        self._tasks: List[BenchmarkTask] = []
        self._results: List[BenchmarkResult] = []

    @abstractmethod
    def load_tasks(self):
        """Load benchmark tasks. Override in subclass."""
        pass

    def get_tasks(
        self,
        difficulty: Optional[TaskDifficulty] = None,
        limit: Optional[int] = None,
    ) -> List[BenchmarkTask]:
        """Get benchmark tasks.

        Args:
            difficulty: Filter by difficulty
            limit: Maximum tasks to return

        Returns:
            List of tasks
        """
        if not self._tasks:
            self.load_tasks()

        tasks = self._tasks

        if difficulty:
            tasks = [t for t in tasks if t.difficulty == difficulty]

        if limit:
            tasks = tasks[:limit]

        return tasks

    def record_result(self, result: BenchmarkResult):
        """Record a task result.

        Args:
            result: Benchmark result
        """
        self._results.append(result)

    def evaluate_output(
        self,
        task: BenchmarkTask,
        output: Any,
    ) -> bool:
        """Evaluate agent output against expected.

        Args:
            task: The task
            output: Agent output

        Returns:
            True if output is correct
        """
        if task.expected_output is None:
            return True  # No expected output defined

        # Simple equality check - subclasses can override
        return output == task.expected_output

    def get_summary(self) -> Dict[str, Any]:
        """Get benchmark summary statistics.

        Returns:
            Summary statistics
        """
        if not self._results:
            return {
                "name": self.name,
                "version": self.version,
                "total_tasks": 0,
            }

        total = len(self._results)
        successes = sum(1 for r in self._results if r.success)
        total_cost = sum(r.cost_usd for r in self._results)
        total_steps = sum(r.steps_taken for r in self._results)
        total_time = sum(r.duration_seconds for r in self._results)

        return {
            "name": self.name,
            "version": self.version,
            "total_tasks": total,
            "successes": successes,
            "success_rate": successes / total,
            "total_cost_usd": total_cost,
            "mean_cost_usd": total_cost / total,
            "total_steps": total_steps,
            "mean_steps": total_steps / total,
            "total_duration_seconds": total_time,
            "mean_duration_seconds": total_time / total,
            "cnsr": (successes / total) / (total_cost / total) if total_cost > 0 else float('inf'),
        }

    def reset(self):
        """Reset results."""
        self._results.clear()


class ReasoningBenchmark(Benchmark):
    """Benchmark for reasoning capabilities.

    Tests:
    - Logical reasoning
    - Mathematical reasoning
    - Commonsense reasoning
    """

    def __init__(self):
        super().__init__(
            name="Reasoning Benchmark",
            description="Evaluates agent reasoning capabilities",
        )

    def load_tasks(self):
        """Load reasoning tasks."""
        self._tasks = [
            # Logical reasoning
            BenchmarkTask(
                task_id="reason_001",
                name="Syllogism",
                description="Evaluate a logical syllogism",
                category=TaskCategory.REASONING,
                difficulty=TaskDifficulty.EASY,
                input_data={
                    "premise1": "All humans are mortal",
                    "premise2": "Socrates is a human",
                    "question": "Is Socrates mortal?",
                },
                expected_output="Yes",
            ),
            BenchmarkTask(
                task_id="reason_002",
                name="Mathematical Reasoning",
                description="Solve a multi-step math problem",
                category=TaskCategory.REASONING,
                difficulty=TaskDifficulty.MEDIUM,
                input_data={
                    "problem": "If x + 5 = 12 and y = 2x, what is y?",
                },
                expected_output="14",
            ),
            BenchmarkTask(
                task_id="reason_003",
                name="Causal Reasoning",
                description="Identify cause and effect",
                category=TaskCategory.REASONING,
                difficulty=TaskDifficulty.MEDIUM,
                input_data={
                    "scenario": "The plant died. The gardener forgot to water it for weeks.",
                    "question": "Why did the plant die?",
                },
                expected_output="lack_of_water",
            ),
            BenchmarkTask(
                task_id="reason_004",
                name="Counterfactual Reasoning",
                description="Reason about hypotheticals",
                category=TaskCategory.REASONING,
                difficulty=TaskDifficulty.HARD,
                input_data={
                    "scenario": "John took an umbrella. It rained heavily.",
                    "question": "If John had not taken an umbrella, what would have happened?",
                },
                expected_output="got_wet",
            ),
        ]


class ToolUseBenchmark(Benchmark):
    """Benchmark for tool use capabilities.

    Tests:
    - Tool selection
    - Parameter extraction
    - Result interpretation
    """

    def __init__(self, mock_tools: Optional[Dict[str, Callable]] = None):
        super().__init__(
            name="Tool Use Benchmark",
            description="Evaluates agent tool use capabilities",
        )
        self.mock_tools = mock_tools or {}

    def load_tasks(self):
        """Load tool use tasks."""
        self._tasks = [
            BenchmarkTask(
                task_id="tool_001",
                name="Simple Tool Call",
                description="Call a single tool with correct parameters",
                category=TaskCategory.TOOL_USE,
                difficulty=TaskDifficulty.EASY,
                input_data={
                    "instruction": "Search for information about Paris",
                    "available_tools": ["search", "calculator", "file_read"],
                },
                expected_output={"tool": "search", "params": {"query": "Paris"}},
            ),
            BenchmarkTask(
                task_id="tool_002",
                name="Tool Chain",
                description="Chain multiple tool calls",
                category=TaskCategory.TOOL_USE,
                difficulty=TaskDifficulty.MEDIUM,
                input_data={
                    "instruction": "Read the file 'data.txt' and count the words",
                    "available_tools": ["file_read", "word_count", "calculator"],
                },
                expected_output=["file_read", "word_count"],
            ),
            BenchmarkTask(
                task_id="tool_003",
                name="Tool Selection",
                description="Choose the right tool among options",
                category=TaskCategory.TOOL_USE,
                difficulty=TaskDifficulty.MEDIUM,
                input_data={
                    "instruction": "Calculate 15% of 250",
                    "available_tools": ["search", "calculator", "translator", "file_write"],
                },
                expected_output="calculator",
            ),
            BenchmarkTask(
                task_id="tool_004",
                name="Error Recovery",
                description="Handle tool errors gracefully",
                category=TaskCategory.TOOL_USE,
                difficulty=TaskDifficulty.HARD,
                input_data={
                    "instruction": "Try to read 'missing.txt', handle the error",
                    "available_tools": ["file_read", "file_exists"],
                    "tool_responses": {"file_read": "ERROR: File not found"},
                },
                expected_output="handled_gracefully",
            ),
        ]


class PlanningBenchmark(Benchmark):
    """Benchmark for planning capabilities.

    Tests:
    - Plan generation
    - Plan execution
    - Plan adaptation
    """

    def __init__(self):
        super().__init__(
            name="Planning Benchmark",
            description="Evaluates agent planning capabilities",
        )

    def load_tasks(self):
        """Load planning tasks."""
        self._tasks = [
            BenchmarkTask(
                task_id="plan_001",
                name="Simple Sequential Plan",
                description="Create a sequential plan",
                category=TaskCategory.PLANNING,
                difficulty=TaskDifficulty.EASY,
                input_data={
                    "goal": "Make a cup of tea",
                    "available_actions": ["boil_water", "get_cup", "add_tea", "pour_water", "wait"],
                },
                max_steps=10,
            ),
            BenchmarkTask(
                task_id="plan_002",
                name="Conditional Plan",
                description="Create a plan with conditions",
                category=TaskCategory.PLANNING,
                difficulty=TaskDifficulty.MEDIUM,
                input_data={
                    "goal": "Go to work",
                    "conditions": {"weather": "unknown", "car_available": "unknown"},
                    "available_actions": ["check_weather", "take_car", "take_bus", "bring_umbrella"],
                },
                max_steps=15,
            ),
            BenchmarkTask(
                task_id="plan_003",
                name="Resource-Constrained Plan",
                description="Plan under resource constraints",
                category=TaskCategory.PLANNING,
                difficulty=TaskDifficulty.HARD,
                input_data={
                    "goal": "Complete project",
                    "subtasks": ["design", "implement", "test", "document"],
                    "resources": {"time": 10, "budget": 100},
                    "task_costs": {"design": 3, "implement": 5, "test": 2, "document": 1},
                },
                max_steps=20,
            ),
            BenchmarkTask(
                task_id="plan_004",
                name="Replanning",
                description="Adapt plan when conditions change",
                category=TaskCategory.PLANNING,
                difficulty=TaskDifficulty.EXPERT,
                input_data={
                    "original_plan": ["step1", "step2", "step3"],
                    "failure_at": "step2",
                    "available_alternatives": ["alt_step2a", "alt_step2b"],
                },
                max_steps=25,
            ),
        ]


class MultiAgentBenchmark(Benchmark):
    """Benchmark for multi-agent capabilities.

    Tests:
    - Agent coordination
    - Communication
    - Conflict resolution
    """

    def __init__(self):
        super().__init__(
            name="Multi-Agent Benchmark",
            description="Evaluates multi-agent coordination",
        )

    def load_tasks(self):
        """Load multi-agent tasks."""
        self._tasks = [
            BenchmarkTask(
                task_id="multi_001",
                name="Simple Delegation",
                description="Delegate task to another agent",
                category=TaskCategory.MULTI_AGENT,
                difficulty=TaskDifficulty.EASY,
                input_data={
                    "task": "Translate document to French",
                    "available_agents": ["translator", "writer", "researcher"],
                },
                expected_output="translator",
            ),
            BenchmarkTask(
                task_id="multi_002",
                name="Parallel Execution",
                description="Coordinate parallel task execution",
                category=TaskCategory.MULTI_AGENT,
                difficulty=TaskDifficulty.MEDIUM,
                input_data={
                    "tasks": ["research_topic", "find_images", "write_outline"],
                    "dependencies": {"write_outline": ["research_topic"]},
                    "available_agents": 3,
                },
            ),
            BenchmarkTask(
                task_id="multi_003",
                name="Conflict Resolution",
                description="Resolve conflicting agent recommendations",
                category=TaskCategory.MULTI_AGENT,
                difficulty=TaskDifficulty.HARD,
                input_data={
                    "agent_recommendations": {
                        "agent_a": "approach_1",
                        "agent_b": "approach_2",
                        "agent_c": "approach_1",
                    },
                    "trust_scores": {"agent_a": 0.8, "agent_b": 0.9, "agent_c": 0.7},
                },
            ),
            BenchmarkTask(
                task_id="multi_004",
                name="Consensus Building",
                description="Build consensus among disagreeing agents",
                category=TaskCategory.MULTI_AGENT,
                difficulty=TaskDifficulty.EXPERT,
                input_data={
                    "positions": {
                        "agent_1": {"choice": "A", "confidence": 0.7},
                        "agent_2": {"choice": "B", "confidence": 0.8},
                        "agent_3": {"choice": "A", "confidence": 0.6},
                    },
                    "required_consensus": 0.8,
                },
            ),
        ]


def create_benchmark_suite() -> List[Benchmark]:
    """Create a full benchmark suite.

    Returns:
        List of all benchmarks
    """
    return [
        ReasoningBenchmark(),
        ToolUseBenchmark(),
        PlanningBenchmark(),
        MultiAgentBenchmark(),
    ]
