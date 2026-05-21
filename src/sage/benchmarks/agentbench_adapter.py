"""
AgentBench Adapter

Adapter for AgentBench OS interaction tasks.

AgentBench is a comprehensive benchmark for evaluating LLM agents across
diverse environments including operating system tasks, database operations,
knowledge graph queries, and more.

Reference: https://github.com/THUDM/AgentBench

This adapter focuses on the OS interaction subset, which evaluates:
- File system operations
- Process management
- System configuration
- Shell command execution

Example:
    >>> adapter = AgentBenchAdapter()
    >>> tasks = adapter.load_tasks(subset="os", n=50)
    >>> for task in tasks:
    ...     result = adapter.evaluate(task, agent.run(task.query))
    ...     print(f"Score: {result.score}")
"""

from __future__ import annotations

import logging
import json
import hashlib
from enum import Enum
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from pathlib import Path

import numpy as np

from .base_adapter import BenchmarkAdapter, BenchmarkTask, BenchmarkResult

logger = logging.getLogger(__name__)


class AgentBenchSubset(Enum):
    """AgentBench task subsets."""
    OS = "os"
    DB = "db"
    KG = "kg"
    ALFWORLD = "alfworld"
    WEBSHOP = "webshop"
    MIND2WEB = "mind2web"


@dataclass
class AgentBenchTask(BenchmarkTask):
    """AgentBench task definition.

    Attributes:
        subset: Task subset (os, db, kg, etc.)
        environment: Environment configuration
        init_state: Initial environment state
        goal: Task goal description
        difficulty: Task difficulty (1-5)
        max_steps: Maximum allowed steps
    """
    subset: AgentBenchSubset = AgentBenchSubset.OS
    environment: Dict[str, Any] = field(default_factory=dict)
    init_state: Dict[str, Any] = field(default_factory=dict)
    goal: str = ""
    difficulty: int = 1
    max_steps: int = 30

    def to_dict(self) -> Dict[str, Any]:
        base = super().to_dict()
        base.update({
            "subset": self.subset.value,
            "environment": self.environment,
            "init_state": self.init_state,
            "goal": self.goal,
            "difficulty": self.difficulty,
            "max_steps": self.max_steps
        })
        return base


@dataclass
class AgentBenchResult(BenchmarkResult):
    """AgentBench evaluation result.

    Attributes:
        steps_taken: Number of steps executed
        commands_executed: List of commands run
        final_state: Final environment state
        goal_achieved: Whether goal was achieved
        partial_credit: Partial credit score
    """
    steps_taken: int = 0
    commands_executed: List[str] = field(default_factory=list)
    final_state: Dict[str, Any] = field(default_factory=dict)
    goal_achieved: bool = False
    partial_credit: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        base = super().to_dict()
        base.update({
            "steps_taken": self.steps_taken,
            "commands_executed": self.commands_executed,
            "final_state": self.final_state,
            "goal_achieved": self.goal_achieved,
            "partial_credit": self.partial_credit
        })
        return base


# Sample OS tasks for demonstration
SAMPLE_OS_TASKS = [
    {
        "id": "os_001",
        "query": "Create a directory called 'project' and inside it create a file 'readme.txt' with the content 'Hello World'",
        "goal": "Directory 'project' exists with file 'readme.txt' containing 'Hello World'",
        "expected_state": {
            "files": {"project/readme.txt": "Hello World"}
        },
        "difficulty": 1
    },
    {
        "id": "os_002",
        "query": "Find all Python files in the current directory and its subdirectories, then count the total number of lines",
        "goal": "Return the total line count of all .py files",
        "expected_output": "line_count",
        "difficulty": 2
    },
    {
        "id": "os_003",
        "query": "Create a backup of all .txt files by copying them to a 'backup' directory with a .bak extension",
        "goal": "Backup directory contains .bak copies of all .txt files",
        "expected_state": {
            "files_pattern": "backup/*.bak"
        },
        "difficulty": 2
    },
    {
        "id": "os_004",
        "query": "Find the process using the most memory and display its name and memory usage",
        "goal": "Display process name and memory information",
        "expected_output": "process_info",
        "difficulty": 3
    },
    {
        "id": "os_005",
        "query": "Create a shell script that monitors disk usage and alerts if any partition exceeds 80%",
        "goal": "Script exists and is executable",
        "expected_state": {
            "files": {"disk_monitor.sh": "executable"}
        },
        "difficulty": 3
    },
    {
        "id": "os_006",
        "query": "Configure a cron job to run a backup script every day at 2 AM",
        "goal": "Cron job is configured correctly",
        "expected_output": "cron_configured",
        "difficulty": 4
    },
    {
        "id": "os_007",
        "query": "Set up a local git repository, create an initial commit, and configure a pre-commit hook that runs tests",
        "goal": "Git repo initialized with pre-commit hook",
        "expected_state": {
            "files": {".git/hooks/pre-commit": "executable"}
        },
        "difficulty": 4
    },
    {
        "id": "os_008",
        "query": "Diagnose why a web server is not responding on port 80 and fix the issue",
        "goal": "Web server is responding on port 80",
        "expected_output": "server_running",
        "difficulty": 5
    },
]


class AgentBenchAdapter(BenchmarkAdapter[AgentBenchTask, AgentBenchResult]):
    """Adapter for AgentBench OS interaction tasks.

    This adapter provides access to AgentBench tasks with focus on
    operating system interactions. It supports:
    - Loading tasks from the benchmark
    - Evaluating agent outputs
    - Computing success metrics

    Example:
        >>> adapter = AgentBenchAdapter()
        >>>
        >>> # Load OS tasks
        >>> tasks = adapter.load_tasks(subset="os", n=50)
        >>>
        >>> # Evaluate agent
        >>> for task in tasks:
        ...     output = agent.run(task.query)
        ...     result = adapter.evaluate(task, output)
        ...     print(f"{task.task_id}: {result.score}")
    """

    def __init__(
        self,
        data_path: Optional[str] = None,
        seed: int = 42,
        use_sample_data: bool = True
    ):
        """Initialize AgentBench adapter.

        Args:
            data_path: Path to AgentBench data
            seed: Random seed
            use_sample_data: Use built-in sample data if real data unavailable
        """
        super().__init__(data_path, seed)
        self.use_sample_data = use_sample_data
        self._rng = np.random.default_rng(seed)

    @property
    def name(self) -> str:
        return "AgentBench"

    @property
    def description(self) -> str:
        return "Benchmark for evaluating LLM agents on OS interaction tasks"

    def load_tasks(
        self,
        subset: str = "os",
        n: Optional[int] = None,
        difficulty: Optional[int] = None,
        **kwargs
    ) -> List[AgentBenchTask]:
        """Load AgentBench tasks.

        Args:
            subset: Task subset ("os", "db", "kg", etc.)
            n: Number of tasks to load
            difficulty: Filter by difficulty level (1-5)

        Returns:
            List of AgentBench tasks
        """
        # Try to load from file
        if self.data_path and Path(self.data_path).exists():
            tasks = self._load_from_file(subset)
        elif self.use_sample_data:
            tasks = self._generate_sample_tasks(subset, n or 50)
        else:
            raise FileNotFoundError(
                f"AgentBench data not found at {self.data_path}. "
                "Set use_sample_data=True to use built-in samples."
            )

        # Filter by difficulty
        if difficulty is not None:
            tasks = [t for t in tasks if t.difficulty == difficulty]

        # Limit number of tasks
        if n is not None and len(tasks) > n:
            indices = self._rng.choice(len(tasks), size=n, replace=False)
            tasks = [tasks[i] for i in sorted(indices)]

        self._tasks = tasks
        self._loaded = True

        logger.info(f"Loaded {len(tasks)} AgentBench tasks (subset={subset})")
        return tasks

    def _load_from_file(self, subset: str) -> List[AgentBenchTask]:
        """Load tasks from file."""
        path = Path(self.data_path) / f"{subset}.json"
        if not path.exists():
            logger.warning(f"File not found: {path}, using sample data")
            return self._generate_sample_tasks(subset, 50)

        with open(path) as f:
            data = json.load(f)

        tasks = []
        for item in data:
            task = AgentBenchTask(
                task_id=item["id"],
                query=item["query"],
                expected_output=item.get("expected_output"),
                metadata=item.get("metadata", {}),
                subset=AgentBenchSubset(subset),
                environment=item.get("environment", {}),
                init_state=item.get("init_state", {}),
                goal=item.get("goal", ""),
                difficulty=item.get("difficulty", 1),
                max_steps=item.get("max_steps", 30)
            )
            tasks.append(task)

        return tasks

    def _generate_sample_tasks(
        self,
        subset: str,
        n: int
    ) -> List[AgentBenchTask]:
        """Generate sample tasks for demonstration."""
        sample_data = SAMPLE_OS_TASKS if subset == "os" else SAMPLE_OS_TASKS

        tasks = []
        for i, item in enumerate(sample_data[:n]):
            task = AgentBenchTask(
                task_id=item["id"],
                query=item["query"],
                expected_output=item.get("expected_output"),
                metadata={"source": "sample"},
                subset=AgentBenchSubset(subset) if subset in [e.value for e in AgentBenchSubset] else AgentBenchSubset.OS,
                goal=item.get("goal", ""),
                difficulty=item.get("difficulty", 1),
                max_steps=30
            )
            tasks.append(task)

        # Pad with generated tasks if needed
        while len(tasks) < n:
            idx = len(tasks)
            task = AgentBenchTask(
                task_id=f"{subset}_{idx:03d}",
                query=f"Sample task {idx} for {subset}",
                metadata={"source": "generated"},
                subset=AgentBenchSubset.OS,
                difficulty=self._rng.integers(1, 5),
                max_steps=30
            )
            tasks.append(task)

        return tasks[:n]

    def evaluate(
        self,
        task: AgentBenchTask,
        output: str,
        execution_trace: Optional[List[Dict]] = None
    ) -> AgentBenchResult:
        """Evaluate agent output for a task.

        Args:
            task: The task
            output: Agent output
            execution_trace: Optional trace of executed commands

        Returns:
            AgentBenchResult with evaluation
        """
        # Parse output to extract commands
        commands = self._extract_commands(output)

        # Check goal achievement
        goal_achieved = self._check_goal_achievement(task, output, commands)

        # Calculate score
        if goal_achieved:
            score = 1.0
        else:
            # Partial credit based on progress
            score = self._calculate_partial_credit(task, output, commands)

        success = score >= 0.5  # Threshold for success

        return AgentBenchResult(
            task_id=task.task_id,
            success=success,
            output=output,
            score=score,
            steps_taken=len(commands),
            commands_executed=commands,
            goal_achieved=goal_achieved,
            partial_credit=score if not goal_achieved else 0.0
        )

    def _extract_commands(self, output: str) -> List[str]:
        """Extract commands from agent output."""
        commands = []

        # Look for code blocks
        import re
        code_blocks = re.findall(r'```(?:bash|sh|shell)?\n(.*?)```', output, re.DOTALL)
        for block in code_blocks:
            for line in block.strip().split('\n'):
                line = line.strip()
                if line and not line.startswith('#'):
                    commands.append(line)

        # Look for inline commands
        inline = re.findall(r'`([^`]+)`', output)
        for cmd in inline:
            if any(kw in cmd for kw in ['cd', 'ls', 'cat', 'echo', 'mkdir', 'cp', 'mv', 'rm']):
                commands.append(cmd)

        return commands

    def _check_goal_achievement(
        self,
        task: AgentBenchTask,
        output: str,
        commands: List[str]
    ) -> bool:
        """Check if goal was achieved."""
        # Simplified goal checking
        if task.expected_output:
            if isinstance(task.expected_output, str):
                return task.expected_output.lower() in output.lower()
            elif isinstance(task.expected_output, dict):
                # Check expected state
                return all(
                    key in output.lower()
                    for key in task.expected_output.keys()
                )

        # Check goal keywords
        if task.goal:
            goal_words = task.goal.lower().split()
            matches = sum(1 for w in goal_words if w in output.lower())
            return matches >= len(goal_words) * 0.5

        return len(commands) > 0

    def _calculate_partial_credit(
        self,
        task: AgentBenchTask,
        output: str,
        commands: List[str]
    ) -> float:
        """Calculate partial credit for incomplete solutions."""
        credit = 0.0

        # Credit for attempting commands
        if commands:
            credit += 0.2

        # Credit for relevant commands
        relevant_keywords = ['mkdir', 'touch', 'echo', 'cat', 'find', 'grep']
        relevant_count = sum(
            1 for cmd in commands
            if any(kw in cmd for kw in relevant_keywords)
        )
        credit += min(0.3, relevant_count * 0.1)

        # Credit for output relevance
        if task.goal:
            goal_words = set(task.goal.lower().split())
            output_words = set(output.lower().split())
            overlap = len(goal_words & output_words) / len(goal_words)
            credit += overlap * 0.3

        return min(1.0, credit)


def download_agentbench_data(
    output_dir: str = "data/agentbench",
    subset: str = "os"
) -> str:
    """Download AgentBench data.

    Note: This function provides instructions for downloading
    AgentBench data. The actual data must be obtained from the
    official repository.

    Args:
        output_dir: Directory to save data
        subset: Subset to download

    Returns:
        Path to downloaded data
    """
    instructions = """
    AgentBench data must be downloaded from the official repository:
    https://github.com/THUDM/AgentBench

    Steps:
    1. Clone the repository: git clone https://github.com/THUDM/AgentBench
    2. Follow the data preparation instructions in the README
    3. Copy the relevant subset data to your local directory

    For OS tasks, look for:
    - data/os_interaction/
    - configs/tasks/os.yaml
    """
    logger.info(instructions)

    # Create output directory
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    return output_dir
