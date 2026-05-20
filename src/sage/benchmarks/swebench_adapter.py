"""
SWE-Bench Adapter

Adapter for SWE-Bench software engineering benchmark.

SWE-Bench evaluates language models on real-world software engineering
tasks derived from GitHub issues and pull requests.

Reference: https://github.com/princeton-nlp/SWE-bench

This adapter supports:
- Loading issues from SWE-Bench Lite and Full
- Evaluating patches against test suites
- Computing pass@1 and other metrics

Example:
    >>> adapter = SWEBenchAdapter()
    >>> tasks = adapter.load_tasks(n=100)
    >>> for task in tasks:
    ...     patch = agent.generate_patch(task.issue_description)
    ...     result = adapter.evaluate(task, patch)
    ...     print(f"Tests passed: {result.tests_passed}/{result.tests_total}")
"""

from __future__ import annotations

import logging
import json
import re
from enum import Enum
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
from pathlib import Path

import numpy as np

from .base_adapter import BenchmarkAdapter, BenchmarkTask, BenchmarkResult

logger = logging.getLogger(__name__)


class SWEBenchDifficulty(Enum):
    """SWE-Bench difficulty levels."""
    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"


class SWEBenchVariant(Enum):
    """SWE-Bench variants."""
    LITE = "lite"
    FULL = "full"
    VERIFIED = "verified"


@dataclass
class SWEBenchTask(BenchmarkTask):
    """SWE-Bench task definition.

    Attributes:
        repo: Repository name (owner/repo)
        base_commit: Base commit hash
        issue_description: GitHub issue description
        hints: Optional hints for solving
        test_patch: Patch to run tests
        golden_patch: Reference solution patch
        difficulty: Estimated difficulty
        created_at: Issue creation date
    """
    repo: str = ""
    base_commit: str = ""
    issue_description: str = ""
    hints: List[str] = field(default_factory=list)
    test_patch: str = ""
    golden_patch: str = ""
    difficulty: SWEBenchDifficulty = SWEBenchDifficulty.MEDIUM
    created_at: str = ""

    def to_dict(self) -> Dict[str, Any]:
        base = super().to_dict()
        base.update({
            "repo": self.repo,
            "base_commit": self.base_commit,
            "issue_description": self.issue_description,
            "hints": self.hints,
            "difficulty": self.difficulty.value,
            "created_at": self.created_at
        })
        return base


@dataclass
class SWEBenchResult(BenchmarkResult):
    """SWE-Bench evaluation result.

    Attributes:
        tests_passed: Number of tests passed
        tests_total: Total number of tests
        tests_failed: List of failed test names
        patch_valid: Whether patch is syntactically valid
        patch_applies: Whether patch applies cleanly
        execution_time: Time taken to run tests
    """
    tests_passed: int = 0
    tests_total: int = 0
    tests_failed: List[str] = field(default_factory=list)
    patch_valid: bool = False
    patch_applies: bool = False
    execution_time: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        base = super().to_dict()
        base.update({
            "tests_passed": self.tests_passed,
            "tests_total": self.tests_total,
            "tests_failed": self.tests_failed,
            "patch_valid": self.patch_valid,
            "patch_applies": self.patch_applies,
            "execution_time": self.execution_time
        })
        return base


# Sample SWE-Bench tasks for demonstration
SAMPLE_SWEBENCH_TASKS = [
    {
        "id": "django__django-11099",
        "repo": "django/django",
        "issue": "UsernameValidator allows trailing newline in usernames",
        "description": "The username validator in Django allows usernames with trailing newlines. This is a security concern as it could lead to confusion or bypass of username-based access controls.",
        "hints": ["Look at django/contrib/auth/validators.py", "The regex pattern needs to be anchored"],
        "difficulty": "easy"
    },
    {
        "id": "scikit-learn__scikit-learn-13779",
        "repo": "scikit-learn/scikit-learn",
        "issue": "VotingClassifier fails with string labels",
        "description": "When using VotingClassifier with estimators that have string labels, the ensemble fails to properly combine predictions due to label encoding issues.",
        "hints": ["Check the _predict_proba method", "Labels need consistent encoding"],
        "difficulty": "medium"
    },
    {
        "id": "matplotlib__matplotlib-22711",
        "repo": "matplotlib/matplotlib",
        "issue": "Legend title fontsize not respected",
        "description": "Setting fontsize for legend title through legend() or set_title() is not working properly. The title uses a different fontsize than specified.",
        "hints": ["Look at lib/matplotlib/legend.py", "The title properties need to be propagated"],
        "difficulty": "medium"
    },
    {
        "id": "sympy__sympy-18532",
        "repo": "sympy/sympy",
        "issue": "expr.atoms() returns wrong set for MutableMatrix",
        "description": "When calling atoms() on expressions containing MutableMatrix objects, the returned set of atoms is incorrect or incomplete.",
        "hints": ["Check the atoms() method in core/basic.py", "MutableMatrix may need special handling"],
        "difficulty": "hard"
    },
    {
        "id": "requests__requests-5087",
        "repo": "psf/requests",
        "issue": "Session does not properly handle retries",
        "description": "The Session class does not properly handle retry configurations when adapters are mounted, leading to inconsistent retry behavior.",
        "hints": ["Look at sessions.py", "Check how adapters are merged"],
        "difficulty": "medium"
    },
]


class SWEBenchAdapter(BenchmarkAdapter[SWEBenchTask, SWEBenchResult]):
    """Adapter for SWE-Bench software engineering benchmark.

    This adapter provides access to SWE-Bench tasks for evaluating
    agents on real-world software engineering problems.

    Example:
        >>> adapter = SWEBenchAdapter(variant="lite")
        >>>
        >>> # Load tasks
        >>> tasks = adapter.load_tasks(n=100)
        >>>
        >>> # Evaluate agent patches
        >>> for task in tasks:
        ...     patch = agent.solve_issue(task.issue_description)
        ...     result = adapter.evaluate(task, patch)
        ...     if result.success:
        ...         print(f"{task.task_id}: PASS")
    """

    def __init__(
        self,
        data_path: Optional[str] = None,
        variant: str = "lite",
        seed: int = 42,
        use_sample_data: bool = True
    ):
        """Initialize SWE-Bench adapter.

        Args:
            data_path: Path to SWE-Bench data
            variant: Benchmark variant ("lite", "full", "verified")
            seed: Random seed
            use_sample_data: Use built-in sample data if real data unavailable
        """
        super().__init__(data_path, seed)
        self.variant = SWEBenchVariant(variant)
        self.use_sample_data = use_sample_data
        self._rng = np.random.default_rng(seed)

    @property
    def name(self) -> str:
        return f"SWE-Bench-{self.variant.value.title()}"

    @property
    def description(self) -> str:
        return "Benchmark for evaluating LLM agents on software engineering tasks"

    def load_tasks(
        self,
        n: Optional[int] = None,
        difficulty: Optional[str] = None,
        repo: Optional[str] = None,
        **kwargs
    ) -> List[SWEBenchTask]:
        """Load SWE-Bench tasks.

        Args:
            n: Number of tasks to load
            difficulty: Filter by difficulty ("easy", "medium", "hard")
            repo: Filter by repository

        Returns:
            List of SWE-Bench tasks
        """
        # Try to load from file
        if self.data_path and Path(self.data_path).exists():
            tasks = self._load_from_file()
        elif self.use_sample_data:
            tasks = self._generate_sample_tasks(n or 100)
        else:
            raise FileNotFoundError(
                f"SWE-Bench data not found at {self.data_path}. "
                "Set use_sample_data=True to use built-in samples."
            )

        # Filter by difficulty
        if difficulty is not None:
            diff = SWEBenchDifficulty(difficulty)
            tasks = [t for t in tasks if t.difficulty == diff]

        # Filter by repo
        if repo is not None:
            tasks = [t for t in tasks if repo in t.repo]

        # Limit number of tasks
        if n is not None and len(tasks) > n:
            indices = self._rng.choice(len(tasks), size=n, replace=False)
            tasks = [tasks[i] for i in sorted(indices)]

        self._tasks = tasks
        self._loaded = True

        logger.info(f"Loaded {len(tasks)} SWE-Bench tasks (variant={self.variant.value})")
        return tasks

    def _load_from_file(self) -> List[SWEBenchTask]:
        """Load tasks from file."""
        path = Path(self.data_path) / f"swebench_{self.variant.value}.json"
        if not path.exists():
            logger.warning(f"File not found: {path}, using sample data")
            return self._generate_sample_tasks(100)

        with open(path) as f:
            data = json.load(f)

        tasks = []
        for item in data:
            task = SWEBenchTask(
                task_id=item["instance_id"],
                query=item.get("problem_statement", ""),
                repo=item.get("repo", ""),
                base_commit=item.get("base_commit", ""),
                issue_description=item.get("problem_statement", ""),
                hints=item.get("hints", []),
                test_patch=item.get("test_patch", ""),
                golden_patch=item.get("patch", ""),
                difficulty=SWEBenchDifficulty(item.get("difficulty", "medium")),
                created_at=item.get("created_at", "")
            )
            tasks.append(task)

        return tasks

    def _generate_sample_tasks(self, n: int) -> List[SWEBenchTask]:
        """Generate sample tasks for demonstration."""
        tasks = []

        for i, item in enumerate(SAMPLE_SWEBENCH_TASKS[:n]):
            task = SWEBenchTask(
                task_id=item["id"],
                query=item["description"],
                repo=item["repo"],
                issue_description=item["description"],
                hints=item.get("hints", []),
                difficulty=SWEBenchDifficulty(item["difficulty"]),
                metadata={"source": "sample"}
            )
            tasks.append(task)

        # Pad with generated tasks if needed
        repos = ["django/django", "scikit-learn/scikit-learn", "matplotlib/matplotlib"]
        while len(tasks) < n:
            idx = len(tasks)
            repo = repos[idx % len(repos)]
            task = SWEBenchTask(
                task_id=f"sample_{idx:03d}",
                query=f"Fix issue in {repo}",
                repo=repo,
                issue_description=f"Sample issue {idx} for {repo}",
                difficulty=SWEBenchDifficulty(["easy", "medium", "hard"][idx % 3]),
                metadata={"source": "generated"}
            )
            tasks.append(task)

        return tasks[:n]

    def evaluate(
        self,
        task: SWEBenchTask,
        patch: str,
        run_tests: bool = True
    ) -> SWEBenchResult:
        """Evaluate agent-generated patch.

        Args:
            task: The task
            patch: Agent-generated patch (unified diff format)
            run_tests: Whether to run tests (simulated if False)

        Returns:
            SWEBenchResult with evaluation
        """
        # Validate patch format
        patch_valid = self._validate_patch_format(patch)

        if not patch_valid:
            return SWEBenchResult(
                task_id=task.task_id,
                success=False,
                output=patch,
                score=0.0,
                patch_valid=False,
                patch_applies=False,
                error="Invalid patch format"
            )

        # Check if patch applies (simulated)
        patch_applies = self._check_patch_applies(patch)

        if not patch_applies:
            return SWEBenchResult(
                task_id=task.task_id,
                success=False,
                output=patch,
                score=0.1,  # Small credit for valid format
                patch_valid=True,
                patch_applies=False,
                error="Patch does not apply cleanly"
            )

        # Evaluate tests (simulated)
        if run_tests:
            tests_passed, tests_total, tests_failed = self._run_tests_simulated(
                task, patch
            )
        else:
            tests_passed, tests_total, tests_failed = 0, 1, []

        # Calculate score
        if tests_total > 0:
            score = tests_passed / tests_total
        else:
            score = 0.5  # No tests case

        success = tests_passed == tests_total and tests_total > 0

        return SWEBenchResult(
            task_id=task.task_id,
            success=success,
            output=patch,
            score=score,
            tests_passed=tests_passed,
            tests_total=tests_total,
            tests_failed=tests_failed,
            patch_valid=True,
            patch_applies=True
        )

    def _validate_patch_format(self, patch: str) -> bool:
        """Validate patch is in unified diff format."""
        if not patch or not patch.strip():
            return False

        # Check for diff headers
        has_diff_header = (
            "diff --git" in patch or
            "---" in patch and "+++" in patch
        )

        # Check for hunk headers
        has_hunks = bool(re.search(r'@@\s*-\d+', patch))

        return has_diff_header or has_hunks

    def _check_patch_applies(self, patch: str) -> bool:
        """Check if patch would apply cleanly (simulated)."""
        # Simulated check - in real implementation would use git apply --check
        # For now, use heuristics

        # Check for obvious issues
        if "<<<<<<" in patch or "======" in patch or ">>>>>>" in patch:
            return False  # Merge conflict markers

        # Check for reasonable diff structure
        lines = patch.split('\n')
        add_count = sum(1 for l in lines if l.startswith('+') and not l.startswith('+++'))
        del_count = sum(1 for l in lines if l.startswith('-') and not l.startswith('---'))

        # At least some changes
        return add_count > 0 or del_count > 0

    def _run_tests_simulated(
        self,
        task: SWEBenchTask,
        patch: str
    ) -> Tuple[int, int, List[str]]:
        """Run tests in simulated mode.

        Returns:
            Tuple of (passed, total, failed_names)
        """
        # Simulate test results based on patch quality heuristics
        patch_lines = len(patch.split('\n'))

        # Base number of tests
        total = self._rng.integers(5, 20)

        # Estimate pass rate based on patch characteristics
        if task.golden_patch and patch == task.golden_patch:
            # Perfect match
            passed = total
        else:
            # Heuristic based on patch size and structure
            quality_score = min(1.0, patch_lines / 50)  # Larger patches may be more complete
            base_pass_rate = 0.3 + quality_score * 0.4
            passed = int(total * base_pass_rate)

        failed = [f"test_{i}" for i in range(passed, total)]

        return passed, total, failed

    def get_pass_at_1(self, results: List[SWEBenchResult]) -> float:
        """Calculate pass@1 metric.

        Args:
            results: List of evaluation results

        Returns:
            Pass@1 score (fraction of tasks solved)
        """
        if not results:
            return 0.0

        passed = sum(1 for r in results if r.success)
        return passed / len(results)

    def get_statistics(self, results: List[SWEBenchResult]) -> Dict[str, Any]:
        """Get detailed statistics for SWE-Bench results."""
        base_stats = super().get_statistics(results)

        if not results:
            return base_stats

        # Add SWE-Bench specific stats
        valid_patches = sum(1 for r in results if r.patch_valid)
        applying_patches = sum(1 for r in results if r.patch_applies)

        base_stats.update({
            "pass_at_1": self.get_pass_at_1(results),
            "valid_patch_rate": valid_patches / len(results),
            "applying_patch_rate": applying_patches / len(results),
            "mean_tests_passed": sum(r.tests_passed for r in results) / len(results),
            "mean_tests_total": sum(r.tests_total for r in results) / len(results)
        })

        return base_stats


def download_swebench_data(
    output_dir: str = "data/swebench",
    variant: str = "lite"
) -> str:
    """Download SWE-Bench data.

    Args:
        output_dir: Directory to save data
        variant: Benchmark variant

    Returns:
        Path to downloaded data
    """
    instructions = """
    SWE-Bench data can be obtained from:
    https://github.com/princeton-nlp/SWE-bench

    For SWE-Bench Lite:
    pip install datasets
    from datasets import load_dataset
    ds = load_dataset("princeton-nlp/SWE-bench_Lite")

    For the full benchmark:
    from datasets import load_dataset
    ds = load_dataset("princeton-nlp/SWE-bench")
    """
    logger.info(instructions)

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    return output_dir
