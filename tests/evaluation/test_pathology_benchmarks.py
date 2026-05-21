"""
Tests for pathology benchmarks.

Verifies:
1. All 8 pathologies have benchmark tasks
2. Tasks are deterministic with fixed seeds
3. Task coverage matches paper Table IX
"""

import pytest

from sage.evaluation.failure_taxonomy import FailurePathology
from sage.evaluation.pathology_benchmarks import (
    ALL_PATHOLOGY_TASKS,
    TASKS_BY_PATHOLOGY,
    PathologyBenchmarkTask,
    get_benchmark_statistics
)


class TestPathologyCoverage:
    """Tests for ensuring all pathologies are covered."""

    def test_all_pathologies_have_tasks(self):
        """Every pathology should have at least one benchmark task."""
        for pathology in FailurePathology:
            tasks = TASKS_BY_PATHOLOGY.get(pathology, [])
            assert len(tasks) >= 1, f"Pathology {pathology.value} has no benchmark tasks"

    def test_minimum_tasks_per_pathology(self):
        """Each pathology should have at least 2 tasks for robustness."""
        # Multi-agent pathologies may have fewer as they're harder to test
        single_agent_pathologies = [
            FailurePathology.HALLUCINATED_AFFORDANCE,
            FailurePathology.STATE_MISESTIMATION,
            FailurePathology.CASCADING_TOOL_FAILURE,
            FailurePathology.ACTION_OBSERVATION_MISMATCH,
            FailurePathology.MEMORY_POISONING,
            FailurePathology.FEEDBACK_AMPLIFICATION,
            FailurePathology.GOAL_DRIFT,
            FailurePathology.PLANNING_MYOPIA,
        ]

        for pathology in single_agent_pathologies:
            tasks = TASKS_BY_PATHOLOGY.get(pathology, [])
            assert len(tasks) >= 2, f"Pathology {pathology.value} needs at least 2 tasks, has {len(tasks)}"

    def test_total_task_count(self):
        """Should have at least 20 total tasks."""
        assert len(ALL_PATHOLOGY_TASKS) >= 20, f"Expected 20+ tasks, got {len(ALL_PATHOLOGY_TASKS)}"


class TestTaskDeterminism:
    """Tests for task determinism/reproducibility."""

    def test_all_tasks_have_seeds(self):
        """Every task should have a fixed seed."""
        for task in ALL_PATHOLOGY_TASKS:
            assert task.seed is not None, f"Task {task.task_id} missing seed"
            assert isinstance(task.seed, int), f"Task {task.task_id} seed must be int"

    def test_seeds_are_unique(self):
        """Each task should have a unique seed to avoid correlation."""
        seeds = [task.seed for task in ALL_PATHOLOGY_TASKS]
        assert len(seeds) == len(set(seeds)), "Duplicate seeds found"

    def test_tasks_have_unique_ids(self):
        """Each task should have a unique ID."""
        ids = [task.task_id for task in ALL_PATHOLOGY_TASKS]
        assert len(ids) == len(set(ids)), "Duplicate task IDs found"


class TestTaskStructure:
    """Tests for task structure completeness."""

    def test_tasks_have_required_fields(self):
        """Every task should have all required fields populated."""
        required_fields = [
            "task_id",
            "pathology",
            "name",
            "description",
            "setup",
            "expected_behavior",
            "pathology_trigger"
        ]

        for task in ALL_PATHOLOGY_TASKS:
            for field in required_fields:
                value = getattr(task, field, None)
                assert value is not None, f"Task {task.task_id} missing {field}"
                if isinstance(value, str):
                    assert len(value) > 0, f"Task {task.task_id} has empty {field}"

    def test_difficulty_values_valid(self):
        """Difficulty should be one of: easy, medium, hard."""
        valid_difficulties = {"easy", "medium", "hard"}

        for task in ALL_PATHOLOGY_TASKS:
            assert task.difficulty in valid_difficulties, \
                f"Task {task.task_id} has invalid difficulty: {task.difficulty}"

    def test_difficulty_distribution(self):
        """Should have mix of difficulties."""
        difficulties = [task.difficulty for task in ALL_PATHOLOGY_TASKS]

        assert "easy" in difficulties, "No easy tasks"
        assert "medium" in difficulties, "No medium tasks"
        assert "hard" in difficulties, "No hard tasks"


class TestPaperAlignment:
    """Tests for alignment with paper Section XV."""

    def test_pathology_categories_match_paper(self):
        """
        Paper Section XV defines these categories:
        - XV-A: Perception and Grounding (hallucinated_affordance, state_misestimation)
        - XV-B: Execution (cascading_tool_failure, action_observation_mismatch)
        - XV-C: Memory and Learning (memory_poisoning, feedback_amplification)
        - XV-D: Goal and Planning (goal_drift, planning_myopia)
        - XV-E: Multi-Agent (emergent_collusion, consensus_deadlock)
        """
        # XV-A: Perception
        assert FailurePathology.HALLUCINATED_AFFORDANCE in TASKS_BY_PATHOLOGY
        assert FailurePathology.STATE_MISESTIMATION in TASKS_BY_PATHOLOGY

        # XV-B: Execution
        assert FailurePathology.CASCADING_TOOL_FAILURE in TASKS_BY_PATHOLOGY
        assert FailurePathology.ACTION_OBSERVATION_MISMATCH in TASKS_BY_PATHOLOGY

        # XV-C: Memory
        assert FailurePathology.MEMORY_POISONING in TASKS_BY_PATHOLOGY
        assert FailurePathology.FEEDBACK_AMPLIFICATION in TASKS_BY_PATHOLOGY

        # XV-D: Planning
        assert FailurePathology.GOAL_DRIFT in TASKS_BY_PATHOLOGY
        assert FailurePathology.PLANNING_MYOPIA in TASKS_BY_PATHOLOGY

        # XV-E: Multi-Agent
        assert FailurePathology.EMERGENT_COLLUSION in TASKS_BY_PATHOLOGY
        assert FailurePathology.CONSENSUS_DEADLOCK in TASKS_BY_PATHOLOGY

    def test_evaluation_methods_per_table_ix(self):
        """
        Paper Table IX specifies evaluation methods for each pathology.
        Verify tasks align with these methods.
        """
        # Table IX: Hallucinated Affordance -> Schema validation tests
        ha_tasks = TASKS_BY_PATHOLOGY[FailurePathology.HALLUCINATED_AFFORDANCE]
        assert any("schema" in task.description.lower() or
                  "schema" in task.pathology_trigger.lower()
                  for task in ha_tasks), \
            "Hallucinated affordance tasks should test schema validation"

        # Table IX: Cascading Tool Failure -> Fault injection testing
        ctf_tasks = TASKS_BY_PATHOLOGY[FailurePathology.CASCADING_TOOL_FAILURE]
        assert any("error" in task.setup.get("injected_failure", {}).get("error", "").lower() or
                  "failure" in task.name.lower()
                  for task in ctf_tasks), \
            "Cascading failure tasks should include fault injection"

        # Table IX: Goal Drift -> Goal drift score tracking
        gd_tasks = TASKS_BY_PATHOLOGY[FailurePathology.GOAL_DRIFT]
        assert any("goal" in task.setup.get("original_goal", "").lower()
                  for task in gd_tasks), \
            "Goal drift tasks should have explicit original goals"


class TestBenchmarkStatistics:
    """Tests for benchmark statistics function."""

    def test_statistics_structure(self):
        """Statistics should have expected structure."""
        stats = get_benchmark_statistics()

        assert "total_tasks" in stats
        assert "pathologies_covered" in stats
        assert "tasks_per_pathology" in stats
        assert "difficulty_distribution" in stats

    def test_statistics_accuracy(self):
        """Statistics should match actual data."""
        stats = get_benchmark_statistics()

        assert stats["total_tasks"] == len(ALL_PATHOLOGY_TASKS)
        assert stats["pathologies_covered"] == len(TASKS_BY_PATHOLOGY)

        total_from_pathologies = sum(stats["tasks_per_pathology"].values())
        assert total_from_pathologies == stats["total_tasks"]


class TestTaskSerialization:
    """Tests for task serialization."""

    def test_to_dict_roundtrip(self):
        """Tasks should serialize to dict correctly."""
        for task in ALL_PATHOLOGY_TASKS[:5]:  # Test first 5
            d = task.to_dict()

            assert d["task_id"] == task.task_id
            assert d["pathology"] == task.pathology.value
            assert d["name"] == task.name
            assert d["seed"] == task.seed
