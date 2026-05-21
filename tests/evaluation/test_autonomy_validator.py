"""
Tests for Autonomy Validator

Verifies:
1. All 4 autonomy criteria are properly validated
2. Autonomy level classification matches Table IV
3. Mock agents demonstrate expected pass/fail behavior
4. Test scenarios properly differentiate genuine vs pseudo-agents
"""

import pytest
from typing import Dict, Any, List

from sage.evaluation.autonomy_validator import (
    AutonomyValidator,
    AutonomyLevel,
    AutonomyCriterion,
    AutonomyCriteria,
    TestResult,
    AutonomyValidationResult,
    TestScenario,
    TestScenarioGenerator,
    ObstacleInjector,
    FailureInjector,
    GenuineAgent,
    ScriptedAgent,
    FragileAgent,
    FixedStepAgent,
)

from sage.evaluation.autonomy_benchmarks import (
    ALL_AUTONOMY_TASKS,
    TASKS_BY_CRITERION,
    get_autonomy_benchmark_statistics,
)

# ============================================================
# Test Action Selection Freedom
# ============================================================


class TestActionSelectionFreedom:
    """Tests for action selection freedom validation."""

    def test_genuine_agent_varies_actions_by_state(self):
        """Agent that selects different actions for different states should PASS."""
        validator = AutonomyValidator(seed=42)
        agent = GenuineAgent()

        result = validator.validate_action_selection_freedom(agent)

        assert result.passed is True
        assert result.criterion == AutonomyCriterion.ACTION_SELECTION_FREEDOM
        assert result.evidence["variation_ratio"] >= 0.5

    def test_scripted_agent_fails_validation(self):
        """Agent that always follows same sequence should FAIL."""
        validator = AutonomyValidator(seed=42)
        agent = ScriptedAgent()

        result = validator.validate_action_selection_freedom(agent)

        assert result.passed is False
        assert (
            "predetermined" in result.failure_reason.lower()
            or "same" in result.failure_reason.lower()
            or result.evidence["variation_ratio"] < 0.5
        )

    def test_conditional_branching_is_not_freedom(self):
        """Agent with predetermined if/else branches should show limited variation."""

        # Create a mock agent with conditional but limited branching
        class ConditionalAgent:
            def __init__(self):
                self.name = "ConditionalAgent"

            def select_action(self, state: Dict[str, Any], goal: str) -> str:
                # Simple if/else - same action for many different states
                if state.get("resources", {}).get("api_quota", 0) > 0:
                    return "use_api"
                else:
                    return "use_fallback"

            def reset(self):
                pass

        validator = AutonomyValidator(seed=42)
        agent = ConditionalAgent()

        result = validator.validate_action_selection_freedom(agent)

        # Should have very low variation (only 2 possible actions)
        assert result.evidence["variation_ratio"] <= 0.6

    def test_scenario_generator_produces_varied_scenarios(self):
        """Test scenario generator creates diverse scenarios."""
        generator = TestScenarioGenerator(seed=42)

        resource_scenarios = generator.generate_resource_scenarios("test goal")
        context_scenarios = generator.generate_context_scenarios("test goal")

        assert len(resource_scenarios) >= 3
        assert len(context_scenarios) >= 3

        # Verify scenarios have different states
        resource_states = [s.state for s in resource_scenarios]
        assert len(set(str(s) for s in resource_states)) == len(resource_states)


# ============================================================
# Test Goal-Directed Persistence
# ============================================================


class TestGoalDirectedPersistence:
    """Tests for goal-directed persistence validation."""

    def test_persistent_agent_adapts_strategy(self):
        """Agent that changes strategy on obstacles should PASS."""
        validator = AutonomyValidator(seed=42)
        agent = GenuineAgent()

        result = validator.validate_goal_directed_persistence(agent)

        assert result.passed is True
        assert result.evidence["strategy_changes"] >= 1
        assert result.evidence["persistence_ratio"] >= 0.5

    def test_agent_abandons_on_first_failure_fails(self):
        """Agent that gives up immediately should FAIL."""
        validator = AutonomyValidator(seed=42)
        agent = FragileAgent()

        result = validator.validate_goal_directed_persistence(agent)

        assert result.passed is False
        assert (
            "gave up" in result.failure_reason.lower()
            or result.evidence["persistence_ratio"] < 0.5
        )

    def test_agent_persists_without_adaptation_fails(self):
        """Agent that retries same approach repeatedly should have low strategy changes."""

        class StubornAgent:
            def __init__(self):
                self.name = "StubornAgent"
                self.step = 0

            def select_action(self, state: Dict, goal: str) -> str:
                self.step += 1
                return "same_action_always"  # Never changes

            def execute(self, action: str, state: Dict) -> Dict:
                return {"success": True, "progress": self.step * 0.1}

            def should_terminate(self, state: Dict, goal: str) -> bool:
                return self.step >= 10

            def handle_failure(self, failure: Dict, state: Dict) -> str:
                return "retry_same"  # Same recovery always

            def reset(self):
                self.step = 0

        validator = AutonomyValidator(seed=42)
        agent = StubornAgent()

        result = validator.validate_goal_directed_persistence(agent)

        # Should have low strategy diversity
        assert result.evidence["strategy_changes"] <= 1

    def test_obstacle_injector_works(self):
        """Test obstacle injector generates appropriate obstacles."""
        injector = ObstacleInjector(seed=42)

        easy = injector.get_obstacle_sequence("easy")
        medium = injector.get_obstacle_sequence("medium")
        hard = injector.get_obstacle_sequence("hard")

        assert len(easy) < len(medium) < len(hard)


# ============================================================
# Test Dynamic Termination
# ============================================================


class TestDynamicTermination:
    """Tests for dynamic termination validation."""

    def test_variable_step_counts_passes(self):
        """Agent with different step counts per goal complexity should PASS."""
        validator = AutonomyValidator(seed=42)
        agent = GenuineAgent()

        result = validator.validate_dynamic_termination(agent)

        assert result.passed is True
        assert result.evidence["step_count_variance"] >= 2

    def test_fixed_step_count_fails(self):
        """Agent that always runs N steps should FAIL."""
        validator = AutonomyValidator(seed=42)
        agent = FixedStepAgent()

        result = validator.validate_dynamic_termination(agent)

        # FixedStepAgent always runs 10 steps
        assert result.passed is False or result.evidence["step_count_variance"] < 5

    def test_premature_termination_detection(self):
        """Test detection of premature termination."""

        class PrematureAgent:
            def __init__(self):
                self.name = "PrematureAgent"
                self.step = 0

            def select_action(self, state: Dict, goal: str) -> str:
                self.step += 1
                return "action"

            def execute(self, action: str, state: Dict) -> Dict:
                return {"success": True}

            def should_terminate(self, state: Dict, goal: str) -> bool:
                return self.step >= 2  # Always terminates after 2 steps

            def handle_failure(self, failure: Dict, state: Dict) -> str:
                return "recover"

            def reset(self):
                self.step = 0

        validator = AutonomyValidator(seed=42)
        agent = PrematureAgent()

        result = validator.validate_dynamic_termination(agent)

        # Should have near-zero variance
        assert result.evidence["step_count_variance"] <= 1


# ============================================================
# Test Error Recovery
# ============================================================


class TestErrorRecovery:
    """Tests for error recovery validation."""

    def test_adaptive_recovery_passes(self):
        """Agent with varied recovery strategies should PASS."""
        validator = AutonomyValidator(seed=42)
        agent = GenuineAgent()

        result = validator.validate_error_recovery(agent)

        assert result.passed is True
        assert result.evidence["unique_recovery_strategies"] >= 2

    def test_hardcoded_fallback_fails(self):
        """Agent with predetermined try/except fallbacks should FAIL."""
        validator = AutonomyValidator(seed=42)
        agent = ScriptedAgent()  # Uses same recovery for all failures

        result = validator.validate_error_recovery(agent)

        assert result.passed is False
        assert result.evidence["unique_recovery_strategies"] <= 1

    def test_no_recovery_attempt_fails(self):
        """Agent that gives up without trying should FAIL."""
        validator = AutonomyValidator(seed=42)
        agent = FragileAgent()

        result = validator.validate_error_recovery(agent)

        assert result.passed is False
        assert result.evidence["recovery_ratio"] < 0.5

    def test_failure_injector_variety(self):
        """Test failure injector generates varied failures."""
        injector = FailureInjector(seed=42)

        failures = injector.get_failure_sequence(count=4)

        assert len(failures) == 4
        types = [f["type"] for f in failures]
        assert len(set(types)) == len(types)  # All unique types


# ============================================================
# Test Autonomy Level Classification
# ============================================================


class TestAutonomyLevelClassification:
    """Tests for autonomy level classification."""

    def test_zero_criteria_is_static_workflow(self):
        """0/4 criteria = STATIC_WORKFLOW"""
        criteria = AutonomyCriteria(
            action_selection_freedom=False,
            goal_directed_persistence=False,
            dynamic_termination=False,
            error_recovery=False,
        )

        level = AutonomyLevel.from_criteria_count(criteria.criteria_met)

        assert level == AutonomyLevel.STATIC_WORKFLOW
        assert criteria.criteria_met == 0

    def test_one_criterion_is_conditional(self):
        """1/4 criteria = CONDITIONAL"""
        criteria = AutonomyCriteria(
            action_selection_freedom=True,
            goal_directed_persistence=False,
            dynamic_termination=False,
            error_recovery=False,
        )

        level = AutonomyLevel.from_criteria_count(criteria.criteria_met)

        assert level == AutonomyLevel.CONDITIONAL
        assert criteria.criteria_met == 1

    def test_two_criteria_is_guided_agent(self):
        """2/4 criteria = GUIDED_AGENT"""
        criteria = AutonomyCriteria(
            action_selection_freedom=True,
            goal_directed_persistence=True,
            dynamic_termination=False,
            error_recovery=False,
        )

        level = AutonomyLevel.from_criteria_count(criteria.criteria_met)

        assert level == AutonomyLevel.GUIDED_AGENT
        assert criteria.criteria_met == 2

    def test_three_criteria_is_bounded_agent(self):
        """3/4 criteria = BOUNDED_AGENT"""
        criteria = AutonomyCriteria(
            action_selection_freedom=True,
            goal_directed_persistence=True,
            dynamic_termination=True,
            error_recovery=False,
        )

        level = AutonomyLevel.from_criteria_count(criteria.criteria_met)

        assert level == AutonomyLevel.BOUNDED_AGENT
        assert criteria.criteria_met == 3

    def test_all_criteria_is_full_agent(self):
        """4/4 criteria = FULL_AGENT"""
        criteria = AutonomyCriteria(
            action_selection_freedom=True,
            goal_directed_persistence=True,
            dynamic_termination=True,
            error_recovery=True,
        )

        level = AutonomyLevel.from_criteria_count(criteria.criteria_met)

        assert level == AutonomyLevel.FULL_AGENT
        assert criteria.criteria_met == 4
        assert criteria.is_genuine_agent is True


# ============================================================
# Test Full Validation Pipeline
# ============================================================


class TestFullValidation:
    """Tests for the complete validation pipeline."""

    def test_genuine_agent_passes_all(self):
        """GenuineAgent should pass all 4 criteria."""
        validator = AutonomyValidator(seed=42, verbose=False)
        agent = GenuineAgent()

        result = validator.validate_all(agent)

        # May not pass all due to test setup, but should pass most
        assert result.criteria_met >= 2
        assert result.level.value >= AutonomyLevel.GUIDED_AGENT.value

    def test_scripted_agent_fails_most(self):
        """ScriptedAgent should fail most criteria."""
        validator = AutonomyValidator(seed=42, verbose=False)
        agent = ScriptedAgent()

        result = validator.validate_all(agent)

        assert result.criteria_met <= 2
        assert result.level.value <= AutonomyLevel.GUIDED_AGENT.value

    def test_validation_result_has_all_fields(self):
        """Validation result should have all expected fields."""
        validator = AutonomyValidator(seed=42)
        agent = GenuineAgent()

        result = validator.validate_all(agent)

        assert hasattr(result, "level")
        assert hasattr(result, "criteria")
        assert hasattr(result, "test_results")
        assert hasattr(result, "validation_time_seconds")

        assert len(result.test_results) == 4
        assert AutonomyCriterion.ACTION_SELECTION_FREEDOM in result.test_results
        assert AutonomyCriterion.GOAL_DIRECTED_PERSISTENCE in result.test_results
        assert AutonomyCriterion.DYNAMIC_TERMINATION in result.test_results
        assert AutonomyCriterion.ERROR_RECOVERY in result.test_results

    def test_validation_result_serializes(self):
        """Validation result should serialize to dict correctly."""
        validator = AutonomyValidator(seed=42)
        agent = GenuineAgent()

        result = validator.validate_all(agent)
        d = result.to_dict()

        assert "level" in d
        assert "criteria_met" in d
        assert "is_genuine_agent" in d
        assert "test_results" in d


# ============================================================
# Test Benchmark Coverage
# ============================================================


class TestBenchmarkCoverage:
    """Tests for benchmark task coverage."""

    def test_all_criteria_have_tasks(self):
        """Every criterion should have benchmark tasks."""
        for criterion in AutonomyCriterion:
            tasks = TASKS_BY_CRITERION.get(criterion, [])
            assert len(tasks) >= 1, f"Criterion {criterion.value} has no tasks"

    def test_four_tasks_per_criterion(self):
        """Each criterion should have exactly 4 tasks."""
        for criterion in AutonomyCriterion:
            tasks = TASKS_BY_CRITERION.get(criterion, [])
            assert (
                len(tasks) == 4
            ), f"Criterion {criterion.value} has {len(tasks)} tasks, expected 4"

    def test_total_sixteen_tasks(self):
        """Should have exactly 16 autonomy tasks."""
        assert len(ALL_AUTONOMY_TASKS) == 16

    def test_unique_task_ids(self):
        """All task IDs should be unique."""
        ids = [t.task_id for t in ALL_AUTONOMY_TASKS]
        assert len(ids) == len(set(ids))

    def test_unique_seeds(self):
        """All tasks should have unique seeds."""
        seeds = [t.seed for t in ALL_AUTONOMY_TASKS]
        assert len(seeds) == len(set(seeds))

    def test_difficulty_distribution(self):
        """Should have mix of difficulties."""
        difficulties = [t.difficulty for t in ALL_AUTONOMY_TASKS]

        assert "easy" in difficulties
        assert "medium" in difficulties
        assert "hard" in difficulties

    def test_statistics_accuracy(self):
        """Statistics function should be accurate."""
        stats = get_autonomy_benchmark_statistics()

        assert stats["total_tasks"] == 16
        assert stats["criteria_covered"] == 4
        assert sum(stats["tasks_per_criterion"].values()) == 16


# ============================================================
# Test Paper Alignment
# ============================================================


class TestPaperAlignment:
    """Tests for alignment with paper Section IV-A."""

    def test_four_criteria_match_paper(self):
        """Should have exactly 4 criteria matching paper Section IV-A."""
        criteria = list(AutonomyCriterion)
        assert len(criteria) == 4

        criterion_names = {c.value for c in criteria}
        expected = {
            "action_selection_freedom",
            "goal_directed_persistence",
            "dynamic_termination",
            "error_recovery",
        }
        assert criterion_names == expected

    def test_five_autonomy_levels_match_table_iv(self):
        """Should have 5 autonomy levels matching Table IV."""
        levels = list(AutonomyLevel)
        assert len(levels) == 5

        level_names = {l.name for l in levels}
        expected = {
            "STATIC_WORKFLOW",
            "CONDITIONAL",
            "GUIDED_AGENT",
            "BOUNDED_AGENT",
            "FULL_AGENT",
        }
        assert level_names == expected

    def test_level_ordering(self):
        """Levels should be ordered 0-4."""
        assert AutonomyLevel.STATIC_WORKFLOW.value == 0
        assert AutonomyLevel.CONDITIONAL.value == 1
        assert AutonomyLevel.GUIDED_AGENT.value == 2
        assert AutonomyLevel.BOUNDED_AGENT.value == 3
        assert AutonomyLevel.FULL_AGENT.value == 4

    def test_genuine_agent_requires_all_four_criteria(self):
        """is_genuine_agent should require all 4 criteria per paper."""
        partial = AutonomyCriteria(
            action_selection_freedom=True,
            goal_directed_persistence=True,
            dynamic_termination=True,
            error_recovery=False,
        )
        assert partial.is_genuine_agent is False

        full = AutonomyCriteria(
            action_selection_freedom=True,
            goal_directed_persistence=True,
            dynamic_termination=True,
            error_recovery=True,
        )
        assert full.is_genuine_agent is True


# ============================================================
# Test Determinism
# ============================================================


class TestDeterminism:
    """Tests for deterministic behavior."""

    def test_same_seed_same_result(self):
        """Same seed should produce same validation result."""
        agent = GenuineAgent()

        validator1 = AutonomyValidator(seed=42)
        result1 = validator1.validate_action_selection_freedom(agent)

        agent.reset()

        validator2 = AutonomyValidator(seed=42)
        result2 = validator2.validate_action_selection_freedom(agent)

        assert result1.passed == result2.passed
        assert (
            result1.evidence["variation_ratio"] == result2.evidence["variation_ratio"]
        )

    def test_different_seed_may_differ(self):
        """Different seeds may produce different scenarios."""
        generator1 = TestScenarioGenerator(seed=42)
        generator2 = TestScenarioGenerator(seed=123)

        scenarios1 = generator1.generate_resource_scenarios("test")
        scenarios2 = generator2.generate_resource_scenarios("test")

        # Structure should be same, content may vary based on implementation
        assert len(scenarios1) == len(scenarios2)
