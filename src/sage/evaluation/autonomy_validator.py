"""
Autonomy Validator

Validates that an agent implementation meets the four minimum autonomy
criteria from Section IV-A of the paper, formally distinguishing genuine
agentic systems from pseudo-agentic workflows.

Paper Section IV-A states:
> "We propose that genuine agentic behavior requires four minimum autonomy criteria:
> 1. Action selection freedom - the system chooses among multiple possible actions
>    based on state assessment, not predetermined branching
> 2. Goal-directed persistence - continued pursuit of objectives across multiple
>    steps with adaptive strategy
> 3. Dynamic termination - self-determined completion based on goal satisfaction
>    rather than fixed step counts
> 4. Error recovery - autonomous response to failures without predetermined
>    fallback scripts"

This module distinguishes genuine agents from pseudo-agentic systems like:
- Scripted chains (fixed LLM call sequences)
- Template-driven workflows (variable substitution without decisions)
- Hard-coded tool sequences (predetermined invocation order)
"""

from __future__ import annotations

import random
import hashlib
import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum, IntEnum
from typing import Any, Dict, List, Optional, Callable, Tuple, Set
from collections import defaultdict

logger = logging.getLogger(__name__)


class AutonomyLevel(IntEnum):
    """
    Autonomy spectrum from Table IV of the paper.

    Levels represent increasing degrees of autonomous decision-making:
    - STATIC_WORKFLOW: Fixed path, no decisions
    - CONDITIONAL: Predetermined branches (if/else)
    - GUIDED_AGENT: Constrained action space
    - BOUNDED_AGENT: Domain-limited autonomy
    - FULL_AGENT: Unrestricted within tool constraints
    """
    STATIC_WORKFLOW = 0
    CONDITIONAL = 1
    GUIDED_AGENT = 2
    BOUNDED_AGENT = 3
    FULL_AGENT = 4

    @classmethod
    def from_criteria_count(cls, count: int) -> "AutonomyLevel":
        """Map number of criteria met to autonomy level."""
        mapping = {
            0: cls.STATIC_WORKFLOW,
            1: cls.CONDITIONAL,
            2: cls.GUIDED_AGENT,
            3: cls.BOUNDED_AGENT,
            4: cls.FULL_AGENT
        }
        return mapping.get(min(count, 4), cls.STATIC_WORKFLOW)


class AutonomyCriterion(Enum):
    """The four minimum autonomy criteria from Section IV-A."""
    ACTION_SELECTION_FREEDOM = "action_selection_freedom"
    GOAL_DIRECTED_PERSISTENCE = "goal_directed_persistence"
    DYNAMIC_TERMINATION = "dynamic_termination"
    ERROR_RECOVERY = "error_recovery"


@dataclass
class AutonomyThresholds:
    """
    Configurable thresholds for autonomy validation (Section IV-A).

    All thresholds are on a 0-1 scale where higher means stricter requirements.

    Attributes:
        action_variation_ratio: Minimum fraction of goals showing action variation
        persistence_obstacle_ratio: Minimum fraction of obstacles continued past
        persistence_strategy_changes: Minimum strategy changes required
        persistence_progress: Minimum goal progress required
        termination_variance: Minimum step count variance (in steps, not 0-1)
        recovery_attempt_ratio: Minimum fraction of failures with recovery attempted
        recovery_strategy_diversity: Minimum unique strategies as fraction of failures
    """
    # Action Selection Freedom thresholds
    action_variation_ratio: float = 0.5  # 50% of goals must show variation

    # Goal-Directed Persistence thresholds
    persistence_obstacle_ratio: float = 0.5  # 50% of obstacles handled
    persistence_strategy_changes: int = 1     # At least 1 strategy change
    persistence_progress: float = 0.3         # 30% progress toward goal

    # Dynamic Termination thresholds
    termination_variance: int = 2             # At least 2 steps variance
    termination_unique_counts: int = 2        # At least 2 unique step counts

    # Error Recovery thresholds
    recovery_attempt_ratio: float = 0.5       # 50% recovery attempts
    recovery_strategy_diversity: int = 2      # At least 2 unique strategies

    # Autonomy level weights (for weighted classification)
    criterion_weights: Dict[str, float] = field(default_factory=lambda: {
        "action_selection_freedom": 1.0,
        "goal_directed_persistence": 1.0,
        "dynamic_termination": 1.0,
        "error_recovery": 1.0
    })

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "action_variation_ratio": self.action_variation_ratio,
            "persistence_obstacle_ratio": self.persistence_obstacle_ratio,
            "persistence_strategy_changes": self.persistence_strategy_changes,
            "persistence_progress": self.persistence_progress,
            "termination_variance": self.termination_variance,
            "termination_unique_counts": self.termination_unique_counts,
            "recovery_attempt_ratio": self.recovery_attempt_ratio,
            "recovery_strategy_diversity": self.recovery_strategy_diversity,
            "criterion_weights": self.criterion_weights
        }

    @classmethod
    def strict(cls) -> "AutonomyThresholds":
        """Return strict thresholds for rigorous validation."""
        return cls(
            action_variation_ratio=0.7,
            persistence_obstacle_ratio=0.7,
            persistence_strategy_changes=2,
            persistence_progress=0.5,
            termination_variance=5,
            termination_unique_counts=3,
            recovery_attempt_ratio=0.8,
            recovery_strategy_diversity=3
        )

    @classmethod
    def lenient(cls) -> "AutonomyThresholds":
        """Return lenient thresholds for basic validation."""
        return cls(
            action_variation_ratio=0.3,
            persistence_obstacle_ratio=0.3,
            persistence_strategy_changes=1,
            persistence_progress=0.2,
            termination_variance=1,
            termination_unique_counts=2,
            recovery_attempt_ratio=0.3,
            recovery_strategy_diversity=1
        )


@dataclass
class TestResult:
    """Result of a single criterion validation test."""
    criterion: AutonomyCriterion
    passed: bool
    confidence: float = 1.0
    failure_reason: Optional[str] = None
    evidence: Dict[str, Any] = field(default_factory=dict)
    execution_trace: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize for logging/storage."""
        return {
            "criterion": self.criterion.value,
            "passed": self.passed,
            "confidence": self.confidence,
            "failure_reason": self.failure_reason,
            "evidence": self.evidence
        }


@dataclass
class AutonomyCriteria:
    """
    Holds boolean flags for each of the 4 autonomy criteria.

    Per Section IV-A, all four must be met for "genuine agentic behavior."
    """
    action_selection_freedom: bool = False
    goal_directed_persistence: bool = False
    dynamic_termination: bool = False
    error_recovery: bool = False

    # Supporting evidence for each criterion
    asf_evidence: Dict[str, Any] = field(default_factory=dict)
    gdp_evidence: Dict[str, Any] = field(default_factory=dict)
    dt_evidence: Dict[str, Any] = field(default_factory=dict)
    er_evidence: Dict[str, Any] = field(default_factory=dict)

    @property
    def criteria_met(self) -> int:
        """Count of criteria that are met."""
        return sum([
            self.action_selection_freedom,
            self.goal_directed_persistence,
            self.dynamic_termination,
            self.error_recovery
        ])

    @property
    def is_genuine_agent(self) -> bool:
        """
        Per paper: genuine agentic behavior requires ALL four criteria.
        """
        return self.criteria_met == 4

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "action_selection_freedom": self.action_selection_freedom,
            "goal_directed_persistence": self.goal_directed_persistence,
            "dynamic_termination": self.dynamic_termination,
            "error_recovery": self.error_recovery,
            "criteria_met": self.criteria_met,
            "is_genuine_agent": self.is_genuine_agent
        }


@dataclass
class AutonomyValidationResult:
    """
    Complete results from autonomy validation.

    Includes:
    - Overall autonomy level classification
    - Criteria met breakdown
    - Detailed test results for each criterion
    """
    level: AutonomyLevel
    criteria: AutonomyCriteria
    test_results: Dict[AutonomyCriterion, TestResult] = field(default_factory=dict)
    validation_time_seconds: float = 0.0
    agent_info: Dict[str, Any] = field(default_factory=dict)

    @property
    def criteria_met(self) -> int:
        """Number of criteria met."""
        return self.criteria.criteria_met

    @property
    def is_genuine_agent(self) -> bool:
        """Whether agent meets all 4 criteria."""
        return self.criteria.is_genuine_agent

    def to_dict(self) -> Dict[str, Any]:
        """Serialize for logging/storage."""
        return {
            "level": self.level.name,
            "level_value": self.level.value,
            "criteria_met": self.criteria_met,
            "is_genuine_agent": self.is_genuine_agent,
            "criteria": self.criteria.to_dict(),
            "test_results": {
                k.value: v.to_dict() for k, v in self.test_results.items()
            },
            "validation_time_seconds": self.validation_time_seconds,
            "agent_info": self.agent_info
        }

    def summary(self) -> str:
        """Human-readable summary."""
        lines = [
            f"Autonomy Level: {self.level.name} ({self.level.value}/4)",
            f"Criteria Met: {self.criteria_met}/4",
            f"Is Genuine Agent: {self.is_genuine_agent}",
            "",
            "Criterion Results:"
        ]

        for criterion, result in self.test_results.items():
            status = "PASS" if result.passed else "FAIL"
            lines.append(f"  {criterion.value}: {status}")
            if not result.passed and result.failure_reason:
                lines.append(f"    Reason: {result.failure_reason}")

        return "\n".join(lines)


# ============================================================
# Test Scenario Generators
# ============================================================

@dataclass
class TestScenario:
    """A test scenario for action selection freedom tests."""
    scenario_id: str
    goal: str
    state: Dict[str, Any]
    available_actions: List[str]
    expected_varied: bool = True  # Whether we expect different actions for different states


class TestScenarioGenerator:
    """
    Generates test scenarios for action selection freedom validation.

    Creates scenarios where the same goal is paired with different states,
    allowing us to verify the agent selects different actions based on
    state assessment rather than predetermined branching.
    """

    def __init__(self, seed: int = 42):
        """Initialize with deterministic seed."""
        self.seed = seed
        random.seed(seed)

    def generate_resource_scenarios(self, goal: str) -> List[TestScenario]:
        """
        Generate scenarios with same goal but different resource availability.

        A genuine agent should choose different actions based on available resources.
        """
        return [
            TestScenario(
                scenario_id="resource_full",
                goal=goal,
                state={
                    "resources": {"api_quota": 100, "memory_mb": 1024, "tools": ["search", "calculator", "file_read"]},
                    "context": "All resources available"
                },
                available_actions=["search", "calculate", "read_file", "ask_user"]
            ),
            TestScenario(
                scenario_id="resource_limited",
                goal=goal,
                state={
                    "resources": {"api_quota": 0, "memory_mb": 1024, "tools": ["calculator"]},
                    "context": "API quota exhausted, only local tools available"
                },
                available_actions=["calculate", "use_cache", "ask_user"]
            ),
            TestScenario(
                scenario_id="resource_minimal",
                goal=goal,
                state={
                    "resources": {"api_quota": 0, "memory_mb": 128, "tools": []},
                    "context": "Minimal resources, must ask for help"
                },
                available_actions=["ask_user", "report_limitation"]
            )
        ]

    def generate_context_scenarios(self, goal: str) -> List[TestScenario]:
        """
        Generate scenarios with same goal but different prior context.

        A genuine agent should adapt approach based on what's already known.
        """
        return [
            TestScenario(
                scenario_id="context_fresh",
                goal=goal,
                state={
                    "prior_knowledge": {},
                    "conversation_history": [],
                    "context": "Fresh start, no prior information"
                },
                available_actions=["search", "ask_clarification", "make_assumption"]
            ),
            TestScenario(
                scenario_id="context_partial",
                goal=goal,
                state={
                    "prior_knowledge": {"topic": "known", "details": "partial"},
                    "conversation_history": ["Previous discussion about topic"],
                    "context": "Some prior knowledge available"
                },
                available_actions=["refine_search", "ask_specific", "proceed_with_known"]
            ),
            TestScenario(
                scenario_id="context_complete",
                goal=goal,
                state={
                    "prior_knowledge": {"topic": "known", "details": "complete", "answer": "cached"},
                    "conversation_history": ["Full discussion", "Answer provided"],
                    "context": "Complete information already available"
                },
                available_actions=["retrieve_cached", "verify_freshness", "present_known"]
            )
        ]

    def generate_constraint_scenarios(self, goal: str) -> List[TestScenario]:
        """
        Generate scenarios with same goal but different constraints.

        A genuine agent should respect different constraint sets.
        """
        return [
            TestScenario(
                scenario_id="unconstrained",
                goal=goal,
                state={
                    "constraints": {},
                    "permissions": ["read", "write", "execute", "network"],
                    "context": "No restrictions"
                },
                available_actions=["full_solution", "comprehensive_approach"]
            ),
            TestScenario(
                scenario_id="time_constrained",
                goal=goal,
                state={
                    "constraints": {"time_limit_seconds": 5},
                    "permissions": ["read", "write", "execute", "network"],
                    "context": "Tight time constraint"
                },
                available_actions=["quick_solution", "approximate_answer", "request_extension"]
            ),
            TestScenario(
                scenario_id="permission_constrained",
                goal=goal,
                state={
                    "constraints": {"no_network": True, "read_only": True},
                    "permissions": ["read"],
                    "context": "Read-only, offline mode"
                },
                available_actions=["local_solution", "cached_answer", "explain_limitation"]
            )
        ]


class ObstacleInjector:
    """
    Injects obstacles during agent execution for persistence testing.

    Obstacles test whether agents can adapt their strategy when
    encountering unexpected difficulties.
    """

    def __init__(self, seed: int = 42):
        """Initialize with deterministic seed."""
        self.seed = seed
        random.seed(seed)
        self.obstacle_log: List[Dict[str, Any]] = []

    def get_obstacle_sequence(self, difficulty: str = "medium") -> List[Dict[str, Any]]:
        """
        Generate a sequence of obstacles to inject.

        Args:
            difficulty: easy, medium, or hard

        Returns:
            List of obstacles with timing and type
        """
        if difficulty == "easy":
            return [
                {"step": 2, "type": "temporary_failure", "description": "Tool returns error once"},
            ]
        elif difficulty == "medium":
            return [
                {"step": 2, "type": "temporary_failure", "description": "Tool returns error once"},
                {"step": 4, "type": "resource_unavailable", "description": "Resource temporarily unavailable"},
                {"step": 6, "type": "unexpected_state", "description": "State differs from expectation"},
            ]
        else:  # hard
            return [
                {"step": 1, "type": "early_failure", "description": "First action fails"},
                {"step": 3, "type": "cascading_failure", "description": "Multiple tools fail"},
                {"step": 5, "type": "state_corruption", "description": "Partial state corruption"},
                {"step": 7, "type": "timeout", "description": "Operation times out"},
                {"step": 9, "type": "permission_revoked", "description": "Permission revoked mid-task"},
            ]

    def inject(self, step: int, obstacles: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """
        Check if an obstacle should be injected at this step.

        Args:
            step: Current execution step
            obstacles: List of planned obstacles

        Returns:
            Obstacle to inject, or None
        """
        for obstacle in obstacles:
            if obstacle["step"] == step:
                self.obstacle_log.append({
                    "step": step,
                    "obstacle": obstacle,
                    "timestamp": time.time()
                })
                return obstacle
        return None

    def get_log(self) -> List[Dict[str, Any]]:
        """Get log of injected obstacles."""
        return self.obstacle_log.copy()

    def clear_log(self) -> None:
        """Clear obstacle log."""
        self.obstacle_log = []


class FailureInjector:
    """
    Injects various failure types for error recovery testing.

    Tests whether agents can recover from failures autonomously
    rather than using predetermined fallback scripts.
    """

    FAILURE_TYPES = [
        "tool_unavailable",
        "invalid_response",
        "timeout",
        "permission_denied",
        "rate_limited",
        "network_error",
        "parse_error",
        "validation_error"
    ]

    def __init__(self, seed: int = 42):
        """Initialize with deterministic seed."""
        self.seed = seed
        random.seed(seed)
        self.failure_log: List[Dict[str, Any]] = []

    def get_failure_sequence(
        self,
        types: Optional[List[str]] = None,
        count: int = 3
    ) -> List[Dict[str, Any]]:
        """
        Generate a sequence of failures to inject.

        Args:
            types: Specific failure types to include
            count: Number of failures to generate

        Returns:
            List of failures with type and details
        """
        types = types or self.FAILURE_TYPES[:4]
        failures = []

        for i, failure_type in enumerate(types[:count]):
            failures.append({
                "step": (i + 1) * 2,  # Inject at steps 2, 4, 6, ...
                "type": failure_type,
                "description": self._get_failure_description(failure_type),
                "recoverable": True
            })

        return failures

    def _get_failure_description(self, failure_type: str) -> str:
        """Get description for failure type."""
        descriptions = {
            "tool_unavailable": "Requested tool is not available",
            "invalid_response": "Tool returned malformed response",
            "timeout": "Operation timed out after 30 seconds",
            "permission_denied": "Insufficient permissions for operation",
            "rate_limited": "API rate limit exceeded",
            "network_error": "Network connection failed",
            "parse_error": "Failed to parse response",
            "validation_error": "Input validation failed"
        }
        return descriptions.get(failure_type, "Unknown failure")

    def inject(
        self,
        step: int,
        failures: List[Dict[str, Any]]
    ) -> Optional[Dict[str, Any]]:
        """
        Check if a failure should be injected at this step.

        Args:
            step: Current execution step
            failures: List of planned failures

        Returns:
            Failure to inject, or None
        """
        for failure in failures:
            if failure["step"] == step:
                self.failure_log.append({
                    "step": step,
                    "failure": failure,
                    "timestamp": time.time()
                })
                return failure
        return None

    def get_log(self) -> List[Dict[str, Any]]:
        """Get log of injected failures."""
        return self.failure_log.copy()

    def clear_log(self) -> None:
        """Clear failure log."""
        self.failure_log = []


# ============================================================
# Mock Agents for Testing
# ============================================================

class MockAgent(ABC):
    """Base class for mock agents used in testing."""

    def __init__(self, name: str):
        self.name = name
        self.execution_trace: List[Dict[str, Any]] = []
        self.step_count = 0

    @abstractmethod
    def select_action(self, state: Dict[str, Any], goal: str) -> str:
        """Select an action based on state and goal."""
        pass

    @abstractmethod
    def execute(self, action: str, state: Dict[str, Any]) -> Dict[str, Any]:
        """Execute an action and return result."""
        pass

    @abstractmethod
    def should_terminate(self, state: Dict[str, Any], goal: str) -> bool:
        """Determine if execution should terminate."""
        pass

    @abstractmethod
    def handle_failure(self, failure: Dict[str, Any], state: Dict[str, Any]) -> str:
        """Handle a failure and return recovery action."""
        pass

    def reset(self) -> None:
        """Reset agent state."""
        self.execution_trace = []
        self.step_count = 0


class GenuineAgent(MockAgent):
    """
    Mock agent that passes all 4 autonomy criteria.

    Demonstrates:
    - Varies actions based on state (action selection freedom)
    - Persists and adapts strategy (goal-directed persistence)
    - Terminates based on goal satisfaction (dynamic termination)
    - Recovers from failures adaptively (error recovery)
    """

    def __init__(self):
        super().__init__("GenuineAgent")
        self.strategy_history: List[str] = []
        self.goal_progress: float = 0.0

    def select_action(self, state: Dict[str, Any], goal: str) -> str:
        """Select action based on state assessment."""
        # Genuine action selection freedom: different actions for different states
        resources = state.get("resources", {})
        constraints = state.get("constraints", {})
        prior = state.get("prior_knowledge", {})

        # Build action based on state
        if not resources.get("api_quota", 100):
            action = "use_local_cache"
        elif constraints.get("time_limit_seconds", float("inf")) < 10:
            action = "quick_approximation"
        elif prior.get("answer"):
            action = "retrieve_known_answer"
        elif prior.get("topic") == "known":
            action = "refine_existing_knowledge"
        else:
            action = "comprehensive_search"

        self.execution_trace.append({
            "step": self.step_count,
            "action": action,
            "state_hash": hashlib.md5(str(state).encode()).hexdigest()[:8]
        })
        self.step_count += 1

        return action

    def execute(self, action: str, state: Dict[str, Any]) -> Dict[str, Any]:
        """Execute action with progress toward goal."""
        # Simulate progress
        self.goal_progress += 0.2

        return {
            "success": True,
            "progress": self.goal_progress,
            "action": action
        }

    def should_terminate(self, state: Dict[str, Any], goal: str) -> bool:
        """Terminate based on goal satisfaction, not fixed steps."""
        # Dynamic termination: based on progress, not step count
        return self.goal_progress >= 1.0

    def handle_failure(self, failure: Dict[str, Any], state: Dict[str, Any]) -> str:
        """Adaptively handle different failure types."""
        failure_type = failure.get("type", "unknown")

        # Different recovery strategies for different failures
        recovery_strategies = {
            "tool_unavailable": "use_alternative_tool",
            "invalid_response": "retry_with_validation",
            "timeout": "increase_timeout_and_retry",
            "permission_denied": "request_permission_or_escalate",
            "rate_limited": "wait_and_retry",
            "network_error": "use_cached_data",
            "parse_error": "request_structured_format",
            "validation_error": "fix_input_and_retry"
        }

        recovery = recovery_strategies.get(failure_type, "generic_retry")

        # Track strategy changes
        if recovery not in self.strategy_history[-3:] if self.strategy_history else True:
            self.strategy_history.append(recovery)

        return recovery

    def reset(self) -> None:
        """Reset agent state."""
        super().reset()
        self.strategy_history = []
        self.goal_progress = 0.0


class ScriptedAgent(MockAgent):
    """
    Mock agent that FAILS action selection freedom.

    Always follows predetermined action sequence regardless of state.
    """

    SCRIPTED_SEQUENCE = ["step_1_search", "step_2_process", "step_3_respond"]

    def __init__(self):
        super().__init__("ScriptedAgent")
        self.sequence_index = 0

    def select_action(self, state: Dict[str, Any], goal: str) -> str:
        """Always return next action in predetermined sequence."""
        # FAILS action selection freedom: ignores state
        action = self.SCRIPTED_SEQUENCE[self.sequence_index % len(self.SCRIPTED_SEQUENCE)]
        self.sequence_index += 1

        self.execution_trace.append({
            "step": self.step_count,
            "action": action,
            "state_hash": hashlib.md5(str(state).encode()).hexdigest()[:8]
        })
        self.step_count += 1

        return action

    def execute(self, action: str, state: Dict[str, Any]) -> Dict[str, Any]:
        """Execute action."""
        return {"success": True, "action": action}

    def should_terminate(self, state: Dict[str, Any], goal: str) -> bool:
        """Terminate after fixed sequence completes."""
        return self.sequence_index >= len(self.SCRIPTED_SEQUENCE)

    def handle_failure(self, failure: Dict[str, Any], state: Dict[str, Any]) -> str:
        """Use predetermined fallback."""
        return "scripted_fallback"  # Same recovery for all failures

    def reset(self) -> None:
        """Reset agent state."""
        super().reset()
        self.sequence_index = 0


class FragileAgent(MockAgent):
    """
    Mock agent that FAILS persistence and error recovery.

    Gives up after first failure, never adapts strategy.
    """

    def __init__(self):
        super().__init__("FragileAgent")
        self.failed = False
        self.attempt_count = 0

    def select_action(self, state: Dict[str, Any], goal: str) -> str:
        """Select action (varies somewhat by state)."""
        if self.failed:
            return "give_up"

        # Some variation based on state
        if state.get("resources", {}).get("api_quota", 100) > 50:
            action = "search"
        else:
            action = "local_search"

        self.execution_trace.append({
            "step": self.step_count,
            "action": action
        })
        self.step_count += 1

        return action

    def execute(self, action: str, state: Dict[str, Any]) -> Dict[str, Any]:
        """Execute action."""
        return {"success": not self.failed, "action": action}

    def should_terminate(self, state: Dict[str, Any], goal: str) -> bool:
        """Terminate on failure or after some steps."""
        return self.failed or self.step_count >= 5

    def handle_failure(self, failure: Dict[str, Any], state: Dict[str, Any]) -> str:
        """FAILS: Give up immediately without recovery attempt."""
        self.failed = True
        return "give_up"  # No recovery, just fails

    def reset(self) -> None:
        """Reset agent state."""
        super().reset()
        self.failed = False
        self.attempt_count = 0


class FixedStepAgent(MockAgent):
    """
    Mock agent that FAILS dynamic termination.

    Always runs exactly N steps regardless of goal satisfaction.
    """

    FIXED_STEPS = 10

    def __init__(self):
        super().__init__("FixedStepAgent")
        self.goal_achieved = False

    def select_action(self, state: Dict[str, Any], goal: str) -> str:
        """Select action (varies by state)."""
        if state.get("prior_knowledge", {}).get("answer"):
            action = "retrieve"
            self.goal_achieved = True
        else:
            action = "search"

        self.execution_trace.append({
            "step": self.step_count,
            "action": action
        })
        self.step_count += 1

        return action

    def execute(self, action: str, state: Dict[str, Any]) -> Dict[str, Any]:
        """Execute action."""
        return {"success": True, "action": action}

    def should_terminate(self, state: Dict[str, Any], goal: str) -> bool:
        """FAILS: Always terminate at fixed step count, ignoring goal."""
        return self.step_count >= self.FIXED_STEPS

    def handle_failure(self, failure: Dict[str, Any], state: Dict[str, Any]) -> str:
        """Handle failure with some variation."""
        return f"recover_from_{failure.get('type', 'unknown')}"

    def reset(self) -> None:
        """Reset agent state."""
        super().reset()
        self.goal_achieved = False


# ============================================================
# Main Validator Class
# ============================================================

class AutonomyValidator:
    """
    Validates that an agent implementation meets the four minimum
    autonomy criteria from Section IV-A of the paper.

    This distinguishes genuine agents from pseudo-agentic systems like:
    - Scripted chains (fixed LLM call sequences)
    - Template-driven workflows (variable substitution without decisions)
    - Hard-coded tool sequences (predetermined invocation order)

    Usage:
        ```python
        validator = AutonomyValidator()
        result = validator.validate_all(agent)

        print(f"Autonomy Level: {result.level}")
        print(f"Is Genuine Agent: {result.is_genuine_agent}")

        for criterion, test_result in result.test_results.items():
            print(f"  {criterion.value}: {'PASS' if test_result.passed else 'FAIL'}")
        ```
    """

    def __init__(
        self,
        seed: int = 42,
        verbose: bool = False,
        thresholds: Optional[AutonomyThresholds] = None
    ):
        """
        Initialize autonomy validator.

        Args:
            seed: Random seed for deterministic testing
            verbose: Enable verbose logging
            thresholds: Custom thresholds for validation criteria
        """
        self.seed = seed
        self.verbose = verbose
        self.thresholds = thresholds or AutonomyThresholds()

        self.scenario_generator = TestScenarioGenerator(seed=seed)
        self.obstacle_injector = ObstacleInjector(seed=seed)
        self.failure_injector = FailureInjector(seed=seed)

        random.seed(seed)

    def validate_action_selection_freedom(
        self,
        agent: Any,
        test_scenarios: Optional[List[TestScenario]] = None
    ) -> TestResult:
        """
        Verify agent selects different actions based on state, not predetermined branching.

        Paper Section IV-A, Criterion 1:
        > "Action selection freedom - the system chooses among multiple possible
        > actions based on state assessment, not predetermined branching"

        Test Logic:
        - Present agent with same goal but different states
        - PASS if agent selects different actions for different states
        - FAIL if agent always follows same action sequence regardless of state

        Args:
            agent: Agent to validate (must have select_action method)
            test_scenarios: Custom scenarios, or None to use generated ones

        Returns:
            TestResult with pass/fail and evidence
        """
        if self.verbose:
            logger.info("Validating action selection freedom...")

        # Generate scenarios if not provided
        if test_scenarios is None:
            goal = "Complete the requested task effectively"
            test_scenarios = (
                self.scenario_generator.generate_resource_scenarios(goal) +
                self.scenario_generator.generate_context_scenarios(goal) +
                self.scenario_generator.generate_constraint_scenarios(goal)
            )

        # Reset agent if possible
        if hasattr(agent, 'reset'):
            agent.reset()

        # Track actions per goal
        actions_by_goal: Dict[str, List[Tuple[str, str]]] = defaultdict(list)
        execution_trace = []

        for scenario in test_scenarios:
            if hasattr(agent, 'reset'):
                agent.reset()

            # Get action for this scenario
            action = agent.select_action(scenario.state, scenario.goal)
            state_hash = hashlib.md5(str(scenario.state).encode()).hexdigest()[:8]

            actions_by_goal[scenario.goal].append((state_hash, action))
            execution_trace.append({
                "scenario_id": scenario.scenario_id,
                "goal": scenario.goal,
                "state_hash": state_hash,
                "action": action
            })

        # Analyze: for each goal, check if actions vary by state
        goals_with_variation = 0
        total_goals = len(actions_by_goal)
        variation_details = {}

        for goal, state_action_pairs in actions_by_goal.items():
            unique_actions = set(action for _, action in state_action_pairs)
            unique_states = set(state_hash for state_hash, _ in state_action_pairs)

            # If we have multiple states and multiple actions, there's variation
            has_variation = len(unique_actions) > 1 and len(unique_states) > 1

            if has_variation:
                goals_with_variation += 1

            variation_details[goal[:30]] = {
                "unique_states": len(unique_states),
                "unique_actions": len(unique_actions),
                "has_variation": has_variation
            }

        # PASS if most goals show action variation based on state
        # Threshold from configuration
        variation_ratio = goals_with_variation / total_goals if total_goals > 0 else 0
        passed = variation_ratio >= self.thresholds.action_variation_ratio

        failure_reason = None
        if not passed:
            failure_reason = (
                f"Agent shows action variation for only {goals_with_variation}/{total_goals} goals. "
                "Actions appear predetermined rather than state-dependent."
            )

        return TestResult(
            criterion=AutonomyCriterion.ACTION_SELECTION_FREEDOM,
            passed=passed,
            confidence=variation_ratio,
            failure_reason=failure_reason,
            evidence={
                "total_scenarios": len(test_scenarios),
                "goals_tested": total_goals,
                "goals_with_variation": goals_with_variation,
                "variation_ratio": variation_ratio,
                "variation_details": variation_details
            },
            execution_trace=execution_trace
        )

    def validate_goal_directed_persistence(
        self,
        agent: Any,
        goal: str = "Complete a multi-step research task",
        max_steps: int = 20,
        obstacle_difficulty: str = "medium"
    ) -> TestResult:
        """
        Verify agent persists toward goal across obstacles with strategy adaptation.

        Paper Section IV-A, Criterion 2:
        > "Goal-directed persistence - continued pursuit of objectives across
        > multiple steps with adaptive strategy"

        Test Logic:
        - Give agent a complex goal requiring multiple steps
        - Inject obstacles/failures during execution
        - PASS if agent: (a) encounters obstacles, (b) changes strategy at least once,
          (c) eventually achieves goal or makes meaningful progress
        - FAIL if agent: abandons goal after first failure OR never adapts strategy

        Args:
            agent: Agent to validate
            goal: Goal to pursue
            max_steps: Maximum execution steps
            obstacle_difficulty: easy, medium, or hard

        Returns:
            TestResult with pass/fail and evidence
        """
        if self.verbose:
            logger.info("Validating goal-directed persistence...")

        # Reset agent and injector
        if hasattr(agent, 'reset'):
            agent.reset()
        self.obstacle_injector.clear_log()

        # Get obstacles to inject
        obstacles = self.obstacle_injector.get_obstacle_sequence(obstacle_difficulty)

        # Track execution
        execution_trace = []
        strategies_used: Set[str] = set()
        obstacles_encountered = 0
        continued_after_obstacle = 0
        goal_progress = 0.0

        state = {"goal": goal, "resources": {"api_quota": 100}, "step": 0}

        for step in range(max_steps):
            # Check for obstacle injection
            obstacle = self.obstacle_injector.inject(step, obstacles)

            if obstacle:
                obstacles_encountered += 1

                # Agent must handle obstacle
                recovery_action = agent.handle_failure(obstacle, state)

                execution_trace.append({
                    "step": step,
                    "type": "obstacle",
                    "obstacle": obstacle,
                    "recovery": recovery_action
                })

                # Check if agent gave up
                if recovery_action in ["give_up", "abort", "terminate"]:
                    break

                continued_after_obstacle += 1
                strategies_used.add(recovery_action)

            # Regular action selection
            action = agent.select_action(state, goal)
            strategies_used.add(action)

            execution_trace.append({
                "step": step,
                "type": "action",
                "action": action
            })

            # Execute and update state
            result = agent.execute(action, state)
            goal_progress = result.get("progress", goal_progress + 0.1)
            state["step"] = step + 1

            # Check termination
            if agent.should_terminate(state, goal):
                execution_trace.append({
                    "step": step,
                    "type": "termination",
                    "progress": goal_progress
                })
                break

        # Analyze results
        strategy_changes = len(strategies_used) - 1  # First strategy doesn't count as change
        persistence_ratio = continued_after_obstacle / max(obstacles_encountered, 1)

        # PASS criteria (using configurable thresholds):
        # 1. Encountered at least one obstacle
        # 2. Continued after obstacles (didn't give up immediately)
        # 3. Changed strategy at least once
        # 4. Made progress toward goal

        passed = (
            obstacles_encountered > 0 and
            persistence_ratio >= self.thresholds.persistence_obstacle_ratio and
            strategy_changes >= self.thresholds.persistence_strategy_changes and
            goal_progress >= self.thresholds.persistence_progress
        )

        failure_reason = None
        if not passed:
            reasons = []
            if obstacles_encountered == 0:
                reasons.append("No obstacles encountered")
            if persistence_ratio < self.thresholds.persistence_obstacle_ratio:
                reasons.append(f"Gave up too easily ({continued_after_obstacle}/{obstacles_encountered} obstacles handled, need {self.thresholds.persistence_obstacle_ratio:.0%})")
            if strategy_changes < self.thresholds.persistence_strategy_changes:
                reasons.append(f"Insufficient strategy changes ({strategy_changes}, need {self.thresholds.persistence_strategy_changes})")
            if goal_progress < self.thresholds.persistence_progress:
                reasons.append(f"Insufficient progress ({goal_progress:.0%}, need {self.thresholds.persistence_progress:.0%})")
            failure_reason = "; ".join(reasons)

        return TestResult(
            criterion=AutonomyCriterion.GOAL_DIRECTED_PERSISTENCE,
            passed=passed,
            confidence=min(persistence_ratio, goal_progress),
            failure_reason=failure_reason,
            evidence={
                "obstacles_encountered": obstacles_encountered,
                "continued_after_obstacle": continued_after_obstacle,
                "persistence_ratio": persistence_ratio,
                "strategy_changes": strategy_changes,
                "strategies_used": list(strategies_used),
                "goal_progress": goal_progress,
                "total_steps": len(execution_trace)
            },
            execution_trace=execution_trace
        )

    def validate_dynamic_termination(
        self,
        agent: Any,
        goals: Optional[List[Dict[str, Any]]] = None
    ) -> TestResult:
        """
        Verify agent terminates based on goal satisfaction, not fixed step count.

        Paper Section IV-A, Criterion 3:
        > "Dynamic termination - self-determined completion based on goal
        > satisfaction rather than fixed step counts"

        Test Logic:
        - Give agent multiple goals of varying complexity
        - PASS if step counts vary based on goal complexity AND agent can recognize completion
        - FAIL if agent always runs same number of steps OR terminates without achieving goal

        Args:
            agent: Agent to validate
            goals: Custom goals, or None to use defaults

        Returns:
            TestResult with pass/fail and evidence
        """
        if self.verbose:
            logger.info("Validating dynamic termination...")

        # Default goals of varying complexity
        if goals is None:
            goals = [
                {
                    "goal": "Return the number 42",
                    "complexity": "simple",
                    "expected_steps_range": (1, 3),
                    "state": {"answer": 42}
                },
                {
                    "goal": "Search for information about Python",
                    "complexity": "medium",
                    "expected_steps_range": (4, 10),
                    "state": {"topic": "Python", "needs_search": True}
                },
                {
                    "goal": "Research and summarize the history of AI, including key milestones",
                    "complexity": "complex",
                    "expected_steps_range": (10, 25),
                    "state": {"topic": "AI history", "requires_multiple_sources": True}
                }
            ]

        # Run each goal and track step counts
        execution_trace = []
        step_counts = []
        complexities = []

        for goal_config in goals:
            if hasattr(agent, 'reset'):
                agent.reset()

            goal = goal_config["goal"]
            state = goal_config.get("state", {})
            complexity = goal_config["complexity"]
            max_steps = goal_config.get("expected_steps_range", (1, 30))[1] + 10

            steps_taken = 0

            for step in range(max_steps):
                action = agent.select_action(state, goal)
                result = agent.execute(action, state)
                steps_taken += 1
                state["step"] = step

                if agent.should_terminate(state, goal):
                    break

            step_counts.append(steps_taken)
            complexities.append(complexity)

            execution_trace.append({
                "goal": goal[:50],
                "complexity": complexity,
                "steps_taken": steps_taken,
                "expected_range": goal_config.get("expected_steps_range", (1, 30))
            })

        # Analyze: check if step counts vary with complexity
        unique_step_counts = len(set(step_counts))
        step_count_variance = max(step_counts) - min(step_counts) if step_counts else 0

        # Check correlation between complexity and steps
        complexity_order = {"simple": 0, "medium": 1, "complex": 2}
        complexity_values = [complexity_order.get(c, 1) for c in complexities]

        # Simple correlation check: do step counts increase with complexity?
        correlation_pairs = list(zip(complexity_values, step_counts))
        correlation_pairs.sort(key=lambda x: x[0])

        is_monotonic = all(
            correlation_pairs[i][1] <= correlation_pairs[i+1][1]
            for i in range(len(correlation_pairs) - 1)
        )

        # PASS criteria (using configurable thresholds):
        # 1. Different step counts for different complexities
        # 2. Step counts roughly correlate with complexity

        passed = (
            unique_step_counts >= self.thresholds.termination_unique_counts and
            step_count_variance >= self.thresholds.termination_variance and
            is_monotonic
        )

        failure_reason = None
        if not passed:
            reasons = []
            if unique_step_counts < self.thresholds.termination_unique_counts:
                reasons.append(f"Same step count ({step_counts[0]}) for all goals (need {self.thresholds.termination_unique_counts} unique)")
            if step_count_variance < self.thresholds.termination_variance:
                reasons.append(f"Insufficient variance (variance={step_count_variance}, need {self.thresholds.termination_variance})")
            if not is_monotonic:
                reasons.append("Step counts don't correlate with goal complexity")
            failure_reason = "; ".join(reasons)

        return TestResult(
            criterion=AutonomyCriterion.DYNAMIC_TERMINATION,
            passed=passed,
            confidence=min(1.0, step_count_variance / 10),
            failure_reason=failure_reason,
            evidence={
                "step_counts": step_counts,
                "complexities": complexities,
                "unique_step_counts": unique_step_counts,
                "step_count_variance": step_count_variance,
                "complexity_correlation": is_monotonic
            },
            execution_trace=execution_trace
        )

    def validate_error_recovery(
        self,
        agent: Any,
        goal: str = "Complete task despite failures",
        failure_types: Optional[List[str]] = None
    ) -> TestResult:
        """
        Verify agent recovers from failures without predetermined fallback scripts.

        Paper Section IV-A, Criterion 4:
        > "Error recovery - autonomous response to failures without predetermined
        > fallback scripts"

        Test Logic:
        - Inject various failure types during execution
        - PASS if agent attempts recovery AND recovery approach varies based on failure type
        - FAIL if agent uses identical recovery for all failures OR gives up without trying

        Args:
            agent: Agent to validate
            goal: Goal to pursue
            failure_types: Types of failures to inject

        Returns:
            TestResult with pass/fail and evidence
        """
        if self.verbose:
            logger.info("Validating error recovery...")

        # Reset agent and injector
        if hasattr(agent, 'reset'):
            agent.reset()
        self.failure_injector.clear_log()

        # Default failure types
        failure_types = failure_types or [
            "tool_unavailable",
            "invalid_response",
            "timeout",
            "permission_denied"
        ]

        # Get failures to inject
        failures = self.failure_injector.get_failure_sequence(
            types=failure_types,
            count=len(failure_types)
        )

        # Track execution
        execution_trace = []
        recovery_actions: Dict[str, str] = {}  # failure_type -> recovery_action
        recovery_attempts = 0
        gave_up_count = 0

        state = {"goal": goal, "resources": {"api_quota": 100}}

        for i, failure in enumerate(failures):
            if hasattr(agent, 'reset'):
                agent.reset()

            # Inject failure
            failure_type = failure["type"]
            recovery_action = agent.handle_failure(failure, state)

            execution_trace.append({
                "failure_type": failure_type,
                "recovery_action": recovery_action
            })

            recovery_actions[failure_type] = recovery_action

            if recovery_action in ["give_up", "abort", "terminate", None]:
                gave_up_count += 1
            else:
                recovery_attempts += 1

        # Analyze: check if recovery strategies vary by failure type
        unique_recoveries = len(set(recovery_actions.values()))
        total_failures = len(failures)

        # PASS criteria (using configurable thresholds):
        # 1. Attempted recovery for most failures (didn't give up immediately)
        # 2. Different recovery strategies for different failure types

        recovery_ratio = recovery_attempts / total_failures if total_failures > 0 else 0
        strategy_diversity = unique_recoveries / total_failures if total_failures > 0 else 0

        passed = (
            recovery_ratio >= self.thresholds.recovery_attempt_ratio and
            unique_recoveries >= min(self.thresholds.recovery_strategy_diversity, total_failures)
        )

        failure_reason = None
        if not passed:
            reasons = []
            if recovery_ratio < self.thresholds.recovery_attempt_ratio:
                reasons.append(f"Gave up too often ({gave_up_count}/{total_failures} failures, need {self.thresholds.recovery_attempt_ratio:.0%} recovery)")
            if unique_recoveries < self.thresholds.recovery_strategy_diversity:
                reasons.append(f"Same recovery for all failures ({unique_recoveries} unique, need {self.thresholds.recovery_strategy_diversity})")
            failure_reason = "; ".join(reasons)

        return TestResult(
            criterion=AutonomyCriterion.ERROR_RECOVERY,
            passed=passed,
            confidence=min(recovery_ratio, strategy_diversity),
            failure_reason=failure_reason,
            evidence={
                "total_failures_injected": total_failures,
                "recovery_attempts": recovery_attempts,
                "gave_up_count": gave_up_count,
                "recovery_ratio": recovery_ratio,
                "unique_recovery_strategies": unique_recoveries,
                "recovery_actions": recovery_actions,
                "strategy_diversity": strategy_diversity
            },
            execution_trace=execution_trace
        )

    def validate_all(self, agent: Any) -> AutonomyValidationResult:
        """
        Run all four autonomy validations.

        Args:
            agent: Agent to validate

        Returns:
            AutonomyValidationResult with comprehensive results
        """
        start_time = time.time()

        if self.verbose:
            logger.info(f"Starting full autonomy validation for {getattr(agent, 'name', 'agent')}...")

        # Run all validations
        asf_result = self.validate_action_selection_freedom(agent)
        gdp_result = self.validate_goal_directed_persistence(agent)
        dt_result = self.validate_dynamic_termination(agent)
        er_result = self.validate_error_recovery(agent)

        # Build criteria from results
        criteria = AutonomyCriteria(
            action_selection_freedom=asf_result.passed,
            goal_directed_persistence=gdp_result.passed,
            dynamic_termination=dt_result.passed,
            error_recovery=er_result.passed,
            asf_evidence=asf_result.evidence,
            gdp_evidence=gdp_result.evidence,
            dt_evidence=dt_result.evidence,
            er_evidence=er_result.evidence
        )

        # Build test results dict for weighted classification
        test_results = {
            AutonomyCriterion.ACTION_SELECTION_FREEDOM: asf_result,
            AutonomyCriterion.GOAL_DIRECTED_PERSISTENCE: gdp_result,
            AutonomyCriterion.DYNAMIC_TERMINATION: dt_result,
            AutonomyCriterion.ERROR_RECOVERY: er_result
        }

        # Classify autonomy level using weighted scoring
        level = self.classify_autonomy_level(criteria, test_results)

        # Build result
        result = AutonomyValidationResult(
            level=level,
            criteria=criteria,
            test_results={
                AutonomyCriterion.ACTION_SELECTION_FREEDOM: asf_result,
                AutonomyCriterion.GOAL_DIRECTED_PERSISTENCE: gdp_result,
                AutonomyCriterion.DYNAMIC_TERMINATION: dt_result,
                AutonomyCriterion.ERROR_RECOVERY: er_result
            },
            validation_time_seconds=time.time() - start_time,
            agent_info={
                "name": getattr(agent, 'name', 'unknown'),
                "type": type(agent).__name__
            }
        )

        if self.verbose:
            logger.info(f"Validation complete: {result.level.name} ({result.criteria_met}/4 criteria)")

        return result

    def classify_autonomy_level(
        self,
        criteria: AutonomyCriteria,
        test_results: Optional[Dict[AutonomyCriterion, TestResult]] = None
    ) -> AutonomyLevel:
        """
        Classify autonomy level using weighted criteria (Table IV).

        Supports two modes:
        1. Simple: Map count of passed criteria to level
        2. Weighted: Use criterion weights and confidence scores

        Args:
            criteria: AutonomyCriteria with pass/fail for each
            test_results: Optional TestResult dict for weighted scoring

        Returns:
            AutonomyLevel classification
        """
        # If no test results, use simple count-based classification
        if test_results is None:
            return AutonomyLevel.from_criteria_count(criteria.criteria_met)

        # Weighted classification using confidence scores
        weighted_score = 0.0
        max_score = 0.0

        weights = self.thresholds.criterion_weights

        criterion_map = {
            AutonomyCriterion.ACTION_SELECTION_FREEDOM: criteria.action_selection_freedom,
            AutonomyCriterion.GOAL_DIRECTED_PERSISTENCE: criteria.goal_directed_persistence,
            AutonomyCriterion.DYNAMIC_TERMINATION: criteria.dynamic_termination,
            AutonomyCriterion.ERROR_RECOVERY: criteria.error_recovery
        }

        for criterion, passed in criterion_map.items():
            weight = weights.get(criterion.value, 1.0)
            max_score += weight

            if passed and criterion in test_results:
                # Weight by both pass status and confidence
                confidence = test_results[criterion].confidence
                weighted_score += weight * confidence
            elif passed:
                weighted_score += weight

        # Normalize to 0-1 scale
        normalized_score = weighted_score / max_score if max_score > 0 else 0

        # Map normalized score to autonomy level
        # Thresholds: 0-0.2=STATIC, 0.2-0.4=CONDITIONAL, 0.4-0.6=GUIDED, 0.6-0.8=BOUNDED, 0.8+=FULL
        if normalized_score < 0.2:
            return AutonomyLevel.STATIC_WORKFLOW
        elif normalized_score < 0.4:
            return AutonomyLevel.CONDITIONAL
        elif normalized_score < 0.6:
            return AutonomyLevel.GUIDED_AGENT
        elif normalized_score < 0.8:
            return AutonomyLevel.BOUNDED_AGENT
        else:
            return AutonomyLevel.FULL_AGENT

    def compute_weighted_autonomy_score(
        self,
        criteria: AutonomyCriteria,
        test_results: Dict[AutonomyCriterion, TestResult]
    ) -> Dict[str, Any]:
        """
        Compute detailed weighted autonomy score with breakdown.

        Returns:
            Dictionary with:
            - weighted_score: Normalized 0-1 score
            - per_criterion_scores: Breakdown by criterion
            - autonomy_level: Classified level
        """
        weights = self.thresholds.criterion_weights
        per_criterion = {}
        total_weighted = 0.0
        max_weighted = 0.0

        criterion_map = {
            AutonomyCriterion.ACTION_SELECTION_FREEDOM: (
                "action_selection_freedom",
                criteria.action_selection_freedom
            ),
            AutonomyCriterion.GOAL_DIRECTED_PERSISTENCE: (
                "goal_directed_persistence",
                criteria.goal_directed_persistence
            ),
            AutonomyCriterion.DYNAMIC_TERMINATION: (
                "dynamic_termination",
                criteria.dynamic_termination
            ),
            AutonomyCriterion.ERROR_RECOVERY: (
                "error_recovery",
                criteria.error_recovery
            )
        }

        for criterion, (name, passed) in criterion_map.items():
            weight = weights.get(name, 1.0)
            max_weighted += weight

            if criterion in test_results:
                confidence = test_results[criterion].confidence
                score = weight * confidence if passed else 0.0
            else:
                score = weight if passed else 0.0

            total_weighted += score
            per_criterion[name] = {
                "passed": passed,
                "weight": weight,
                "confidence": test_results[criterion].confidence if criterion in test_results else 1.0,
                "weighted_score": score
            }

        normalized = total_weighted / max_weighted if max_weighted > 0 else 0

        return {
            "weighted_score": normalized,
            "raw_score": total_weighted,
            "max_score": max_weighted,
            "per_criterion_scores": per_criterion,
            "autonomy_level": self.classify_autonomy_level(criteria, test_results).name
        }
