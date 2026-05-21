"""Conflict resolution for multi-agent systems."""

import logging
from typing import Optional, List, Dict, Any, Callable
from dataclasses import dataclass, field
from enum import Enum

logger = logging.getLogger(__name__)


class ConflictType(Enum):
    """Types of conflicts between agents."""
    RESOURCE = "resource"  # Same resource access
    ACTION = "action"  # Conflicting actions
    GOAL = "goal"  # Conflicting goals
    DATA = "data"  # Conflicting data modifications
    PRIORITY = "priority"  # Priority conflicts


class ResolutionStrategy(Enum):
    """Strategies for resolving conflicts."""
    PRIORITY = "priority"  # Higher priority wins
    UTILITY = "utility"  # Maximize utility
    CONSTRAINT = "constraint"  # Apply constraints
    NEGOTIATION = "negotiation"  # Agent negotiation
    RANDOM = "random"  # Random selection


@dataclass
class ConflictAction:
    """An action involved in a conflict."""
    agent_name: str
    action: str
    resource: Optional[str] = None
    priority: float = 0.5
    utility: float = 0.5
    parameters: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Conflict:
    """Represents a conflict between actions."""
    conflict_id: str
    conflict_type: ConflictType
    actions: List[ConflictAction] = field(default_factory=list)
    description: str = ""
    is_resolved: bool = False
    resolution: Optional[str] = None
    winning_agent: Optional[str] = None


@dataclass
class ResolutionResult:
    """Result of conflict resolution."""
    success: bool
    winning_action: Optional[ConflictAction] = None
    blocked_actions: List[ConflictAction] = field(default_factory=list)
    strategy_used: Optional[ResolutionStrategy] = None
    reasoning: str = ""


class ConflictDetector:
    """Detects conflicts between agent actions."""

    def __init__(self):
        """Initialize the conflict detector."""
        self._resource_locks: Dict[str, str] = {}  # resource -> agent
        self._conflict_count = 0

    def detect_conflicts(
        self,
        actions: List[ConflictAction],
    ) -> List[Conflict]:
        """Detect conflicts in a set of actions.

        Args:
            actions: List of proposed actions

        Returns:
            List of detected conflicts
        """
        conflicts = []

        # Check for resource conflicts
        resource_actions: Dict[str, List[ConflictAction]] = {}
        for action in actions:
            if action.resource:
                if action.resource not in resource_actions:
                    resource_actions[action.resource] = []
                resource_actions[action.resource].append(action)

        for resource, res_actions in resource_actions.items():
            if len(res_actions) > 1:
                self._conflict_count += 1
                conflicts.append(Conflict(
                    conflict_id=f"conflict_{self._conflict_count}",
                    conflict_type=ConflictType.RESOURCE,
                    actions=res_actions,
                    description=f"Multiple agents accessing resource: {resource}",
                ))

        # Check for action conflicts (e.g., write vs delete)
        conflicting_pairs = self._find_conflicting_actions(actions)
        for action1, action2 in conflicting_pairs:
            self._conflict_count += 1
            conflicts.append(Conflict(
                conflict_id=f"conflict_{self._conflict_count}",
                conflict_type=ConflictType.ACTION,
                actions=[action1, action2],
                description=f"Conflicting actions: {action1.action} vs {action2.action}",
            ))

        return conflicts

    def _find_conflicting_actions(
        self,
        actions: List[ConflictAction],
    ) -> List[tuple]:
        """Find pairs of conflicting actions."""
        conflicts = []

        # Define conflicting action pairs
        conflict_patterns = [
            ("write", "delete"),
            ("create", "delete"),
            ("update", "delete"),
            ("read", "delete"),
        ]

        for i, action1 in enumerate(actions):
            for action2 in actions[i + 1:]:
                for pattern in conflict_patterns:
                    if ((pattern[0] in action1.action.lower() and pattern[1] in action2.action.lower()) or
                        (pattern[1] in action1.action.lower() and pattern[0] in action2.action.lower())):
                        conflicts.append((action1, action2))

        return conflicts


class ConflictResolver:
    """Resolves conflicts between agent actions.

    Example:
        >>> resolver = ConflictResolver(strategy=ResolutionStrategy.UTILITY)
        >>> result = resolver.resolve(conflict)
        >>> if result.success:
        ...     execute(result.winning_action)
    """

    def __init__(
        self,
        strategy: ResolutionStrategy = ResolutionStrategy.UTILITY,
        utility_function: Optional[Callable[[ConflictAction], float]] = None,
    ):
        """Initialize the resolver.

        Args:
            strategy: Resolution strategy
            utility_function: Custom utility function for UTILITY strategy
        """
        self.strategy = strategy
        self.utility_function = utility_function or self._default_utility

    def resolve(self, conflict: Conflict) -> ResolutionResult:
        """Resolve a conflict.

        Args:
            conflict: Conflict to resolve

        Returns:
            ResolutionResult
        """
        if not conflict.actions:
            return ResolutionResult(success=False, reasoning="No actions in conflict")

        if len(conflict.actions) == 1:
            return ResolutionResult(
                success=True,
                winning_action=conflict.actions[0],
                strategy_used=self.strategy,
            )

        if self.strategy == ResolutionStrategy.PRIORITY:
            return self._resolve_by_priority(conflict)
        elif self.strategy == ResolutionStrategy.UTILITY:
            return self._resolve_by_utility(conflict)
        elif self.strategy == ResolutionStrategy.CONSTRAINT:
            return self._resolve_by_constraint(conflict)
        elif self.strategy == ResolutionStrategy.RANDOM:
            return self._resolve_random(conflict)
        else:
            return self._resolve_by_utility(conflict)

    def _resolve_by_priority(self, conflict: Conflict) -> ResolutionResult:
        """Resolve by agent priority."""
        sorted_actions = sorted(
            conflict.actions,
            key=lambda a: a.priority,
            reverse=True,
        )

        winner = sorted_actions[0]
        blocked = sorted_actions[1:]

        conflict.is_resolved = True
        conflict.winning_agent = winner.agent_name

        return ResolutionResult(
            success=True,
            winning_action=winner,
            blocked_actions=blocked,
            strategy_used=ResolutionStrategy.PRIORITY,
            reasoning=f"Agent {winner.agent_name} has highest priority ({winner.priority})",
        )

    def _resolve_by_utility(self, conflict: Conflict) -> ResolutionResult:
        """Resolve by maximizing utility."""
        utilities = [
            (action, self.utility_function(action))
            for action in conflict.actions
        ]
        utilities.sort(key=lambda x: x[1], reverse=True)

        winner = utilities[0][0]
        blocked = [u[0] for u in utilities[1:]]

        conflict.is_resolved = True
        conflict.winning_agent = winner.agent_name

        return ResolutionResult(
            success=True,
            winning_action=winner,
            blocked_actions=blocked,
            strategy_used=ResolutionStrategy.UTILITY,
            reasoning=f"Agent {winner.agent_name} maximizes utility ({utilities[0][1]:.2f})",
        )

    def _resolve_by_constraint(self, conflict: Conflict) -> ResolutionResult:
        """Resolve by applying constraints."""
        # Simple constraint: prefer read over write over delete
        action_order = {"read": 0, "write": 1, "delete": 2}

        def action_rank(a: ConflictAction) -> int:
            for key, rank in action_order.items():
                if key in a.action.lower():
                    return rank
            return 99

        sorted_actions = sorted(conflict.actions, key=action_rank)
        winner = sorted_actions[0]
        blocked = sorted_actions[1:]

        conflict.is_resolved = True
        conflict.winning_agent = winner.agent_name

        return ResolutionResult(
            success=True,
            winning_action=winner,
            blocked_actions=blocked,
            strategy_used=ResolutionStrategy.CONSTRAINT,
            reasoning=f"Action {winner.action} preferred by constraint rules",
        )

    def _resolve_random(self, conflict: Conflict) -> ResolutionResult:
        """Resolve randomly."""
        import random
        winner = random.choice(conflict.actions)
        blocked = [a for a in conflict.actions if a != winner]

        conflict.is_resolved = True
        conflict.winning_agent = winner.agent_name

        return ResolutionResult(
            success=True,
            winning_action=winner,
            blocked_actions=blocked,
            strategy_used=ResolutionStrategy.RANDOM,
            reasoning="Selected randomly",
        )

    def _default_utility(self, action: ConflictAction) -> float:
        """Default utility function."""
        # Combine priority and explicit utility
        return 0.5 * action.priority + 0.5 * action.utility
