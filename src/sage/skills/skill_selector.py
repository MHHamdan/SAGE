"""Skill selector for intelligent skill ranking and selection."""

import logging
from typing import Optional, List, Dict, Any, Tuple
from dataclasses import dataclass
from enum import Enum

from .skill_base import Skill
from .skill_registry import SkillRegistry

logger = logging.getLogger(__name__)


class SelectionStrategy(Enum):
    """Strategy for skill selection."""
    RELEVANCE = "relevance"  # Pure relevance ranking
    TRUST = "trust"  # Trust-weighted ranking
    COST = "cost"  # Cost-optimized ranking
    BALANCED = "balanced"  # Balance all factors


@dataclass
class SkillScore:
    """Score breakdown for a skill."""
    skill: Skill
    relevance_score: float = 0.0
    trust_score: float = 0.0
    cost_score: float = 0.0
    history_score: float = 0.0
    total_score: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "skill_name": self.skill.name,
            "relevance": self.relevance_score,
            "trust": self.trust_score,
            "cost": self.cost_score,
            "history": self.history_score,
            "total": self.total_score,
        }


class SkillSelector:
    """Intelligent skill selector based on relevance, trust, and cost.

    Ranks and selects skills based on:
    - Relevance to the query (semantic similarity)
    - Trust score (based on historical performance)
    - Cost (estimated execution cost)
    - Success history (past performance)

    Example:
        >>> selector = SkillSelector(registry)
        >>> top_skills = selector.select(
        ...     query="search for AI papers",
        ...     top_k=3,
        ...     strategy=SelectionStrategy.BALANCED,
        ... )
        >>> for score in top_skills:
        ...     print(f"{score.skill.name}: {score.total_score:.2f}")
    """

    def __init__(
        self,
        registry: SkillRegistry,
        relevance_weight: float = 0.4,
        trust_weight: float = 0.3,
        cost_weight: float = 0.15,
        history_weight: float = 0.15,
    ):
        """Initialize the skill selector.

        Args:
            registry: Skill registry to select from
            relevance_weight: Weight for relevance score
            trust_weight: Weight for trust score
            cost_weight: Weight for cost score (inverse)
            history_weight: Weight for history score
        """
        self.registry = registry
        self.relevance_weight = relevance_weight
        self.trust_weight = trust_weight
        self.cost_weight = cost_weight
        self.history_weight = history_weight

    def select(
        self,
        query: str,
        top_k: int = 5,
        strategy: SelectionStrategy = SelectionStrategy.BALANCED,
        min_trust: float = 0.0,
        max_cost: Optional[float] = None,
        exclude_deprecated: bool = True,
    ) -> List[SkillScore]:
        """Select top skills for a query.

        Args:
            query: Query describing the task
            top_k: Number of skills to return
            strategy: Selection strategy
            min_trust: Minimum trust threshold
            max_cost: Maximum cost threshold
            exclude_deprecated: Exclude deprecated skills

        Returns:
            List of SkillScore objects, sorted by total score
        """
        # Get candidate skills
        candidates = self._get_candidates(
            min_trust=min_trust,
            max_cost=max_cost,
            exclude_deprecated=exclude_deprecated,
        )

        if not candidates:
            return []

        # Score each skill
        scores = []
        for skill in candidates:
            score = self._score_skill(skill, query, strategy)
            scores.append(score)

        # Sort by total score (descending)
        scores.sort(key=lambda s: s.total_score, reverse=True)

        return scores[:top_k]

    def _get_candidates(
        self,
        min_trust: float,
        max_cost: Optional[float],
        exclude_deprecated: bool,
    ) -> List[Skill]:
        """Get candidate skills based on filters."""
        candidates = []

        for skill in self.registry:
            if exclude_deprecated and skill.is_deprecated:
                continue
            if skill.trust_score < min_trust:
                continue
            if max_cost is not None and skill.estimated_cost > max_cost:
                continue
            candidates.append(skill)

        return candidates

    def _score_skill(
        self,
        skill: Skill,
        query: str,
        strategy: SelectionStrategy,
    ) -> SkillScore:
        """Score a skill for a query."""
        score = SkillScore(skill=skill)

        # Calculate individual scores
        score.relevance_score = self._calculate_relevance(skill, query)
        score.trust_score = skill.trust_score
        score.cost_score = self._calculate_cost_score(skill)
        score.history_score = self._calculate_history_score(skill)

        # Calculate total based on strategy
        if strategy == SelectionStrategy.RELEVANCE:
            score.total_score = score.relevance_score

        elif strategy == SelectionStrategy.TRUST:
            score.total_score = 0.5 * score.relevance_score + 0.5 * score.trust_score

        elif strategy == SelectionStrategy.COST:
            score.total_score = 0.4 * score.relevance_score + 0.6 * score.cost_score

        elif strategy == SelectionStrategy.BALANCED:
            score.total_score = (
                self.relevance_weight * score.relevance_score +
                self.trust_weight * score.trust_score +
                self.cost_weight * score.cost_score +
                self.history_weight * score.history_score
            )

        return score

    def _calculate_relevance(self, skill: Skill, query: str) -> float:
        """Calculate relevance score using keyword matching.

        In production, this could use semantic similarity with embeddings.
        """
        query_lower = query.lower()
        skill_text = f"{skill.name} {skill.description}".lower()

        # Simple keyword overlap
        query_words = set(query_lower.split())
        skill_words = set(skill_text.split())

        if not query_words:
            return 0.0

        overlap = len(query_words & skill_words)
        return min(1.0, overlap / len(query_words))

    def _calculate_cost_score(self, skill: Skill) -> float:
        """Calculate cost score (higher is better = lower cost)."""
        # Normalize cost to 0-1 scale (inverse)
        max_cost = 1.0  # Assume max cost of $1
        if skill.estimated_cost <= 0:
            return 1.0
        return max(0, 1 - skill.estimated_cost / max_cost)

    def _calculate_history_score(self, skill: Skill) -> float:
        """Calculate history score based on past performance."""
        stats = skill.stats

        if stats.total_invocations == 0:
            return 0.5  # Neutral for new skills

        # Combine success rate and recency
        success_score = stats.success_rate

        # Recency bonus (skills used recently get a boost)
        recency_bonus = 0.0
        if stats.last_used:
            from datetime import datetime
            try:
                last_used = datetime.fromisoformat(stats.last_used)
                age_hours = (datetime.now() - last_used).total_seconds() / 3600
                if age_hours < 24:
                    recency_bonus = 0.2
                elif age_hours < 168:  # 1 week
                    recency_bonus = 0.1
            except:
                pass

        return min(1.0, success_score + recency_bonus)

    def select_best(
        self,
        query: str,
        **kwargs,
    ) -> Optional[Skill]:
        """Select the single best skill for a query.

        Args:
            query: Query describing the task
            **kwargs: Additional arguments for select()

        Returns:
            Best skill or None
        """
        results = self.select(query, top_k=1, **kwargs)
        if results:
            return results[0].skill
        return None

    def explain_selection(
        self,
        query: str,
        top_k: int = 3,
    ) -> str:
        """Explain skill selection reasoning.

        Args:
            query: Query
            top_k: Number of skills to explain

        Returns:
            Explanation string
        """
        results = self.select(query, top_k=top_k)

        if not results:
            return f"No skills found for query: '{query}'"

        lines = [f"Skill selection for: '{query}'", ""]

        for i, score in enumerate(results, 1):
            lines.append(f"{i}. {score.skill.name} (total: {score.total_score:.3f})")
            lines.append(f"   - Relevance: {score.relevance_score:.3f}")
            lines.append(f"   - Trust: {score.trust_score:.3f}")
            lines.append(f"   - Cost: {score.cost_score:.3f}")
            lines.append(f"   - History: {score.history_score:.3f}")
            lines.append("")

        return "\n".join(lines)
