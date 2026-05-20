"""Skills module for skill-centric agent architecture.

Provides:
- Skill: Base skill class with versioning and trust
- SkillRegistry: Skill management and registration
- SkillSelector: Intelligent skill selection based on relevance, trust, cost

Usage:
    >>> from sage.skills import Skill, SkillRegistry, SkillSelector
    >>>
    >>> registry = SkillRegistry()
    >>> registry.register(Skill(
    ...     name="search",
    ...     description="Search the web",
    ...     version="1.0.0",
    ... ))
    >>>
    >>> selector = SkillSelector(registry)
    >>> best_skills = selector.select("find information about AI", top_k=3)
"""

from .skill_base import Skill, SkillMetadata, SkillPermission
from .skill_registry import SkillRegistry
from .skill_selector import SkillSelector, SelectionStrategy
from .versioning import SkillVersion, VersionedSkill

__all__ = [
    "Skill",
    "SkillMetadata",
    "SkillPermission",
    "SkillRegistry",
    "SkillSelector",
    "SelectionStrategy",
    "SkillVersion",
    "VersionedSkill",
]
