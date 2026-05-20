"""Skill registry for managing agent skills."""

import logging
from typing import Optional, List, Dict, Any, Callable
from dataclasses import dataclass, field

from .skill_base import Skill, SkillPermission

logger = logging.getLogger(__name__)


class SkillRegistry:
    """Registry for managing and organizing skills.

    Provides skill registration, lookup, filtering, and lifecycle management.

    Example:
        >>> registry = SkillRegistry()
        >>> registry.register(search_skill)
        >>> registry.register(analyze_skill)
        >>>
        >>> all_skills = registry.get_all()
        >>> network_skills = registry.filter_by_permission(SkillPermission.NETWORK)
    """

    def __init__(self):
        """Initialize the skill registry."""
        self._skills: Dict[str, Skill] = {}
        self._tags_index: Dict[str, List[str]] = {}  # tag -> skill names
        self._version_history: Dict[str, List[str]] = {}  # skill name -> versions

    def register(
        self,
        skill: Skill,
        replace: bool = False,
    ) -> bool:
        """Register a skill.

        Args:
            skill: Skill to register
            replace: Replace if exists

        Returns:
            True if registered successfully
        """
        key = f"{skill.name}:{skill.version}"

        if skill.name in self._skills and not replace:
            existing = self._skills[skill.name]
            if existing.version == skill.version:
                logger.warning(f"Skill '{skill.name}' v{skill.version} already registered")
                return False

        self._skills[skill.name] = skill

        # Index by tags
        for tag in skill.metadata.tags:
            if tag not in self._tags_index:
                self._tags_index[tag] = []
            if skill.name not in self._tags_index[tag]:
                self._tags_index[tag].append(skill.name)

        # Track version history
        if skill.name not in self._version_history:
            self._version_history[skill.name] = []
        if skill.version not in self._version_history[skill.name]:
            self._version_history[skill.name].append(skill.version)

        logger.debug(f"Registered skill: {skill.name} v{skill.version}")
        return True

    def unregister(self, name: str) -> bool:
        """Unregister a skill.

        Args:
            name: Skill name

        Returns:
            True if unregistered
        """
        if name not in self._skills:
            return False

        skill = self._skills[name]

        # Remove from tag index
        for tag in skill.metadata.tags:
            if tag in self._tags_index and name in self._tags_index[tag]:
                self._tags_index[tag].remove(name)

        del self._skills[name]
        logger.debug(f"Unregistered skill: {name}")
        return True

    def get(self, name: str) -> Optional[Skill]:
        """Get a skill by name.

        Args:
            name: Skill name

        Returns:
            Skill if found, None otherwise
        """
        return self._skills.get(name)

    def get_all(self) -> List[Skill]:
        """Get all registered skills.

        Returns:
            List of all skills
        """
        return list(self._skills.values())

    def get_by_tag(self, tag: str) -> List[Skill]:
        """Get skills by tag.

        Args:
            tag: Tag to filter by

        Returns:
            List of skills with the tag
        """
        names = self._tags_index.get(tag, [])
        return [self._skills[name] for name in names if name in self._skills]

    def filter_by_permission(
        self,
        permission: SkillPermission,
        include: bool = True,
    ) -> List[Skill]:
        """Filter skills by permission.

        Args:
            permission: Permission to filter by
            include: If True, include skills with permission; if False, exclude

        Returns:
            Filtered list of skills
        """
        result = []
        for skill in self._skills.values():
            has_permission = permission in skill.permissions
            if (include and has_permission) or (not include and not has_permission):
                result.append(skill)
        return result

    def filter_by_trust(self, min_trust: float = 0.5) -> List[Skill]:
        """Filter skills by minimum trust score.

        Args:
            min_trust: Minimum trust score

        Returns:
            List of skills meeting trust threshold
        """
        return [s for s in self._skills.values() if s.trust_score >= min_trust]

    def filter_active(self) -> List[Skill]:
        """Get non-deprecated skills.

        Returns:
            List of active skills
        """
        return [s for s in self._skills.values() if not s.is_deprecated]

    def search(self, query: str) -> List[Skill]:
        """Search skills by name or description.

        Args:
            query: Search query

        Returns:
            List of matching skills
        """
        query_lower = query.lower()
        results = []

        for skill in self._skills.values():
            if (query_lower in skill.name.lower() or
                query_lower in skill.description.lower()):
                results.append(skill)

        return results

    def get_versions(self, name: str) -> List[str]:
        """Get version history for a skill.

        Args:
            name: Skill name

        Returns:
            List of versions
        """
        return self._version_history.get(name, [])

    def list_tags(self) -> List[str]:
        """List all tags.

        Returns:
            List of all tags
        """
        return list(self._tags_index.keys())

    def get_stats(self) -> Dict[str, Any]:
        """Get registry statistics.

        Returns:
            Statistics dictionary
        """
        skills = list(self._skills.values())

        return {
            "total_skills": len(skills),
            "active_skills": len([s for s in skills if not s.is_deprecated]),
            "deprecated_skills": len([s for s in skills if s.is_deprecated]),
            "total_tags": len(self._tags_index),
            "avg_trust_score": sum(s.trust_score for s in skills) / len(skills) if skills else 0,
            "total_invocations": sum(s.stats.total_invocations for s in skills),
        }

    def to_dict(self) -> Dict[str, Any]:
        """Convert registry to dictionary.

        Returns:
            Dictionary representation
        """
        return {
            "skills": {name: skill.to_dict() for name, skill in self._skills.items()},
            "tags": self._tags_index,
            "stats": self.get_stats(),
        }

    def __len__(self) -> int:
        return len(self._skills)

    def __contains__(self, name: str) -> bool:
        return name in self._skills

    def __iter__(self):
        return iter(self._skills.values())


# Global default registry
_default_registry = SkillRegistry()


def get_default_registry() -> SkillRegistry:
    """Get the default global skill registry."""
    return _default_registry


def register_skill(skill: Skill) -> bool:
    """Register a skill with the default registry."""
    return _default_registry.register(skill)
