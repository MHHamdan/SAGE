"""Skill versioning and lifecycle management."""

import re
import logging
from typing import Optional, List, Dict, Any, Tuple
from dataclasses import dataclass, field
from datetime import datetime

from .skill_base import Skill

logger = logging.getLogger(__name__)


@dataclass
class SkillVersion:
    """Semantic version for a skill."""
    major: int = 1
    minor: int = 0
    patch: int = 0

    def __str__(self) -> str:
        return f"{self.major}.{self.minor}.{self.patch}"

    def __lt__(self, other: "SkillVersion") -> bool:
        return (self.major, self.minor, self.patch) < (other.major, other.minor, other.patch)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, SkillVersion):
            return False
        return (self.major, self.minor, self.patch) == (other.major, other.minor, other.patch)

    @classmethod
    def from_string(cls, version_str: str) -> "SkillVersion":
        """Parse version from string."""
        match = re.match(r"(\d+)\.(\d+)\.(\d+)", version_str)
        if not match:
            raise ValueError(f"Invalid version string: {version_str}")
        return cls(
            major=int(match.group(1)),
            minor=int(match.group(2)),
            patch=int(match.group(3)),
        )

    def bump_major(self) -> "SkillVersion":
        return SkillVersion(self.major + 1, 0, 0)

    def bump_minor(self) -> "SkillVersion":
        return SkillVersion(self.major, self.minor + 1, 0)

    def bump_patch(self) -> "SkillVersion":
        return SkillVersion(self.major, self.minor, self.patch + 1)


@dataclass
class VersionedSkill:
    """A skill with version history management."""
    name: str
    current_version: Skill
    version_history: Dict[str, Skill] = field(default_factory=dict)
    deprecation_schedule: Optional[str] = None

    def __post_init__(self):
        # Add current to history
        self.version_history[self.current_version.version] = self.current_version

    def add_version(
        self,
        skill: Skill,
        deprecate_previous: bool = False,
    ):
        """Add a new version.

        Args:
            skill: New version of the skill
            deprecate_previous: Deprecate the previous version
        """
        if skill.name != self.name:
            raise ValueError(f"Skill name mismatch: {skill.name} != {self.name}")

        # Validate version is newer
        new_ver = SkillVersion.from_string(skill.version)
        cur_ver = SkillVersion.from_string(self.current_version.version)

        if new_ver <= cur_ver:
            raise ValueError(f"New version {skill.version} must be newer than {self.current_version.version}")

        # Deprecate previous if requested
        if deprecate_previous:
            self.current_version.deprecate(
                reason=f"Superseded by version {skill.version}",
                successor=f"{self.name}:{skill.version}",
            )

        # Update
        self.version_history[skill.version] = skill
        self.current_version = skill

        logger.info(f"Added new version {skill.version} for skill {self.name}")

    def get_version(self, version: str) -> Optional[Skill]:
        """Get a specific version.

        Args:
            version: Version string

        Returns:
            Skill if found
        """
        return self.version_history.get(version)

    def get_latest(self) -> Skill:
        """Get the latest version."""
        return self.current_version

    def get_all_versions(self) -> List[str]:
        """Get all version strings, sorted."""
        versions = list(self.version_history.keys())
        versions.sort(key=lambda v: SkillVersion.from_string(v))
        return versions

    def rollback(self, version: str) -> bool:
        """Rollback to a previous version.

        Args:
            version: Version to rollback to

        Returns:
            True if successful
        """
        if version not in self.version_history:
            logger.error(f"Version {version} not found in history")
            return False

        old_skill = self.version_history[version]
        if old_skill.is_deprecated:
            old_skill.is_deprecated = False
            old_skill.deprecation_reason = None

        self.current_version = old_skill
        logger.info(f"Rolled back {self.name} to version {version}")
        return True

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "current_version": self.current_version.version,
            "versions": self.get_all_versions(),
            "deprecation_schedule": self.deprecation_schedule,
        }


class VersionManager:
    """Manager for versioned skills."""

    def __init__(self):
        """Initialize the version manager."""
        self._skills: Dict[str, VersionedSkill] = {}

    def register(self, skill: Skill) -> VersionedSkill:
        """Register a skill, creating or updating versioned entry.

        Args:
            skill: Skill to register

        Returns:
            VersionedSkill instance
        """
        if skill.name in self._skills:
            versioned = self._skills[skill.name]
            try:
                versioned.add_version(skill)
            except ValueError as e:
                logger.warning(f"Could not add version: {e}")
        else:
            versioned = VersionedSkill(
                name=skill.name,
                current_version=skill,
            )
            self._skills[skill.name] = versioned

        return versioned

    def get(self, name: str) -> Optional[VersionedSkill]:
        """Get versioned skill by name."""
        return self._skills.get(name)

    def get_skill(
        self,
        name: str,
        version: Optional[str] = None,
    ) -> Optional[Skill]:
        """Get a specific skill version.

        Args:
            name: Skill name
            version: Version string (latest if None)

        Returns:
            Skill if found
        """
        versioned = self._skills.get(name)
        if not versioned:
            return None

        if version:
            return versioned.get_version(version)
        return versioned.get_latest()

    def list_skills(self) -> List[str]:
        """List all skill names."""
        return list(self._skills.keys())

    def get_all_current(self) -> List[Skill]:
        """Get all current versions."""
        return [v.current_version for v in self._skills.values()]
