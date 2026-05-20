"""Base skill class with versioning, permissions, and trust.

Skills are higher-level capabilities than tools, encapsulating
a coherent set of actions with associated metadata.
"""

import logging
from typing import Optional, List, Dict, Any, Callable, Set
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime

logger = logging.getLogger(__name__)


class SkillPermission(Enum):
    """Permissions that a skill may require."""
    READ = "read"
    WRITE = "write"
    EXECUTE = "execute"
    NETWORK = "network"
    FILE_SYSTEM = "file_system"
    SENSITIVE_DATA = "sensitive_data"


@dataclass
class SkillMetadata:
    """Metadata associated with a skill."""
    author: str = "unknown"
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    updated_at: Optional[str] = None
    tags: List[str] = field(default_factory=list)
    documentation_url: Optional[str] = None
    source_url: Optional[str] = None
    license: str = "MIT"


@dataclass
class SkillStats:
    """Runtime statistics for a skill."""
    total_invocations: int = 0
    successful_invocations: int = 0
    failed_invocations: int = 0
    total_latency_ms: float = 0.0
    last_used: Optional[str] = None

    @property
    def success_rate(self) -> float:
        if self.total_invocations == 0:
            return 1.0
        return self.successful_invocations / self.total_invocations

    @property
    def average_latency_ms(self) -> float:
        if self.total_invocations == 0:
            return 0.0
        return self.total_latency_ms / self.total_invocations

    def record_invocation(self, success: bool, latency_ms: float):
        """Record an invocation."""
        self.total_invocations += 1
        self.total_latency_ms += latency_ms
        if success:
            self.successful_invocations += 1
        else:
            self.failed_invocations += 1
        self.last_used = datetime.now().isoformat()


@dataclass
class Skill:
    """A skill representing a higher-level agent capability.

    Skills encapsulate:
    - Functionality (via tools)
    - Permissions required
    - Trust score (based on history)
    - Version information
    - Cost estimates

    Example:
        >>> skill = Skill(
        ...     name="web_search",
        ...     description="Search the web for information",
        ...     version="1.0.0",
        ...     permissions=[SkillPermission.NETWORK],
        ... )
        >>> skill.execute(query="AI research papers")
    """
    name: str
    description: str
    version: str = "1.0.0"

    # Permissions
    permissions: List[SkillPermission] = field(default_factory=list)

    # Trust and reliability
    trust_score: float = 1.0  # 0.0 to 1.0
    min_trust: float = 0.1

    # Cost
    estimated_cost: float = 0.0

    # Execution
    handler: Optional[Callable] = None
    tools: List[str] = field(default_factory=list)

    # Metadata
    metadata: SkillMetadata = field(default_factory=SkillMetadata)

    # Runtime stats
    stats: SkillStats = field(default_factory=SkillStats)

    # State
    is_deprecated: bool = False
    deprecation_reason: Optional[str] = None
    successor_skill: Optional[str] = None

    def execute(self, **kwargs) -> Any:
        """Execute the skill.

        Args:
            **kwargs: Arguments to pass to the handler

        Returns:
            Skill execution result
        """
        if not self.handler:
            raise RuntimeError(f"Skill '{self.name}' has no handler")

        if self.is_deprecated:
            logger.warning(f"Using deprecated skill: {self.name}")

        import time
        start_time = time.time()

        try:
            result = self.handler(**kwargs)
            latency_ms = (time.time() - start_time) * 1000
            self.stats.record_invocation(True, latency_ms)
            self._update_trust(True)
            return result

        except Exception as e:
            latency_ms = (time.time() - start_time) * 1000
            self.stats.record_invocation(False, latency_ms)
            self._update_trust(False)
            raise

    def _update_trust(self, success: bool, decay: float = 0.1):
        """Update trust score based on execution result.

        Uses exponential moving average.
        """
        outcome = 1.0 if success else 0.0
        self.trust_score = (1 - decay) * self.trust_score + decay * outcome
        self.trust_score = max(self.min_trust, min(1.0, self.trust_score))

    def requires_permission(self, permission: SkillPermission) -> bool:
        """Check if skill requires a permission.

        Args:
            permission: Permission to check

        Returns:
            True if skill requires the permission
        """
        return permission in self.permissions

    def has_all_permissions(self, available: Set[SkillPermission]) -> bool:
        """Check if all required permissions are available.

        Args:
            available: Set of available permissions

        Returns:
            True if all required permissions are available
        """
        return all(p in available for p in self.permissions)

    def deprecate(self, reason: str, successor: Optional[str] = None):
        """Mark the skill as deprecated.

        Args:
            reason: Reason for deprecation
            successor: Name of successor skill
        """
        self.is_deprecated = True
        self.deprecation_reason = reason
        self.successor_skill = successor
        self.metadata.updated_at = datetime.now().isoformat()

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "name": self.name,
            "description": self.description,
            "version": self.version,
            "permissions": [p.value for p in self.permissions],
            "trust_score": self.trust_score,
            "estimated_cost": self.estimated_cost,
            "tools": self.tools,
            "is_deprecated": self.is_deprecated,
            "stats": {
                "total_invocations": self.stats.total_invocations,
                "success_rate": self.stats.success_rate,
                "average_latency_ms": self.stats.average_latency_ms,
            },
        }

    def __repr__(self) -> str:
        return f"Skill(name='{self.name}', version='{self.version}', trust={self.trust_score:.2f})"


def create_skill(
    name: str,
    handler: Callable,
    description: str = "",
    permissions: Optional[List[SkillPermission]] = None,
    **kwargs,
) -> Skill:
    """Factory function to create a skill.

    Args:
        name: Skill name
        handler: Skill handler function
        description: Skill description
        permissions: Required permissions
        **kwargs: Additional skill attributes

    Returns:
        Configured Skill instance
    """
    return Skill(
        name=name,
        description=description or handler.__doc__ or "",
        handler=handler,
        permissions=permissions or [],
        **kwargs,
    )


def skill_decorator(
    name: Optional[str] = None,
    description: Optional[str] = None,
    permissions: Optional[List[SkillPermission]] = None,
    version: str = "1.0.0",
):
    """Decorator to create a skill from a function.

    Example:
        >>> @skill_decorator(name="search", permissions=[SkillPermission.NETWORK])
        ... def search_web(query: str) -> str:
        ...     '''Search the web for information.'''
        ...     return f"Results for {query}"
    """
    def decorator(func: Callable) -> Skill:
        return create_skill(
            name=name or func.__name__,
            handler=func,
            description=description or func.__doc__ or "",
            permissions=permissions or [],
            version=version,
        )
    return decorator
