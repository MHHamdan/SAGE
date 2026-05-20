"""Scoped permissions for tools.

Implements least-privilege access control for tool execution.
"""

import logging
from typing import Optional, List, Dict, Any, Set
from dataclasses import dataclass, field
from enum import Enum, auto

logger = logging.getLogger(__name__)


class Permission(Enum):
    """Available permissions."""
    READ = auto()
    WRITE = auto()
    EXECUTE = auto()
    DELETE = auto()
    NETWORK = auto()
    FILE_SYSTEM = auto()
    PROCESS = auto()
    ADMIN = auto()


@dataclass
class PermissionScope:
    """Scope for a permission."""
    permission: Permission
    resource_pattern: str = "*"  # Pattern for resources this applies to
    allow: bool = True


@dataclass
class PermissionSet:
    """Set of permissions for an entity."""
    name: str
    permissions: Set[Permission] = field(default_factory=set)
    scopes: List[PermissionScope] = field(default_factory=list)

    def has_permission(self, permission: Permission) -> bool:
        """Check if permission is granted."""
        return permission in self.permissions

    def grant(self, permission: Permission):
        """Grant a permission."""
        self.permissions.add(permission)

    def revoke(self, permission: Permission):
        """Revoke a permission."""
        self.permissions.discard(permission)

    def add_scope(self, scope: PermissionScope):
        """Add a scoped permission."""
        self.scopes.append(scope)


class PermissionManager:
    """Manager for tool permissions.

    Example:
        >>> manager = PermissionManager()
        >>> manager.grant_permission("search_tool", Permission.NETWORK)
        >>> if manager.check_permission("search_tool", Permission.NETWORK):
        ...     tool.execute()
    """

    def __init__(self):
        """Initialize the permission manager."""
        self._permissions: Dict[str, PermissionSet] = {}
        self._default_permissions: Set[Permission] = {Permission.READ}

    def register_tool(
        self,
        tool_name: str,
        permissions: Optional[Set[Permission]] = None,
    ):
        """Register a tool with permissions.

        Args:
            tool_name: Tool name
            permissions: Initial permissions
        """
        self._permissions[tool_name] = PermissionSet(
            name=tool_name,
            permissions=permissions or self._default_permissions.copy(),
        )

    def grant_permission(
        self,
        tool_name: str,
        permission: Permission,
    ):
        """Grant a permission to a tool.

        Args:
            tool_name: Tool name
            permission: Permission to grant
        """
        if tool_name not in self._permissions:
            self.register_tool(tool_name)

        self._permissions[tool_name].grant(permission)
        logger.debug(f"Granted {permission.name} to {tool_name}")

    def revoke_permission(
        self,
        tool_name: str,
        permission: Permission,
    ):
        """Revoke a permission from a tool.

        Args:
            tool_name: Tool name
            permission: Permission to revoke
        """
        if tool_name in self._permissions:
            self._permissions[tool_name].revoke(permission)
            logger.debug(f"Revoked {permission.name} from {tool_name}")

    def check_permission(
        self,
        tool_name: str,
        permission: Permission,
    ) -> bool:
        """Check if a tool has a permission.

        Args:
            tool_name: Tool name
            permission: Permission to check

        Returns:
            True if permission is granted
        """
        if tool_name not in self._permissions:
            return permission in self._default_permissions

        return self._permissions[tool_name].has_permission(permission)

    def get_permissions(self, tool_name: str) -> Set[Permission]:
        """Get all permissions for a tool."""
        if tool_name not in self._permissions:
            return self._default_permissions.copy()

        return self._permissions[tool_name].permissions.copy()

    def set_default_permissions(self, permissions: Set[Permission]):
        """Set default permissions for new tools."""
        self._default_permissions = permissions


# Predefined permission sets
READONLY_PERMISSIONS = {Permission.READ}
STANDARD_PERMISSIONS = {Permission.READ, Permission.WRITE}
FULL_PERMISSIONS = {Permission.READ, Permission.WRITE, Permission.EXECUTE, Permission.DELETE}
NETWORK_PERMISSIONS = {Permission.READ, Permission.NETWORK}
