"""Tool registry for managing and organizing agent tools."""

from typing import List, Dict, Any, Callable, Optional
import logging


logger = logging.getLogger(__name__)


class ToolRegistry:
    """Registry for managing agent tools.

    Provides centralized tool management with features like
    categorization, validation, and discovery.

    Example:
        >>> registry = ToolRegistry()
        >>> registry.register(my_tool, category="search")
        >>> tools = registry.get_by_category("search")
    """

    def __init__(self):
        """Initialize the tool registry."""
        self._tools: Dict[str, Callable] = {}
        self._categories: Dict[str, List[str]] = {}
        self._metadata: Dict[str, Dict[str, Any]] = {}

    def register(
        self,
        tool: Callable,
        name: Optional[str] = None,
        category: str = "general",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Register a tool.

        Args:
            tool: Tool function or callable
            name: Optional name (defaults to function name)
            category: Tool category for organization
            metadata: Additional tool metadata
        """
        tool_name = name or getattr(tool, "__name__", str(tool))

        if tool_name in self._tools:
            logger.warning(f"Tool '{tool_name}' already registered, overwriting")

        self._tools[tool_name] = tool
        self._metadata[tool_name] = metadata or {}
        self._metadata[tool_name]["category"] = category

        # Add to category
        if category not in self._categories:
            self._categories[category] = []
        if tool_name not in self._categories[category]:
            self._categories[category].append(tool_name)

        logger.debug(f"Registered tool: {tool_name} (category: {category})")

    def unregister(self, name: str) -> bool:
        """Unregister a tool.

        Args:
            name: Tool name to remove

        Returns:
            True if tool was removed, False if not found
        """
        if name not in self._tools:
            return False

        del self._tools[name]

        # Remove from categories
        for category, tools in self._categories.items():
            if name in tools:
                tools.remove(name)

        if name in self._metadata:
            del self._metadata[name]

        return True

    def get(self, name: str) -> Optional[Callable]:
        """Get a tool by name.

        Args:
            name: Tool name

        Returns:
            Tool callable or None
        """
        return self._tools.get(name)

    def get_all(self) -> List[Callable]:
        """Get all registered tools.

        Returns:
            List of all tool callables
        """
        return list(self._tools.values())

    def get_by_category(self, category: str) -> List[Callable]:
        """Get tools by category.

        Args:
            category: Category name

        Returns:
            List of tools in the category
        """
        tool_names = self._categories.get(category, [])
        return [self._tools[name] for name in tool_names if name in self._tools]

    def list_tools(self) -> List[Dict[str, Any]]:
        """List all tools with metadata.

        Returns:
            List of tool info dictionaries
        """
        return [
            {
                "name": name,
                "description": getattr(tool, "__doc__", ""),
                **self._metadata.get(name, {}),
            }
            for name, tool in self._tools.items()
        ]

    def list_categories(self) -> List[str]:
        """List all categories.

        Returns:
            List of category names
        """
        return list(self._categories.keys())

    def __len__(self) -> int:
        """Return number of registered tools."""
        return len(self._tools)

    def __contains__(self, name: str) -> bool:
        """Check if tool is registered."""
        return name in self._tools

    def __repr__(self) -> str:
        return f"ToolRegistry(tools={len(self._tools)}, categories={len(self._categories)})"


# Global default registry
_default_registry = ToolRegistry()


def get_default_registry() -> ToolRegistry:
    """Get the default global tool registry.

    Returns:
        Default ToolRegistry instance
    """
    return _default_registry


def register_tool(
    tool: Callable,
    name: Optional[str] = None,
    category: str = "general",
) -> Callable:
    """Decorator to register a tool with the default registry.

    Args:
        tool: Tool function
        name: Optional tool name
        category: Tool category

    Returns:
        The original tool function
    """
    _default_registry.register(tool, name=name, category=category)
    return tool
