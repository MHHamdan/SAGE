"""Base tool class and decorator for creating agent tools."""

from typing import Callable, Any, Optional, Dict
from functools import wraps
from pydantic import BaseModel, Field

# Re-export LangChain tool decorator for convenience
from langchain_core.tools import tool


class BaseTool(BaseModel):
    """Base class for structured tools.

    Tools represent capabilities that agents can invoke to interact
    with external systems, perform calculations, or execute actions.

    Example:
        >>> class SearchTool(BaseTool):
        ...     name: str = "search"
        ...     description: str = "Search the web for information"
        ...
        ...     def run(self, query: str) -> str:
        ...         return f"Results for: {query}"
    """

    name: str = Field(description="Tool name")
    description: str = Field(description="Tool description")
    return_direct: bool = Field(
        default=False,
        description="Return tool output directly without LLM processing",
    )

    def run(self, *args, **kwargs) -> Any:
        """Execute the tool.

        Override this method in subclasses to implement tool logic.
        """
        raise NotImplementedError("Subclasses must implement run()")

    def __call__(self, *args, **kwargs) -> Any:
        """Make the tool callable."""
        return self.run(*args, **kwargs)

    class Config:
        arbitrary_types_allowed = True


def create_tool(
    func: Callable,
    name: Optional[str] = None,
    description: Optional[str] = None,
) -> Callable:
    """Create a tool from a function.

    This is an alternative to the @tool decorator that allows
    programmatic tool creation.

    Args:
        func: Function to convert to a tool
        name: Optional tool name (defaults to function name)
        description: Optional description (defaults to docstring)

    Returns:
        Tool-wrapped function
    """
    tool_name = name or func.__name__
    tool_description = description or func.__doc__ or "No description"

    @wraps(func)
    def wrapper(*args, **kwargs):
        return func(*args, **kwargs)

    wrapper.__name__ = tool_name
    wrapper.__doc__ = tool_description

    return tool(wrapper)
