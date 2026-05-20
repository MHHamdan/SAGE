"""MCP (Model Context Protocol) implementation.

Provides secure tool and resource sharing between agents and models.
"""

from .client import MCPClient
from .server import MCPServer
from .validation import MCPValidationError, validate_tool_call, validate_resource

__all__ = [
    "MCPClient",
    "MCPServer",
    "MCPValidationError",
    "validate_tool_call",
    "validate_resource",
]
