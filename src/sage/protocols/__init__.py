"""Protocol implementations for agent communication.

This module provides hardened implementations for:
- MCP (Model Context Protocol) - Tool and resource sharing
- A2A (Agent-to-Agent) - Inter-agent communication

Security features:
- Agent card validation
- Replay protection (nonce/timestamps)
- Capability authentication
"""

from .mcp import MCPClient, MCPServer, MCPValidationError
from .a2a import AgentCard, AgentCardValidator, ReplayProtection, CapabilityAuth

__all__ = [
    "MCPClient",
    "MCPServer",
    "MCPValidationError",
    "AgentCard",
    "AgentCardValidator",
    "ReplayProtection",
    "CapabilityAuth",
]
