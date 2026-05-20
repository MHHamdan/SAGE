"""Agent-to-Agent (A2A) Protocol implementation.

Provides secure inter-agent communication with:
- Agent card validation
- Replay protection
- Capability authentication
"""

from .agent_card import AgentCard, AgentCardValidator, AgentCardError
from .replay_protection import ReplayProtection, NonceManager, ReplayAttackDetected
from .capability_auth import CapabilityAuth, CapabilityToken, CapabilityAuthError

__all__ = [
    "AgentCard",
    "AgentCardValidator",
    "AgentCardError",
    "ReplayProtection",
    "NonceManager",
    "ReplayAttackDetected",
    "CapabilityAuth",
    "CapabilityToken",
    "CapabilityAuthError",
]
