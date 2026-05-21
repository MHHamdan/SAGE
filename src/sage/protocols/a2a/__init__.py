"""Agent-to-Agent (A2A) Protocol implementation.

Provides secure inter-agent communication with:
- Agent card validation
- Replay protection
- Capability authentication
"""

from .agent_card import AgentCard, AgentCardError, AgentCardValidator
from .capability_auth import CapabilityAuth, CapabilityAuthError, CapabilityToken
from .replay_protection import NonceManager, ReplayAttackDetected, ReplayProtection

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
