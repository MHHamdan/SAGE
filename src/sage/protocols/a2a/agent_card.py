"""Agent Card implementation for A2A protocol.

Agent Cards provide identity and capability declarations for agents.
"""

import logging
import hashlib
import json
from typing import Optional, List, Dict, Any
from dataclasses import dataclass, field
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


class AgentCardError(Exception):
    """Exception for agent card errors."""
    pass


@dataclass
class AgentCapability:
    """A capability declared by an agent."""
    name: str
    version: str = "1.0.0"
    description: str = ""
    input_schema: Dict[str, Any] = field(default_factory=dict)
    output_schema: Dict[str, Any] = field(default_factory=dict)
    permissions_required: List[str] = field(default_factory=list)


@dataclass
class AgentCard:
    """Agent identity and capability card.

    Provides:
    - Unique agent identification
    - Capability declarations
    - Trust metadata
    - Expiration management

    Example:
        >>> card = AgentCard(
        ...     agent_id="search-agent-001",
        ...     name="Search Agent",
        ...     capabilities=[AgentCapability(name="web_search")],
        ... )
        >>> validator = AgentCardValidator()
        >>> if validator.validate(card):
        ...     print("Card is valid")
    """
    agent_id: str
    name: str
    version: str = "1.0.0"
    description: str = ""
    capabilities: List[AgentCapability] = field(default_factory=list)
    trust_level: float = 0.5  # 0.0 to 1.0
    owner: str = ""
    created_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    expires_at: Optional[str] = None
    public_key: Optional[str] = None  # For cryptographic verification
    metadata: Dict[str, Any] = field(default_factory=dict)
    signature: Optional[str] = None  # Card signature

    def __post_init__(self):
        """Set default expiration if not provided."""
        if self.expires_at is None:
            default_expiry = datetime.utcnow() + timedelta(days=365)
            self.expires_at = default_expiry.isoformat()

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "agent_id": self.agent_id,
            "name": self.name,
            "version": self.version,
            "description": self.description,
            "capabilities": [
                {
                    "name": c.name,
                    "version": c.version,
                    "description": c.description,
                    "input_schema": c.input_schema,
                    "output_schema": c.output_schema,
                    "permissions_required": c.permissions_required,
                }
                for c in self.capabilities
            ],
            "trust_level": self.trust_level,
            "owner": self.owner,
            "created_at": self.created_at,
            "expires_at": self.expires_at,
            "public_key": self.public_key,
            "metadata": self.metadata,
            "signature": self.signature,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AgentCard":
        """Create from dictionary."""
        capabilities = [
            AgentCapability(
                name=c["name"],
                version=c.get("version", "1.0.0"),
                description=c.get("description", ""),
                input_schema=c.get("input_schema", {}),
                output_schema=c.get("output_schema", {}),
                permissions_required=c.get("permissions_required", []),
            )
            for c in data.get("capabilities", [])
        ]

        return cls(
            agent_id=data["agent_id"],
            name=data["name"],
            version=data.get("version", "1.0.0"),
            description=data.get("description", ""),
            capabilities=capabilities,
            trust_level=data.get("trust_level", 0.5),
            owner=data.get("owner", ""),
            created_at=data.get("created_at", datetime.utcnow().isoformat()),
            expires_at=data.get("expires_at"),
            public_key=data.get("public_key"),
            metadata=data.get("metadata", {}),
            signature=data.get("signature"),
        )

    def compute_hash(self) -> str:
        """Compute card hash for integrity checking."""
        # Exclude signature from hash
        data = self.to_dict()
        data.pop("signature", None)
        content = json.dumps(data, sort_keys=True)
        return hashlib.sha256(content.encode()).hexdigest()

    def has_capability(self, capability_name: str) -> bool:
        """Check if agent has a capability."""
        return any(c.name == capability_name for c in self.capabilities)

    def get_capability(self, name: str) -> Optional[AgentCapability]:
        """Get capability by name."""
        for cap in self.capabilities:
            if cap.name == name:
                return cap
        return None

    def is_expired(self) -> bool:
        """Check if card is expired."""
        if self.expires_at is None:
            return False

        try:
            expiry = datetime.fromisoformat(self.expires_at.replace("Z", "+00:00"))
            return datetime.utcnow() > expiry.replace(tzinfo=None)
        except Exception:
            return True  # If we can't parse, consider expired


class AgentCardValidator:
    """Validates agent cards.

    Checks:
    - Required fields
    - Expiration
    - Trust level
    - Signature (if public key available)
    """

    def __init__(
        self,
        min_trust_level: float = 0.0,
        require_signature: bool = False,
        allowed_capabilities: Optional[List[str]] = None,
    ):
        """Initialize validator.

        Args:
            min_trust_level: Minimum required trust level
            require_signature: Whether signature is required
            allowed_capabilities: Allowlist of capabilities (None = all allowed)
        """
        self.min_trust_level = min_trust_level
        self.require_signature = require_signature
        self.allowed_capabilities = allowed_capabilities
        self._trusted_keys: Dict[str, str] = {}  # agent_id -> public_key

    def register_trusted_key(self, agent_id: str, public_key: str):
        """Register a trusted public key for an agent.

        Args:
            agent_id: Agent identifier
            public_key: Agent's public key
        """
        self._trusted_keys[agent_id] = public_key

    def validate(self, card: AgentCard) -> bool:
        """Validate an agent card.

        Args:
            card: Agent card to validate

        Returns:
            True if valid

        Raises:
            AgentCardError: If validation fails
        """
        # Check required fields
        if not card.agent_id:
            raise AgentCardError("Missing agent_id")

        if not card.name:
            raise AgentCardError("Missing name")

        # Check expiration
        if card.is_expired():
            raise AgentCardError(f"Card expired at {card.expires_at}")

        # Check trust level
        if card.trust_level < self.min_trust_level:
            raise AgentCardError(
                f"Trust level {card.trust_level} below minimum {self.min_trust_level}"
            )

        # Check capabilities allowlist
        if self.allowed_capabilities is not None:
            for cap in card.capabilities:
                if cap.name not in self.allowed_capabilities:
                    raise AgentCardError(
                        f"Capability '{cap.name}' not in allowlist"
                    )

        # Check signature if required
        if self.require_signature:
            if not card.signature:
                raise AgentCardError("Signature required but not provided")

            if not self._verify_signature(card):
                raise AgentCardError("Invalid signature")

        return True

    def _verify_signature(self, card: AgentCard) -> bool:
        """Verify card signature.

        Args:
            card: Card with signature to verify

        Returns:
            True if signature is valid
        """
        # Check if we have a trusted key for this agent
        if card.agent_id in self._trusted_keys:
            expected_key = self._trusted_keys[card.agent_id]
            if card.public_key != expected_key:
                logger.warning(f"Public key mismatch for {card.agent_id}")
                return False

        # Placeholder for actual cryptographic verification
        # In production, this would use proper signature verification
        # For now, check that signature matches expected hash format
        if card.signature:
            expected_hash = card.compute_hash()
            # Simple verification: signature should contain the hash
            # Real implementation would use asymmetric crypto
            return expected_hash[:16] in card.signature

        return False

    def validate_capability_request(
        self,
        card: AgentCard,
        capability_name: str,
    ) -> bool:
        """Validate a capability request.

        Args:
            card: Agent's card
            capability_name: Requested capability

        Returns:
            True if agent can use the capability

        Raises:
            AgentCardError: If validation fails
        """
        # First validate the card itself
        self.validate(card)

        # Check agent has the capability
        if not card.has_capability(capability_name):
            raise AgentCardError(
                f"Agent {card.agent_id} doesn't have capability '{capability_name}'"
            )

        return True


class AgentCardRegistry:
    """Registry for known agent cards."""

    def __init__(self):
        """Initialize the registry."""
        self._cards: Dict[str, AgentCard] = {}
        self._validator = AgentCardValidator()

    def register(self, card: AgentCard):
        """Register an agent card.

        Args:
            card: Card to register

        Raises:
            AgentCardError: If card is invalid
        """
        self._validator.validate(card)
        self._cards[card.agent_id] = card
        logger.info(f"Registered agent card: {card.agent_id}")

    def get(self, agent_id: str) -> Optional[AgentCard]:
        """Get a card by agent ID.

        Args:
            agent_id: Agent identifier

        Returns:
            Agent card or None
        """
        return self._cards.get(agent_id)

    def remove(self, agent_id: str):
        """Remove a card from registry.

        Args:
            agent_id: Agent identifier
        """
        if agent_id in self._cards:
            del self._cards[agent_id]
            logger.info(f"Removed agent card: {agent_id}")

    def list_agents(self) -> List[str]:
        """List all registered agent IDs."""
        return list(self._cards.keys())

    def find_by_capability(self, capability_name: str) -> List[AgentCard]:
        """Find agents with a specific capability.

        Args:
            capability_name: Capability to search for

        Returns:
            List of matching agent cards
        """
        return [
            card for card in self._cards.values()
            if card.has_capability(capability_name) and not card.is_expired()
        ]
