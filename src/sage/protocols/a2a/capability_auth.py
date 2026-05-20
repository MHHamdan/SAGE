"""Capability authentication for A2A protocol.

Provides token-based capability authentication and authorization.
"""

import logging
import time
import hashlib
import secrets
import json
from typing import Optional, List, Dict, Any, Set
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum

logger = logging.getLogger(__name__)


class CapabilityAuthError(Exception):
    """Exception for capability authentication errors."""
    pass


class TokenStatus(Enum):
    """Token status values."""
    VALID = "valid"
    EXPIRED = "expired"
    REVOKED = "revoked"
    INVALID = "invalid"


@dataclass
class CapabilityToken:
    """Token granting specific capabilities.

    Example:
        >>> token = CapabilityToken.create(
        ...     agent_id="agent-001",
        ...     capabilities=["read", "write"],
        ...     issuer="coordinator",
        ...     validity_seconds=3600,
        ... )
    """
    token_id: str
    agent_id: str
    capabilities: List[str]
    issuer: str
    issued_at: float
    expires_at: float
    signature: str
    scope: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def create(
        cls,
        agent_id: str,
        capabilities: List[str],
        issuer: str,
        secret_key: str,
        validity_seconds: float = 3600.0,
        scope: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> "CapabilityToken":
        """Create a new capability token.

        Args:
            agent_id: Agent the token is for
            capabilities: List of granted capabilities
            issuer: Token issuer
            secret_key: Secret key for signing
            validity_seconds: Token validity in seconds
            scope: Optional scope limitation
            metadata: Optional metadata

        Returns:
            New CapabilityToken
        """
        now = time.time()
        token_id = secrets.token_hex(16)

        # Create signature
        payload = f"{token_id}:{agent_id}:{','.join(sorted(capabilities))}:{issuer}:{now}"
        signature = hashlib.sha256(f"{payload}:{secret_key}".encode()).hexdigest()

        return cls(
            token_id=token_id,
            agent_id=agent_id,
            capabilities=capabilities,
            issuer=issuer,
            issued_at=now,
            expires_at=now + validity_seconds,
            signature=signature,
            scope=scope,
            metadata=metadata or {},
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "token_id": self.token_id,
            "agent_id": self.agent_id,
            "capabilities": self.capabilities,
            "issuer": self.issuer,
            "issued_at": self.issued_at,
            "expires_at": self.expires_at,
            "signature": self.signature,
            "scope": self.scope,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CapabilityToken":
        """Create from dictionary."""
        return cls(
            token_id=data["token_id"],
            agent_id=data["agent_id"],
            capabilities=data["capabilities"],
            issuer=data["issuer"],
            issued_at=data["issued_at"],
            expires_at=data["expires_at"],
            signature=data["signature"],
            scope=data.get("scope"),
            metadata=data.get("metadata", {}),
        )

    def is_expired(self) -> bool:
        """Check if token is expired."""
        return time.time() > self.expires_at

    def has_capability(self, capability: str) -> bool:
        """Check if token grants a capability."""
        return capability in self.capabilities

    def remaining_validity(self) -> float:
        """Get remaining validity in seconds."""
        return max(0, self.expires_at - time.time())


class CapabilityAuth:
    """Capability authentication and authorization manager.

    Features:
    - Token issuance and validation
    - Token revocation
    - Capability checking
    - Audit logging

    Example:
        >>> auth = CapabilityAuth(secret_key="my-secret")
        >>> token = auth.issue_token(
        ...     agent_id="agent-001",
        ...     capabilities=["search", "read"],
        ... )
        >>> auth.validate_token(token)
        >>> auth.check_capability(token, "search")
    """

    def __init__(
        self,
        secret_key: str,
        issuer_id: str = "capability-auth",
        default_validity_seconds: float = 3600.0,
    ):
        """Initialize capability auth.

        Args:
            secret_key: Secret key for signing tokens
            issuer_id: Identifier for this issuer
            default_validity_seconds: Default token validity
        """
        self.secret_key = secret_key
        self.issuer_id = issuer_id
        self.default_validity_seconds = default_validity_seconds

        self._revoked_tokens: Set[str] = set()
        self._issued_tokens: Dict[str, CapabilityToken] = {}
        self._capability_grants: Dict[str, Set[str]] = {}  # agent_id -> capabilities

    def issue_token(
        self,
        agent_id: str,
        capabilities: List[str],
        validity_seconds: Optional[float] = None,
        scope: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> CapabilityToken:
        """Issue a new capability token.

        Args:
            agent_id: Agent to issue token for
            capabilities: Capabilities to grant
            validity_seconds: Token validity (uses default if None)
            scope: Optional scope limitation
            metadata: Optional metadata

        Returns:
            New CapabilityToken
        """
        # Check agent is allowed these capabilities
        allowed = self._capability_grants.get(agent_id, set())
        for cap in capabilities:
            if allowed and cap not in allowed:
                raise CapabilityAuthError(
                    f"Agent {agent_id} not authorized for capability '{cap}'"
                )

        token = CapabilityToken.create(
            agent_id=agent_id,
            capabilities=capabilities,
            issuer=self.issuer_id,
            secret_key=self.secret_key,
            validity_seconds=validity_seconds or self.default_validity_seconds,
            scope=scope,
            metadata=metadata,
        )

        self._issued_tokens[token.token_id] = token
        logger.info(f"Issued token {token.token_id[:8]}... for agent {agent_id}")

        return token

    def validate_token(self, token: CapabilityToken) -> TokenStatus:
        """Validate a capability token.

        Args:
            token: Token to validate

        Returns:
            TokenStatus

        Raises:
            CapabilityAuthError: If token is invalid
        """
        # Check if revoked
        if token.token_id in self._revoked_tokens:
            return TokenStatus.REVOKED

        # Check if expired
        if token.is_expired():
            return TokenStatus.EXPIRED

        # Verify signature
        expected_payload = (
            f"{token.token_id}:{token.agent_id}:"
            f"{','.join(sorted(token.capabilities))}:{token.issuer}:{token.issued_at}"
        )
        expected_sig = hashlib.sha256(
            f"{expected_payload}:{self.secret_key}".encode()
        ).hexdigest()

        if token.signature != expected_sig:
            logger.warning(f"Invalid signature for token {token.token_id[:8]}...")
            return TokenStatus.INVALID

        return TokenStatus.VALID

    def check_capability(
        self,
        token: CapabilityToken,
        capability: str,
        scope: Optional[str] = None,
    ) -> bool:
        """Check if token grants a specific capability.

        Args:
            token: Token to check
            capability: Capability to check for
            scope: Optional scope to verify

        Returns:
            True if capability is granted

        Raises:
            CapabilityAuthError: If token is invalid or doesn't have capability
        """
        # Validate token first
        status = self.validate_token(token)
        if status != TokenStatus.VALID:
            raise CapabilityAuthError(f"Token is {status.value}")

        # Check capability
        if not token.has_capability(capability):
            raise CapabilityAuthError(
                f"Token doesn't grant capability '{capability}'"
            )

        # Check scope if specified
        if scope and token.scope and token.scope != scope:
            raise CapabilityAuthError(
                f"Token scope '{token.scope}' doesn't match required '{scope}'"
            )

        return True

    def revoke_token(self, token_id: str):
        """Revoke a token.

        Args:
            token_id: ID of token to revoke
        """
        self._revoked_tokens.add(token_id)
        if token_id in self._issued_tokens:
            del self._issued_tokens[token_id]
        logger.info(f"Revoked token {token_id[:8]}...")

    def revoke_all_for_agent(self, agent_id: str):
        """Revoke all tokens for an agent.

        Args:
            agent_id: Agent whose tokens to revoke
        """
        to_revoke = [
            token_id for token_id, token in self._issued_tokens.items()
            if token.agent_id == agent_id
        ]

        for token_id in to_revoke:
            self.revoke_token(token_id)

        logger.info(f"Revoked {len(to_revoke)} tokens for agent {agent_id}")

    def grant_capability(self, agent_id: str, capability: str):
        """Grant a capability to an agent (for future token issuance).

        Args:
            agent_id: Agent to grant to
            capability: Capability to grant
        """
        if agent_id not in self._capability_grants:
            self._capability_grants[agent_id] = set()
        self._capability_grants[agent_id].add(capability)

    def revoke_capability(self, agent_id: str, capability: str):
        """Revoke a capability from an agent.

        Args:
            agent_id: Agent to revoke from
            capability: Capability to revoke
        """
        if agent_id in self._capability_grants:
            self._capability_grants[agent_id].discard(capability)

    def get_agent_capabilities(self, agent_id: str) -> Set[str]:
        """Get capabilities granted to an agent.

        Args:
            agent_id: Agent to query

        Returns:
            Set of capability names
        """
        return self._capability_grants.get(agent_id, set()).copy()

    def cleanup_expired_tokens(self):
        """Remove expired tokens from tracking."""
        expired = [
            token_id for token_id, token in self._issued_tokens.items()
            if token.is_expired()
        ]

        for token_id in expired:
            del self._issued_tokens[token_id]

        logger.debug(f"Cleaned up {len(expired)} expired tokens")


class DelegatedCapabilityAuth:
    """Support for delegated capability tokens.

    Allows agents to create sub-tokens with reduced capabilities.
    """

    def __init__(self, auth: CapabilityAuth):
        """Initialize delegated auth.

        Args:
            auth: Parent capability auth
        """
        self.auth = auth
        self._delegation_chains: Dict[str, str] = {}  # token_id -> parent_token_id

    def delegate(
        self,
        parent_token: CapabilityToken,
        capabilities: List[str],
        agent_id: str,
        validity_seconds: Optional[float] = None,
    ) -> CapabilityToken:
        """Create a delegated token with subset of capabilities.

        Args:
            parent_token: Token to delegate from
            capabilities: Capabilities to delegate (must be subset)
            agent_id: Agent to delegate to
            validity_seconds: Validity (must be <= parent remaining)

        Returns:
            Delegated token

        Raises:
            CapabilityAuthError: If delegation is invalid
        """
        # Validate parent token
        status = self.auth.validate_token(parent_token)
        if status != TokenStatus.VALID:
            raise CapabilityAuthError(f"Parent token is {status.value}")

        # Check capabilities are subset
        for cap in capabilities:
            if not parent_token.has_capability(cap):
                raise CapabilityAuthError(
                    f"Cannot delegate capability '{cap}' not in parent token"
                )

        # Check validity doesn't exceed parent
        parent_remaining = parent_token.remaining_validity()
        if validity_seconds is None:
            validity_seconds = parent_remaining
        elif validity_seconds > parent_remaining:
            validity_seconds = parent_remaining

        # Create delegated token
        token = CapabilityToken.create(
            agent_id=agent_id,
            capabilities=capabilities,
            issuer=parent_token.agent_id,  # Original agent is issuer
            secret_key=self.auth.secret_key,
            validity_seconds=validity_seconds,
            scope=parent_token.scope,
            metadata={"delegated_from": parent_token.token_id},
        )

        self._delegation_chains[token.token_id] = parent_token.token_id
        logger.info(
            f"Delegated token {token.token_id[:8]}... "
            f"from {parent_token.token_id[:8]}..."
        )

        return token

    def get_delegation_chain(self, token_id: str) -> List[str]:
        """Get the delegation chain for a token.

        Args:
            token_id: Token to trace

        Returns:
            List of token IDs in chain (root first)
        """
        chain = []
        current = token_id

        while current in self._delegation_chains:
            chain.append(current)
            current = self._delegation_chains[current]

        chain.append(current)  # Add root
        chain.reverse()

        return chain
