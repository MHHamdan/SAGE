"""Tests for capability authentication."""

import time

import pytest


class TestCapabilityToken:
    """Test capability token creation and validation."""

    def test_token_creation(self):
        """Test token creation with capabilities."""
        from sage.protocols.a2a import CapabilityAuth, CapabilityToken

        auth = CapabilityAuth(secret_key="test-secret")
        auth.grant_capability("agent-1", "read")
        auth.grant_capability("agent-1", "write")

        token = auth.issue_token(
            agent_id="agent-1",
            capabilities=["read", "write"],
            validity_seconds=3600,
        )

        assert token.agent_id == "agent-1"
        assert "read" in token.capabilities
        assert "write" in token.capabilities
        assert not token.is_expired()

    def test_token_validation(self):
        """Test token signature validation."""
        from sage.protocols.a2a import CapabilityAuth, TokenStatus

        auth = CapabilityAuth(secret_key="test-secret")

        token = auth.issue_token(
            agent_id="agent-1",
            capabilities=["read"],
        )

        status = auth.validate_token(token)
        assert status == TokenStatus.VALID

    def test_invalid_signature_rejected(self):
        """Test that tampered tokens are rejected."""
        from sage.protocols.a2a import CapabilityAuth, TokenStatus

        auth = CapabilityAuth(secret_key="test-secret")

        token = auth.issue_token(
            agent_id="agent-1",
            capabilities=["read"],
        )

        # Tamper with the token
        token.signature = "tampered-signature"

        status = auth.validate_token(token)
        assert status == TokenStatus.INVALID

    def test_expired_token_rejected(self):
        """Test that expired tokens are rejected."""
        from sage.protocols.a2a import CapabilityAuth, TokenStatus

        auth = CapabilityAuth(secret_key="test-secret")

        token = auth.issue_token(
            agent_id="agent-1",
            capabilities=["read"],
            validity_seconds=0.01,  # Very short validity
        )

        time.sleep(0.1)  # Wait for expiration

        status = auth.validate_token(token)
        assert status == TokenStatus.EXPIRED

    def test_revoked_token_rejected(self):
        """Test that revoked tokens are rejected."""
        from sage.protocols.a2a import CapabilityAuth, TokenStatus

        auth = CapabilityAuth(secret_key="test-secret")

        token = auth.issue_token(
            agent_id="agent-1",
            capabilities=["read"],
        )

        auth.revoke_token(token.token_id)

        status = auth.validate_token(token)
        assert status == TokenStatus.REVOKED

    def test_capability_check(self):
        """Test capability checking."""
        from sage.protocols.a2a import CapabilityAuth, CapabilityAuthError

        auth = CapabilityAuth(secret_key="test-secret")

        token = auth.issue_token(
            agent_id="agent-1",
            capabilities=["read", "search"],
        )

        # Valid capability
        assert auth.check_capability(token, "read")
        assert auth.check_capability(token, "search")

        # Invalid capability
        with pytest.raises(CapabilityAuthError):
            auth.check_capability(token, "delete")


class TestAgentCardValidation:
    """Test agent card validation."""

    def test_valid_card(self):
        """Test valid agent card passes validation."""
        from sage.protocols.a2a import AgentCard, AgentCardValidator

        card = AgentCard(
            agent_id="agent-001",
            name="Test Agent",
            trust_level=0.8,
        )

        validator = AgentCardValidator()
        assert validator.validate(card)

    def test_expired_card_rejected(self):
        """Test expired cards are rejected."""
        from datetime import datetime, timedelta

        from sage.protocols.a2a import (
            AgentCard,
            AgentCardError,
            AgentCardValidator,
        )

        expired = (datetime.utcnow() - timedelta(days=1)).isoformat()
        card = AgentCard(
            agent_id="agent-001",
            name="Expired Agent",
            expires_at=expired,
        )

        validator = AgentCardValidator()
        with pytest.raises(AgentCardError, match="expired"):
            validator.validate(card)

    def test_low_trust_rejected(self):
        """Test low trust cards are rejected when threshold set."""
        from sage.protocols.a2a import (
            AgentCard,
            AgentCardError,
            AgentCardValidator,
        )

        card = AgentCard(
            agent_id="agent-001",
            name="Low Trust Agent",
            trust_level=0.3,
        )

        validator = AgentCardValidator(min_trust_level=0.5)
        with pytest.raises(AgentCardError, match="Trust level"):
            validator.validate(card)

    def test_capability_validation(self):
        """Test capability request validation."""
        from sage.protocols.a2a import (
            AgentCapability,
            AgentCard,
            AgentCardError,
            AgentCardValidator,
        )

        card = AgentCard(
            agent_id="agent-001",
            name="Limited Agent",
            capabilities=[
                AgentCapability(name="read"),
                AgentCapability(name="search"),
            ],
        )

        validator = AgentCardValidator()

        # Valid capability
        assert validator.validate_capability_request(card, "read")

        # Invalid capability
        with pytest.raises(AgentCardError, match="doesn't have capability"):
            validator.validate_capability_request(card, "delete")
