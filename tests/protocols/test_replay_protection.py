"""Tests for replay protection."""

import pytest
import time


class TestReplayProtection:
    """Test replay protection functionality."""

    def test_valid_message(self):
        """Test valid message passes validation."""
        from sage.protocols.a2a import ReplayProtection, MessageMetadata

        protection = ReplayProtection()
        metadata = MessageMetadata.generate(sender_id="agent-1")

        # First message should be valid
        assert protection.validate(metadata)

    def test_replay_detection(self):
        """Test replay attack detection."""
        from sage.protocols.a2a import (
            ReplayProtection,
            MessageMetadata,
            ReplayAttackDetected,
        )

        protection = ReplayProtection()
        metadata = MessageMetadata.generate(sender_id="agent-1")

        # First use: valid
        protection.validate(metadata)

        # Second use: replay attack
        with pytest.raises(ReplayAttackDetected):
            protection.validate(metadata)

    def test_old_message_rejected(self):
        """Test old messages are rejected."""
        from sage.protocols.a2a import (
            ReplayProtection,
            MessageMetadata,
            ReplayAttackDetected,
        )

        protection = ReplayProtection(max_message_age_seconds=0.1)
        metadata = MessageMetadata.generate(sender_id="agent-1")
        metadata.timestamp = time.time() - 10  # 10 seconds ago

        with pytest.raises(ReplayAttackDetected):
            protection.validate(metadata)

    def test_future_message_rejected(self):
        """Test messages with future timestamps are rejected."""
        from sage.protocols.a2a import (
            ReplayProtection,
            MessageMetadata,
            ReplayAttackDetected,
        )

        protection = ReplayProtection(max_clock_skew_seconds=30)
        metadata = MessageMetadata.generate(sender_id="agent-1")
        metadata.timestamp = time.time() + 120  # 2 minutes in future

        with pytest.raises(ReplayAttackDetected):
            protection.validate(metadata)

    def test_sequence_validation(self):
        """Test sequence number validation."""
        from sage.protocols.a2a import (
            ReplayProtection,
            MessageMetadata,
            ReplayAttackDetected,
        )

        protection = ReplayProtection(track_sequences=True)

        # Send messages with increasing sequence numbers
        for seq in [1, 2, 3]:
            metadata = MessageMetadata.generate(sender_id="agent-1", sequence_number=seq)
            protection.validate(metadata)

        # Replay with old sequence should fail
        metadata = MessageMetadata.generate(sender_id="agent-1", sequence_number=2)
        with pytest.raises(ReplayAttackDetected):
            protection.validate(metadata)


class TestNonceManager:
    """Test nonce manager."""

    def test_nonce_tracking(self):
        """Test nonce tracking."""
        from sage.protocols.a2a import NonceManager

        manager = NonceManager()

        # First use: valid
        assert manager.check_and_add("nonce1", time.time())

        # Second use: replay
        from sage.protocols.a2a import ReplayAttackDetected
        with pytest.raises(ReplayAttackDetected):
            manager.check_and_add("nonce1", time.time())

    def test_nonce_cleanup(self):
        """Test old nonces are cleaned up."""
        from sage.protocols.a2a import NonceManager

        manager = NonceManager(max_age_seconds=0.1, cleanup_interval=0.05)

        # Add some nonces
        manager.check_and_add("nonce1", time.time() - 1)  # Already old

        # Wait for cleanup
        time.sleep(0.2)

        # Check that old nonces are gone
        # (implementation detail: we can re-add them if they were cleaned)
