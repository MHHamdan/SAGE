"""Replay protection for A2A protocol.

Prevents replay attacks through nonce and timestamp validation.
"""

import logging
import time
import secrets
import threading
from typing import Optional, Dict, Set
from dataclasses import dataclass, field
from collections import OrderedDict

logger = logging.getLogger(__name__)


class ReplayAttackDetected(Exception):
    """Exception raised when a replay attack is detected."""
    pass


@dataclass
class MessageMetadata:
    """Metadata for replay protection."""
    message_id: str
    nonce: str
    timestamp: float
    sender_id: str
    sequence_number: Optional[int] = None

    @classmethod
    def generate(cls, sender_id: str, sequence_number: Optional[int] = None) -> "MessageMetadata":
        """Generate new message metadata.

        Args:
            sender_id: ID of the sending agent
            sequence_number: Optional sequence number

        Returns:
            New MessageMetadata
        """
        return cls(
            message_id=secrets.token_hex(16),
            nonce=secrets.token_hex(16),
            timestamp=time.time(),
            sender_id=sender_id,
            sequence_number=sequence_number,
        )


class NonceManager:
    """Manages nonces for replay protection.

    Features:
    - Nonce uniqueness checking
    - Time-based expiration
    - Memory-efficient cleanup
    """

    def __init__(
        self,
        max_age_seconds: float = 300.0,  # 5 minutes
        max_nonces: int = 100000,
        cleanup_interval: float = 60.0,
    ):
        """Initialize the nonce manager.

        Args:
            max_age_seconds: Maximum age for valid nonces
            max_nonces: Maximum number of nonces to store
            cleanup_interval: Interval for cleanup in seconds
        """
        self.max_age_seconds = max_age_seconds
        self.max_nonces = max_nonces
        self.cleanup_interval = cleanup_interval

        self._nonces: OrderedDict[str, float] = OrderedDict()  # nonce -> timestamp
        self._lock = threading.Lock()
        self._last_cleanup = time.time()

    def check_and_add(self, nonce: str, timestamp: float) -> bool:
        """Check if nonce is valid and add it.

        Args:
            nonce: Nonce to check
            timestamp: Associated timestamp

        Returns:
            True if nonce is valid (not seen before)

        Raises:
            ReplayAttackDetected: If nonce was already used
        """
        with self._lock:
            # Periodic cleanup
            if time.time() - self._last_cleanup > self.cleanup_interval:
                self._cleanup()

            # Check if nonce exists
            if nonce in self._nonces:
                raise ReplayAttackDetected(f"Nonce {nonce[:8]}... already used")

            # Add nonce
            self._nonces[nonce] = timestamp

            # Enforce max size
            while len(self._nonces) > self.max_nonces:
                self._nonces.popitem(last=False)

            return True

    def _cleanup(self):
        """Remove expired nonces."""
        cutoff = time.time() - self.max_age_seconds
        expired = [
            nonce for nonce, ts in self._nonces.items()
            if ts < cutoff
        ]

        for nonce in expired:
            del self._nonces[nonce]

        self._last_cleanup = time.time()
        logger.debug(f"Cleaned up {len(expired)} expired nonces")

    def clear(self):
        """Clear all stored nonces."""
        with self._lock:
            self._nonces.clear()


class ReplayProtection:
    """Replay protection for agent messages.

    Implements:
    - Nonce-based protection
    - Timestamp validation
    - Sequence number tracking (optional)
    - Clock skew tolerance

    Example:
        >>> protection = ReplayProtection()
        >>> metadata = MessageMetadata.generate(sender_id="agent-1")
        >>> protection.validate(metadata)  # First time: OK
        >>> protection.validate(metadata)  # Second time: raises ReplayAttackDetected
    """

    def __init__(
        self,
        max_clock_skew_seconds: float = 60.0,
        max_message_age_seconds: float = 300.0,
        track_sequences: bool = True,
    ):
        """Initialize replay protection.

        Args:
            max_clock_skew_seconds: Maximum allowed clock skew
            max_message_age_seconds: Maximum message age
            track_sequences: Whether to track sequence numbers
        """
        self.max_clock_skew_seconds = max_clock_skew_seconds
        self.max_message_age_seconds = max_message_age_seconds
        self.track_sequences = track_sequences

        self._nonce_manager = NonceManager(max_age_seconds=max_message_age_seconds)
        self._sequence_numbers: Dict[str, int] = {}  # sender_id -> last sequence
        self._lock = threading.Lock()

    def validate(self, metadata: MessageMetadata) -> bool:
        """Validate message metadata for replay protection.

        Args:
            metadata: Message metadata to validate

        Returns:
            True if message is valid

        Raises:
            ReplayAttackDetected: If replay attack detected
        """
        current_time = time.time()

        # Check timestamp isn't in the future (with skew tolerance)
        if metadata.timestamp > current_time + self.max_clock_skew_seconds:
            raise ReplayAttackDetected(
                f"Message timestamp {metadata.timestamp} is in the future"
            )

        # Check message isn't too old
        age = current_time - metadata.timestamp
        if age > self.max_message_age_seconds:
            raise ReplayAttackDetected(
                f"Message too old: {age:.1f}s > {self.max_message_age_seconds}s"
            )

        # Check nonce uniqueness
        self._nonce_manager.check_and_add(metadata.nonce, metadata.timestamp)

        # Check sequence number if tracking
        if self.track_sequences and metadata.sequence_number is not None:
            self._validate_sequence(metadata)

        return True

    def _validate_sequence(self, metadata: MessageMetadata):
        """Validate sequence number.

        Args:
            metadata: Message metadata

        Raises:
            ReplayAttackDetected: If sequence is invalid
        """
        with self._lock:
            last_seq = self._sequence_numbers.get(metadata.sender_id, -1)

            # Sequence must be strictly increasing
            if metadata.sequence_number <= last_seq:
                raise ReplayAttackDetected(
                    f"Sequence number {metadata.sequence_number} <= last seen {last_seq}"
                )

            self._sequence_numbers[metadata.sender_id] = metadata.sequence_number

    def reset_sender(self, sender_id: str):
        """Reset tracking for a sender (e.g., on reconnection).

        Args:
            sender_id: Sender to reset
        """
        with self._lock:
            if sender_id in self._sequence_numbers:
                del self._sequence_numbers[sender_id]

    def get_last_sequence(self, sender_id: str) -> Optional[int]:
        """Get last seen sequence number for a sender.

        Args:
            sender_id: Sender to query

        Returns:
            Last sequence number or None
        """
        return self._sequence_numbers.get(sender_id)


class WindowedReplayProtection:
    """Sliding window replay protection.

    More memory-efficient for high-throughput scenarios.
    Uses a sliding window of recent message IDs.
    """

    def __init__(
        self,
        window_size: int = 1000,
        max_message_age_seconds: float = 300.0,
    ):
        """Initialize windowed protection.

        Args:
            window_size: Size of the sliding window
            max_message_age_seconds: Maximum message age
        """
        self.window_size = window_size
        self.max_message_age_seconds = max_message_age_seconds

        self._window: OrderedDict[str, float] = OrderedDict()
        self._lock = threading.Lock()

    def validate(self, message_id: str, timestamp: float) -> bool:
        """Validate a message ID.

        Args:
            message_id: Unique message identifier
            timestamp: Message timestamp

        Returns:
            True if valid

        Raises:
            ReplayAttackDetected: If replay detected
        """
        current_time = time.time()

        # Check age
        if current_time - timestamp > self.max_message_age_seconds:
            raise ReplayAttackDetected("Message too old")

        with self._lock:
            # Check if in window
            if message_id in self._window:
                raise ReplayAttackDetected(f"Message {message_id[:8]}... already seen")

            # Add to window
            self._window[message_id] = timestamp

            # Trim window
            while len(self._window) > self.window_size:
                self._window.popitem(last=False)

            return True
