"""
Experience Buffer for Agent Learning

Stores and manages agent experiences for replay-based learning.
Supports both standard and prioritized experience replay.

Experience buffers enable agents to learn from past interactions
by storing (state, action, reward, next_state) tuples for later
training.

Example:
    >>> buffer = ExperienceBuffer(max_size=10000)
    >>> buffer.add(Experience(
    ...     state={"context": "user query"},
    ...     action={"response": "helpful answer"},
    ...     reward=1.0,
    ...     next_state={"context": "follow-up"},
    ... ))
    >>>
    >>> batch = buffer.sample(batch_size=32)
    >>> for exp in batch:
    ...     train_on_experience(exp)
"""

from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List, Union
from datetime import datetime
from collections import deque
import random
import logging
import json

logger = logging.getLogger(__name__)


@dataclass
class Experience:
    """Single experience tuple for learning.

    Attributes:
        experience_id: Unique identifier
        timestamp: When experience was recorded
        state: State before action
        action: Action taken
        reward: Reward received
        next_state: State after action
        done: Whether episode ended
        metadata: Additional metadata
        priority: Priority for sampling (higher = more likely)
    """
    experience_id: str = ""
    timestamp: datetime = field(default_factory=datetime.now)
    state: Dict[str, Any] = field(default_factory=dict)
    action: Dict[str, Any] = field(default_factory=dict)
    reward: float = 0.0
    next_state: Optional[Dict[str, Any]] = None
    done: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)
    priority: float = 1.0

    def __post_init__(self):
        if not self.experience_id:
            self.experience_id = f"exp-{id(self)}"

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "experience_id": self.experience_id,
            "timestamp": self.timestamp.isoformat(),
            "state": self.state,
            "action": self.action,
            "reward": self.reward,
            "next_state": self.next_state,
            "done": self.done,
            "metadata": self.metadata,
            "priority": self.priority,
        }


@dataclass
class ExperienceBatch:
    """Batch of experiences for training.

    Attributes:
        experiences: List of experiences
        batch_id: Unique batch identifier
        sampled_at: When batch was sampled
    """
    experiences: List[Experience]
    batch_id: str = ""
    sampled_at: datetime = field(default_factory=datetime.now)

    def __post_init__(self):
        if not self.batch_id:
            self.batch_id = f"batch-{id(self)}"

    def __len__(self) -> int:
        return len(self.experiences)

    def __iter__(self):
        return iter(self.experiences)

    def get_states(self) -> List[Dict[str, Any]]:
        """Get all states in batch."""
        return [e.state for e in self.experiences]

    def get_actions(self) -> List[Dict[str, Any]]:
        """Get all actions in batch."""
        return [e.action for e in self.experiences]

    def get_rewards(self) -> List[float]:
        """Get all rewards in batch."""
        return [e.reward for e in self.experiences]

    def mean_reward(self) -> float:
        """Get mean reward of batch."""
        if not self.experiences:
            return 0.0
        return sum(e.reward for e in self.experiences) / len(self.experiences)


class ExperienceBuffer:
    """
    Fixed-size buffer for storing agent experiences.

    Implements a ring buffer that overwrites oldest experiences
    when capacity is reached. Supports random sampling for
    experience replay training.

    Example:
        >>> buffer = ExperienceBuffer(max_size=10000)
        >>>
        >>> # Add experiences
        >>> for state, action, reward, next_state in trajectory:
        ...     buffer.add(Experience(
        ...         state=state,
        ...         action=action,
        ...         reward=reward,
        ...         next_state=next_state
        ...     ))
        >>>
        >>> # Sample batch for training
        >>> batch = buffer.sample(32)
        >>> print(f"Batch mean reward: {batch.mean_reward():.3f}")

    Attributes:
        max_size: Maximum buffer capacity
    """

    def __init__(self, max_size: int = 10000):
        """Initialize experience buffer.

        Args:
            max_size: Maximum number of experiences to store
        """
        self.max_size = max_size
        self._buffer: deque = deque(maxlen=max_size)
        self._experience_counter = 0

    def add(self, experience: Experience) -> Experience:
        """Add an experience to the buffer.

        Args:
            experience: Experience to add

        Returns:
            Added experience with ID assigned
        """
        self._experience_counter += 1
        if not experience.experience_id or experience.experience_id.startswith("exp-"):
            experience.experience_id = f"exp-{self._experience_counter:08d}"

        self._buffer.append(experience)
        return experience

    def add_transition(
        self,
        state: Dict[str, Any],
        action: Dict[str, Any],
        reward: float,
        next_state: Optional[Dict[str, Any]] = None,
        done: bool = False,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Experience:
        """Add a transition as an experience.

        Convenience method for adding (s, a, r, s') tuples.

        Args:
            state: State before action
            action: Action taken
            reward: Reward received
            next_state: State after action
            done: Whether episode ended
            metadata: Additional metadata

        Returns:
            Created experience
        """
        experience = Experience(
            state=state,
            action=action,
            reward=reward,
            next_state=next_state,
            done=done,
            metadata=metadata or {},
        )
        return self.add(experience)

    def sample(self, batch_size: int) -> ExperienceBatch:
        """Sample a random batch of experiences.

        Args:
            batch_size: Number of experiences to sample

        Returns:
            ExperienceBatch with sampled experiences

        Raises:
            ValueError: If buffer has fewer than batch_size experiences
        """
        if len(self._buffer) < batch_size:
            raise ValueError(
                f"Buffer has {len(self._buffer)} experiences, "
                f"cannot sample {batch_size}"
            )

        sampled = random.sample(list(self._buffer), batch_size)
        return ExperienceBatch(experiences=sampled)

    def sample_recent(
        self,
        batch_size: int,
        recency_bias: float = 0.5
    ) -> ExperienceBatch:
        """Sample with bias toward recent experiences.

        Args:
            batch_size: Number of experiences to sample
            recency_bias: How much to favor recent experiences (0-1)

        Returns:
            ExperienceBatch with sampled experiences
        """
        if len(self._buffer) < batch_size:
            raise ValueError(f"Not enough experiences in buffer")

        # Create weights favoring recent experiences
        n = len(self._buffer)
        weights = [
            (1 - recency_bias) + recency_bias * (i / n)
            for i in range(n)
        ]

        sampled = random.choices(list(self._buffer), weights=weights, k=batch_size)
        return ExperienceBatch(experiences=sampled)

    def get_recent(self, n: int) -> List[Experience]:
        """Get the n most recent experiences.

        Args:
            n: Number of experiences to get

        Returns:
            List of recent experiences
        """
        return list(self._buffer)[-n:]

    def get_high_reward(
        self,
        n: int,
        min_reward: Optional[float] = None
    ) -> List[Experience]:
        """Get experiences with highest rewards.

        Args:
            n: Number of experiences to return
            min_reward: Minimum reward threshold

        Returns:
            List of high-reward experiences
        """
        experiences = list(self._buffer)

        if min_reward is not None:
            experiences = [e for e in experiences if e.reward >= min_reward]

        experiences.sort(key=lambda e: e.reward, reverse=True)
        return experiences[:n]

    def get_statistics(self) -> Dict[str, Any]:
        """Get buffer statistics.

        Returns:
            Statistics dictionary
        """
        if not self._buffer:
            return {
                "size": 0,
                "max_size": self.max_size,
                "utilization": 0.0,
            }

        rewards = [e.reward for e in self._buffer]
        return {
            "size": len(self._buffer),
            "max_size": self.max_size,
            "utilization": len(self._buffer) / self.max_size,
            "mean_reward": sum(rewards) / len(rewards),
            "min_reward": min(rewards),
            "max_reward": max(rewards),
            "done_count": sum(1 for e in self._buffer if e.done),
        }

    def clear(self) -> None:
        """Clear all experiences."""
        self._buffer.clear()
        self._experience_counter = 0

    def __len__(self) -> int:
        """Get number of experiences in buffer."""
        return len(self._buffer)

    def __iter__(self):
        """Iterate over experiences."""
        return iter(self._buffer)


class PrioritizedExperienceBuffer:
    """
    Experience buffer with prioritized sampling.

    Experiences are sampled proportional to their priority,
    allowing important experiences (high TD error, rare events)
    to be replayed more frequently.

    Example:
        >>> buffer = PrioritizedExperienceBuffer(max_size=10000)
        >>>
        >>> # Add with priority
        >>> buffer.add(experience, priority=2.0)
        >>>
        >>> # Sample prioritized batch
        >>> batch = buffer.sample(32)
        >>>
        >>> # Update priorities after learning
        >>> buffer.update_priorities(
        ...     {exp.experience_id: new_priority for exp in batch}
        ... )

    Attributes:
        max_size: Maximum buffer capacity
        alpha: Priority exponent (0 = uniform, 1 = full priority)
        beta: Importance sampling exponent
    """

    def __init__(
        self,
        max_size: int = 10000,
        alpha: float = 0.6,
        beta: float = 0.4,
        beta_increment: float = 0.001,
    ):
        """Initialize prioritized buffer.

        Args:
            max_size: Maximum capacity
            alpha: Priority exponent
            beta: Initial importance sampling exponent
            beta_increment: How much to increase beta per sample
        """
        self.max_size = max_size
        self.alpha = alpha
        self.beta = beta
        self.beta_increment = beta_increment

        self._buffer: Dict[str, Experience] = {}
        self._priorities: Dict[str, float] = {}
        self._experience_counter = 0
        self._max_priority = 1.0

    def add(
        self,
        experience: Experience,
        priority: Optional[float] = None
    ) -> Experience:
        """Add experience with priority.

        Args:
            experience: Experience to add
            priority: Initial priority (None uses max_priority)

        Returns:
            Added experience
        """
        self._experience_counter += 1
        if not experience.experience_id or experience.experience_id.startswith("exp-"):
            experience.experience_id = f"pexp-{self._experience_counter:08d}"

        # Use max priority for new experiences
        if priority is None:
            priority = self._max_priority

        # Remove oldest if at capacity
        if len(self._buffer) >= self.max_size:
            oldest_id = min(
                self._buffer.keys(),
                key=lambda k: self._buffer[k].timestamp
            )
            del self._buffer[oldest_id]
            del self._priorities[oldest_id]

        self._buffer[experience.experience_id] = experience
        self._priorities[experience.experience_id] = priority ** self.alpha

        return experience

    def sample(self, batch_size: int) -> ExperienceBatch:
        """Sample batch with priority-weighted probabilities.

        Args:
            batch_size: Number of experiences to sample

        Returns:
            ExperienceBatch with sampled experiences
        """
        if len(self._buffer) < batch_size:
            raise ValueError(f"Not enough experiences in buffer")

        # Calculate sampling probabilities
        total_priority = sum(self._priorities.values())
        probs = {
            k: p / total_priority
            for k, p in self._priorities.items()
        }

        # Sample
        ids = list(self._buffer.keys())
        weights = [probs[id_] for id_ in ids]
        sampled_ids = random.choices(ids, weights=weights, k=batch_size)

        experiences = [self._buffer[id_] for id_ in sampled_ids]

        # Increment beta
        self.beta = min(1.0, self.beta + self.beta_increment)

        return ExperienceBatch(experiences=experiences)

    def update_priorities(self, priority_updates: Dict[str, float]) -> None:
        """Update priorities for experiences.

        Args:
            priority_updates: Dict mapping experience_id to new priority
        """
        for exp_id, priority in priority_updates.items():
            if exp_id in self._priorities:
                self._priorities[exp_id] = priority ** self.alpha
                self._max_priority = max(self._max_priority, priority)

    def get_experience(self, experience_id: str) -> Optional[Experience]:
        """Get experience by ID."""
        return self._buffer.get(experience_id)

    def get_statistics(self) -> Dict[str, Any]:
        """Get buffer statistics."""
        if not self._buffer:
            return {
                "size": 0,
                "max_size": self.max_size,
                "utilization": 0.0,
            }

        priorities = list(self._priorities.values())
        rewards = [e.reward for e in self._buffer.values()]

        return {
            "size": len(self._buffer),
            "max_size": self.max_size,
            "utilization": len(self._buffer) / self.max_size,
            "mean_priority": sum(priorities) / len(priorities),
            "max_priority": self._max_priority,
            "mean_reward": sum(rewards) / len(rewards),
            "alpha": self.alpha,
            "beta": self.beta,
        }

    def clear(self) -> None:
        """Clear all experiences."""
        self._buffer.clear()
        self._priorities.clear()
        self._experience_counter = 0
        self._max_priority = 1.0

    def __len__(self) -> int:
        """Get number of experiences."""
        return len(self._buffer)
