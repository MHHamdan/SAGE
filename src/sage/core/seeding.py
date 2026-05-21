"""Deterministic seeding for reproducible experiments.

This module provides utilities to ensure reproducible results across
all random number generators used in the toolkit.
"""

import random
import os
import hashlib
import logging
from typing import Optional, Dict, Any
from dataclasses import dataclass, field
from datetime import datetime

logger = logging.getLogger(__name__)


@dataclass
class SeedState:
    """Container for seed state information."""

    seed: int
    python_random: bool = True
    numpy: bool = False
    torch: bool = False
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for logging."""
        return {
            "seed": self.seed,
            "python_random": self.python_random,
            "numpy": self.numpy,
            "torch": self.torch,
            "timestamp": self.timestamp,
        }


_global_seed_state: Optional[SeedState] = None


def set_global_seed(seed: int = 42) -> SeedState:
    """Set seed for all random number generators.

    This function sets seeds for:
    - Python's random module
    - NumPy (if available)
    - PyTorch (if available)
    - Environment variable for hash seed

    Args:
        seed: The seed value (default: 42)

    Returns:
        SeedState with information about what was seeded

    Example:
        >>> from sage.core.seeding import set_global_seed
        >>> state = set_global_seed(42)
        >>> print(f"Seed set: {state.seed}")
    """
    global _global_seed_state

    state = SeedState(seed=seed)

    # Set Python random seed
    random.seed(seed)
    state.python_random = True
    logger.debug(f"Set Python random seed to {seed}")

    # Set hash seed (affects dict ordering in Python < 3.7)
    os.environ["PYTHONHASHSEED"] = str(seed)

    # Try to set NumPy seed
    try:
        import numpy as np

        np.random.seed(seed)
        state.numpy = True
        logger.debug(f"Set NumPy random seed to {seed}")
    except ImportError:
        pass

    # Try to set PyTorch seed
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        state.torch = True
        logger.debug(f"Set PyTorch random seed to {seed}")
    except ImportError:
        pass

    _global_seed_state = state
    logger.info(f"Global seed set to {seed}")

    return state


def get_global_seed() -> Optional[int]:
    """Get the current global seed.

    Returns:
        Current seed value or None if not set
    """
    if _global_seed_state:
        return _global_seed_state.seed
    return None


def get_seed_state() -> Optional[SeedState]:
    """Get the current seed state.

    Returns:
        Current SeedState or None if not set
    """
    return _global_seed_state


def derive_seed(base_seed: int, *components: str) -> int:
    """Derive a deterministic seed from a base seed and components.

    Useful for creating different but reproducible seeds for different
    parts of an experiment.

    Args:
        base_seed: The base seed value
        *components: String components to hash

    Returns:
        Derived seed value

    Example:
        >>> seed1 = derive_seed(42, "planner")
        >>> seed2 = derive_seed(42, "evaluator")
        >>> # seed1 != seed2, but both are deterministic
    """
    hash_input = f"{base_seed}:" + ":".join(components)
    hash_value = hashlib.sha256(hash_input.encode()).hexdigest()
    return int(hash_value[:8], 16)


def random_with_seed(seed: int):
    """Context manager for temporarily using a specific seed.

    Args:
        seed: Seed to use within the context

    Example:
        >>> with random_with_seed(123):
        ...     value = random.random()
        >>> # Original seed state is restored
    """
    return SeedContext(seed)


class SeedContext:
    """Context manager for temporary seed changes."""

    def __init__(self, seed: int):
        self.seed = seed
        self.old_state = None

    def __enter__(self):
        self.old_state = random.getstate()
        random.seed(self.seed)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.old_state:
            random.setstate(self.old_state)


class DeterministicRandom:
    """A deterministic random number generator for specific use cases.

    Provides isolation from the global random state.

    Example:
        >>> rng = DeterministicRandom(seed=42)
        >>> value1 = rng.random()
        >>> rng.reset()
        >>> value2 = rng.random()
        >>> assert value1 == value2
    """

    def __init__(self, seed: int = 42):
        """Initialize with a seed.

        Args:
            seed: The seed value
        """
        self.initial_seed = seed
        self._random = random.Random(seed)

    def random(self) -> float:
        """Generate a random float in [0.0, 1.0)."""
        return self._random.random()

    def randint(self, a: int, b: int) -> int:
        """Generate a random integer in [a, b]."""
        return self._random.randint(a, b)

    def choice(self, seq):
        """Choose a random element from a sequence."""
        return self._random.choice(seq)

    def shuffle(self, seq):
        """Shuffle a sequence in place."""
        self._random.shuffle(seq)

    def sample(self, population, k: int):
        """Return k unique elements from population."""
        return self._random.sample(population, k)

    def uniform(self, a: float, b: float) -> float:
        """Generate a random float in [a, b]."""
        return self._random.uniform(a, b)

    def gauss(self, mu: float, sigma: float) -> float:
        """Generate a random number from Gaussian distribution."""
        return self._random.gauss(mu, sigma)

    def reset(self):
        """Reset to initial seed state."""
        self._random = random.Random(self.initial_seed)

    def get_state(self):
        """Get current state."""
        return self._random.getstate()

    def set_state(self, state):
        """Set state."""
        self._random.setstate(state)


def get_reproducibility_info() -> Dict[str, Any]:
    """Get information for reproducibility reporting.

    Returns:
        Dictionary with version and seed information
    """
    import sys
    import platform

    info = {
        "python_version": sys.version,
        "platform": platform.platform(),
        "seed_state": _global_seed_state.to_dict() if _global_seed_state else None,
    }

    # Add package versions
    packages = ["numpy", "torch", "langchain", "httpx"]
    for pkg in packages:
        try:
            mod = __import__(pkg)
            info[f"{pkg}_version"] = getattr(mod, "__version__", "unknown")
        except ImportError:
            info[f"{pkg}_version"] = "not installed"

    return info
