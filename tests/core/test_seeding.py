"""Tests for deterministic seeding module."""

import random

import pytest


class TestSeeding:
    """Test seeding functionality."""

    def test_set_global_seed(self, seed):
        """Test that global seed produces deterministic results."""
        from sage.core.seeding import set_global_seed

        set_global_seed(seed)
        values1 = [random.random() for _ in range(5)]

        set_global_seed(seed)
        values2 = [random.random() for _ in range(5)]

        assert values1 == values2

    def test_deterministic_random(self, seed):
        """Test DeterministicRandom class."""
        from sage.core.seeding import DeterministicRandom

        rng = DeterministicRandom(seed)
        values1 = [rng.random() for _ in range(5)]

        rng2 = DeterministicRandom(seed)
        values2 = [rng2.random() for _ in range(5)]

        assert values1 == values2

    def test_derive_seed(self, seed):
        """Test seed derivation."""
        from sage.core.seeding import derive_seed

        derived1 = derive_seed(seed, "component_a")
        derived2 = derive_seed(seed, "component_b")

        assert derived1 != derived2
        assert derive_seed(seed, "component_a") == derived1

    def test_deterministic_choice(self, seed):
        """Test deterministic choice."""
        from sage.core.seeding import DeterministicRandom

        rng = DeterministicRandom(seed)
        items = ["a", "b", "c", "d", "e"]

        choices1 = [rng.choice(items) for _ in range(5)]

        rng2 = DeterministicRandom(seed)
        choices2 = [rng2.choice(items) for _ in range(5)]

        assert choices1 == choices2
