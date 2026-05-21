"""Tests for cost tracking module."""

import pytest


class TestCostTracker:
    """Test cost tracking functionality."""

    def test_token_cost_calculation(self):
        """Test token cost calculation."""
        from sage.core.cost import CostTracker

        tracker = CostTracker()
        tracker.add_tokens("gpt-4o", input_tokens=1000, output_tokens=500)

        # GPT-4o pricing: $5/M input, $15/M output
        expected = (1000 * 5 / 1_000_000) + (500 * 15 / 1_000_000)
        assert abs(tracker.token_cost - expected) < 0.0001

    def test_ollama_zero_cost(self):
        """Test Ollama models have zero cost."""
        from sage.core.cost import CostTracker

        tracker = CostTracker()
        tracker.add_tokens("llama3.1:8b", input_tokens=10000, output_tokens=5000)

        assert tracker.token_cost == 0.0

    def test_tool_cost(self):
        """Test tool cost tracking."""
        from sage.core.cost import CostTracker

        tracker = CostTracker()
        tracker.add_tool_call("search", cost=0.01)
        tracker.add_tool_call("search", cost=0.01)
        tracker.add_tool_call("database_query", cost=0.05)

        assert tracker.tool_cost == 0.07
        assert tracker.tool_calls == 3

    def test_total_cost(self):
        """Test total cost calculation."""
        from sage.core.cost import CostTracker

        tracker = CostTracker()
        tracker.add_tokens("gpt-4o-mini", input_tokens=1000, output_tokens=500)
        tracker.add_tool_call("api_call", cost=0.02)
        tracker.add_human_intervention()

        total = tracker.calculate_total_cost()
        assert total > 0

    def test_cost_summary(self):
        """Test cost summary generation."""
        from sage.core.cost import CostTracker

        tracker = CostTracker()
        tracker.add_tokens("llama3.1:8b", input_tokens=500, output_tokens=200)
        tracker.add_tool_call("test_tool")

        summary = tracker.get_summary()

        assert "total_tokens" in summary
        assert "tool_calls" in summary
        assert "total_cost" in summary
