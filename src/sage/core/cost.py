"""Cost tracking for agentic AI operations.

This module provides comprehensive cost tracking including:
- Token usage (estimated or actual)
- Tool invocation costs
- Latency measurements
- Human intervention tracking

For Ollama (local models), token costs are $0, but usage is still tracked
for comparison with API models.
"""

import time
import logging
from typing import Optional, Dict, Any, List
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime
import threading

logger = logging.getLogger(__name__)


class CostCategory(Enum):
    """Categories of costs in agentic systems."""
    TOKEN_INPUT = "token_input"
    TOKEN_OUTPUT = "token_output"
    TOOL_CALL = "tool_call"
    LATENCY = "latency"
    HUMAN_INTERVENTION = "human_intervention"
    API_CALL = "api_call"


@dataclass
class TokenUsage:
    """Token usage record."""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0

    def __add__(self, other: "TokenUsage") -> "TokenUsage":
        return TokenUsage(
            prompt_tokens=self.prompt_tokens + other.prompt_tokens,
            completion_tokens=self.completion_tokens + other.completion_tokens,
            total_tokens=self.total_tokens + other.total_tokens,
        )


@dataclass
class CostRecord:
    """Single cost record."""
    category: CostCategory
    value: float
    unit: str
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CostSummary:
    """Summary of all costs."""
    total_tokens: TokenUsage = field(default_factory=TokenUsage)
    total_tool_calls: int = 0
    total_latency_ms: float = 0.0
    total_human_interventions: int = 0
    total_api_calls: int = 0
    estimated_cost_usd: float = 0.0
    records: List[CostRecord] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "total_tokens": {
                "prompt": self.total_tokens.prompt_tokens,
                "completion": self.total_tokens.completion_tokens,
                "total": self.total_tokens.total_tokens,
            },
            "total_tool_calls": self.total_tool_calls,
            "total_latency_ms": self.total_latency_ms,
            "total_human_interventions": self.total_human_interventions,
            "total_api_calls": self.total_api_calls,
            "estimated_cost_usd": self.estimated_cost_usd,
            "num_records": len(self.records),
        }


# Pricing for common models (per 1K tokens)
# Ollama models have $0 cost since they run locally
MODEL_PRICING = {
    # Ollama (local - free)
    "llama3.1:8b": {"input": 0.0, "output": 0.0},
    "llama3.2:3b": {"input": 0.0, "output": 0.0},
    "llama3.2:latest": {"input": 0.0, "output": 0.0},
    "qwen2.5:14b": {"input": 0.0, "output": 0.0},
    "mistral:latest": {"input": 0.0, "output": 0.0},
    "phi3:latest": {"input": 0.0, "output": 0.0},
    "gemma2:2b": {"input": 0.0, "output": 0.0},
    "llava:7b": {"input": 0.0, "output": 0.0},
    # OpenAI (API - paid)
    "gpt-4o-mini": {"input": 0.00015, "output": 0.0006},
    "gpt-4o": {"input": 0.005, "output": 0.015},
    "gpt-4-turbo": {"input": 0.01, "output": 0.03},
    "gpt-3.5-turbo": {"input": 0.0005, "output": 0.0015},
    # Default for unknown models
    "default": {"input": 0.0, "output": 0.0},
}


class CostTracker:
    """Track costs across agent operations.

    Thread-safe tracker for accumulating costs throughout an experiment.

    Example:
        >>> tracker = CostTracker(model="llama3.1:8b")
        >>> tracker.record_tokens(prompt_tokens=100, completion_tokens=50)
        >>> tracker.record_tool_call("search", latency_ms=150)
        >>> summary = tracker.get_summary()
        >>> print(f"Total tokens: {summary.total_tokens.total_tokens}")
    """

    def __init__(
        self,
        model: str = "llama3.1:8b",
        custom_pricing: Optional[Dict[str, Dict[str, float]]] = None,
    ):
        """Initialize the cost tracker.

        Args:
            model: Model name for pricing lookup
            custom_pricing: Optional custom pricing overrides
        """
        self.model = model
        self._lock = threading.Lock()
        self._records: List[CostRecord] = []
        self._tokens = TokenUsage()
        self._tool_calls = 0
        self._latency_ms = 0.0
        self._human_interventions = 0
        self._api_calls = 0

        # Merge custom pricing
        self._pricing = MODEL_PRICING.copy()
        if custom_pricing:
            self._pricing.update(custom_pricing)

    def _get_pricing(self) -> Dict[str, float]:
        """Get pricing for current model."""
        return self._pricing.get(self.model, self._pricing["default"])

    def record_tokens(
        self,
        prompt_tokens: int = 0,
        completion_tokens: int = 0,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> float:
        """Record token usage.

        Args:
            prompt_tokens: Number of input tokens
            completion_tokens: Number of output tokens
            metadata: Optional additional metadata

        Returns:
            Estimated cost in USD
        """
        with self._lock:
            self._tokens.prompt_tokens += prompt_tokens
            self._tokens.completion_tokens += completion_tokens
            self._tokens.total_tokens += prompt_tokens + completion_tokens

            pricing = self._get_pricing()
            cost = (
                (prompt_tokens / 1000) * pricing["input"] +
                (completion_tokens / 1000) * pricing["output"]
            )

            self._records.append(CostRecord(
                category=CostCategory.TOKEN_INPUT,
                value=prompt_tokens,
                unit="tokens",
                metadata={"completion_tokens": completion_tokens, **(metadata or {})},
            ))

            self._api_calls += 1
            return cost

    def record_tool_call(
        self,
        tool_name: str,
        latency_ms: float = 0.0,
        success: bool = True,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        """Record a tool invocation.

        Args:
            tool_name: Name of the tool called
            latency_ms: Latency of the call in milliseconds
            success: Whether the call succeeded
            metadata: Optional additional metadata
        """
        with self._lock:
            self._tool_calls += 1
            self._latency_ms += latency_ms

            self._records.append(CostRecord(
                category=CostCategory.TOOL_CALL,
                value=latency_ms,
                unit="ms",
                metadata={
                    "tool_name": tool_name,
                    "success": success,
                    **(metadata or {}),
                },
            ))

    def record_latency(
        self,
        latency_ms: float,
        operation: str = "unknown",
        metadata: Optional[Dict[str, Any]] = None,
    ):
        """Record latency for an operation.

        Args:
            latency_ms: Latency in milliseconds
            operation: Name of the operation
            metadata: Optional additional metadata
        """
        with self._lock:
            self._latency_ms += latency_ms

            self._records.append(CostRecord(
                category=CostCategory.LATENCY,
                value=latency_ms,
                unit="ms",
                metadata={"operation": operation, **(metadata or {})},
            ))

    def record_human_intervention(
        self,
        reason: str = "approval_required",
        metadata: Optional[Dict[str, Any]] = None,
    ):
        """Record a human intervention event.

        Args:
            reason: Reason for intervention
            metadata: Optional additional metadata
        """
        with self._lock:
            self._human_interventions += 1

            self._records.append(CostRecord(
                category=CostCategory.HUMAN_INTERVENTION,
                value=1,
                unit="count",
                metadata={"reason": reason, **(metadata or {})},
            ))

    def get_summary(self) -> CostSummary:
        """Get cost summary.

        Returns:
            CostSummary with all tracked costs
        """
        with self._lock:
            pricing = self._get_pricing()
            estimated_cost = (
                (self._tokens.prompt_tokens / 1000) * pricing["input"] +
                (self._tokens.completion_tokens / 1000) * pricing["output"]
            )

            return CostSummary(
                total_tokens=TokenUsage(
                    prompt_tokens=self._tokens.prompt_tokens,
                    completion_tokens=self._tokens.completion_tokens,
                    total_tokens=self._tokens.total_tokens,
                ),
                total_tool_calls=self._tool_calls,
                total_latency_ms=self._latency_ms,
                total_human_interventions=self._human_interventions,
                total_api_calls=self._api_calls,
                estimated_cost_usd=estimated_cost,
                records=self._records.copy(),
            )

    def reset(self):
        """Reset all tracked costs."""
        with self._lock:
            self._records = []
            self._tokens = TokenUsage()
            self._tool_calls = 0
            self._latency_ms = 0.0
            self._human_interventions = 0
            self._api_calls = 0


class CostTimer:
    """Context manager for timing operations.

    Example:
        >>> tracker = CostTracker()
        >>> with CostTimer(tracker, "planning"):
        ...     # Do planning work
        ...     pass
    """

    def __init__(
        self,
        tracker: CostTracker,
        operation: str,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        self.tracker = tracker
        self.operation = operation
        self.metadata = metadata or {}
        self.start_time = None

    def __enter__(self):
        self.start_time = time.time()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        elapsed_ms = (time.time() - self.start_time) * 1000
        self.tracker.record_latency(
            latency_ms=elapsed_ms,
            operation=self.operation,
            metadata={
                "success": exc_type is None,
                **self.metadata,
            },
        )


def estimate_tokens(text: str, method: str = "char") -> int:
    """Estimate token count from text.

    Args:
        text: Input text
        method: Estimation method ("char" or "word")

    Returns:
        Estimated token count
    """
    if method == "char":
        # Roughly 4 characters per token for English
        return len(text) // 4
    elif method == "word":
        # Roughly 0.75 tokens per word
        words = len(text.split())
        return int(words * 0.75)
    else:
        return len(text) // 4


def calculate_total_cost(
    token_usage: TokenUsage,
    tool_calls: int,
    latency_ms: float,
    human_interventions: int,
    model: str = "llama3.1:8b",
    weights: Optional[Dict[str, float]] = None,
) -> Dict[str, float]:
    """Calculate total cost with decomposition.

    Implements the cost equation from the paper:
    Total Cost = Token Cost + Tool Cost + Latency Cost + Human Cost

    Args:
        token_usage: Token usage record
        tool_calls: Number of tool calls
        latency_ms: Total latency
        human_interventions: Number of human interventions
        model: Model name for pricing
        weights: Optional custom weights for cost components

    Returns:
        Dictionary with cost breakdown
    """
    weights = weights or {
        "token": 1.0,
        "tool": 0.001,  # $0.001 per tool call (negligible for local)
        "latency": 0.00001,  # $0.00001 per ms
        "human": 1.0,  # $1.00 per intervention (time cost)
    }

    pricing = MODEL_PRICING.get(model, MODEL_PRICING["default"])

    token_cost = (
        (token_usage.prompt_tokens / 1000) * pricing["input"] +
        (token_usage.completion_tokens / 1000) * pricing["output"]
    ) * weights["token"]

    tool_cost = tool_calls * weights["tool"]
    latency_cost = latency_ms * weights["latency"]
    human_cost = human_interventions * weights["human"]

    return {
        "token_cost": token_cost,
        "tool_cost": tool_cost,
        "latency_cost": latency_cost,
        "human_cost": human_cost,
        "total_cost": token_cost + tool_cost + latency_cost + human_cost,
    }
