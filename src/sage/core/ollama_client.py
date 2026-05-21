"""Ollama-first LLM client for local inference.

This module provides a unified interface for Ollama models with fallback
support for API-based models (optional). Ollama is the default backend.
"""

import json
import logging
import time
from typing import Optional, List, Dict, Any, Union, Iterator, Callable
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
import httpx

logger = logging.getLogger(__name__)


@dataclass
class GenerationConfig:
    """Configuration for text generation."""
    temperature: float = 0.1
    top_p: float = 0.9
    max_tokens: int = 4096
    stop: Optional[List[str]] = None
    seed: Optional[int] = None

    def to_ollama_options(self) -> Dict[str, Any]:
        """Convert to Ollama options format."""
        opts = {
            "temperature": self.temperature,
            "top_p": self.top_p,
            "num_predict": self.max_tokens,
        }
        if self.stop:
            opts["stop"] = self.stop
        if self.seed is not None:
            opts["seed"] = self.seed
        return opts


@dataclass
class GenerationResult:
    """Result from text generation."""
    content: str
    model: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    latency_ms: float = 0.0
    raw_response: Optional[Dict[str, Any]] = None

    @property
    def estimated_cost(self) -> float:
        """Estimate cost (0 for local Ollama models)."""
        return 0.0


@dataclass
class Message:
    """Chat message."""
    role: str  # system, user, assistant
    content: str
    tool_calls: Optional[List[Dict]] = None
    tool_call_id: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        d = {"role": self.role, "content": self.content}
        if self.tool_calls:
            d["tool_calls"] = self.tool_calls
        if self.tool_call_id:
            d["tool_call_id"] = self.tool_call_id
        return d


class BaseLLMProvider(ABC):
    """Abstract base class for LLM providers."""

    @abstractmethod
    def generate(
        self,
        prompt: str,
        system: Optional[str] = None,
        config: Optional[GenerationConfig] = None,
    ) -> GenerationResult:
        """Generate text from a prompt."""
        pass

    @abstractmethod
    def chat(
        self,
        messages: List[Message],
        config: Optional[GenerationConfig] = None,
        tools: Optional[List[Dict]] = None,
    ) -> GenerationResult:
        """Chat completion with messages."""
        pass

    @abstractmethod
    def stream(
        self,
        prompt: str,
        system: Optional[str] = None,
        config: Optional[GenerationConfig] = None,
    ) -> Iterator[str]:
        """Stream text generation."""
        pass


class OllamaClient(BaseLLMProvider):
    """Ollama LLM client for local model inference.

    This is the default and recommended client for the toolkit.
    No API keys required - runs entirely locally.

    Example:
        >>> client = OllamaClient(model="llama3.1:8b")
        >>> result = client.generate("What is 2+2?")
        >>> print(result.content)
    """

    def __init__(
        self,
        model: str = "llama3.1:8b",
        base_url: str = "http://localhost:11434",
        timeout: float = 120.0,
        max_retries: int = 3,
        retry_delay: float = 1.0,
    ):
        """Initialize the Ollama client.

        Args:
            model: Ollama model name (e.g., llama3.1:8b, mistral, qwen2.5:14b)
            base_url: Ollama server URL
            timeout: Request timeout in seconds
            max_retries: Maximum retry attempts
            retry_delay: Delay between retries in seconds
        """
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self._client = httpx.Client(timeout=timeout)

        logger.info(f"Initialized OllamaClient with model={model}, base_url={base_url}")

    def _make_request(
        self,
        endpoint: str,
        payload: Dict[str, Any],
        stream: bool = False,
    ) -> Union[Dict[str, Any], Iterator[Dict[str, Any]]]:
        """Make HTTP request to Ollama API with retries."""
        url = f"{self.base_url}{endpoint}"

        for attempt in range(self.max_retries):
            try:
                if stream:
                    return self._stream_request(url, payload)

                response = self._client.post(url, json=payload)
                response.raise_for_status()
                return response.json()

            except httpx.HTTPError as e:
                logger.warning(f"Request failed (attempt {attempt + 1}): {e}")
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay * (attempt + 1))
                else:
                    raise RuntimeError(f"Ollama request failed after {self.max_retries} attempts: {e}")

    def _stream_request(
        self,
        url: str,
        payload: Dict[str, Any],
    ) -> Iterator[Dict[str, Any]]:
        """Stream response from Ollama."""
        payload["stream"] = True

        with self._client.stream("POST", url, json=payload) as response:
            response.raise_for_status()
            for line in response.iter_lines():
                if line:
                    yield json.loads(line)

    def _estimate_tokens(self, text: str) -> int:
        """Estimate token count (roughly 4 chars per token for English)."""
        return len(text) // 4

    def generate(
        self,
        prompt: str,
        system: Optional[str] = None,
        config: Optional[GenerationConfig] = None,
    ) -> GenerationResult:
        """Generate text from a prompt.

        Args:
            prompt: The input prompt
            system: Optional system message
            config: Generation configuration

        Returns:
            GenerationResult with content and metadata
        """
        config = config or GenerationConfig()

        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": config.to_ollama_options(),
        }

        if system:
            payload["system"] = system

        start_time = time.time()
        response = self._make_request("/api/generate", payload)
        latency_ms = (time.time() - start_time) * 1000

        content = response.get("response", "")

        # Get token counts from response or estimate
        prompt_tokens = response.get("prompt_eval_count", self._estimate_tokens(prompt))
        completion_tokens = response.get("eval_count", self._estimate_tokens(content))

        return GenerationResult(
            content=content,
            model=self.model,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
            latency_ms=latency_ms,
            raw_response=response,
        )

    def chat(
        self,
        messages: List[Message],
        config: Optional[GenerationConfig] = None,
        tools: Optional[List[Dict]] = None,
    ) -> GenerationResult:
        """Chat completion with messages.

        Args:
            messages: List of chat messages
            config: Generation configuration
            tools: Optional list of tools for function calling

        Returns:
            GenerationResult with content and metadata
        """
        config = config or GenerationConfig()

        payload = {
            "model": self.model,
            "messages": [m.to_dict() for m in messages],
            "stream": False,
            "options": config.to_ollama_options(),
        }

        if tools:
            payload["tools"] = tools

        start_time = time.time()
        response = self._make_request("/api/chat", payload)
        latency_ms = (time.time() - start_time) * 1000

        message = response.get("message", {})
        content = message.get("content", "")

        # Estimate tokens
        prompt_text = " ".join(m.content for m in messages)
        prompt_tokens = response.get("prompt_eval_count", self._estimate_tokens(prompt_text))
        completion_tokens = response.get("eval_count", self._estimate_tokens(content))

        result = GenerationResult(
            content=content,
            model=self.model,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
            latency_ms=latency_ms,
            raw_response=response,
        )

        return result

    def stream(
        self,
        prompt: str,
        system: Optional[str] = None,
        config: Optional[GenerationConfig] = None,
    ) -> Iterator[str]:
        """Stream text generation.

        Args:
            prompt: The input prompt
            system: Optional system message
            config: Generation configuration

        Yields:
            Text chunks as they are generated
        """
        config = config or GenerationConfig()

        payload = {
            "model": self.model,
            "prompt": prompt,
            "options": config.to_ollama_options(),
        }

        if system:
            payload["system"] = system

        for chunk in self._make_request("/api/generate", payload, stream=True):
            if "response" in chunk:
                yield chunk["response"]

    def list_models(self) -> List[Dict[str, Any]]:
        """List available Ollama models.

        Returns:
            List of model information dictionaries
        """
        response = self._client.get(f"{self.base_url}/api/tags")
        response.raise_for_status()
        return response.json().get("models", [])

    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the current model.

        Returns:
            Model information dictionary
        """
        payload = {"name": self.model}
        response = self._client.post(f"{self.base_url}/api/show", json=payload)
        response.raise_for_status()
        return response.json()

    def is_available(self) -> bool:
        """Check if Ollama server is available.

        Returns:
            True if server is reachable
        """
        try:
            response = self._client.get(f"{self.base_url}/api/tags")
            return response.status_code == 200
        except Exception:
            return False

    def close(self):
        """Close the HTTP client."""
        self._client.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


class UnifiedLLMClient:
    """Unified LLM client supporting multiple backends.

    Provides a consistent interface across Ollama (default) and optional
    API-based providers (OpenAI, Anthropic, etc.).

    Example:
        >>> # Default: Ollama
        >>> client = UnifiedLLMClient()
        >>>
        >>> # With OpenAI (optional)
        >>> client = UnifiedLLMClient(
        ...     provider="openai",
        ...     api_key="sk-...",
        ...     model="gpt-4o-mini"
        ... )
    """

    PROVIDERS = {
        "ollama": OllamaClient,
    }

    def __init__(
        self,
        provider: str = "ollama",
        model: Optional[str] = None,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        **kwargs,
    ):
        """Initialize the unified client.

        Args:
            provider: LLM provider ("ollama" is default and recommended)
            model: Model name (provider-specific)
            base_url: Base URL for the provider
            api_key: API key (only needed for API providers)
            **kwargs: Additional provider-specific arguments
        """
        self.provider_name = provider

        if provider == "ollama":
            model = model or "llama3.1:8b"
            base_url = base_url or "http://localhost:11434"
            self._provider = OllamaClient(
                model=model,
                base_url=base_url,
                **kwargs,
            )
        elif provider == "openai":
            # Optional OpenAI support - requires langchain-openai
            try:
                from .llm_client import LLMClient as OpenAILLMClient
                self._provider = OpenAILLMClient(
                    model=model or "gpt-4o-mini",
                    api_key=api_key,
                    **kwargs,
                )
                self._is_langchain = True
            except ImportError:
                raise ImportError(
                    "OpenAI provider requires langchain-openai. "
                    "Install with: pip install langchain-openai"
                )
        else:
            raise ValueError(f"Unknown provider: {provider}")

        self.model = model
        self._is_langchain = provider != "ollama"

    def generate(
        self,
        prompt: str,
        system: Optional[str] = None,
        config: Optional[GenerationConfig] = None,
        **kwargs,
    ) -> GenerationResult:
        """Generate text from a prompt.

        Args:
            prompt: The input prompt
            system: Optional system message
            config: Generation configuration
            **kwargs: Additional arguments

        Returns:
            GenerationResult with content and metadata
        """
        if self._is_langchain:
            # Wrap LangChain response
            from langchain_core.messages import HumanMessage, SystemMessage
            messages = []
            if system:
                messages.append(SystemMessage(content=system))
            messages.append(HumanMessage(content=prompt))

            start_time = time.time()
            response = self._provider.invoke(messages)
            latency_ms = (time.time() - start_time) * 1000

            return GenerationResult(
                content=response.content,
                model=self.model,
                prompt_tokens=len(prompt) // 4,
                completion_tokens=len(response.content) // 4,
                total_tokens=(len(prompt) + len(response.content)) // 4,
                latency_ms=latency_ms,
            )
        else:
            return self._provider.generate(prompt, system, config)

    def chat(
        self,
        messages: List[Message],
        config: Optional[GenerationConfig] = None,
        tools: Optional[List[Dict]] = None,
        **kwargs,
    ) -> GenerationResult:
        """Chat completion with messages.

        Args:
            messages: List of chat messages
            config: Generation configuration
            tools: Optional tools for function calling
            **kwargs: Additional arguments

        Returns:
            GenerationResult with content and metadata
        """
        if self._is_langchain:
            from langchain_core.messages import HumanMessage, SystemMessage, AIMessage

            lc_messages = []
            for m in messages:
                if m.role == "system":
                    lc_messages.append(SystemMessage(content=m.content))
                elif m.role == "user":
                    lc_messages.append(HumanMessage(content=m.content))
                elif m.role == "assistant":
                    lc_messages.append(AIMessage(content=m.content))

            start_time = time.time()
            response = self._provider.invoke(lc_messages)
            latency_ms = (time.time() - start_time) * 1000

            return GenerationResult(
                content=response.content,
                model=self.model,
                latency_ms=latency_ms,
            )
        else:
            return self._provider.chat(messages, config, tools)

    def stream(
        self,
        prompt: str,
        system: Optional[str] = None,
        config: Optional[GenerationConfig] = None,
    ) -> Iterator[str]:
        """Stream text generation.

        Args:
            prompt: The input prompt
            system: Optional system message
            config: Generation configuration

        Yields:
            Text chunks as they are generated
        """
        if self._is_langchain:
            for chunk in self._provider.stream(prompt):
                yield chunk.content
        else:
            yield from self._provider.stream(prompt, system, config)

    def is_available(self) -> bool:
        """Check if the provider is available."""
        if hasattr(self._provider, "is_available"):
            return self._provider.is_available()
        return True

    def close(self):
        """Close the client."""
        if hasattr(self._provider, "close"):
            self._provider.close()


def create_ollama_client(
    model: str = "llama3.1:8b",
    **kwargs,
) -> OllamaClient:
    """Factory function to create an Ollama client.

    Args:
        model: Ollama model name
        **kwargs: Additional arguments

    Returns:
        Configured OllamaClient
    """
    return OllamaClient(model=model, **kwargs)


def create_unified_client(
    provider: str = "ollama",
    **kwargs,
) -> UnifiedLLMClient:
    """Factory function to create a unified LLM client.

    Args:
        provider: LLM provider (default: ollama)
        **kwargs: Additional arguments

    Returns:
        Configured UnifiedLLMClient
    """
    return UnifiedLLMClient(provider=provider, **kwargs)
