"""LLM client wrapper for unified model access."""

from typing import Optional, List, Dict, Any, Union
from abc import ABC, abstractmethod

from langchain_openai import ChatOpenAI
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage, AIMessage


class BaseLLMClient(ABC):
    """Abstract base class for LLM clients."""

    @abstractmethod
    def invoke(
        self, messages: List[BaseMessage], **kwargs
    ) -> Union[str, BaseMessage]:
        """Invoke the LLM with messages."""
        pass

    @abstractmethod
    def bind_tools(self, tools: List[Any]) -> "BaseLLMClient":
        """Bind tools to the LLM."""
        pass


class LLMClient(BaseLLMClient):
    """Unified LLM client supporting multiple providers.

    This client provides a consistent interface for interacting with
    different LLM providers (OpenAI, Google, etc.).

    Example:
        >>> client = LLMClient(model="gpt-4o-mini", api_key="...")
        >>> response = client.invoke([HumanMessage(content="Hello")])
        >>> print(response.content)
    """

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        api_key: Optional[str] = None,
        temperature: float = 0.1,
        max_tokens: int = 4096,
        **kwargs,
    ):
        """Initialize the LLM client.

        Args:
            model: Model name to use
            api_key: API key for the provider
            temperature: Sampling temperature (0.0 to 1.0)
            max_tokens: Maximum tokens in response
            **kwargs: Additional provider-specific arguments
        """
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens

        # Initialize the underlying client based on model prefix
        if model.startswith("gpt") or model.startswith("o1"):
            self._client = ChatOpenAI(
                model=model,
                api_key=api_key,
                temperature=temperature,
                max_tokens=max_tokens,
                **kwargs,
            )
        else:
            # Default to OpenAI-compatible client
            self._client = ChatOpenAI(
                model=model,
                api_key=api_key,
                temperature=temperature,
                max_tokens=max_tokens,
                **kwargs,
            )

        self._tools_bound = False

    def invoke(
        self,
        messages: Union[List[BaseMessage], str],
        **kwargs,
    ) -> AIMessage:
        """Invoke the LLM with messages.

        Args:
            messages: List of messages or single string query
            **kwargs: Additional invocation arguments

        Returns:
            AI response message
        """
        if isinstance(messages, str):
            messages = [HumanMessage(content=messages)]

        return self._client.invoke(messages, **kwargs)

    def bind_tools(self, tools: List[Any]) -> "LLMClient":
        """Bind tools to the LLM for function calling.

        Args:
            tools: List of tool definitions

        Returns:
            Self with tools bound
        """
        self._client = self._client.bind_tools(tools)
        self._tools_bound = True
        return self

    def stream(
        self,
        messages: Union[List[BaseMessage], str],
        **kwargs,
    ):
        """Stream LLM responses.

        Args:
            messages: List of messages or single string query
            **kwargs: Additional arguments

        Yields:
            Response chunks
        """
        if isinstance(messages, str):
            messages = [HumanMessage(content=messages)]

        for chunk in self._client.stream(messages, **kwargs):
            yield chunk

    async def ainvoke(
        self,
        messages: Union[List[BaseMessage], str],
        **kwargs,
    ) -> AIMessage:
        """Async invoke the LLM.

        Args:
            messages: List of messages or single string query
            **kwargs: Additional arguments

        Returns:
            AI response message
        """
        if isinstance(messages, str):
            messages = [HumanMessage(content=messages)]

        return await self._client.ainvoke(messages, **kwargs)

    @property
    def has_tools(self) -> bool:
        """Check if tools are bound."""
        return self._tools_bound

    def get_model_info(self) -> Dict[str, Any]:
        """Get model information.

        Returns:
            Dictionary with model details
        """
        return {
            "model": self.model,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "tools_bound": self._tools_bound,
        }


def create_llm_client(
    model: str = "gpt-4o-mini",
    api_key: Optional[str] = None,
    **kwargs,
) -> LLMClient:
    """Factory function to create an LLM client.

    Args:
        model: Model name
        api_key: API key
        **kwargs: Additional arguments

    Returns:
        Configured LLM client
    """
    return LLMClient(model=model, api_key=api_key, **kwargs)
