"""Configuration management for the agentic toolkit."""

import os
from dataclasses import dataclass, field
from typing import Optional, Dict, Any

from dotenv import load_dotenv


@dataclass
class LLMConfig:
    """Configuration for LLM providers."""

    model: str = "gpt-4o-mini"
    temperature: float = 0.1
    max_tokens: int = 4096
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    timeout: float = 60.0
    max_retries: int = 3


@dataclass
class MemoryConfig:
    """Configuration for memory systems."""

    buffer_size: int = 10
    max_tokens: int = 4000
    vector_store_type: str = "chroma"
    embedding_model: str = "text-embedding-3-small"
    persist_directory: Optional[str] = None


@dataclass
class ObservabilityConfig:
    """Configuration for observability and tracing."""

    enabled: bool = True
    langsmith_api_key: Optional[str] = None
    project_name: str = "agentic-toolkit"
    trace_all: bool = True


@dataclass
class Config:
    """Main configuration class for the agentic toolkit.

    Example:
        >>> config = Config.from_env()
        >>> print(config.llm.model)
        'gpt-4o-mini'
    """

    llm: LLMConfig = field(default_factory=LLMConfig)
    memory: MemoryConfig = field(default_factory=MemoryConfig)
    observability: ObservabilityConfig = field(default_factory=ObservabilityConfig)

    # Additional settings
    debug: bool = False
    log_level: str = "INFO"

    @classmethod
    def from_env(cls, env_file: Optional[str] = None) -> "Config":
        """Create configuration from environment variables.

        Args:
            env_file: Optional path to .env file

        Returns:
            Config instance populated from environment
        """
        if env_file:
            load_dotenv(env_file)
        else:
            load_dotenv()

        llm_config = LLMConfig(
            model=os.getenv("LLM_MODEL", "gpt-4o-mini"),
            temperature=float(os.getenv("LLM_TEMPERATURE", "0.1")),
            max_tokens=int(os.getenv("LLM_MAX_TOKENS", "4096")),
            api_key=os.getenv("OPENAI_API_KEY"),
            timeout=float(os.getenv("LLM_TIMEOUT", "60.0")),
            max_retries=int(os.getenv("LLM_MAX_RETRIES", "3")),
        )

        memory_config = MemoryConfig(
            buffer_size=int(os.getenv("MEMORY_BUFFER_SIZE", "10")),
            max_tokens=int(os.getenv("MEMORY_MAX_TOKENS", "4000")),
            vector_store_type=os.getenv("VECTOR_STORE_TYPE", "chroma"),
            embedding_model=os.getenv("EMBEDDING_MODEL", "text-embedding-3-small"),
            persist_directory=os.getenv("VECTOR_PERSIST_DIR"),
        )

        observability_config = ObservabilityConfig(
            enabled=os.getenv("OBSERVABILITY_ENABLED", "true").lower() == "true",
            langsmith_api_key=os.getenv("LANGSMITH_API_KEY"),
            project_name=os.getenv("LANGSMITH_PROJECT", "agentic-toolkit"),
            trace_all=os.getenv("TRACE_ALL", "true").lower() == "true",
        )

        return cls(
            llm=llm_config,
            memory=memory_config,
            observability=observability_config,
            debug=os.getenv("DEBUG", "false").lower() == "true",
            log_level=os.getenv("LOG_LEVEL", "INFO"),
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary.

        Returns:
            Dictionary representation of configuration
        """
        return {
            "llm": {
                "model": self.llm.model,
                "temperature": self.llm.temperature,
                "max_tokens": self.llm.max_tokens,
                "timeout": self.llm.timeout,
                "max_retries": self.llm.max_retries,
            },
            "memory": {
                "buffer_size": self.memory.buffer_size,
                "max_tokens": self.memory.max_tokens,
                "vector_store_type": self.memory.vector_store_type,
                "embedding_model": self.memory.embedding_model,
            },
            "observability": {
                "enabled": self.observability.enabled,
                "project_name": self.observability.project_name,
                "trace_all": self.observability.trace_all,
            },
            "debug": self.debug,
            "log_level": self.log_level,
        }
