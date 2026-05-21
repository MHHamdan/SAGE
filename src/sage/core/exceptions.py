"""Custom exceptions for the agentic toolkit."""


class AgentError(Exception):
    """Base exception for agent-related errors."""

    def __init__(self, message: str, agent_name: str = None):
        self.agent_name = agent_name
        super().__init__(f"[{agent_name}] {message}" if agent_name else message)


class ToolExecutionError(AgentError):
    """Exception raised when tool execution fails."""

    def __init__(self, message: str, tool_name: str, agent_name: str = None):
        self.tool_name = tool_name
        super().__init__(f"Tool '{tool_name}' failed: {message}", agent_name)


class MemoryError(AgentError):
    """Exception raised for memory-related errors."""

    def __init__(self, message: str, memory_type: str = None, agent_name: str = None):
        self.memory_type = memory_type
        prefix = f"Memory[{memory_type}]" if memory_type else "Memory"
        super().__init__(f"{prefix}: {message}", agent_name)


class ConfigurationError(Exception):
    """Exception raised for configuration errors."""

    def __init__(self, message: str, config_key: str = None):
        self.config_key = config_key
        super().__init__(
            f"Configuration error for '{config_key}': {message}"
            if config_key
            else f"Configuration error: {message}"
        )


class ProtocolError(AgentError):
    """Exception raised for protocol-related errors (MCP, A2A)."""

    def __init__(self, message: str, protocol: str, agent_name: str = None):
        self.protocol = protocol
        super().__init__(f"Protocol[{protocol}]: {message}", agent_name)


class ContextOverflowError(AgentError):
    """Exception raised when context exceeds limits."""

    def __init__(
        self,
        message: str,
        current_tokens: int,
        max_tokens: int,
        agent_name: str = None,
    ):
        self.current_tokens = current_tokens
        self.max_tokens = max_tokens
        super().__init__(
            f"Context overflow ({current_tokens}/{max_tokens} tokens): {message}",
            agent_name,
        )


class EvaluationError(Exception):
    """Exception raised during agent evaluation."""

    def __init__(self, message: str, evaluator_name: str = None):
        self.evaluator_name = evaluator_name
        super().__init__(
            f"Evaluation[{evaluator_name}]: {message}"
            if evaluator_name
            else f"Evaluation error: {message}"
        )


class GuardrailViolationError(AgentError):
    """Exception raised when guardrails are violated."""

    def __init__(
        self,
        message: str,
        guardrail_type: str,
        severity: str = "warning",
        agent_name: str = None,
    ):
        self.guardrail_type = guardrail_type
        self.severity = severity
        super().__init__(
            f"Guardrail[{guardrail_type}] ({severity}): {message}", agent_name
        )
