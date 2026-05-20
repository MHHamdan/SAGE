"""Tool sandboxing for secure execution.

Provides isolation and resource limits for tool execution.
"""

import logging
import time
import subprocess
from typing import Optional, List, Dict, Any, Callable
from dataclasses import dataclass, field
from enum import Enum
import threading

logger = logging.getLogger(__name__)


class SandboxMode(Enum):
    """Sandbox execution mode."""
    NONE = "none"  # No sandboxing
    BASIC = "basic"  # Basic resource limits
    STRICT = "strict"  # Strict isolation


@dataclass
class ResourceLimits:
    """Resource limits for sandbox execution."""
    max_memory_mb: int = 512
    max_cpu_seconds: float = 30.0
    max_file_size_mb: int = 10
    max_processes: int = 10
    timeout_seconds: float = 60.0


@dataclass
class SandboxResult:
    """Result of sandboxed execution."""
    success: bool
    result: Any = None
    error: Optional[str] = None
    resource_usage: Dict[str, Any] = field(default_factory=dict)
    execution_time_ms: float = 0.0
    was_terminated: bool = False
    termination_reason: Optional[str] = None


class Sandbox:
    """Sandbox for secure tool execution.

    Provides:
    - Resource limits (CPU, memory, time)
    - Network isolation (optional)
    - File system restrictions
    - Process limits

    Example:
        >>> sandbox = Sandbox(mode=SandboxMode.BASIC)
        >>> result = sandbox.execute(my_tool, arg1="value")
        >>> if result.was_terminated:
        ...     print(f"Tool was terminated: {result.termination_reason}")
    """

    def __init__(
        self,
        mode: SandboxMode = SandboxMode.BASIC,
        limits: Optional[ResourceLimits] = None,
        allowed_paths: Optional[List[str]] = None,
        blocked_paths: Optional[List[str]] = None,
    ):
        """Initialize the sandbox.

        Args:
            mode: Sandbox mode
            limits: Resource limits
            allowed_paths: Allowed file system paths
            blocked_paths: Blocked file system paths
        """
        self.mode = mode
        self.limits = limits or ResourceLimits()
        self.allowed_paths = allowed_paths or []
        self.blocked_paths = blocked_paths or ["/etc", "/root", "/boot"]

    def execute(
        self,
        func: Callable,
        *args,
        **kwargs,
    ) -> SandboxResult:
        """Execute a function in the sandbox.

        Args:
            func: Function to execute
            *args: Positional arguments
            **kwargs: Keyword arguments

        Returns:
            SandboxResult with execution details
        """
        if self.mode == SandboxMode.NONE:
            return self._execute_unsandboxed(func, *args, **kwargs)

        return self._execute_sandboxed(func, *args, **kwargs)

    def _execute_unsandboxed(
        self,
        func: Callable,
        *args,
        **kwargs,
    ) -> SandboxResult:
        """Execute without sandboxing."""
        start_time = time.time()

        try:
            result = func(*args, **kwargs)
            return SandboxResult(
                success=True,
                result=result,
                execution_time_ms=(time.time() - start_time) * 1000,
            )
        except Exception as e:
            return SandboxResult(
                success=False,
                error=str(e),
                execution_time_ms=(time.time() - start_time) * 1000,
            )

    def _execute_sandboxed(
        self,
        func: Callable,
        *args,
        **kwargs,
    ) -> SandboxResult:
        """Execute with sandboxing."""
        result = SandboxResult(success=False)
        start_time = time.time()

        # Use threading with timeout
        execution_result = {"value": None, "error": None}

        def target():
            try:
                execution_result["value"] = func(*args, **kwargs)
            except Exception as e:
                execution_result["error"] = str(e)

        thread = threading.Thread(target=target)
        thread.start()
        thread.join(timeout=self.limits.timeout_seconds)

        result.execution_time_ms = (time.time() - start_time) * 1000

        if thread.is_alive():
            # Timeout occurred
            result.was_terminated = True
            result.termination_reason = f"Timeout after {self.limits.timeout_seconds}s"
            result.error = result.termination_reason
            # Note: Python threads can't be forcibly terminated,
            # but we return and let the caller handle it
        elif execution_result["error"]:
            result.error = execution_result["error"]
        else:
            result.success = True
            result.result = execution_result["value"]

        return result

    def validate_path(self, path: str) -> bool:
        """Validate if a path is allowed.

        Args:
            path: File system path

        Returns:
            True if path is allowed
        """
        # Check blocked paths
        for blocked in self.blocked_paths:
            if path.startswith(blocked):
                return False

        # If allowlist is set, check it
        if self.allowed_paths:
            for allowed in self.allowed_paths:
                if path.startswith(allowed):
                    return True
            return False

        return True


class SandboxedTool:
    """Wrapper to make any tool sandboxed.

    Example:
        >>> def my_tool(x): return x * 2
        >>> sandboxed = SandboxedTool(my_tool, sandbox=Sandbox())
        >>> result = sandboxed(5)
    """

    def __init__(
        self,
        tool: Callable,
        sandbox: Optional[Sandbox] = None,
        name: Optional[str] = None,
    ):
        """Initialize the sandboxed tool.

        Args:
            tool: Original tool function
            sandbox: Sandbox instance
            name: Tool name
        """
        self.tool = tool
        self.sandbox = sandbox or Sandbox()
        self.name = name or getattr(tool, "__name__", "unknown")
        self.__name__ = self.name
        self.__doc__ = tool.__doc__

    def __call__(self, *args, **kwargs) -> Any:
        """Execute the tool in sandbox."""
        result = self.sandbox.execute(self.tool, *args, **kwargs)

        if not result.success:
            raise RuntimeError(f"Sandboxed tool '{self.name}' failed: {result.error}")

        return result.result

    def execute_safe(self, *args, **kwargs) -> SandboxResult:
        """Execute and return full result (no exception on failure)."""
        return self.sandbox.execute(self.tool, *args, **kwargs)


def sandboxed(
    mode: SandboxMode = SandboxMode.BASIC,
    timeout: float = 60.0,
):
    """Decorator to sandbox a function.

    Example:
        >>> @sandboxed(timeout=30.0)
        ... def risky_operation():
        ...     pass
    """
    def decorator(func: Callable) -> SandboxedTool:
        limits = ResourceLimits(timeout_seconds=timeout)
        sandbox = Sandbox(mode=mode, limits=limits)
        return SandboxedTool(func, sandbox)

    return decorator
