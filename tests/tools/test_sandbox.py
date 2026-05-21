"""Tests for tool sandboxing."""

import time

import pytest


class TestSandbox:
    """Test sandbox functionality."""

    def test_basic_execution(self):
        """Test basic sandboxed execution."""
        from sage.tools.sandbox import Sandbox, SandboxMode

        sandbox = Sandbox(mode=SandboxMode.BASIC)

        def simple_func(x):
            return x * 2

        result = sandbox.execute(simple_func, 5)

        assert result.success
        assert result.result == 10

    def test_timeout(self):
        """Test timeout enforcement."""
        from sage.tools.sandbox import ResourceLimits, Sandbox

        limits = ResourceLimits(timeout_seconds=0.1)
        sandbox = Sandbox(limits=limits)

        def slow_func():
            time.sleep(2)
            return "done"

        result = sandbox.execute(slow_func)

        assert not result.success
        assert result.was_terminated
        assert "Timeout" in result.termination_reason

    def test_no_sandbox_mode(self):
        """Test execution without sandboxing."""
        from sage.tools.sandbox import Sandbox, SandboxMode

        sandbox = Sandbox(mode=SandboxMode.NONE)

        def func():
            return "result"

        result = sandbox.execute(func)

        assert result.success
        assert result.result == "result"

    def test_exception_handling(self):
        """Test exception handling in sandbox."""
        from sage.tools.sandbox import Sandbox

        sandbox = Sandbox()

        def failing_func():
            raise ValueError("Test error")

        result = sandbox.execute(failing_func)

        assert not result.success
        assert "Test error" in result.error

    def test_path_validation(self):
        """Test path validation."""
        from sage.tools.sandbox import Sandbox

        sandbox = Sandbox(blocked_paths=["/etc", "/root"])

        assert not sandbox.validate_path("/etc/passwd")
        assert not sandbox.validate_path("/root/.ssh")
        assert sandbox.validate_path("/home/user/data.txt")

    def test_allowed_paths(self):
        """Test allowed paths."""
        from sage.tools.sandbox import Sandbox

        sandbox = Sandbox(allowed_paths=["/app/data"])

        assert sandbox.validate_path("/app/data/file.txt")
        assert not sandbox.validate_path("/other/path")


class TestSandboxedTool:
    """Test SandboxedTool wrapper."""

    def test_sandboxed_tool_creation(self):
        """Test sandboxed tool creation."""
        from sage.tools.sandbox import SandboxedTool

        def my_tool(x):
            return x + 1

        sandboxed = SandboxedTool(my_tool)

        assert sandboxed.name == "my_tool"
        assert sandboxed(5) == 6

    def test_sandboxed_tool_safe_execute(self):
        """Test safe execution method."""
        from sage.tools.sandbox import SandboxedTool

        def risky_tool():
            raise RuntimeError("Oops")

        sandboxed = SandboxedTool(risky_tool)
        result = sandboxed.execute_safe()

        assert not result.success
        assert "Oops" in result.error


class TestSandboxedDecorator:
    """Test @sandboxed decorator."""

    def test_decorator(self):
        """Test sandboxed decorator."""
        from sage.tools.sandbox import sandboxed

        @sandboxed(timeout=5.0)
        def my_func(x):
            return x * 3

        result = my_func(4)
        assert result == 12
