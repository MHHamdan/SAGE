"""Tests proving security enforcement is real, not cosmetic."""

import pytest


class TestPermissionEnforcement:
    """Test that permission checks actually block unauthorized access."""

    def test_forbidden_permission_blocked(self):
        """Verify tools without required permissions are blocked."""
        from sage.tools.permissions import (
            PermissionManager,
            Permission,
        )

        manager = PermissionManager()
        manager.register_tool("read_tool", permissions={Permission.READ})

        # Tool has READ, but not WRITE
        assert manager.check_permission("read_tool", Permission.READ)
        assert not manager.check_permission("read_tool", Permission.WRITE)

    def test_revoked_permission_blocked(self):
        """Verify revoked permissions are actually revoked."""
        from sage.tools.permissions import (
            PermissionManager,
            Permission,
        )

        manager = PermissionManager()
        manager.grant_permission("test_tool", Permission.DELETE)

        # Permission granted
        assert manager.check_permission("test_tool", Permission.DELETE)

        # Revoke permission
        manager.revoke_permission("test_tool", Permission.DELETE)

        # Permission denied
        assert not manager.check_permission("test_tool", Permission.DELETE)


class TestSandboxEnforcement:
    """Test that sandbox actually restricts execution."""

    def test_blocked_path_rejected(self):
        """Verify blocked paths are actually blocked."""
        from sage.tools.sandbox import Sandbox

        sandbox = Sandbox(blocked_paths=["/etc", "/root", "/boot"])

        assert not sandbox.validate_path("/etc/passwd")
        assert not sandbox.validate_path("/etc/shadow")
        assert not sandbox.validate_path("/root/.bashrc")
        assert not sandbox.validate_path("/boot/vmlinuz")

    def test_allowed_path_only(self):
        """Verify allowlist is enforced."""
        from sage.tools.sandbox import Sandbox

        sandbox = Sandbox(allowed_paths=["/app/data", "/tmp"])

        assert sandbox.validate_path("/app/data/file.txt")
        assert sandbox.validate_path("/tmp/cache")
        assert not sandbox.validate_path("/home/user/secret")
        assert not sandbox.validate_path("/var/log/system")

    def test_timeout_enforced(self):
        """Verify timeout actually terminates execution."""
        import time
        from sage.tools.sandbox import Sandbox, ResourceLimits

        limits = ResourceLimits(timeout_seconds=0.1)
        sandbox = Sandbox(limits=limits)

        def slow_operation():
            time.sleep(5)
            return "should not reach"

        result = sandbox.execute(slow_operation)

        assert not result.success
        assert result.was_terminated
        assert "Timeout" in str(result.termination_reason)


class TestPolicyEnforcement:
    """Test that policy engine actually enforces policies."""

    def test_deny_policy_blocks_action(self):
        """Verify DENY policies block execution."""
        from sage.verification.policies import (
            PolicyEngine,
            PolicyRule,
            PolicyDecision,
        )

        engine = PolicyEngine()
        engine.add_rule(PolicyRule(
            name="no_delete",
            condition=lambda ctx: "delete" in ctx.get("action", ""),
            action="*",
            decision=PolicyDecision.DENY,
            reason="Delete operations forbidden",
        ))

        # Delete should be denied
        context = {"action": "delete_file", "path": "/data/important.txt"}
        decision = engine.evaluate(context)
        assert decision.decision == PolicyDecision.DENY

        # Read should be allowed
        context = {"action": "read_file", "path": "/data/important.txt"}
        decision = engine.evaluate(context)
        assert decision.decision == PolicyDecision.ALLOW

    def test_approval_required_policy(self):
        """Verify REQUIRE_APPROVAL policies trigger approval flow."""
        from sage.verification.policies import (
            PolicyEngine,
            PolicyRule,
            PolicyDecision,
        )

        engine = PolicyEngine()
        engine.add_rule(PolicyRule(
            name="approve_external",
            condition=lambda ctx: ctx.get("external", False),
            action="*",
            decision=PolicyDecision.REQUIRE_APPROVAL,
            reason="External operations require approval",
        ))

        context = {"action": "api_call", "external": True}
        decision = engine.evaluate(context)
        assert decision.decision == PolicyDecision.REQUIRE_APPROVAL


class TestAuditLogging:
    """Test that audit logging correctly redacts and records."""

    def test_sensitive_data_redacted(self):
        """Verify sensitive fields are redacted in audit logs."""
        from sage.tools.audit import AuditEntry

        entry = AuditEntry(
            timestamp="2024-01-01T00:00:00",
            tool_name="auth",
            action="login",
            parameters={
                "username": "user123",
                "password": "secret123",
                "api_key": "sk-1234567890",
                "auth_token": "bearer-xyz",
            },
        )

        sanitized = entry.to_dict()
        params = sanitized["parameters"]

        assert params["username"] == "user123"  # Not sensitive
        assert params["password"] == "***REDACTED***"
        assert params["api_key"] == "***REDACTED***"
        assert params["auth_token"] == "***REDACTED***"

    def test_long_values_truncated(self):
        """Verify long values are truncated."""
        from sage.tools.audit import AuditEntry

        long_value = "x" * 1000

        entry = AuditEntry(
            timestamp="2024-01-01T00:00:00",
            tool_name="process",
            action="execute",
            parameters={"data": long_value},
        )

        sanitized = entry.to_dict()
        assert len(sanitized["parameters"]["data"]) <= 503  # 500 + "..."

    def test_audit_logging_records_all(self):
        """Verify all invocations are logged."""
        from sage.tools.audit import AuditLogger

        logger = AuditLogger(enabled=True)

        # Log multiple invocations
        for i in range(5):
            logger.log_invocation(
                tool_name=f"tool_{i}",
                action="execute",
                success=i % 2 == 0,
            )

        entries = logger.get_entries()
        assert len(entries) == 5

        stats = logger.get_statistics()
        assert stats["total_entries"] == 5
        assert stats["success_count"] == 3  # 0, 2, 4


class TestApprovalCost:
    """Test that approval-required actions incur human cost."""

    def test_approval_adds_human_cost(self):
        """Verify approval-gated actions add human intervention cost."""
        from sage.core.cost import CostTracker

        tracker = CostTracker()

        # Simulate an approval-required action
        # When approval is required and granted, it counts as human intervention
        tracker.add_human_intervention()

        summary = tracker.get_summary()
        assert summary["human_interventions"] == 1
        assert summary["total_cost"] > 0  # Human cost is non-zero
