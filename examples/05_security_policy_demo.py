#!/usr/bin/env python3
"""Security Policy Demonstration.

This example demonstrates:
1. Policy-based access control (ALLOW/DENY/REQUIRE_APPROVAL)
2. Permission enforcement
3. Sandboxed execution
4. Audit logging with redaction

Run with: python examples/05_security_policy_demo.py
"""

import json
from datetime import datetime


def main():
    print("=" * 60)
    print("Security Policy Demonstration")
    print("=" * 60)
    print()

    # Import security components
    from sage.tools.permissions import (
        PermissionManager,
        Permission,
        READONLY_PERMISSIONS,
        STANDARD_PERMISSIONS,
    )
    from sage.tools.sandbox import Sandbox, SandboxMode, ResourceLimits
    from sage.tools.audit import AuditLogger
    from sage.verification.policies import (
        PolicyEngine,
        PolicyRule,
        PolicyDecision,
    )

    # ==========================================================================
    # 1. Permission Management
    # ==========================================================================
    print("1. Permission Management")
    print("-" * 40)

    manager = PermissionManager()

    # Register tools with different permission levels
    manager.register_tool("search_tool", permissions=READONLY_PERMISSIONS)
    manager.register_tool("file_tool", permissions=STANDARD_PERMISSIONS)
    manager.grant_permission("admin_tool", Permission.ADMIN)

    print(f"search_tool permissions: {manager.get_permissions('search_tool')}")
    print(f"file_tool permissions: {manager.get_permissions('file_tool')}")
    print()

    # Test permission checks
    tests = [
        ("search_tool", Permission.READ, True),
        ("search_tool", Permission.WRITE, False),
        ("file_tool", Permission.WRITE, True),
        ("file_tool", Permission.DELETE, False),
        ("admin_tool", Permission.ADMIN, True),
    ]

    for tool, perm, expected in tests:
        result = manager.check_permission(tool, perm)
        status = "✓" if result == expected else "✗"
        print(f"  {status} {tool} has {perm.name}: {result}")

    print()

    # ==========================================================================
    # 2. Policy Engine (ALLOW/DENY/REQUIRE_APPROVAL)
    # ==========================================================================
    print("2. Policy Engine Demo")
    print("-" * 40)

    engine = PolicyEngine()

    # Policy 1: Deny all delete operations in production
    engine.add_rule(PolicyRule(
        name="no_production_delete",
        condition=lambda ctx: ctx.get("env") == "production" and "delete" in ctx.get("action", ""),
        action="delete",
        decision=PolicyDecision.DENY,
        reason="Delete operations forbidden in production",
    ))

    # Policy 2: Require approval for external API calls
    engine.add_rule(PolicyRule(
        name="approve_external_api",
        condition=lambda ctx: ctx.get("external", False),
        action="api_call",
        decision=PolicyDecision.REQUIRE_APPROVAL,
        reason="External API calls require human approval",
    ))

    # Policy 3: Allow read operations
    engine.add_rule(PolicyRule(
        name="allow_read",
        condition=lambda ctx: ctx.get("action", "").startswith("read"),
        action="read",
        decision=PolicyDecision.ALLOW,
    ))

    # Test cases
    test_contexts = [
        {"action": "read_file", "path": "/data/config.json"},
        {"action": "delete_user", "env": "production", "user_id": 123},
        {"action": "api_call", "external": True, "endpoint": "https://api.example.com"},
        {"action": "write_file", "path": "/tmp/output.txt"},
    ]

    for ctx in test_contexts:
        decision = engine.evaluate(ctx)
        icon = {"ALLOW": "✓", "DENY": "✗", "REQUIRE_APPROVAL": "⚠"}[decision.decision.name]
        print(f"  {icon} {ctx['action']}: {decision.decision.name}")
        if decision.reason:
            print(f"      Reason: {decision.reason}")

    print()

    # ==========================================================================
    # 3. Sandboxed Execution
    # ==========================================================================
    print("3. Sandboxed Execution")
    print("-" * 40)

    # Configure sandbox
    sandbox = Sandbox(
        mode=SandboxMode.STRICT,
        limits=ResourceLimits(
            max_memory_mb=256,
            timeout_seconds=5.0,
            max_processes=5,
        ),
        blocked_paths=["/etc", "/root", "/boot"],
        allowed_paths=["/tmp", "/app/data"],
    )

    # Test path validation
    paths = [
        "/app/data/file.txt",
        "/tmp/cache.json",
        "/etc/passwd",
        "/root/.ssh/id_rsa",
    ]

    print("  Path validation:")
    for path in paths:
        allowed = sandbox.validate_path(path)
        icon = "✓" if allowed else "✗"
        print(f"    {icon} {path}")

    print()

    # Test execution
    def safe_operation():
        return {"status": "success", "data": [1, 2, 3]}

    def risky_operation():
        import time
        time.sleep(10)  # Will timeout
        return "should not reach"

    print("  Execution tests:")

    result = sandbox.execute(safe_operation)
    print(f"    ✓ safe_operation: success={result.success}, result={result.result}")

    result = sandbox.execute(risky_operation)
    print(f"    ✗ risky_operation: success={result.success}, terminated={result.was_terminated}")

    print()

    # ==========================================================================
    # 4. Audit Logging with Redaction
    # ==========================================================================
    print("4. Audit Logging with Redaction")
    print("-" * 40)

    audit = AuditLogger(enabled=True)

    # Log various invocations
    audit.log_invocation(
        tool_name="auth_service",
        action="login",
        parameters={
            "username": "admin",
            "password": "super_secret_password",  # Should be redacted
            "api_key": "sk-1234567890abcdef",  # Should be redacted
        },
        success=True,
        duration_ms=150,
    )

    audit.log_invocation(
        tool_name="database",
        action="query",
        parameters={
            "table": "users",
            "query": "SELECT * FROM users",
            "credential": "db_pass_123",  # Should be redacted
        },
        success=True,
        duration_ms=50,
    )

    audit.log_invocation(
        tool_name="file_service",
        action="write",
        parameters={"path": "/data/output.txt"},
        success=False,
        error="Permission denied",
        duration_ms=10,
    )

    print("  Audit entries (with redaction):")
    for entry in audit.get_entries():
        sanitized = entry.to_dict()
        print(f"    - {sanitized['tool_name']}.{sanitized['action']}: "
              f"success={sanitized['success']}")
        if "password" in sanitized["parameters"]:
            print(f"      password: {sanitized['parameters']['password']}")
        if "api_key" in sanitized["parameters"]:
            print(f"      api_key: {sanitized['parameters']['api_key']}")

    print()
    print("  Audit statistics:")
    stats = audit.get_statistics()
    print(f"    Total entries: {stats['total_entries']}")
    print(f"    Success rate: {stats['success_rate']:.1%}")
    print(f"    Avg duration: {stats['avg_duration_ms']:.1f}ms")

    print()

    # ==========================================================================
    # 5. Complete Security Flow
    # ==========================================================================
    print("5. Complete Security Flow Demo")
    print("-" * 40)

    def execute_with_security(action: str, params: dict, context: dict):
        """Execute an action with full security checks."""

        print(f"\n  Attempting: {action}")
        print(f"  Context: {context}")

        # Step 1: Check policy
        ctx = {"action": action, **context}
        decision = engine.evaluate(ctx)

        if decision.decision == PolicyDecision.DENY:
            audit.log_invocation(
                tool_name=action,
                action="execute",
                parameters=params,
                success=False,
                error=f"Policy denied: {decision.reason}",
            )
            print(f"  ✗ DENIED by policy: {decision.reason}")
            return {"status": "denied", "reason": decision.reason}

        if decision.decision == PolicyDecision.REQUIRE_APPROVAL:
            print(f"  ⚠ Requires approval: {decision.reason}")
            # In real system, would wait for human approval
            # For demo, we'll simulate approval
            print(f"  ✓ Approval granted (simulated)")

        # Step 2: Check permissions
        tool_name = action.split("_")[0] + "_tool"
        required_perm = Permission.WRITE if "write" in action else Permission.READ
        if not manager.check_permission(tool_name, required_perm):
            # Auto-grant for demo
            manager.grant_permission(tool_name, required_perm)

        # Step 3: Execute in sandbox
        def operation():
            return {"status": "success", "action": action}

        result = sandbox.execute(operation)

        # Step 4: Log to audit
        audit.log_invocation(
            tool_name=action,
            action="execute",
            parameters=params,
            success=result.success,
            duration_ms=result.execution_time_ms,
        )

        if result.success:
            print(f"  ✓ Executed successfully")
        else:
            print(f"  ✗ Execution failed: {result.error}")

        return result.result

    # Run complete flow tests
    execute_with_security(
        "read_file",
        {"path": "/data/config.json"},
        {"env": "development"},
    )

    execute_with_security(
        "delete_record",
        {"table": "logs", "id": 123},
        {"env": "production"},  # Will be denied
    )

    execute_with_security(
        "api_call",
        {"endpoint": "https://external.api/data"},
        {"external": True},  # Requires approval
    )

    print()
    print("=" * 60)
    print("Security demonstration complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
