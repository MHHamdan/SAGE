#!/usr/bin/env python3
"""MCP and A2A Protocol Demonstration.

This example demonstrates:
1. MCP (Model Context Protocol) client-server loopback
2. A2A (Agent-to-Agent) agent-card exchange
3. Capability token issuance and validation
4. Replay protection with nonces
5. Signature verification

Run with: python examples/06_protocols_demo.py
"""

import time
import uuid
from datetime import datetime, timedelta


def main():
    print("=" * 60)
    print("Protocol Demonstration (MCP + A2A)")
    print("=" * 60)
    print()

    # ==========================================================================
    # 1. MCP Protocol Demo - Client/Server Loopback
    # ==========================================================================
    print("1. MCP Protocol - Client/Server Loopback")
    print("-" * 40)

    from sage.protocols.mcp import (
        MCPServer,
        MCPClient,
        MCPTool,
        MCPMessage,
        MCPMessageType,
    )

    # Create a local MCP server with tools
    server = MCPServer(name="demo-server")

    # Register tools
    @server.tool(name="greet", description="Greet a user")
    def greet_tool(name: str) -> str:
        return f"Hello, {name}!"

    @server.tool(name="calculate", description="Simple calculator")
    def calc_tool(expression: str) -> float:
        # Safe evaluation for demo
        allowed = set("0123456789+-*/.()")
        if all(c in allowed or c.isspace() for c in expression):
            return eval(expression)
        return 0.0

    print(f"  Server registered tools: {[t.name for t in server.list_tools()]}")

    # Create client and connect to server (loopback)
    client = MCPClient()
    client.connect(server)
    print(f"  Client connected to: {server.name}")

    # Invoke tools via MCP protocol
    print()
    print("  Tool invocations:")

    result = client.invoke("greet", {"name": "Researcher"})
    print(f"    greet('Researcher') -> {result.result}")

    result = client.invoke("calculate", {"expression": "2 + 3 * 4"})
    print(f"    calculate('2 + 3 * 4') -> {result.result}")

    # Show message signing
    print()
    print("  Message signing demonstration:")
    msg = MCPMessage(
        type=MCPMessageType.TOOL_INVOKE,
        tool_name="greet",
        parameters={"name": "Test"},
        nonce=str(uuid.uuid4()),
        timestamp=datetime.utcnow().isoformat(),
    )
    signed_msg = client.sign_message(msg)
    print(f"    Message nonce: {msg.nonce[:8]}...")
    print(f"    Signature: {signed_msg.signature[:16]}...")
    print(f"    Signature valid: {server.verify_signature(signed_msg)}")

    print()

    # ==========================================================================
    # 2. MCP Replay Protection
    # ==========================================================================
    print("2. MCP Replay Protection")
    print("-" * 40)

    from sage.protocols.mcp import ReplayProtector, ReplayError

    protector = ReplayProtector(window_seconds=60)

    # First message - should succeed
    nonce1 = str(uuid.uuid4())
    protector.check_nonce(nonce1)
    print(f"  First use of nonce {nonce1[:8]}...: Accepted")

    # Replay same nonce - should fail
    try:
        protector.check_nonce(nonce1)
        print(f"  Replay of nonce {nonce1[:8]}...: ERROR - Should have been rejected!")
    except ReplayError as e:
        print(f"  Replay of nonce {nonce1[:8]}...: Rejected ('{e}')")

    # Different nonce - should succeed
    nonce2 = str(uuid.uuid4())
    protector.check_nonce(nonce2)
    print(f"  New nonce {nonce2[:8]}...: Accepted")

    print()

    # ==========================================================================
    # 3. A2A Agent Card Exchange
    # ==========================================================================
    print("3. A2A Agent Card Exchange")
    print("-" * 40)

    from sage.protocols.a2a import (
        AgentCard,
        AgentCapability,
        AgentCardValidator,
        AgentCardError,
    )

    # Create agent cards
    researcher_card = AgentCard(
        agent_id="researcher-001",
        name="Research Agent",
        description="Specialized in literature review and synthesis",
        trust_level=0.9,
        capabilities=[
            AgentCapability(name="search", description="Search academic databases"),
            AgentCapability(name="summarize", description="Summarize papers"),
            AgentCapability(name="cite", description="Generate citations"),
        ],
        expires_at=(datetime.utcnow() + timedelta(days=30)).isoformat(),
    )

    coder_card = AgentCard(
        agent_id="coder-001",
        name="Code Agent",
        description="Specialized in code generation and review",
        trust_level=0.85,
        capabilities=[
            AgentCapability(name="generate", description="Generate code"),
            AgentCapability(name="review", description="Review code"),
            AgentCapability(name="test", description="Write tests"),
        ],
    )

    print("  Agent cards created:")
    print(f"    - {researcher_card.name} (trust: {researcher_card.trust_level})")
    print(f"      Capabilities: {[c.name for c in researcher_card.capabilities]}")
    print(f"    - {coder_card.name} (trust: {coder_card.trust_level})")
    print(f"      Capabilities: {[c.name for c in coder_card.capabilities]}")

    # Validate cards
    print()
    print("  Card validation:")
    validator = AgentCardValidator(min_trust_level=0.5)

    for card in [researcher_card, coder_card]:
        try:
            validator.validate(card)
            print(f"    {card.name}: Valid")
        except AgentCardError as e:
            print(f"    {card.name}: Invalid - {e}")

    # Validate capability requests
    print()
    print("  Capability validation:")
    test_cases = [
        (researcher_card, "search", True),
        (researcher_card, "generate", False),
        (coder_card, "review", True),
        (coder_card, "cite", False),
    ]

    for card, capability, expected in test_cases:
        try:
            validator.validate_capability_request(card, capability)
            status = "Allowed"
        except AgentCardError:
            status = "Denied"
        icon = "ok" if (status == "Allowed") == expected else "!!"
        print(f"    [{icon}] {card.name} -> {capability}: {status}")

    print()

    # ==========================================================================
    # 4. A2A Capability Token Flow
    # ==========================================================================
    print("4. A2A Capability Token Flow")
    print("-" * 40)

    from sage.protocols.a2a import (
        CapabilityAuth,
        CapabilityToken,
        TokenStatus,
    )

    # Create auth system
    auth = CapabilityAuth(secret_key="demo-secret-key-12345")

    # Grant capabilities to agents
    auth.grant_capability("researcher-001", "search")
    auth.grant_capability("researcher-001", "summarize")
    auth.grant_capability("coder-001", "generate")
    auth.grant_capability("coder-001", "review")

    print("  Capabilities granted:")
    print("    researcher-001: search, summarize")
    print("    coder-001: generate, review")

    # Issue tokens
    researcher_token = auth.issue_token(
        agent_id="researcher-001",
        capabilities=["search", "summarize"],
        validity_seconds=3600,
    )

    coder_token = auth.issue_token(
        agent_id="coder-001",
        capabilities=["generate"],
        validity_seconds=1800,
    )

    print()
    print("  Tokens issued:")
    print(f"    researcher-001: {researcher_token.token_id[:16]}...")
    print(f"      Capabilities: {researcher_token.capabilities}")
    print(f"      Expires: {researcher_token.expires_at}")
    print(f"    coder-001: {coder_token.token_id[:16]}...")
    print(f"      Capabilities: {coder_token.capabilities}")

    # Validate tokens
    print()
    print("  Token validation:")
    for token, name in [(researcher_token, "researcher"), (coder_token, "coder")]:
        status = auth.validate_token(token)
        print(f"    {name}_token: {status.name}")

    # Test capability checking
    print()
    print("  Capability checks:")
    checks = [
        (researcher_token, "search", True),
        (researcher_token, "generate", False),
        (coder_token, "generate", True),
        (coder_token, "search", False),
    ]

    from sage.protocols.a2a import CapabilityAuthError

    for token, cap, expected in checks:
        try:
            auth.check_capability(token, cap)
            result = "Allowed"
        except CapabilityAuthError:
            result = "Denied"
        icon = "ok" if (result == "Allowed") == expected else "!!"
        print(f"    [{icon}] {token.agent_id} -> {cap}: {result}")

    print()

    # ==========================================================================
    # 5. Token Security Demo
    # ==========================================================================
    print("5. Token Security Demonstration")
    print("-" * 40)

    # Test tampered token
    print("  Tampered token test:")
    tampered_token = CapabilityToken(
        token_id=researcher_token.token_id,
        agent_id=researcher_token.agent_id,
        capabilities=["search", "summarize", "admin"],  # Added unauthorized cap
        issued_at=researcher_token.issued_at,
        expires_at=researcher_token.expires_at,
        signature="fake-signature",
    )
    status = auth.validate_token(tampered_token)
    print(f"    Tampered token status: {status.name}")
    assert status == TokenStatus.INVALID, "Tampered token should be invalid"
    print("    Result: Correctly rejected")

    # Test expired token
    print()
    print("  Expired token test:")
    expired_token = auth.issue_token(
        agent_id="test-agent",
        capabilities=["test"],
        validity_seconds=0.01,
    )
    time.sleep(0.1)
    status = auth.validate_token(expired_token)
    print(f"    Expired token status: {status.name}")
    assert status == TokenStatus.EXPIRED, "Expired token should be rejected"
    print("    Result: Correctly rejected")

    # Test revoked token
    print()
    print("  Revoked token test:")
    revoke_token = auth.issue_token(
        agent_id="revoke-test",
        capabilities=["test"],
        validity_seconds=3600,
    )
    print(f"    Before revocation: {auth.validate_token(revoke_token).name}")
    auth.revoke_token(revoke_token.token_id)
    print(f"    After revocation: {auth.validate_token(revoke_token).name}")
    assert auth.validate_token(revoke_token) == TokenStatus.REVOKED
    print("    Result: Correctly rejected")

    print()

    # ==========================================================================
    # 6. Complete A2A Handshake
    # ==========================================================================
    print("6. Complete A2A Handshake Simulation")
    print("-" * 40)

    from sage.protocols.a2a import A2AHandshake, HandshakeStatus

    # Simulate handshake between two agents
    handshake = A2AHandshake(auth=auth, validator=validator)

    print("  Initiating handshake: researcher-001 -> coder-001")
    print()

    # Step 1: Exchange cards
    result = handshake.exchange_cards(researcher_card, coder_card)
    print(f"  Step 1 - Card exchange: {result.status.name}")

    # Step 2: Request capabilities
    result = handshake.request_capabilities(
        requester=researcher_card,
        target=coder_card,
        capabilities=["generate"],
    )
    print(f"  Step 2 - Capability request: {result.status.name}")
    if result.granted_token:
        print(f"          Token issued: {result.granted_token.token_id[:16]}...")

    # Step 3: Verify handshake
    result = handshake.verify_handshake(
        researcher_card.agent_id,
        coder_card.agent_id,
    )
    print(f"  Step 3 - Handshake verified: {result.status.name}")

    print()
    print("=" * 60)
    print("Protocol demonstration complete!")
    print("=" * 60)
    print()
    print("Summary:")
    print("  - MCP loopback: Client successfully invoked server tools")
    print("  - Replay protection: Nonce reuse correctly rejected")
    print("  - Agent cards: Validation and capability checks working")
    print("  - Capability tokens: Issue, validate, revoke all working")
    print("  - Security: Tampered, expired, and revoked tokens rejected")
    print("  - A2A handshake: Complete agent-to-agent authentication flow")


if __name__ == "__main__":
    main()
