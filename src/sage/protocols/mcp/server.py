"""MCP Server implementation with security hardening.

Model Context Protocol (MCP) server implementation providing secure
tool execution, resource access, and prompt management.

Features:
- Tool registration and execution with validation
- Resource management with dynamic content handlers
- Prompt templates with argument substitution
- Authentication and rate limiting
- Server-Sent Events (SSE) for notifications
- Health checks and capability reporting

Example:
    >>> server = MCPServer(config=MCPServerConfig(name="MyServer"))
    >>>
    >>> # Register tools
    >>> server.register_tool(
    ...     MCPToolDefinition(name="search", description="Search documents"),
    ...     search_handler
    ... )
    >>>
    >>> # Register prompts
    >>> server.register_prompt(MCPPrompt(
    ...     name="summarize",
    ...     description="Summarize text",
    ...     template="Summarize the following: {text}"
    ... ))
    >>>
    >>> await server.start()
"""

import logging
import time
import hashlib
import asyncio
from typing import Optional, List, Dict, Any, Callable, AsyncIterator
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

from .client import (
    MCPToolDefinition,
    MCPResource,
    MCPRequest,
    MCPResponse,
)
from .validation import validate_tool_call, MCPValidationError

logger = logging.getLogger(__name__)


class ServerEventType(Enum):
    """Types of server-sent events."""
    TOOL_REGISTERED = "tool_registered"
    TOOL_EXECUTED = "tool_executed"
    RESOURCE_UPDATED = "resource_updated"
    CLIENT_CONNECTED = "client_connected"
    CLIENT_DISCONNECTED = "client_disconnected"
    ERROR = "error"
    HEARTBEAT = "heartbeat"


@dataclass
class MCPPrompt:
    """MCP prompt template definition.

    Attributes:
        name: Unique prompt name
        description: Human-readable description
        template: Prompt template with {arg} placeholders
        arguments: List of argument definitions
    """
    name: str
    description: str
    template: str
    arguments: List[Dict[str, Any]] = field(default_factory=list)

    def render(self, args: Dict[str, str]) -> str:
        """Render the prompt with arguments.

        Args:
            args: Argument values to substitute

        Returns:
            Rendered prompt string
        """
        result = self.template
        for arg_name, arg_value in args.items():
            result = result.replace(f"{{{arg_name}}}", str(arg_value))
        return result


@dataclass
class ServerEvent:
    """Server-sent event.

    Attributes:
        event_type: Type of event
        data: Event payload
        timestamp: When event occurred
        client_id: Target client (None for broadcast)
    """
    event_type: ServerEventType
    data: Dict[str, Any]
    timestamp: datetime = field(default_factory=datetime.now)
    client_id: Optional[str] = None


@dataclass
class MCPServerConfig:
    """MCP Server configuration.

    Attributes:
        name: Server name
        version: Server version
        max_request_age_seconds: Max age for request validation
        require_authentication: Whether to require client auth
        rate_limit_requests_per_minute: Rate limit per client
        enable_sse: Enable Server-Sent Events
        heartbeat_interval_seconds: SSE heartbeat interval
        max_tool_execution_time: Maximum tool execution time
        enable_completions: Enable completions endpoint
    """
    name: str = "MCP Server"
    version: str = "1.0.0"
    max_request_age_seconds: float = 300.0  # 5 minutes
    require_authentication: bool = True
    rate_limit_requests_per_minute: int = 100
    enable_sse: bool = True
    heartbeat_interval_seconds: float = 30.0
    max_tool_execution_time: float = 60.0
    enable_completions: bool = True


class MCPServer:
    """MCP Server for exposing tools and resources.

    Complete MCP server implementation with:
    - Tool registration and execution
    - Resource management
    - Prompt templates
    - Authentication and authorization
    - Rate limiting
    - Server-Sent Events for notifications
    - Health checks and capability reporting

    Security features:
    - Request signature validation
    - Replay attack protection
    - Rate limiting
    - Tool permission checking

    Example:
        >>> server = MCPServer(config=MCPServerConfig(name="MyServer"))
        >>>
        >>> # Register tools
        >>> server.register_tool(my_tool_def, my_tool_handler)
        >>>
        >>> # Register prompts
        >>> server.register_prompt(MCPPrompt(
        ...     name="analyze",
        ...     description="Analyze data",
        ...     template="Analyze this: {input}"
        ... ))
        >>>
        >>> # Start server
        >>> await server.start()
    """

    def __init__(
        self,
        config: Optional[MCPServerConfig] = None,
        secret_keys: Optional[Dict[str, str]] = None,
        completion_handler: Optional[Callable] = None,
    ):
        """Initialize the MCP server.

        Args:
            config: Server configuration
            secret_keys: Client ID -> secret key mapping
            completion_handler: Handler for completion requests
        """
        self.config = config or MCPServerConfig()
        self.secret_keys = secret_keys or {}
        self._completion_handler = completion_handler

        # Registry
        self._tools: Dict[str, MCPToolDefinition] = {}
        self._tool_handlers: Dict[str, Callable] = {}
        self._resources: Dict[str, MCPResource] = {}
        self._resource_handlers: Dict[str, Callable] = {}
        self._prompts: Dict[str, MCPPrompt] = {}

        # Security state
        self._used_nonces: Dict[str, float] = {}  # nonce -> timestamp
        self._request_counts: Dict[str, List[float]] = {}  # client_id -> timestamps
        self._authenticated_clients: set = set()
        self._client_permissions: Dict[str, List[str]] = {}  # client_id -> allowed tools

        # Event system
        self._event_queues: Dict[str, asyncio.Queue] = {}  # client_id -> event queue
        self._event_subscribers: List[Callable[[ServerEvent], None]] = []

        # Statistics
        self._stats = {
            "requests_handled": 0,
            "tools_executed": 0,
            "resources_accessed": 0,
            "prompts_rendered": 0,
            "auth_failures": 0,
            "rate_limit_hits": 0,
            "errors": 0,
        }

        self._running = False
        self._start_time: Optional[datetime] = None
        self._heartbeat_task: Optional[asyncio.Task] = None

    @property
    def is_running(self) -> bool:
        """Check if server is running."""
        return self._running

    async def start(self):
        """Start the MCP server."""
        self._running = True
        self._start_time = datetime.now()

        # Start heartbeat if SSE enabled
        if self.config.enable_sse:
            self._heartbeat_task = asyncio.create_task(self._heartbeat_loop())

        logger.info(f"MCP Server '{self.config.name}' v{self.config.version} started")

    async def stop(self):
        """Stop the MCP server."""
        self._running = False

        # Cancel heartbeat
        if self._heartbeat_task:
            self._heartbeat_task.cancel()
            try:
                await self._heartbeat_task
            except asyncio.CancelledError:
                pass

        logger.info(f"MCP Server '{self.config.name}' stopped")

    async def _heartbeat_loop(self):
        """Send periodic heartbeat events."""
        while self._running:
            await asyncio.sleep(self.config.heartbeat_interval_seconds)
            if self._running:
                await self._emit_event(ServerEvent(
                    event_type=ServerEventType.HEARTBEAT,
                    data={"timestamp": datetime.now().isoformat()},
                ))

    def register_tool(
        self,
        tool: MCPToolDefinition,
        handler: Callable,
    ):
        """Register a tool with its handler.

        Args:
            tool: Tool definition
            handler: Function to handle tool calls
        """
        self._tools[tool.name] = tool
        self._tool_handlers[tool.name] = handler
        logger.info(f"Registered tool: {tool.name}")

    def register_resource(
        self,
        resource: MCPResource,
        handler: Optional[Callable] = None,
    ):
        """Register a resource.

        Args:
            resource: Resource definition
            handler: Optional handler for dynamic resources
        """
        self._resources[resource.uri] = resource
        if handler:
            self._resource_handlers[resource.uri] = handler
        logger.info(f"Registered resource: {resource.uri}")

    def register_prompt(self, prompt: MCPPrompt):
        """Register a prompt template.

        Args:
            prompt: Prompt definition
        """
        self._prompts[prompt.name] = prompt
        logger.info(f"Registered prompt: {prompt.name}")

    def register_client(
        self,
        client_id: str,
        secret_key: str,
        allowed_tools: Optional[List[str]] = None,
    ):
        """Register a client with its secret key.

        Args:
            client_id: Client identifier
            secret_key: Client's secret key
            allowed_tools: List of tool names client can access (None = all)
        """
        self.secret_keys[client_id] = secret_key
        if allowed_tools is not None:
            self._client_permissions[client_id] = allowed_tools
        logger.info(f"Registered client: {client_id}")

    def subscribe_to_events(self, callback: Callable[[ServerEvent], None]):
        """Subscribe to server events.

        Args:
            callback: Function to call when events occur
        """
        self._event_subscribers.append(callback)

    async def _emit_event(self, event: ServerEvent):
        """Emit an event to subscribers and queues.

        Args:
            event: Event to emit
        """
        # Notify subscribers
        for callback in self._event_subscribers:
            try:
                callback(event)
            except Exception as e:
                logger.error(f"Event callback error: {e}")

        # Add to client queues
        if event.client_id:
            # Specific client
            if event.client_id in self._event_queues:
                await self._event_queues[event.client_id].put(event)
        else:
            # Broadcast to all
            for queue in self._event_queues.values():
                await queue.put(event)

    async def get_events(self, client_id: str) -> AsyncIterator[ServerEvent]:
        """Get events for a client (SSE stream).

        Args:
            client_id: Client to get events for

        Yields:
            ServerEvent objects
        """
        if client_id not in self._event_queues:
            self._event_queues[client_id] = asyncio.Queue()

        while self._running:
            try:
                event = await asyncio.wait_for(
                    self._event_queues[client_id].get(),
                    timeout=self.config.heartbeat_interval_seconds
                )
                yield event
            except asyncio.TimeoutError:
                # Yield heartbeat on timeout
                yield ServerEvent(
                    event_type=ServerEventType.HEARTBEAT,
                    data={},
                    client_id=client_id,
                )

    async def handle_request(
        self,
        request: MCPRequest,
        client_id: str,
    ) -> MCPResponse:
        """Handle an incoming MCP request.

        Args:
            request: The MCP request
            client_id: ID of the requesting client

        Returns:
            MCPResponse
        """
        self._stats["requests_handled"] += 1

        # Handle auth requests without full security check
        if request.method.startswith("auth/"):
            return await self._handle_auth_request(request, client_id)

        # Security checks
        try:
            self._validate_request_security(request, client_id)
        except MCPValidationError as e:
            self._stats["auth_failures"] += 1
            return MCPResponse(
                request_id=request.request_id,
                success=False,
                error=f"Security validation failed: {e}",
            )

        # Route request
        try:
            if request.method == "tools/list":
                return await self._handle_list_tools(request, client_id)
            elif request.method.startswith("tools/call/"):
                tool_name = request.method.split("/")[-1]
                return await self._handle_call_tool(request, tool_name, client_id)
            elif request.method == "resources/list":
                return await self._handle_list_resources(request)
            elif request.method == "resources/read":
                return await self._handle_read_resource(request)
            elif request.method == "prompts/list":
                return await self._handle_list_prompts(request)
            elif request.method.startswith("prompts/get/"):
                prompt_name = request.method.split("/")[-1]
                return await self._handle_get_prompt(request, prompt_name)
            elif request.method == "completions/complete":
                return await self._handle_completion(request)
            elif request.method == "server/capabilities":
                return await self._handle_capabilities(request)
            elif request.method == "server/health":
                return await self._handle_health(request)
            else:
                return MCPResponse(
                    request_id=request.request_id,
                    success=False,
                    error=f"Unknown method: {request.method}",
                )
        except Exception as e:
            self._stats["errors"] += 1
            logger.error(f"Request handling error: {e}")
            return MCPResponse(
                request_id=request.request_id,
                success=False,
                error=str(e),
            )

    async def _handle_auth_request(
        self,
        request: MCPRequest,
        client_id: str,
    ) -> MCPResponse:
        """Handle authentication requests.

        Args:
            request: The auth request
            client_id: Client ID

        Returns:
            MCPResponse
        """
        if request.method == "auth/login":
            # Verify client secret
            provided_secret = request.params.get("secret")
            if client_id not in self.secret_keys:
                return MCPResponse(
                    request_id=request.request_id,
                    success=False,
                    error="Unknown client",
                )

            expected_secret = self.secret_keys[client_id]
            if provided_secret != expected_secret:
                self._stats["auth_failures"] += 1
                return MCPResponse(
                    request_id=request.request_id,
                    success=False,
                    error="Invalid credentials",
                )

            self._authenticated_clients.add(client_id)

            # Emit event
            await self._emit_event(ServerEvent(
                event_type=ServerEventType.CLIENT_CONNECTED,
                data={"client_id": client_id},
            ))

            return MCPResponse(
                request_id=request.request_id,
                success=True,
                result={"authenticated": True, "client_id": client_id},
            )

        elif request.method == "auth/logout":
            self._authenticated_clients.discard(client_id)

            await self._emit_event(ServerEvent(
                event_type=ServerEventType.CLIENT_DISCONNECTED,
                data={"client_id": client_id},
            ))

            return MCPResponse(
                request_id=request.request_id,
                success=True,
                result={"logged_out": True},
            )

        return MCPResponse(
            request_id=request.request_id,
            success=False,
            error=f"Unknown auth method: {request.method}",
        )

    def _validate_request_security(
        self,
        request: MCPRequest,
        client_id: str,
    ):
        """Validate request security.

        Args:
            request: The request to validate
            client_id: Client ID

        Raises:
            MCPValidationError: If validation fails
        """
        # Check client is registered
        if client_id not in self.secret_keys:
            raise MCPValidationError(f"Unknown client: {client_id}")

        # Check authentication if required
        if self.config.require_authentication:
            if client_id not in self._authenticated_clients:
                if not request.method.startswith("auth/"):
                    raise MCPValidationError("Client not authenticated")

        # Verify signature
        expected_sig = request.compute_signature(self.secret_keys[client_id])
        if request.signature != expected_sig:
            raise MCPValidationError("Invalid request signature")

        # Check request age (replay protection)
        age = time.time() - request.timestamp
        if age > self.config.max_request_age_seconds:
            raise MCPValidationError(f"Request too old: {age:.1f}s")

        if age < -60:  # Allow 1 minute clock skew
            raise MCPValidationError("Request timestamp in future")

        # Check nonce hasn't been used
        if request.nonce in self._used_nonces:
            raise MCPValidationError("Nonce already used (replay attack?)")

        self._used_nonces[request.nonce] = request.timestamp
        self._cleanup_old_nonces()

        # Rate limiting
        self._check_rate_limit(client_id)

    def _check_rate_limit(self, client_id: str):
        """Check and update rate limit.

        Args:
            client_id: Client to check

        Raises:
            MCPValidationError: If rate limit exceeded
        """
        now = time.time()
        minute_ago = now - 60

        if client_id not in self._request_counts:
            self._request_counts[client_id] = []

        # Remove old timestamps
        self._request_counts[client_id] = [
            ts for ts in self._request_counts[client_id]
            if ts > minute_ago
        ]

        # Check limit
        if len(self._request_counts[client_id]) >= self.config.rate_limit_requests_per_minute:
            raise MCPValidationError("Rate limit exceeded")

        # Add current request
        self._request_counts[client_id].append(now)

    def _cleanup_old_nonces(self):
        """Remove old nonces to prevent memory growth."""
        cutoff = time.time() - (self.config.max_request_age_seconds * 2)
        self._used_nonces = {
            nonce: ts for nonce, ts in self._used_nonces.items()
            if ts > cutoff
        }

    async def _handle_list_tools(
        self,
        request: MCPRequest,
        client_id: str,
    ) -> MCPResponse:
        """Handle tools/list request."""
        # Filter by client permissions
        allowed_tools = self._client_permissions.get(client_id)

        tools = []
        for t in self._tools.values():
            if allowed_tools is None or t.name in allowed_tools:
                tools.append({
                    "name": t.name,
                    "description": t.description,
                    "parameters": t.parameters,
                    "version": t.version,
                })

        return MCPResponse(
            request_id=request.request_id,
            success=True,
            result={"tools": tools},
        )

    async def _handle_call_tool(
        self,
        request: MCPRequest,
        tool_name: str,
        client_id: str,
    ) -> MCPResponse:
        """Handle tools/call request."""
        if tool_name not in self._tools:
            return MCPResponse(
                request_id=request.request_id,
                success=False,
                error=f"Unknown tool: {tool_name}",
            )

        # Check client permissions
        allowed_tools = self._client_permissions.get(client_id)
        if allowed_tools is not None and tool_name not in allowed_tools:
            return MCPResponse(
                request_id=request.request_id,
                success=False,
                error=f"Permission denied for tool: {tool_name}",
            )

        # Validate the tool call
        tool = self._tools[tool_name]
        arguments = request.params.get("arguments", {})

        try:
            validate_tool_call(tool, arguments)
        except MCPValidationError as e:
            return MCPResponse(
                request_id=request.request_id,
                success=False,
                error=f"Tool validation failed: {e}",
            )

        # Execute the tool with timeout
        handler = self._tool_handlers[tool_name]
        try:
            if asyncio.iscoroutinefunction(handler):
                result = await asyncio.wait_for(
                    handler(**arguments),
                    timeout=self.config.max_tool_execution_time
                )
            else:
                result = handler(**arguments)

            self._stats["tools_executed"] += 1

            # Emit event
            await self._emit_event(ServerEvent(
                event_type=ServerEventType.TOOL_EXECUTED,
                data={"tool": tool_name, "client_id": client_id},
            ))

            return MCPResponse(
                request_id=request.request_id,
                success=True,
                result=result,
            )
        except asyncio.TimeoutError:
            logger.error(f"Tool {tool_name} timed out")
            return MCPResponse(
                request_id=request.request_id,
                success=False,
                error=f"Tool execution timed out after {self.config.max_tool_execution_time}s",
            )
        except Exception as e:
            logger.error(f"Tool {tool_name} failed: {e}")
            return MCPResponse(
                request_id=request.request_id,
                success=False,
                error=str(e),
            )

    async def _handle_list_resources(self, request: MCPRequest) -> MCPResponse:
        """Handle resources/list request."""
        resources = [
            {
                "uri": r.uri,
                "name": r.name,
                "mime_type": r.mime_type,
                "description": r.description,
            }
            for r in self._resources.values()
        ]

        return MCPResponse(
            request_id=request.request_id,
            success=True,
            result={"resources": resources},
        )

    async def _handle_read_resource(self, request: MCPRequest) -> MCPResponse:
        """Handle resources/read request."""
        uri = request.params.get("uri")

        if uri not in self._resources:
            return MCPResponse(
                request_id=request.request_id,
                success=False,
                error=f"Unknown resource: {uri}",
            )

        # Use handler if available
        if uri in self._resource_handlers:
            try:
                content = self._resource_handlers[uri]()
                return MCPResponse(
                    request_id=request.request_id,
                    success=True,
                    result={"content": content},
                )
            except Exception as e:
                return MCPResponse(
                    request_id=request.request_id,
                    success=False,
                    error=str(e),
                )

        return MCPResponse(
            request_id=request.request_id,
            success=True,
            result={"content": None},
        )

    async def _handle_list_prompts(self, request: MCPRequest) -> MCPResponse:
        """Handle prompts/list request."""
        prompts = [
            {
                "name": p.name,
                "description": p.description,
                "arguments": p.arguments,
            }
            for p in self._prompts.values()
        ]

        return MCPResponse(
            request_id=request.request_id,
            success=True,
            result={"prompts": prompts},
        )

    async def _handle_get_prompt(
        self,
        request: MCPRequest,
        prompt_name: str,
    ) -> MCPResponse:
        """Handle prompts/get request."""
        if prompt_name not in self._prompts:
            return MCPResponse(
                request_id=request.request_id,
                success=False,
                error=f"Unknown prompt: {prompt_name}",
            )

        prompt = self._prompts[prompt_name]
        args = request.params.get("arguments", {})

        try:
            rendered = prompt.render(args)
            self._stats["prompts_rendered"] += 1

            return MCPResponse(
                request_id=request.request_id,
                success=True,
                result={
                    "name": prompt.name,
                    "description": prompt.description,
                    "messages": [{"role": "user", "content": rendered}],
                },
            )
        except Exception as e:
            return MCPResponse(
                request_id=request.request_id,
                success=False,
                error=f"Prompt rendering failed: {e}",
            )

    async def _handle_completion(self, request: MCPRequest) -> MCPResponse:
        """Handle completions/complete request."""
        if not self.config.enable_completions:
            return MCPResponse(
                request_id=request.request_id,
                success=False,
                error="Completions not enabled",
            )

        if not self._completion_handler:
            return MCPResponse(
                request_id=request.request_id,
                success=False,
                error="No completion handler configured",
            )

        ref = request.params.get("ref", {})
        argument = request.params.get("argument", {})

        try:
            if asyncio.iscoroutinefunction(self._completion_handler):
                completions = await self._completion_handler(ref, argument)
            else:
                completions = self._completion_handler(ref, argument)

            return MCPResponse(
                request_id=request.request_id,
                success=True,
                result={"completion": completions},
            )
        except Exception as e:
            return MCPResponse(
                request_id=request.request_id,
                success=False,
                error=f"Completion failed: {e}",
            )

    async def _handle_capabilities(self, request: MCPRequest) -> MCPResponse:
        """Handle server/capabilities request."""
        capabilities = {
            "name": self.config.name,
            "version": self.config.version,
            "tools": {
                "listChanged": True,
            },
            "resources": {
                "subscribe": False,
                "listChanged": True,
            },
            "prompts": {
                "listChanged": True,
            },
            "logging": {},
        }

        if self.config.enable_completions:
            capabilities["completions"] = {}

        return MCPResponse(
            request_id=request.request_id,
            success=True,
            result={"capabilities": capabilities},
        )

    async def _handle_health(self, request: MCPRequest) -> MCPResponse:
        """Handle server/health request."""
        uptime = None
        if self._start_time:
            uptime = (datetime.now() - self._start_time).total_seconds()

        return MCPResponse(
            request_id=request.request_id,
            success=True,
            result={
                "status": "healthy" if self._running else "stopped",
                "uptime_seconds": uptime,
                "stats": self._stats.copy(),
                "registered_tools": len(self._tools),
                "registered_resources": len(self._resources),
                "registered_prompts": len(self._prompts),
                "authenticated_clients": len(self._authenticated_clients),
            },
        )

    def get_stats(self) -> Dict[str, Any]:
        """Get server statistics.

        Returns:
            Dictionary with server statistics
        """
        return {
            **self._stats,
            "is_running": self._running,
            "uptime_seconds": (
                (datetime.now() - self._start_time).total_seconds()
                if self._start_time else None
            ),
            "registered_tools": list(self._tools.keys()),
            "registered_resources": list(self._resources.keys()),
            "registered_prompts": list(self._prompts.keys()),
            "authenticated_clients": list(self._authenticated_clients),
        }
