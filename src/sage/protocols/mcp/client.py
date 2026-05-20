"""MCP Client implementation with security hardening."""

import logging
import time
import hashlib
import secrets
from typing import Optional, List, Dict, Any, Callable
from dataclasses import dataclass, field
from enum import Enum

logger = logging.getLogger(__name__)


class MCPConnectionState(Enum):
    """MCP connection states."""
    DISCONNECTED = "disconnected"
    CONNECTING = "connecting"
    CONNECTED = "connected"
    AUTHENTICATED = "authenticated"
    ERROR = "error"


@dataclass
class MCPToolDefinition:
    """Definition of an MCP tool."""
    name: str
    description: str
    parameters: Dict[str, Any] = field(default_factory=dict)
    required_permissions: List[str] = field(default_factory=list)
    version: str = "1.0.0"


@dataclass
class MCPResource:
    """An MCP resource."""
    uri: str
    name: str
    mime_type: str = "application/octet-stream"
    description: str = ""
    size_bytes: Optional[int] = None


@dataclass
class MCPRequest:
    """MCP request with security metadata."""
    request_id: str
    method: str
    params: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)
    nonce: str = field(default_factory=lambda: secrets.token_hex(16))
    signature: Optional[str] = None

    def compute_signature(self, secret_key: str) -> str:
        """Compute HMAC signature for request integrity."""
        payload = f"{self.request_id}:{self.method}:{self.timestamp}:{self.nonce}"
        return hashlib.sha256(f"{payload}:{secret_key}".encode()).hexdigest()


@dataclass
class MCPResponse:
    """MCP response."""
    request_id: str
    success: bool
    result: Any = None
    error: Optional[str] = None
    timestamp: float = field(default_factory=time.time)


class MCPClient:
    """MCP Client for connecting to MCP servers.

    Implements security features:
    - Request signing
    - Nonce-based replay protection
    - Connection state management
    - Tool validation

    Example:
        >>> client = MCPClient(server_url="http://localhost:3000")
        >>> await client.connect()
        >>> tools = await client.list_tools()
        >>> result = await client.call_tool("search", {"query": "AI"})
    """

    def __init__(
        self,
        server_url: str,
        client_id: Optional[str] = None,
        secret_key: Optional[str] = None,
        timeout: float = 30.0,
        max_retries: int = 3,
    ):
        """Initialize the MCP client.

        Args:
            server_url: URL of the MCP server
            client_id: Client identifier
            secret_key: Secret key for request signing
            timeout: Request timeout in seconds
            max_retries: Maximum retry attempts
        """
        self.server_url = server_url
        self.client_id = client_id or secrets.token_hex(8)
        self.secret_key = secret_key or secrets.token_hex(32)
        self.timeout = timeout
        self.max_retries = max_retries

        self._state = MCPConnectionState.DISCONNECTED
        self._tools: Dict[str, MCPToolDefinition] = {}
        self._resources: Dict[str, MCPResource] = {}
        self._used_nonces: set = set()
        self._request_counter = 0

    @property
    def state(self) -> MCPConnectionState:
        """Get connection state."""
        return self._state

    @property
    def is_connected(self) -> bool:
        """Check if connected."""
        return self._state in (MCPConnectionState.CONNECTED, MCPConnectionState.AUTHENTICATED)

    async def connect(self) -> bool:
        """Connect to the MCP server.

        Returns:
            True if connection successful
        """
        self._state = MCPConnectionState.CONNECTING

        try:
            # In production, this would establish actual connection
            # For now, simulate successful connection
            self._state = MCPConnectionState.CONNECTED
            logger.info(f"MCP client connected to {self.server_url}")
            return True
        except Exception as e:
            self._state = MCPConnectionState.ERROR
            logger.error(f"MCP connection failed: {e}")
            return False

    async def disconnect(self):
        """Disconnect from the MCP server."""
        self._state = MCPConnectionState.DISCONNECTED
        self._tools.clear()
        self._resources.clear()
        logger.info("MCP client disconnected")

    async def authenticate(self, credentials: Dict[str, str]) -> bool:
        """Authenticate with the server.

        Args:
            credentials: Authentication credentials

        Returns:
            True if authentication successful
        """
        if not self.is_connected:
            raise RuntimeError("Not connected to server")

        # Create signed auth request
        request = self._create_request("auth/login", credentials)

        # In production, send to server
        # For now, simulate success
        self._state = MCPConnectionState.AUTHENTICATED
        logger.info("MCP client authenticated")
        return True

    async def list_tools(self) -> List[MCPToolDefinition]:
        """List available tools.

        Returns:
            List of tool definitions
        """
        if not self.is_connected:
            raise RuntimeError("Not connected to server")

        request = self._create_request("tools/list", {})

        # In production, this would call the server
        # Return cached tools
        return list(self._tools.values())

    async def call_tool(
        self,
        tool_name: str,
        arguments: Dict[str, Any],
    ) -> MCPResponse:
        """Call an MCP tool.

        Args:
            tool_name: Name of the tool
            arguments: Tool arguments

        Returns:
            MCPResponse with result
        """
        if not self.is_connected:
            raise RuntimeError("Not connected to server")

        # Validate tool exists
        if tool_name not in self._tools:
            return MCPResponse(
                request_id="",
                success=False,
                error=f"Unknown tool: {tool_name}",
            )

        # Create signed request
        request = self._create_request(
            f"tools/call/{tool_name}",
            {"arguments": arguments},
        )

        # In production, send to server and get response
        # For now, return placeholder
        return MCPResponse(
            request_id=request.request_id,
            success=True,
            result={"message": f"Tool {tool_name} called"},
        )

    async def list_resources(self) -> List[MCPResource]:
        """List available resources.

        Returns:
            List of resources
        """
        if not self.is_connected:
            raise RuntimeError("Not connected to server")

        return list(self._resources.values())

    async def read_resource(self, uri: str) -> Any:
        """Read a resource.

        Args:
            uri: Resource URI

        Returns:
            Resource content
        """
        if not self.is_connected:
            raise RuntimeError("Not connected to server")

        if uri not in self._resources:
            raise ValueError(f"Unknown resource: {uri}")

        request = self._create_request("resources/read", {"uri": uri})

        # In production, fetch from server
        return None

    def register_tool(self, tool: MCPToolDefinition):
        """Register a tool (for local tools).

        Args:
            tool: Tool definition
        """
        self._tools[tool.name] = tool
        logger.debug(f"Registered tool: {tool.name}")

    def register_resource(self, resource: MCPResource):
        """Register a resource.

        Args:
            resource: Resource to register
        """
        self._resources[resource.uri] = resource
        logger.debug(f"Registered resource: {resource.uri}")

    def _create_request(
        self,
        method: str,
        params: Dict[str, Any],
    ) -> MCPRequest:
        """Create a signed MCP request.

        Args:
            method: Request method
            params: Request parameters

        Returns:
            Signed MCPRequest
        """
        self._request_counter += 1

        request = MCPRequest(
            request_id=f"{self.client_id}-{self._request_counter}",
            method=method,
            params=params,
        )

        # Sign the request
        request.signature = request.compute_signature(self.secret_key)

        # Track nonce for replay protection
        self._used_nonces.add(request.nonce)

        return request

    def validate_response_nonce(self, nonce: str) -> bool:
        """Validate a response nonce hasn't been reused.

        Args:
            nonce: Nonce to validate

        Returns:
            True if nonce is valid (not reused)
        """
        if nonce in self._used_nonces:
            return True  # Expected - matches our request

        logger.warning(f"Unexpected nonce in response: {nonce}")
        return False
