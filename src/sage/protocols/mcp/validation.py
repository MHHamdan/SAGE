"""MCP validation utilities."""

import logging
import re
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class MCPValidationError(Exception):
    """Exception raised for MCP validation failures."""
    pass


def validate_tool_call(
    tool,  # MCPToolDefinition
    arguments: Dict[str, Any],
) -> bool:
    """Validate a tool call against its definition.

    Args:
        tool: Tool definition
        arguments: Provided arguments

    Returns:
        True if valid

    Raises:
        MCPValidationError: If validation fails
    """
    params_schema = tool.parameters

    # Check required parameters
    required = params_schema.get("required", [])
    for param in required:
        if param not in arguments:
            raise MCPValidationError(f"Missing required parameter: {param}")

    # Check parameter types
    properties = params_schema.get("properties", {})
    for arg_name, arg_value in arguments.items():
        if arg_name in properties:
            prop_schema = properties[arg_name]
            _validate_parameter_type(arg_name, arg_value, prop_schema)

    return True


def _validate_parameter_type(
    name: str,
    value: Any,
    schema: Dict[str, Any],
):
    """Validate a parameter against its schema.

    Args:
        name: Parameter name
        value: Parameter value
        schema: JSON schema for parameter

    Raises:
        MCPValidationError: If type doesn't match
    """
    expected_type = schema.get("type")

    if expected_type is None:
        return  # No type constraint

    type_map = {
        "string": str,
        "integer": int,
        "number": (int, float),
        "boolean": bool,
        "array": list,
        "object": dict,
    }

    python_type = type_map.get(expected_type)
    if python_type and not isinstance(value, python_type):
        raise MCPValidationError(
            f"Parameter '{name}' should be {expected_type}, got {type(value).__name__}"
        )

    # Additional validations
    if expected_type == "string":
        _validate_string(name, value, schema)
    elif expected_type == "array":
        _validate_array(name, value, schema)
    elif expected_type in ("integer", "number"):
        _validate_number(name, value, schema)


def _validate_string(name: str, value: str, schema: Dict[str, Any]):
    """Validate string constraints."""
    min_len = schema.get("minLength")
    max_len = schema.get("maxLength")
    pattern = schema.get("pattern")

    if min_len is not None and len(value) < min_len:
        raise MCPValidationError(
            f"Parameter '{name}' must be at least {min_len} characters"
        )

    if max_len is not None and len(value) > max_len:
        raise MCPValidationError(
            f"Parameter '{name}' must be at most {max_len} characters"
        )

    if pattern is not None:
        if not re.match(pattern, value):
            raise MCPValidationError(
                f"Parameter '{name}' doesn't match pattern: {pattern}"
            )


def _validate_array(name: str, value: list, schema: Dict[str, Any]):
    """Validate array constraints."""
    min_items = schema.get("minItems")
    max_items = schema.get("maxItems")

    if min_items is not None and len(value) < min_items:
        raise MCPValidationError(
            f"Parameter '{name}' must have at least {min_items} items"
        )

    if max_items is not None and len(value) > max_items:
        raise MCPValidationError(
            f"Parameter '{name}' must have at most {max_items} items"
        )


def _validate_number(name: str, value: Any, schema: Dict[str, Any]):
    """Validate number constraints."""
    minimum = schema.get("minimum")
    maximum = schema.get("maximum")

    if minimum is not None and value < minimum:
        raise MCPValidationError(
            f"Parameter '{name}' must be >= {minimum}"
        )

    if maximum is not None and value > maximum:
        raise MCPValidationError(
            f"Parameter '{name}' must be <= {maximum}"
        )


def validate_resource(
    uri: str,
    allowed_schemes: Optional[List[str]] = None,
    allowed_hosts: Optional[List[str]] = None,
) -> bool:
    """Validate a resource URI.

    Args:
        uri: Resource URI to validate
        allowed_schemes: List of allowed schemes (e.g., ['file', 'http'])
        allowed_hosts: List of allowed hosts

    Returns:
        True if valid

    Raises:
        MCPValidationError: If validation fails
    """
    from urllib.parse import urlparse

    try:
        parsed = urlparse(uri)
    except Exception as e:
        raise MCPValidationError(f"Invalid URI: {e}")

    # Check scheme
    if allowed_schemes and parsed.scheme not in allowed_schemes:
        raise MCPValidationError(
            f"Scheme '{parsed.scheme}' not allowed. Allowed: {allowed_schemes}"
        )

    # Check host
    if allowed_hosts and parsed.netloc not in allowed_hosts:
        raise MCPValidationError(
            f"Host '{parsed.netloc}' not allowed. Allowed: {allowed_hosts}"
        )

    # Prevent path traversal
    if ".." in parsed.path:
        raise MCPValidationError("Path traversal not allowed")

    return True


def sanitize_tool_input(value: Any, max_length: int = 10000) -> Any:
    """Sanitize tool input to prevent injection attacks.

    Args:
        value: Input value
        max_length: Maximum string length

    Returns:
        Sanitized value
    """
    if isinstance(value, str):
        # Truncate long strings
        if len(value) > max_length:
            value = value[:max_length]

        # Remove null bytes
        value = value.replace("\x00", "")

        return value

    elif isinstance(value, dict):
        return {k: sanitize_tool_input(v, max_length) for k, v in value.items()}

    elif isinstance(value, list):
        return [sanitize_tool_input(v, max_length) for v in value]

    return value


def validate_tool_definition(tool_def: Dict[str, Any]) -> bool:
    """Validate a tool definition structure.

    Args:
        tool_def: Tool definition dictionary

    Returns:
        True if valid

    Raises:
        MCPValidationError: If invalid
    """
    required_fields = ["name", "description"]

    for field in required_fields:
        if field not in tool_def:
            raise MCPValidationError(f"Missing required field: {field}")

    # Validate name format
    name = tool_def["name"]
    if not re.match(r"^[a-zA-Z_][a-zA-Z0-9_]*$", name):
        raise MCPValidationError(
            f"Invalid tool name '{name}': must be a valid identifier"
        )

    # Validate parameters schema if present
    if "parameters" in tool_def:
        params = tool_def["parameters"]
        if not isinstance(params, dict):
            raise MCPValidationError("Parameters must be a dictionary")

        if "type" in params and params["type"] != "object":
            raise MCPValidationError("Parameters type must be 'object'")

    return True
