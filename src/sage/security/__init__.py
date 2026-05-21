"""
Security Module for Agentic AI Systems

Provides security validation and threat modeling capabilities including:
- STRIDE threat analysis for MCP and A2A protocols
- Attack tree generation
- Vulnerability assessment
- Security recommendations

This module implements the formal threat modeling framework described in Section X-C.
"""

from .threat_validator import (
    # Core classes
    ThreatValidator,
    # Data classes
    ThreatDefinition,
    ThreatResult,
    AttackTreeNode,
    STRIDEReport,
    # Enums
    STRIDECategory,
    ThreatSeverity,
    MitigationStatus,
    # Threat definitions
    MCP_THREATS,
    A2A_THREATS,
    ALL_THREATS,
    # Convenience functions
    quick_security_assessment,
    get_threat_by_id,
    get_stride_mcp_table,
)

__all__ = [
    # Core classes
    "ThreatValidator",
    # Data classes
    "ThreatDefinition",
    "ThreatResult",
    "AttackTreeNode",
    "STRIDEReport",
    # Enums
    "STRIDECategory",
    "ThreatSeverity",
    "MitigationStatus",
    # Threat definitions
    "MCP_THREATS",
    "A2A_THREATS",
    "ALL_THREATS",
    # Convenience functions
    "quick_security_assessment",
    "get_threat_by_id",
    "get_stride_mcp_table",
]
