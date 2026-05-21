"""
Monitoring Module for Agentic AI Systems

Provides real-time monitoring capabilities for agent execution, including:
- Stability monitoring (formal conditions from Section III-F-5)
- Goal convergence tracking
- Oscillation detection
- Progress monotonicity verification

This module implements the control-theoretic framework described in the paper.
"""

from .stability_monitor import (  # Core classes; Status types; Enums; Convenience functions
    ConvergenceStatus,
    FidelityStatus,
    LimitCycleDetector,
    MonotonicityStatus,
    OscillationStatus,
    StabilityMonitor,
    StabilityReport,
    StabilitySeverity,
    StabilityStatus,
    StabilityViolation,
    ViolationType,
    check_stability_conditions,
    create_stability_monitor,
)

__all__ = [
    # Core classes
    "StabilityMonitor",
    "LimitCycleDetector",
    # Status types
    "StabilityStatus",
    "ConvergenceStatus",
    "OscillationStatus",
    "MonotonicityStatus",
    "FidelityStatus",
    "StabilityReport",
    "StabilityViolation",
    # Enums
    "ViolationType",
    "StabilitySeverity",
    # Convenience functions
    "create_stability_monitor",
    "check_stability_conditions",
]
