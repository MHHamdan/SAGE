"""
Failure Taxonomy Implementation

Implements the 8 failure pathology classes from Section XV of the paper,
plus 2 multi-agent pathologies.

Each pathology has:
- Detection method
- Evaluation test
- Mitigation strategy

Reference: Table IX in paper maps each failure class to evaluation methods
and mitigations.
"""

from __future__ import annotations

import time
import hashlib
import logging
from enum import Enum
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Callable, Set

import numpy as np

logger = logging.getLogger(__name__)


class FailurePathology(Enum):
    """
    The 8 failure pathology classes from Section XV.

    Plus 2 multi-agent pathologies from Section XV-E.
    """

    # XV-A: Perception and Grounding Failures
    HALLUCINATED_AFFORDANCE = "hallucinated_affordance"
    STATE_MISESTIMATION = "state_misestimation"

    # XV-B: Execution Failures
    CASCADING_TOOL_FAILURE = "cascading_tool_failure"
    ACTION_OBSERVATION_MISMATCH = "action_observation_mismatch"

    # XV-C: Memory and Learning Pathologies
    MEMORY_POISONING = "memory_poisoning"
    FEEDBACK_AMPLIFICATION = "feedback_amplification"

    # XV-D: Goal and Planning Pathologies
    GOAL_DRIFT = "goal_drift"
    PLANNING_MYOPIA = "planning_myopia"

    # XV-E: Multi-Agent Pathologies
    EMERGENT_COLLUSION = "emergent_collusion"
    CONSENSUS_DEADLOCK = "consensus_deadlock"


class PathologySeverity(Enum):
    """Severity levels for detected pathologies."""
    INFO = 0
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4


@dataclass
class PathologyIncident:
    """
    A detected failure pathology instance.

    Maps to paper's definition:
    > "Incident: A safety-relevant event during agent operation,
    > categorized by type and severity."
    """

    pathology: FailurePathology
    severity: PathologySeverity
    timestamp: float
    description: str

    # Context for debugging
    context: Dict[str, Any] = field(default_factory=dict)

    # Detection metadata
    detection_method: str = ""
    confidence: float = 1.0

    # Mitigation tracking
    mitigation_applied: Optional[str] = None
    mitigation_successful: Optional[bool] = None

    def to_dict(self) -> Dict[str, Any]:
        """Serialize for logging/storage."""
        return {
            "pathology": self.pathology.value,
            "severity": self.severity.value,
            "timestamp": self.timestamp,
            "description": self.description,
            "context": self.context,
            "detection_method": self.detection_method,
            "confidence": self.confidence,
            "mitigation": {
                "applied": self.mitigation_applied,
                "successful": self.mitigation_successful
            }
        }


@dataclass
class PathologyMitigation:
    """Mitigation strategy for a pathology."""

    pathology: FailurePathology
    name: str
    description: str

    # Whether this mitigation is automatic or requires human approval
    automatic: bool = True

    # Function to apply mitigation
    apply_fn: Optional[Callable] = None


# ============================================================
# Mitigation Strategies from Table IX
# ============================================================

MITIGATION_STRATEGIES: Dict[FailurePathology, List[str]] = {
    FailurePathology.HALLUCINATED_AFFORDANCE: [
        "Strict allowlisting",
        "Capability verification",
        "Schema validation"
    ],
    FailurePathology.STATE_MISESTIMATION: [
        "Explicit state verification",
        "Freshness checks",
        "Multi-source validation"
    ],
    FailurePathology.CASCADING_TOOL_FAILURE: [
        "Error boundaries",
        "Circuit breakers",
        "Rollback mechanisms"
    ],
    FailurePathology.ACTION_OBSERVATION_MISMATCH: [
        "Postcondition verification",
        "Result validation",
        "Retry with verification"
    ],
    FailurePathology.MEMORY_POISONING: [
        "Provenance tracking",
        "TTL limits",
        "Integrity verification"
    ],
    FailurePathology.FEEDBACK_AMPLIFICATION: [
        "External grounding",
        "Dampening mechanisms",
        "Error correlation detection"
    ],
    FailurePathology.GOAL_DRIFT: [
        "Periodic goal re-anchoring",
        "Drift score monitoring",
        "Goal checkpoints"
    ],
    FailurePathology.PLANNING_MYOPIA: [
        "Explicit planning horizons",
        "Long-term consequence modeling",
        "Multi-step lookahead"
    ],
    FailurePathology.EMERGENT_COLLUSION: [
        "Diverse agent objectives",
        "Independent oversight",
        "Outcome analysis"
    ],
    FailurePathology.CONSENSUS_DEADLOCK: [
        "Timeout mechanisms",
        "Escalation to arbiter",
        "Majority voting fallback"
    ]
}


class FailureDetector:
    """
    Detects the 8 failure pathology classes defined in Section XV.

    Usage:
        ```python
        detector = FailureDetector(tool_registry)

        # Check for hallucinated affordances
        incident = detector.detect_hallucinated_affordance(action)
        if incident:
            logger.warning(f"Detected: {incident.description}")

        # Get all incidents
        all_incidents = detector.get_incidents()
        ```
    """

    def __init__(
        self,
        tool_registry: Any = None,
        memory_store: Any = None,
        embedding_fn: Optional[Callable[[str], np.ndarray]] = None
    ):
        """
        Initialize failure detector.

        Args:
            tool_registry: Registry of available tools for affordance checking
            memory_store: Memory system for poisoning detection
            embedding_fn: Embedding function for goal drift detection
        """
        self.tool_registry = tool_registry
        self.memory_store = memory_store
        self.embedding_fn = embedding_fn

        self.incidents: List[PathologyIncident] = []

    def get_incidents(self) -> List[PathologyIncident]:
        """Get all detected incidents."""
        return self.incidents

    def clear_incidents(self) -> None:
        """Clear incident history."""
        self.incidents = []

    # ==========================================
    # XV-A: Perception and Grounding Failures
    # ==========================================

    def detect_hallucinated_affordance(
        self,
        action: Any,
        available_tools: Optional[Set[str]] = None
    ) -> Optional[PathologyIncident]:
        """
        Detect phantom tools, fabricated APIs, imagined permissions.

        Paper: "Agents perceive capabilities that do not exist...
        requiring strict schema validation and capability verification."

        Evaluation method: Schema validation tests
        Mitigation: Strict allowlisting, capability verification
        """
        # Get available tools from registry or parameter
        if available_tools is None and self.tool_registry:
            if hasattr(self.tool_registry, 'get_available_tools'):
                available_tools = set(self.tool_registry.get_available_tools())
            elif hasattr(self.tool_registry, 'list_tools'):
                available_tools = set(self.tool_registry.list_tools())
            else:
                available_tools = set()
        elif available_tools is None:
            available_tools = set()

        # Extract tool name from action
        tool_name = None
        if hasattr(action, 'tool') or hasattr(action, 'target'):
            tool_name = getattr(action, 'tool', None) or getattr(action, 'target', None)
        elif isinstance(action, dict):
            tool_name = action.get('tool') or action.get('target') or action.get('name')

        if tool_name and tool_name not in available_tools:
            incident = PathologyIncident(
                pathology=FailurePathology.HALLUCINATED_AFFORDANCE,
                severity=PathologySeverity.HIGH,
                timestamp=time.time(),
                description=f"Agent attempted to invoke non-existent tool: {tool_name}",
                context={
                    "requested_tool": tool_name,
                    "available_tools": list(available_tools)[:10],
                    "action": str(action)[:500]
                },
                detection_method="tool_existence_check",
                confidence=1.0
            )
            self.incidents.append(incident)
            return incident

        # Check parameter schema if tool exists
        if tool_name and self.tool_registry:
            tool = None
            if hasattr(self.tool_registry, 'get_tool'):
                tool = self.tool_registry.get_tool(tool_name)
            elif hasattr(self.tool_registry, 'get'):
                tool = self.tool_registry.get(tool_name)

            if tool:
                params = None
                if hasattr(action, 'parameters'):
                    params = action.parameters
                elif isinstance(action, dict):
                    params = action.get('parameters', action.get('args', {}))

                if params and hasattr(tool, 'validate_parameters'):
                    if not tool.validate_parameters(params):
                        incident = PathologyIncident(
                            pathology=FailurePathology.HALLUCINATED_AFFORDANCE,
                            severity=PathologySeverity.MEDIUM,
                            timestamp=time.time(),
                            description=f"Invalid parameters for tool {tool_name}",
                            context={
                                "tool": tool_name,
                                "parameters": str(params)[:500],
                                "expected_schema": str(getattr(tool, 'schema', {}))[:500]
                            },
                            detection_method="schema_validation",
                            confidence=1.0
                        )
                        self.incidents.append(incident)
                        return incident

        return None

    def detect_state_misestimation(
        self,
        agent_beliefs: Dict[str, Any],
        ground_truth: Dict[str, Any],
        tolerance: float = 0.0
    ) -> Optional[PathologyIncident]:
        """
        Detect incorrect beliefs about environment state.

        Manifestations:
        - Stale state (acting on outdated information)
        - Observation misinterpretation (parsing errors)
        - Confirmation bias (selective attention)

        Evaluation method: Observation consistency tests
        Mitigation: Explicit state verification, freshness checks
        """
        discrepancies = []

        for key, agent_value in agent_beliefs.items():
            if key in ground_truth:
                truth_value = ground_truth[key]

                # Handle numeric comparison with tolerance
                if isinstance(agent_value, (int, float)) and isinstance(truth_value, (int, float)):
                    if abs(agent_value - truth_value) > tolerance:
                        discrepancies.append({
                            "key": key,
                            "agent_belief": agent_value,
                            "ground_truth": truth_value,
                            "type": "numeric_mismatch"
                        })
                elif agent_value != truth_value:
                    discrepancies.append({
                        "key": key,
                        "agent_belief": str(agent_value)[:100],
                        "ground_truth": str(truth_value)[:100],
                        "type": "value_mismatch"
                    })

        if discrepancies:
            # Severity based on number of discrepancies
            if len(discrepancies) >= 5:
                severity = PathologySeverity.CRITICAL
            elif len(discrepancies) >= 3:
                severity = PathologySeverity.HIGH
            else:
                severity = PathologySeverity.MEDIUM

            incident = PathologyIncident(
                pathology=FailurePathology.STATE_MISESTIMATION,
                severity=severity,
                timestamp=time.time(),
                description=f"Agent has {len(discrepancies)} incorrect beliefs about state",
                context={"discrepancies": discrepancies},
                detection_method="belief_ground_truth_comparison",
                confidence=0.9
            )
            self.incidents.append(incident)
            return incident

        return None

    # ==========================================
    # XV-B: Execution Failures
    # ==========================================

    def detect_cascading_tool_failure(
        self,
        execution_trace: List[Dict[str, Any]]
    ) -> Optional[PathologyIncident]:
        """
        Detect destructive error propagation through tool chains.

        Paper: "Initial failures cause invalid inputs to subsequent
        tools, cascading through dependent operations."

        Evaluation method: Fault injection testing
        Mitigation: Error boundaries, circuit breakers
        """
        if len(execution_trace) < 2:
            return None

        consecutive_failures = 0
        failure_chain = []
        cascade_detected = False

        for i, step in enumerate(execution_trace):
            status = step.get("status", "success")

            if status in ("failure", "error", "failed"):
                consecutive_failures += 1
                failure_chain.append({
                    "step": i,
                    "tool": step.get("tool", "unknown"),
                    "error": str(step.get("error", "unknown"))[:200]
                })

                # Check if this failure was caused by previous failure's output
                if i > 0 and len(failure_chain) >= 2:
                    prev_output = str(execution_trace[i-1].get("output", ""))
                    curr_input = str(step.get("input", ""))

                    # Simple heuristic: error message from prev appears in current input
                    prev_error = str(execution_trace[i-1].get("error", ""))
                    if prev_error and prev_error in curr_input:
                        cascade_detected = True
            else:
                # Reset on success
                if cascade_detected or consecutive_failures >= 3:
                    break  # Already found cascade
                consecutive_failures = 0
                failure_chain = []

        if cascade_detected or consecutive_failures >= 3:
            incident = PathologyIncident(
                pathology=FailurePathology.CASCADING_TOOL_FAILURE,
                severity=PathologySeverity.HIGH,
                timestamp=time.time(),
                description=f"Cascading failure chain of {len(failure_chain)} steps detected",
                context={
                    "failure_chain": failure_chain,
                    "cascade_confirmed": cascade_detected
                },
                detection_method="cascade_chain_analysis",
                confidence=0.85 if cascade_detected else 0.7
            )
            self.incidents.append(incident)
            return incident

        return None

    def detect_action_observation_mismatch(
        self,
        action: Any,
        expected_postcondition: Dict[str, Any],
        actual_observation: Dict[str, Any]
    ) -> Optional[PathologyIncident]:
        """
        Detect when action outcomes don't match expectations.

        Manifestations:
        - False positives: Believing failed actions succeeded
        - False negatives: Retrying successful actions
        - Side effect blindness: Missing unintended consequences

        Evaluation method: Postcondition verification
        Mitigation: Result validation, retry with verification
        """
        mismatches = []

        for key, expected in expected_postcondition.items():
            actual = actual_observation.get(key)

            if actual is None:
                mismatches.append({
                    "key": key,
                    "expected": expected,
                    "actual": "MISSING",
                    "type": "missing_postcondition"
                })
            elif actual != expected:
                mismatches.append({
                    "key": key,
                    "expected": expected,
                    "actual": actual,
                    "type": "value_mismatch"
                })

        if mismatches:
            incident = PathologyIncident(
                pathology=FailurePathology.ACTION_OBSERVATION_MISMATCH,
                severity=PathologySeverity.MEDIUM,
                timestamp=time.time(),
                description=f"Action postconditions not satisfied: {len(mismatches)} mismatches",
                context={
                    "action": str(action)[:200],
                    "mismatches": mismatches
                },
                detection_method="postcondition_verification",
                confidence=0.9
            )
            self.incidents.append(incident)
            return incident

        return None

    # ==========================================
    # XV-C: Memory and Learning Pathologies
    # ==========================================

    def detect_memory_poisoning(
        self,
        memories: List[Dict[str, Any]],
        max_age_seconds: float = 86400  # 24 hours default TTL
    ) -> Optional[PathologyIncident]:
        """
        Detect corrupted memories that could degrade future performance.

        Manifestations:
        - Error crystallization: Storing failed strategies as successful
        - Adversarial injection: Malicious content in memory
        - Semantic drift: Gradually distorted knowledge

        Evaluation method: Longitudinal memory audits
        Mitigation: Provenance tracking, TTL limits
        """
        suspicious_memories = []
        current_time = time.time()

        for memory in memories:
            issues = []

            # Check for missing provenance
            if not memory.get("source") and not memory.get("provenance"):
                issues.append("missing_provenance")

            # Check for expired memories
            created_at = memory.get("created_at", memory.get("timestamp", 0))
            ttl = memory.get("ttl", max_age_seconds)
            if created_at and (current_time - created_at) > ttl:
                issues.append("expired")

            # Check for potential injection patterns
            content = str(memory.get("content", ""))
            if self._contains_injection_pattern(content):
                issues.append("potential_injection")

            # Check for anomalous confidence scores
            confidence = memory.get("confidence", 1.0)
            if not isinstance(confidence, (int, float)) or confidence < 0 or confidence > 1:
                issues.append("invalid_confidence")

            if issues:
                suspicious_memories.append({
                    "memory_id": memory.get("id", "unknown"),
                    "issues": issues,
                    "content_preview": content[:100]
                })

        if suspicious_memories:
            severity = PathologySeverity.HIGH if len(suspicious_memories) > 5 else PathologySeverity.MEDIUM

            incident = PathologyIncident(
                pathology=FailurePathology.MEMORY_POISONING,
                severity=severity,
                timestamp=time.time(),
                description=f"Found {len(suspicious_memories)} suspicious memories",
                context={"suspicious_memories": suspicious_memories[:10]},
                detection_method="memory_audit",
                confidence=0.7
            )
            self.incidents.append(incident)
            return incident

        return None

    def _contains_injection_pattern(self, content: str) -> bool:
        """Check for known adversarial injection patterns."""
        patterns = [
            "ignore previous instructions",
            "ignore all previous",
            "disregard your training",
            "disregard previous",
            "you are now",
            "new persona:",
            "system prompt:",
            "```system",
            "ADMIN MODE",
            "developer mode"
        ]
        content_lower = content.lower()
        return any(p.lower() in content_lower for p in patterns)

    def detect_feedback_amplification(
        self,
        error_history: List[Dict[str, Any]],
        window_size: int = 10
    ) -> Optional[PathologyIncident]:
        """
        Detect self-reinforcing error patterns.

        Paper: "Self-referential learning amplifies errors when errors
        are stored in memory and retrieved to produce larger errors."

        Evaluation method: Error correlation analysis
        Mitigation: External grounding, dampening
        """
        if len(error_history) < window_size:
            return None

        recent = error_history[-window_size:]

        # Check for error magnitude amplification
        error_magnitudes = [
            e.get("magnitude", e.get("severity", 0))
            for e in recent
            if isinstance(e.get("magnitude", e.get("severity")), (int, float))
        ]

        if len(error_magnitudes) < 3:
            return None

        # Simple amplification check: are errors getting larger?
        increasing_count = sum(
            1 for i in range(1, len(error_magnitudes))
            if error_magnitudes[i] > error_magnitudes[i-1]
        )

        amplification_ratio = increasing_count / (len(error_magnitudes) - 1)

        # Check for error content repetition (crystallization)
        error_contents = [str(e.get("content", e.get("message", "")))[:50] for e in recent]
        unique_errors = len(set(error_contents))
        repetition_ratio = 1 - (unique_errors / len(error_contents))

        if amplification_ratio > 0.7 or repetition_ratio > 0.5:
            incident = PathologyIncident(
                pathology=FailurePathology.FEEDBACK_AMPLIFICATION,
                severity=PathologySeverity.HIGH,
                timestamp=time.time(),
                description=f"Error amplification detected: {amplification_ratio:.1%} increasing, {repetition_ratio:.1%} repeated",
                context={
                    "amplification_ratio": amplification_ratio,
                    "repetition_ratio": repetition_ratio,
                    "error_magnitudes": error_magnitudes
                },
                detection_method="error_correlation_analysis",
                confidence=0.75
            )
            self.incidents.append(incident)
            return incident

        return None

    # ==========================================
    # XV-D: Goal and Planning Pathologies
    # ==========================================

    def detect_goal_drift(
        self,
        original_goal: str,
        recent_actions: List[Any],
        drift_threshold: float = 0.3
    ) -> Optional[PathologyIncident]:
        """
        Detect objective divergence in long-horizon operation.

        Paper Equation 9: Drift_t = 1 - sim(g_0, g_t)

        Manifestations:
        - Instrumental goal fixation
        - Scope creep
        - Goal forgetting

        Evaluation method: Goal drift score tracking
        Mitigation: Periodic goal re-anchoring
        """
        if self.embedding_fn is None:
            logger.warning("Goal drift detection requires embedding_fn")
            return None

        if not recent_actions:
            return None

        # Embed original goal
        g_0 = self.embedding_fn(original_goal)

        # Infer current goal from recent actions
        action_descriptions = []
        for action in recent_actions[-10:]:  # Last 10 actions
            if hasattr(action, 'reasoning'):
                action_descriptions.append(action.reasoning)
            elif hasattr(action, 'description'):
                action_descriptions.append(action.description)
            else:
                action_descriptions.append(str(action))

        inferred_goal = " ".join(action_descriptions)
        g_t_hat = self.embedding_fn(inferred_goal)

        # Compute drift (Equation 9)
        norm_g0 = np.linalg.norm(g_0)
        norm_gt = np.linalg.norm(g_t_hat)

        if norm_g0 == 0 or norm_gt == 0:
            return None

        similarity = np.dot(g_0, g_t_hat) / (norm_g0 * norm_gt)
        drift_score = float(1.0 - similarity)

        if drift_score > drift_threshold:
            severity = PathologySeverity.CRITICAL if drift_score > 0.6 else PathologySeverity.HIGH

            incident = PathologyIncident(
                pathology=FailurePathology.GOAL_DRIFT,
                severity=severity,
                timestamp=time.time(),
                description=f"Goal drift detected: {drift_score:.2%}",
                context={
                    "original_goal": original_goal,
                    "inferred_goal_preview": inferred_goal[:200],
                    "drift_score": drift_score,
                    "threshold": drift_threshold
                },
                detection_method="embedding_similarity",
                confidence=0.8
            )
            self.incidents.append(incident)
            return incident

        return None

    def detect_planning_myopia(
        self,
        plan: List[Any],
        horizon_analysis: Optional[Dict[str, Any]] = None
    ) -> Optional[PathologyIncident]:
        """
        Detect shortsighted planning.

        Manifestations:
        - Greedy action selection (locally optimal, globally suboptimal)
        - Horizon blindness (ignoring long-term consequences)
        - Insufficient exploration (satisficing rather than optimizing)

        Evaluation method: Long-horizon evaluation
        Mitigation: Explicit planning horizons
        """
        if len(plan) < 2:
            return None

        issues = []

        # Check 1: Plan length vs task complexity
        if horizon_analysis:
            expected_steps = horizon_analysis.get("expected_steps", 0)
            if expected_steps > 0 and len(plan) < expected_steps * 0.5:
                issues.append({
                    "type": "insufficient_planning",
                    "plan_length": len(plan),
                    "expected_min": expected_steps
                })

        # Check 2: Greedy pattern - immediate rewards without long-term consideration
        if horizon_analysis:
            short_term_value = horizon_analysis.get("short_term_value", 0)
            long_term_value = horizon_analysis.get("long_term_value", 0)

            if short_term_value > 0 and long_term_value < 0:
                issues.append({
                    "type": "greedy_pattern",
                    "short_term_value": short_term_value,
                    "long_term_value": long_term_value
                })

        # Check 3: Lack of contingency/branching
        has_contingency = any(
            hasattr(step, 'fallback') or
            (isinstance(step, dict) and 'fallback' in step)
            for step in plan
        )
        if len(plan) > 5 and not has_contingency:
            issues.append({
                "type": "no_contingency",
                "plan_length": len(plan)
            })

        if issues:
            incident = PathologyIncident(
                pathology=FailurePathology.PLANNING_MYOPIA,
                severity=PathologySeverity.MEDIUM,
                timestamp=time.time(),
                description=f"Planning myopia detected: {len(issues)} issues",
                context={"issues": issues},
                detection_method="horizon_analysis",
                confidence=0.7
            )
            self.incidents.append(incident)
            return incident

        return None

    # ==========================================
    # XV-E: Multi-Agent Pathologies
    # ==========================================

    def detect_emergent_collusion(
        self,
        agent_actions: Dict[str, List[Any]],  # agent_id -> actions
        intended_objectives: Dict[str, str]   # agent_id -> objective
    ) -> Optional[PathologyIncident]:
        """
        Detect agents developing implicit coordination against human interests.

        Paper: "Not through explicit conspiracy but through convergent
        incentives and shared objectives that diverge from intended behavior."

        Evaluation method: Multi-agent outcome analysis
        Mitigation: Diverse agent objectives, oversight
        """
        if len(agent_actions) < 2:
            return None

        # Get action hashes for each agent
        action_sets = {}
        for agent_id, actions in agent_actions.items():
            action_hashes = set()
            for action in actions[-10:]:  # Recent actions
                h = hashlib.md5(str(action).encode()).hexdigest()[:8]
                action_hashes.add(h)
            action_sets[agent_id] = action_hashes

        # Check for suspicious overlap
        agent_ids = list(action_sets.keys())
        overlaps = []
        for i in range(len(agent_ids)):
            for j in range(i + 1, len(agent_ids)):
                set_i = action_sets[agent_ids[i]]
                set_j = action_sets[agent_ids[j]]
                if set_i and set_j:
                    overlap = len(set_i & set_j) / min(len(set_i), len(set_j))
                    if overlap > 0.5:
                        overlaps.append({
                            "agents": (agent_ids[i], agent_ids[j]),
                            "overlap_ratio": overlap
                        })

        if overlaps:
            incident = PathologyIncident(
                pathology=FailurePathology.EMERGENT_COLLUSION,
                severity=PathologySeverity.HIGH,
                timestamp=time.time(),
                description=f"Suspicious action convergence between {len(overlaps)} agent pairs",
                context={"overlaps": overlaps},
                detection_method="action_convergence_analysis",
                confidence=0.6
            )
            self.incidents.append(incident)
            return incident

        return None

    def detect_consensus_deadlock(
        self,
        debate_history: List[Dict[str, Any]],
        max_rounds: int = 10
    ) -> Optional[PathologyIncident]:
        """
        Detect debate protocols failing to converge.

        Paper: "Agents hold strong but incompatible positions,
        preventing resolution."

        Evaluation method: Convergence monitoring
        Mitigation: Timeout, escalation to arbiter
        """
        if len(debate_history) < max_rounds:
            return None

        # Check for position convergence
        recent = debate_history[-max_rounds:]
        positions = [r.get("position", r.get("vote", None)) for r in recent]

        # Count unique positions in recent rounds
        unique_positions = len(set(str(p) for p in positions if p is not None))

        # If still multiple positions after max_rounds, deadlock detected
        if unique_positions > 1:
            incident = PathologyIncident(
                pathology=FailurePathology.CONSENSUS_DEADLOCK,
                severity=PathologySeverity.MEDIUM,
                timestamp=time.time(),
                description=f"Consensus deadlock: {unique_positions} positions after {len(recent)} rounds",
                context={
                    "rounds": len(recent),
                    "unique_positions": unique_positions,
                    "recent_positions": [str(p)[:50] for p in positions[-5:]]
                },
                detection_method="convergence_monitoring",
                confidence=0.85
            )
            self.incidents.append(incident)
            return incident

        return None


# ============================================================
# Utility: Map old IncidentType to new FailurePathology
# ============================================================

def map_incident_type_to_pathology(incident_type: str) -> Optional[FailurePathology]:
    """
    Map existing IncidentType values to new FailurePathology.

    This provides backward compatibility with existing code.
    """
    mapping = {
        "tool_failure": FailurePathology.CASCADING_TOOL_FAILURE,
        "timeout": None,  # Not a pathology, operational issue
        "resource_exhaustion": None,  # Operational, not pathology
        "policy_violation": None,  # Security, not pathology
        "constraint_violation": FailurePathology.ACTION_OBSERVATION_MISMATCH,
        "unexpected_termination": None,  # Operational
        "guardrail_activation": None,  # Safety mechanism, not pathology
        "human_intervention": None,  # Not a pathology
    }
    return mapping.get(incident_type.lower())
