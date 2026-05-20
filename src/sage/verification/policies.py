"""Policy-as-code for guarded execution.

Provides a lightweight Python DSL for defining security and
execution policies. Optionally supports OPA/Rego integration.
"""

import re
import yaml
import logging
from typing import Optional, List, Dict, Any, Callable, Union
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

logger = logging.getLogger(__name__)


class PolicyDecision(Enum):
    """Decision from policy evaluation."""
    ALLOW = "allow"
    DENY = "deny"
    REQUIRE_APPROVAL = "require_approval"


@dataclass
class PolicyResult:
    """Result of policy evaluation."""
    decision: PolicyDecision
    rule_name: Optional[str] = None
    reason: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def allowed(self) -> bool:
        return self.decision == PolicyDecision.ALLOW

    @property
    def denied(self) -> bool:
        return self.decision == PolicyDecision.DENY

    @property
    def needs_approval(self) -> bool:
        return self.decision == PolicyDecision.REQUIRE_APPROVAL


@dataclass
class PolicyRule:
    """A single policy rule.

    Rules are evaluated in order. First matching rule determines the decision.
    """
    name: str
    description: str = ""
    condition: Optional[Callable[[Dict[str, Any]], bool]] = None
    condition_pattern: Optional[str] = None  # Regex pattern for action matching
    decision: PolicyDecision = PolicyDecision.ALLOW
    priority: int = 0  # Higher priority evaluated first
    metadata: Dict[str, Any] = field(default_factory=dict)

    def matches(self, context: Dict[str, Any]) -> bool:
        """Check if rule matches the context.

        Args:
            context: Evaluation context with action, parameters, etc.

        Returns:
            True if rule matches
        """
        # Check condition function
        if self.condition is not None:
            try:
                return self.condition(context)
            except Exception as e:
                logger.warning(f"Rule '{self.name}' condition failed: {e}")
                return False

        # Check pattern
        if self.condition_pattern:
            action = context.get("action", "")
            if re.search(self.condition_pattern, action, re.IGNORECASE):
                return True

        return False


class Policy:
    """Policy-as-code for execution control.

    Defines rules for allowing, denying, or requiring approval
    for agent actions.

    Example:
        >>> policy = Policy()
        >>> policy.add_rule(PolicyRule(
        ...     name="block_delete",
        ...     condition_pattern=r"delete|remove|rm",
        ...     decision=PolicyDecision.DENY,
        ... ))
        >>> result = policy.evaluate({"action": "delete_file"})
        >>> print(result.decision)  # PolicyDecision.DENY
    """

    def __init__(
        self,
        name: str = "default",
        default_decision: PolicyDecision = PolicyDecision.ALLOW,
    ):
        """Initialize the policy.

        Args:
            name: Policy name
            default_decision: Decision when no rules match
        """
        self.name = name
        self.default_decision = default_decision
        self._rules: List[PolicyRule] = []

        # Default blocklist
        self._blocklist: List[str] = []
        self._allowlist: List[str] = []

    def add_rule(self, rule: PolicyRule):
        """Add a rule to the policy.

        Args:
            rule: Rule to add
        """
        self._rules.append(rule)
        # Sort by priority (higher first)
        self._rules.sort(key=lambda r: -r.priority)
        logger.debug(f"Added policy rule: {rule.name}")

    def add_blocklist(self, patterns: List[str]):
        """Add patterns to blocklist.

        Args:
            patterns: Patterns to block
        """
        self._blocklist.extend(patterns)
        for pattern in patterns:
            self.add_rule(PolicyRule(
                name=f"blocklist_{pattern}",
                condition_pattern=re.escape(pattern),
                decision=PolicyDecision.DENY,
                priority=100,  # High priority
            ))

    def add_allowlist(self, patterns: List[str]):
        """Add patterns to allowlist.

        Args:
            patterns: Patterns to allow
        """
        self._allowlist.extend(patterns)
        for pattern in patterns:
            self.add_rule(PolicyRule(
                name=f"allowlist_{pattern}",
                condition_pattern=re.escape(pattern),
                decision=PolicyDecision.ALLOW,
                priority=90,  # High priority but below blocklist
            ))

    def evaluate(self, context: Dict[str, Any]) -> PolicyResult:
        """Evaluate the policy for a given context.

        Args:
            context: Evaluation context containing:
                - action: Action name
                - parameters: Action parameters
                - agent: Agent name
                - risk_score: Risk score
                - etc.

        Returns:
            PolicyResult with decision
        """
        for rule in self._rules:
            if rule.matches(context):
                return PolicyResult(
                    decision=rule.decision,
                    rule_name=rule.name,
                    reason=rule.description or f"Matched rule: {rule.name}",
                    metadata=rule.metadata,
                )

        return PolicyResult(
            decision=self.default_decision,
            reason="No rules matched, using default decision",
        )

    def evaluate_action(
        self,
        action: str,
        parameters: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> PolicyResult:
        """Convenience method to evaluate an action.

        Args:
            action: Action name
            parameters: Action parameters
            **kwargs: Additional context

        Returns:
            PolicyResult
        """
        context = {
            "action": action,
            "parameters": parameters or {},
            **kwargs,
        }
        return self.evaluate(context)

    @classmethod
    def from_yaml(cls, filepath: str) -> "Policy":
        """Load policy from YAML file.

        Args:
            filepath: Path to YAML file

        Returns:
            Policy instance

        YAML format:
            name: my_policy
            default: allow
            blocklist:
              - rm -rf
              - sudo
            allowlist:
              - read
              - search
            rules:
              - name: high_risk_approval
                pattern: deploy|execute
                decision: require_approval
                priority: 50
        """
        with open(filepath, "r") as f:
            data = yaml.safe_load(f)

        default_str = data.get("default", "allow").upper()
        default_decision = PolicyDecision[default_str]

        policy = cls(
            name=data.get("name", "loaded_policy"),
            default_decision=default_decision,
        )

        # Add blocklist
        if "blocklist" in data:
            policy.add_blocklist(data["blocklist"])

        # Add allowlist
        if "allowlist" in data:
            policy.add_allowlist(data["allowlist"])

        # Add custom rules
        for rule_data in data.get("rules", []):
            decision_str = rule_data.get("decision", "allow").upper()
            rule = PolicyRule(
                name=rule_data.get("name", "unnamed"),
                description=rule_data.get("description", ""),
                condition_pattern=rule_data.get("pattern"),
                decision=PolicyDecision[decision_str],
                priority=rule_data.get("priority", 0),
            )
            policy.add_rule(rule)

        return policy

    def to_yaml(self, filepath: str):
        """Save policy to YAML file.

        Args:
            filepath: Path to save to
        """
        data = {
            "name": self.name,
            "default": self.default_decision.value,
            "blocklist": self._blocklist,
            "allowlist": self._allowlist,
            "rules": [
                {
                    "name": r.name,
                    "description": r.description,
                    "pattern": r.condition_pattern,
                    "decision": r.decision.value,
                    "priority": r.priority,
                }
                for r in self._rules
                if not r.name.startswith(("blocklist_", "allowlist_"))
            ],
        }

        with open(filepath, "w") as f:
            yaml.dump(data, f, default_flow_style=False)

    def get_rules(self) -> List[Dict[str, Any]]:
        """Get list of all rules.

        Returns:
            List of rule dictionaries
        """
        return [
            {
                "name": r.name,
                "description": r.description,
                "pattern": r.condition_pattern,
                "decision": r.decision.value,
                "priority": r.priority,
            }
            for r in self._rules
        ]


# Default security policy
def create_default_policy() -> Policy:
    """Create a default security policy.

    Returns:
        Policy with sensible defaults
    """
    policy = Policy(name="default_security")

    # Block dangerous operations
    policy.add_blocklist([
        "rm -rf",
        "sudo",
        "chmod 777",
        "eval(",
        "exec(",
        "DROP TABLE",
        "DELETE FROM",
        "format c:",
    ])

    # Require approval for risky operations
    policy.add_rule(PolicyRule(
        name="production_deployment",
        condition_pattern=r"deploy.*prod|production",
        decision=PolicyDecision.REQUIRE_APPROVAL,
        priority=80,
        description="Production deployments require approval",
    ))

    policy.add_rule(PolicyRule(
        name="data_deletion",
        condition_pattern=r"delete|truncate|drop",
        decision=PolicyDecision.REQUIRE_APPROVAL,
        priority=70,
        description="Data deletion requires approval",
    ))

    policy.add_rule(PolicyRule(
        name="high_risk",
        condition=lambda ctx: ctx.get("risk_score", 0) >= 0.7,
        decision=PolicyDecision.REQUIRE_APPROVAL,
        priority=60,
        description="High-risk actions require approval",
    ))

    return policy
