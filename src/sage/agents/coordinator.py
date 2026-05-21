"""Multi-agent coordinator with arbitration modes."""

import logging
from typing import Optional, List, Dict, Any
from dataclasses import dataclass, field
from enum import Enum

from ..core.base_agent import BaseAgent

logger = logging.getLogger(__name__)


class ArbitrationMode(Enum):
    """Modes for resolving agent decisions."""
    MAJORITY_VOTE = "majority_vote"
    WEIGHTED_VOTE = "weighted_vote"
    SUPERVISOR_OVERRIDE = "supervisor_override"
    CONSENSUS = "consensus"
    PRIORITY = "priority"


@dataclass
class AgentVote:
    """Vote from an agent."""
    agent_name: str
    decision: str
    confidence: float = 1.0
    weight: float = 1.0
    reasoning: str = ""


@dataclass
class ArbitrationResult:
    """Result of arbitration."""
    final_decision: str
    mode_used: ArbitrationMode
    votes: List[AgentVote] = field(default_factory=list)
    agreement_score: float = 0.0
    winning_margin: float = 0.0


class CoordinatorAgent:
    """Coordinator for multi-agent systems.

    Provides arbitration between agents using various strategies.

    Example:
        >>> coordinator = CoordinatorAgent(
        ...     agents=[agent1, agent2, agent3],
        ...     mode=ArbitrationMode.WEIGHTED_VOTE,
        ... )
        >>> result = coordinator.coordinate(task="Analyze data")
    """

    def __init__(
        self,
        agents: List[BaseAgent],
        mode: ArbitrationMode = ArbitrationMode.WEIGHTED_VOTE,
        agent_weights: Optional[Dict[str, float]] = None,
        supervisor: Optional[BaseAgent] = None,
    ):
        """Initialize the coordinator.

        Args:
            agents: List of agents to coordinate
            mode: Arbitration mode
            agent_weights: Weights for weighted voting
            supervisor: Supervisor agent for override mode
        """
        self.agents = agents
        self.mode = mode
        self.agent_weights = agent_weights or {}
        self.supervisor = supervisor

        self._agent_map = {a.name: a for a in agents}

    def coordinate(
        self,
        task: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Coordinate agents on a task.

        Args:
            task: Task description
            context: Additional context

        Returns:
            Coordination result
        """
        context = context or {}

        # Collect responses from all agents
        responses = []
        for agent in self.agents:
            try:
                response = agent.run(task)
                responses.append({
                    "agent": agent.name,
                    "response": response,
                    "success": True,
                })
            except Exception as e:
                responses.append({
                    "agent": agent.name,
                    "error": str(e),
                    "success": False,
                })

        # Arbitrate if multiple successful responses
        successful = [r for r in responses if r["success"]]

        if len(successful) == 0:
            return {"error": "All agents failed", "responses": responses}

        if len(successful) == 1:
            return {
                "result": successful[0]["response"],
                "agent": successful[0]["agent"],
                "responses": responses,
            }

        # Need arbitration
        arbitration = self._arbitrate(successful, task)

        return {
            "result": arbitration.final_decision,
            "arbitration": arbitration,
            "responses": responses,
        }

    def _arbitrate(
        self,
        responses: List[Dict],
        task: str,
    ) -> ArbitrationResult:
        """Arbitrate between responses.

        Args:
            responses: List of agent responses
            task: Original task

        Returns:
            ArbitrationResult
        """
        votes = [
            AgentVote(
                agent_name=r["agent"],
                decision=r["response"],
                weight=self.agent_weights.get(r["agent"], 1.0),
            )
            for r in responses
        ]

        if self.mode == ArbitrationMode.MAJORITY_VOTE:
            return self._majority_vote(votes)
        elif self.mode == ArbitrationMode.WEIGHTED_VOTE:
            return self._weighted_vote(votes)
        elif self.mode == ArbitrationMode.SUPERVISOR_OVERRIDE:
            return self._supervisor_override(votes, task)
        elif self.mode == ArbitrationMode.PRIORITY:
            return self._priority_vote(votes)
        else:
            return self._majority_vote(votes)

    def _majority_vote(self, votes: List[AgentVote]) -> ArbitrationResult:
        """Simple majority vote."""
        decision_counts: Dict[str, int] = {}
        for vote in votes:
            decision_counts[vote.decision] = decision_counts.get(vote.decision, 0) + 1

        winner = max(decision_counts.items(), key=lambda x: x[1])

        return ArbitrationResult(
            final_decision=winner[0],
            mode_used=ArbitrationMode.MAJORITY_VOTE,
            votes=votes,
            agreement_score=winner[1] / len(votes),
        )

    def _weighted_vote(self, votes: List[AgentVote]) -> ArbitrationResult:
        """Weighted vote based on agent weights."""
        decision_weights: Dict[str, float] = {}
        total_weight = sum(v.weight for v in votes)

        for vote in votes:
            decision_weights[vote.decision] = (
                decision_weights.get(vote.decision, 0) + vote.weight
            )

        winner = max(decision_weights.items(), key=lambda x: x[1])

        return ArbitrationResult(
            final_decision=winner[0],
            mode_used=ArbitrationMode.WEIGHTED_VOTE,
            votes=votes,
            agreement_score=winner[1] / total_weight if total_weight > 0 else 0,
        )

    def _supervisor_override(
        self,
        votes: List[AgentVote],
        task: str,
    ) -> ArbitrationResult:
        """Let supervisor make final decision."""
        if not self.supervisor:
            return self._weighted_vote(votes)

        # Provide supervisor with agent responses
        summary = "\n".join(
            f"- {v.agent_name}: {v.decision[:200]}"
            for v in votes
        )

        supervisor_prompt = f"""Task: {task}

Agent responses:
{summary}

Choose the best response or synthesize a final answer."""

        decision = self.supervisor.run(supervisor_prompt)

        return ArbitrationResult(
            final_decision=decision,
            mode_used=ArbitrationMode.SUPERVISOR_OVERRIDE,
            votes=votes,
            agreement_score=1.0,
        )

    def _priority_vote(self, votes: List[AgentVote]) -> ArbitrationResult:
        """Select by agent priority (weight as priority)."""
        sorted_votes = sorted(votes, key=lambda v: v.weight, reverse=True)
        winner = sorted_votes[0]

        return ArbitrationResult(
            final_decision=winner.decision,
            mode_used=ArbitrationMode.PRIORITY,
            votes=votes,
            agreement_score=1.0,
        )

    def set_agent_weight(self, agent_name: str, weight: float):
        """Set weight for an agent."""
        self.agent_weights[agent_name] = weight

    def get_agent(self, name: str) -> Optional[BaseAgent]:
        """Get agent by name."""
        return self._agent_map.get(name)
