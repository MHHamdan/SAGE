"""Base agent class providing foundational agent capabilities."""

from abc import ABC, abstractmethod
from typing import Optional, List, Dict, Any, Callable
from dataclasses import dataclass, field
import logging

from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage

from sage.core.llm_client import LLMClient
from sage.core.exceptions import AgentError


logger = logging.getLogger(__name__)


@dataclass
class AgentState:
    """State container for agent execution."""

    messages: List[BaseMessage] = field(default_factory=list)
    current_step: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)
    completed: bool = False
    error: Optional[str] = None


class BaseAgent(ABC):
    """Abstract base class for all agents.

    This class provides the foundational structure for building agents
    following the SPAR (Sense, Plan, Act, Reflect) framework.

    Example:
        >>> class MyAgent(BaseAgent):
        ...     def run(self, query: str) -> str:
        ...         # Implementation
        ...         pass
        >>> agent = MyAgent(name="my_agent", llm=llm_client)
        >>> result = agent.run("Hello")
    """

    def __init__(
        self,
        name: str,
        llm: Optional[LLMClient] = None,
        description: str = "",
        instructions: str = "",
        tools: Optional[List[Callable]] = None,
        max_iterations: int = 10,
        verbose: bool = False,
    ):
        """Initialize the base agent.

        Args:
            name: Unique agent identifier
            llm: LLM client for the agent
            description: Brief description of agent capabilities
            instructions: System instructions for the agent
            tools: List of tools available to the agent
            max_iterations: Maximum execution iterations
            verbose: Enable verbose logging
        """
        self.name = name
        self.llm = llm
        self.description = description
        self.instructions = instructions
        self.tools = tools or []
        self.max_iterations = max_iterations
        self.verbose = verbose

        self._state: Optional[AgentState] = None
        self._history: List[Dict[str, Any]] = []

        if verbose:
            logging.basicConfig(level=logging.DEBUG)

    @abstractmethod
    def run(self, query: str, **kwargs) -> str:
        """Execute the agent with a query.

        Args:
            query: User query or task
            **kwargs: Additional execution arguments

        Returns:
            Agent response
        """
        pass

    def sense(self, input_data: Any) -> Dict[str, Any]:
        """Sense phase: Process and interpret input.

        Args:
            input_data: Raw input to process

        Returns:
            Processed input representation
        """
        logger.debug(f"[{self.name}] Sensing input: {type(input_data)}")
        return {"raw_input": input_data, "type": type(input_data).__name__}

    def plan(self, sensed_data: Dict[str, Any]) -> List[str]:
        """Plan phase: Generate action plan.

        Args:
            sensed_data: Processed input from sense phase

        Returns:
            List of planned actions
        """
        logger.debug(f"[{self.name}] Planning based on sensed data")
        return ["analyze", "respond"]

    def act(self, action: str, **kwargs) -> Any:
        """Act phase: Execute an action.

        Args:
            action: Action to execute
            **kwargs: Action parameters

        Returns:
            Action result
        """
        logger.debug(f"[{self.name}] Executing action: {action}")
        return {"action": action, "status": "completed"}

    def reflect(self, result: Any) -> Dict[str, Any]:
        """Reflect phase: Evaluate action results.

        Args:
            result: Result from act phase

        Returns:
            Reflection analysis
        """
        logger.debug(f"[{self.name}] Reflecting on result")
        return {
            "success": True,
            "result": result,
            "learnings": [],
        }

    def _build_system_message(self) -> SystemMessage:
        """Build the system message from instructions.

        Returns:
            System message for LLM
        """
        system_content = self.instructions
        if self.description:
            system_content = f"{self.description}\n\n{system_content}"
        return SystemMessage(content=system_content)

    def _log_step(
        self,
        step_type: str,
        input_data: Any,
        output_data: Any,
    ) -> None:
        """Log an execution step.

        Args:
            step_type: Type of step (sense, plan, act, reflect)
            input_data: Step input
            output_data: Step output
        """
        step_record = {
            "step_type": step_type,
            "input": str(input_data)[:200],
            "output": str(output_data)[:200],
            "step_number": len(self._history) + 1,
        }
        self._history.append(step_record)

        if self.verbose:
            logger.info(f"[{self.name}] Step {step_record['step_number']}: {step_type}")

    def get_history(self) -> List[Dict[str, Any]]:
        """Get execution history.

        Returns:
            List of execution steps
        """
        return self._history.copy()

    def reset(self) -> None:
        """Reset agent state."""
        self._state = None
        self._history = []
        logger.debug(f"[{self.name}] Agent reset")

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"name='{self.name}', "
            f"tools={len(self.tools)}, "
            f"max_iterations={self.max_iterations})"
        )
