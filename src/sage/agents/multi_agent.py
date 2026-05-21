"""Multi-agent system implementations."""

from typing import Optional, List, Dict, Any, Annotated, TypedDict
from abc import ABC, abstractmethod
import logging

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

from sage.core.base_agent import BaseAgent
from sage.core.llm_client import LLMClient
from sage.core.exceptions import AgentError

logger = logging.getLogger(__name__)


class MultiAgentState(TypedDict):
    """Shared state for multi-agent workflows."""

    messages: Annotated[list, add_messages]
    current_agent: str
    workflow_step: int
    metadata: Dict[str, Any]


class MultiAgentOrchestrator(ABC):
    """Abstract base class for multi-agent orchestrators."""

    def __init__(
        self,
        name: str,
        agents: List[BaseAgent],
        verbose: bool = False,
    ):
        """Initialize the orchestrator.

        Args:
            name: Orchestrator name
            agents: List of agents to coordinate
            verbose: Enable verbose logging
        """
        self.name = name
        self.agents = agents
        self.verbose = verbose
        self._agent_map = {agent.name: agent for agent in agents}

    @abstractmethod
    def run(self, query: str, **kwargs) -> str:
        """Execute the multi-agent workflow."""
        pass

    def get_agent(self, name: str) -> Optional[BaseAgent]:
        """Get agent by name."""
        return self._agent_map.get(name)


class SequentialPipeline(MultiAgentOrchestrator):
    """Sequential multi-agent pipeline.

    Agents execute in a fixed order, each processing the output
    of the previous agent.

    Example:
        >>> pipeline = SequentialPipeline(
        ...     name="processing_pipeline",
        ...     agents=[collector, validator, processor],
        ... )
        >>> result = pipeline.run("Process this data")
    """

    def __init__(
        self,
        name: str,
        agents: List[BaseAgent],
        verbose: bool = False,
    ):
        """Initialize the sequential pipeline.

        Args:
            name: Pipeline name
            agents: Ordered list of agents
            verbose: Enable verbose logging
        """
        super().__init__(name=name, agents=agents, verbose=verbose)
        self._graph = None
        self._compiled = None
        self._build_graph()

    def _build_graph(self) -> None:
        """Build the sequential workflow graph."""
        if not self.agents:
            return

        def create_agent_node(agent: BaseAgent):
            """Create a node function for an agent."""

            def node_fn(state: MultiAgentState) -> dict:
                """Process state through the agent."""
                last_message = state["messages"][-1]

                # Get agent response
                try:
                    response = agent.run(last_message.content)
                except Exception as e:
                    response = f"Error in {agent.name}: {e}"

                if self.verbose:
                    logger.info(f"[{self.name}] {agent.name}: {response[:100]}...")

                return {
                    "messages": [AIMessage(content=response)],
                    "current_agent": agent.name,
                    "workflow_step": state["workflow_step"] + 1,
                }

            return node_fn

        # Build the graph
        workflow = StateGraph(MultiAgentState)

        # Add nodes for each agent
        for agent in self.agents:
            workflow.add_node(agent.name, create_agent_node(agent))

        # Connect agents sequentially
        workflow.add_edge(START, self.agents[0].name)
        for i in range(len(self.agents) - 1):
            workflow.add_edge(self.agents[i].name, self.agents[i + 1].name)
        workflow.add_edge(self.agents[-1].name, END)

        self._compiled = workflow.compile()
        logger.debug(f"[{self.name}] Sequential pipeline compiled")

    def run(self, query: str, **kwargs) -> str:
        """Execute the sequential pipeline.

        Args:
            query: Initial query
            **kwargs: Additional arguments

        Returns:
            Final pipeline output
        """
        if not self._compiled:
            raise AgentError("Pipeline not compiled", agent_name=self.name)

        initial_state = {
            "messages": [HumanMessage(content=query)],
            "current_agent": "",
            "workflow_step": 0,
            "metadata": kwargs.get("metadata", {}),
        }

        result = self._compiled.invoke(initial_state)
        return result["messages"][-1].content


class SupervisorAgent(MultiAgentOrchestrator):
    """Supervisor-based multi-agent system.

    A central supervisor agent coordinates specialized worker agents,
    delegating tasks based on query analysis.

    Example:
        >>> supervisor = SupervisorAgent(
        ...     name="financial_advisor",
        ...     agents=[investment_agent, risk_agent, tax_agent],
        ...     supervisor_llm=llm,
        ...     supervisor_instructions="Route to appropriate specialist...",
        ... )
        >>> result = supervisor.run("Analyze my investment portfolio")
    """

    def __init__(
        self,
        name: str,
        agents: List[BaseAgent],
        supervisor_llm: LLMClient,
        supervisor_instructions: str = "",
        max_iterations: int = 10,
        verbose: bool = False,
    ):
        """Initialize the supervisor agent.

        Args:
            name: Supervisor name
            agents: Worker agents to coordinate
            supervisor_llm: LLM for supervisor reasoning
            supervisor_instructions: Instructions for routing
            max_iterations: Maximum coordination iterations
            verbose: Enable verbose logging
        """
        super().__init__(name=name, agents=agents, verbose=verbose)
        self.supervisor_llm = supervisor_llm
        self.supervisor_instructions = supervisor_instructions
        self.max_iterations = max_iterations

    def _select_agent(self, query: str, history: List[Dict]) -> Optional[str]:
        """Select the appropriate agent for a query.

        Args:
            query: User query
            history: Previous interactions

        Returns:
            Selected agent name or None if complete
        """
        agent_descriptions = "\n".join(
            f"- {agent.name}: {agent.description}" for agent in self.agents
        )

        selection_prompt = f"""
{self.supervisor_instructions}

Available agents:
{agent_descriptions}

Query: {query}

Previous interactions:
{history if history else "None"}

Select the most appropriate agent to handle this query.
Respond with just the agent name, or "COMPLETE" if the task is done.
"""

        response = self.supervisor_llm.invoke(selection_prompt)
        selected = response.content.strip()

        if selected.upper() == "COMPLETE":
            return None

        # Find matching agent
        for agent in self.agents:
            if agent.name.lower() in selected.lower():
                return agent.name

        # Default to first agent if no match
        if self.agents:
            return self.agents[0].name
        return None

    def run(self, query: str, **kwargs) -> str:
        """Execute the supervisor workflow.

        Args:
            query: User query
            **kwargs: Additional arguments

        Returns:
            Final aggregated response
        """
        history = []
        iterations = 0
        responses = []

        while iterations < self.max_iterations:
            # Select agent
            selected_name = self._select_agent(query, history)

            if selected_name is None:
                break

            agent = self.get_agent(selected_name)
            if not agent:
                logger.warning(f"Agent {selected_name} not found")
                break

            if self.verbose:
                logger.info(f"[{self.name}] Routing to: {selected_name}")

            # Execute agent
            try:
                response = agent.run(query)
                responses.append(f"{selected_name}: {response}")
                history.append(
                    {
                        "agent": selected_name,
                        "response": response[:500],
                    }
                )
            except Exception as e:
                logger.error(f"Agent {selected_name} failed: {e}")
                history.append(
                    {
                        "agent": selected_name,
                        "error": str(e),
                    }
                )

            iterations += 1

        # Synthesize final response
        if responses:
            return self._synthesize_response(query, responses)
        return "Unable to process query with available agents."

    def _synthesize_response(self, query: str, responses: List[str]) -> str:
        """Synthesize final response from agent outputs.

        Args:
            query: Original query
            responses: List of agent responses

        Returns:
            Synthesized response
        """
        synthesis_prompt = f"""
Original query: {query}

Agent responses:
{chr(10).join(responses)}

Synthesize these responses into a coherent final answer.
"""

        response = self.supervisor_llm.invoke(synthesis_prompt)
        return response.content
