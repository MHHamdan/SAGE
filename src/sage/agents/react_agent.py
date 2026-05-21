"""ReAct (Reasoning and Acting) agent implementation."""

from typing import Optional, List, Callable, Any, Literal
import logging

from langgraph.prebuilt import ToolNode
from langgraph.graph import StateGraph, MessagesState, START, END
from langchain_core.messages import HumanMessage

from sage.core.base_agent import BaseAgent
from sage.core.llm_client import LLMClient
from sage.core.exceptions import AgentError


logger = logging.getLogger(__name__)


class ReActAgent(BaseAgent):
    """ReAct agent implementing the Reasoning and Acting paradigm.

    This agent follows the ReAct pattern:
    1. Thought: Reason about the current state
    2. Action: Execute a tool or provide response
    3. Observation: Process action results
    4. Repeat until task is complete

    Example:
        >>> from langchain_core.tools import tool
        >>> @tool
        ... def search(query: str) -> str:
        ...     '''Search for information.'''
        ...     return f"Results for {query}"
        >>> agent = ReActAgent(
        ...     name="research_agent",
        ...     llm=llm_client,
        ...     tools=[search],
        ... )
        >>> result = agent.run("Find information about AI")
    """

    def __init__(
        self,
        name: str,
        llm: LLMClient,
        tools: Optional[List[Callable]] = None,
        instructions: str = "",
        max_iterations: int = 10,
        verbose: bool = False,
    ):
        """Initialize the ReAct agent.

        Args:
            name: Agent name
            llm: LLM client for reasoning
            tools: List of tools the agent can use
            instructions: System instructions
            max_iterations: Maximum reasoning iterations
            verbose: Enable verbose output
        """
        super().__init__(
            name=name,
            llm=llm,
            description="ReAct agent for reasoning and acting",
            instructions=instructions,
            tools=tools,
            max_iterations=max_iterations,
            verbose=verbose,
        )

        self._graph = None
        self._compiled_agent = None
        self._build_graph()

    def _build_graph(self) -> None:
        """Build the LangGraph workflow for ReAct pattern."""
        if not self.tools:
            logger.warning(f"[{self.name}] No tools provided, agent will be limited")
            return

        # Bind tools to LLM
        llm_with_tools = self.llm._client.bind_tools(self.tools)

        # Create tool node
        tool_node = ToolNode(self.tools)

        def call_model(state: MessagesState) -> dict:
            """Call the LLM to reason and decide on action."""
            messages = state["messages"]

            # Add system instructions if provided
            if self.instructions:
                from langchain_core.messages import SystemMessage
                system_msg = SystemMessage(content=self.instructions)
                messages = [system_msg] + list(messages)

            response = llm_with_tools.invoke(messages)

            if self.verbose:
                logger.debug(f"[{self.name}] Model response: {response.content[:100]}...")

            return {"messages": [response]}

        def should_continue(state: MessagesState) -> Literal["tools", END]:
            """Determine whether to continue with tools or end."""
            messages = state["messages"]
            last_message = messages[-1]

            if hasattr(last_message, "tool_calls") and last_message.tool_calls:
                if self.verbose:
                    logger.debug(
                        f"[{self.name}] Tool calls detected: "
                        f"{len(last_message.tool_calls)}"
                    )
                return "tools"

            return END

        # Build the graph
        workflow = StateGraph(MessagesState)
        workflow.add_node("agent", call_model)
        workflow.add_node("tools", tool_node)

        workflow.add_edge(START, "agent")
        workflow.add_conditional_edges("agent", should_continue)
        workflow.add_edge("tools", "agent")

        self._compiled_agent = workflow.compile()
        logger.debug(f"[{self.name}] ReAct graph compiled successfully")

    def run(self, query: str, **kwargs) -> str:
        """Execute the ReAct agent with a query.

        Args:
            query: User query or task
            **kwargs: Additional arguments

        Returns:
            Agent response

        Raises:
            AgentError: If execution fails
        """
        if not self._compiled_agent:
            # Fallback to simple LLM call if no tools
            response = self.llm.invoke(query)
            return response.content

        try:
            messages = [HumanMessage(content=query)]

            result = self._compiled_agent.invoke(
                {"messages": messages},
                {"recursion_limit": self.max_iterations * 2},
            )

            # Extract final response
            final_message = result["messages"][-1]
            response = final_message.content

            self._log_step("run", query, response)
            return response

        except Exception as e:
            logger.error(f"[{self.name}] Execution error: {e}")
            raise AgentError(f"Execution failed: {e}", agent_name=self.name)

    def stream(self, query: str, **kwargs):
        """Stream agent execution.

        Args:
            query: User query
            **kwargs: Additional arguments

        Yields:
            Execution events and responses
        """
        if not self._compiled_agent:
            for chunk in self.llm.stream(query):
                yield chunk
            return

        messages = [HumanMessage(content=query)]

        for event in self._compiled_agent.stream(
            {"messages": messages},
            stream_mode="values",
        ):
            yield event

    def get_graph_visualization(self):
        """Get graph visualization if available.

        Returns:
            Graph visualization object or None
        """
        if self._compiled_agent:
            try:
                return self._compiled_agent.get_graph().draw_mermaid_png()
            except Exception:
                return None
        return None


def create_react_agent(
    name: str,
    llm: LLMClient,
    tools: List[Callable],
    instructions: str = "",
    **kwargs,
) -> ReActAgent:
    """Factory function to create a ReAct agent.

    Args:
        name: Agent name
        llm: LLM client
        tools: List of tools
        instructions: System instructions
        **kwargs: Additional arguments

    Returns:
        Configured ReAct agent
    """
    return ReActAgent(
        name=name,
        llm=llm,
        tools=tools,
        instructions=instructions,
        **kwargs,
    )
