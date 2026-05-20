#!/usr/bin/env python
"""
Example: Multi-Agent Systems

This example demonstrates how to create multi-agent systems
with different coordination patterns: sequential pipelines
and supervisor architectures.
"""

import os
from dotenv import load_dotenv
from langchain_core.tools import tool

load_dotenv()

from sage.core.llm_client import LLMClient
from sage.agents.react_agent import ReActAgent


# Define specialized tools for different agents

@tool
def search_web(query: str) -> str:
    """Search the web for information.

    Args:
        query: Search query

    Returns:
        Search results
    """
    # Mock search results
    results = {
        "renewable energy": "Solar and wind power are growing rapidly. "
                           "Global renewable capacity increased 50% in 2023.",
        "ai trends": "Large language models and agentic AI are key trends. "
                    "Enterprise adoption is accelerating.",
        "python": "Python remains the most popular language for AI/ML. "
                 "New features in Python 3.12 improve performance.",
    }

    for key, value in results.items():
        if key in query.lower():
            return value
    return f"Search results for: {query}"


@tool
def analyze_data(text: str) -> str:
    """Analyze text data and extract insights.

    Args:
        text: Text to analyze

    Returns:
        Analysis results
    """
    word_count = len(text.split())
    return f"Analysis: {word_count} words. Key themes: growth, technology, trends."


@tool
def write_summary(content: str, style: str = "professional") -> str:
    """Write a formatted summary.

    Args:
        content: Content to summarize
        style: Writing style (professional, casual, technical)

    Returns:
        Formatted summary
    """
    return f"[{style.upper()} SUMMARY]\n{content[:200]}..."


@tool
def fact_check(claim: str) -> str:
    """Verify a factual claim.

    Args:
        claim: Claim to verify

    Returns:
        Verification result
    """
    return f"Claim verified: '{claim[:50]}...' - Source: Industry reports 2024"


def demo_sequential_pipeline():
    """Demonstrate sequential multi-agent pipeline."""
    print("\n" + "=" * 60)
    print("SEQUENTIAL PIPELINE DEMO")
    print("=" * 60)
    print("\nPattern: Researcher → Analyst → Writer")
    print("Each agent processes the output of the previous one.\n")

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("Error: OPENAI_API_KEY not set")
        return

    llm = LLMClient(model="gpt-4o-mini", api_key=api_key, temperature=0.1)

    # Create specialized agents
    researcher = ReActAgent(
        name="researcher",
        llm=llm,
        tools=[search_web],
        instructions="""You are a research specialist. Your job is to:
1. Search for relevant information on the given topic
2. Gather key facts and data points
3. Pass the raw research to the next agent

Output your findings in a structured format.""",
        verbose=True,
    )

    analyst = ReActAgent(
        name="analyst",
        llm=llm,
        tools=[analyze_data, fact_check],
        instructions="""You are a data analyst. Your job is to:
1. Analyze the research provided by the researcher
2. Verify key claims
3. Extract insights and patterns
4. Pass your analysis to the writer

Be thorough and cite your analysis.""",
        verbose=True,
    )

    writer = ReActAgent(
        name="writer",
        llm=llm,
        tools=[write_summary],
        instructions="""You are a professional writer. Your job is to:
1. Take the analysis and create a polished summary
2. Write in a clear, professional style
3. Structure the content logically

Create a final deliverable.""",
        verbose=True,
    )

    # Execute pipeline manually
    topic = "renewable energy trends"
    print(f"Topic: {topic}")
    print("-" * 60)

    print("\n[1] RESEARCHER")
    research_output = researcher.run(f"Research: {topic}")
    print(f"Output: {research_output[:200]}...")

    print("\n[2] ANALYST")
    analysis_output = analyst.run(f"Analyze this research: {research_output}")
    print(f"Output: {analysis_output[:200]}...")

    print("\n[3] WRITER")
    final_output = writer.run(f"Write a summary based on: {analysis_output}")
    print(f"Output: {final_output}")


def demo_parallel_specialists():
    """Demonstrate parallel specialist agents."""
    print("\n" + "=" * 60)
    print("PARALLEL SPECIALISTS DEMO")
    print("=" * 60)
    print("\nPattern: Multiple specialists work on different aspects")
    print("Results are combined at the end.\n")

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("Error: OPENAI_API_KEY not set")
        return

    llm = LLMClient(model="gpt-4o-mini", api_key=api_key, temperature=0.1)

    # Create specialist agents
    tech_specialist = ReActAgent(
        name="tech_specialist",
        llm=llm,
        tools=[search_web],
        instructions="You specialize in technology trends. Focus on technical aspects.",
        verbose=True,
    )

    market_specialist = ReActAgent(
        name="market_specialist",
        llm=llm,
        tools=[search_web, analyze_data],
        instructions="You specialize in market analysis. Focus on business impact.",
        verbose=True,
    )

    # Run specialists in parallel (simulated)
    topic = "AI in enterprise"
    print(f"Topic: {topic}")
    print("-" * 60)

    print("\n[TECH SPECIALIST]")
    tech_analysis = tech_specialist.run(f"Analyze technical aspects of: {topic}")
    print(f"Output: {tech_analysis[:200]}...")

    print("\n[MARKET SPECIALIST]")
    market_analysis = market_specialist.run(f"Analyze market aspects of: {topic}")
    print(f"Output: {market_analysis[:200]}...")

    print("\n[COMBINED RESULTS]")
    print("Technical View:", tech_analysis[:100])
    print("Market View:", market_analysis[:100])


def demo_agent_with_memory():
    """Demonstrate agent with persistent memory across tasks."""
    print("\n" + "=" * 60)
    print("AGENT WITH MEMORY DEMO")
    print("=" * 60)
    print("\nAgent retains context across multiple interactions.\n")

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("Error: OPENAI_API_KEY not set")
        return

    llm = LLMClient(model="gpt-4o-mini", api_key=api_key, temperature=0.1)

    from sage.memory import BufferMemory

    # Create agent with memory
    memory = BufferMemory(max_items=10)

    agent = ReActAgent(
        name="assistant",
        llm=llm,
        tools=[search_web],
        instructions="You are a helpful assistant. Remember previous interactions.",
        verbose=True,
    )

    # Simulate conversation
    queries = [
        "Search for information about Python",
        "What did we just discuss?",
        "Tell me more about that topic",
    ]

    for query in queries:
        print(f"\nUser: {query}")
        print("-" * 40)

        # Add to memory
        memory.add_user_message(query)

        # Run agent
        response = agent.run(query)
        print(f"Agent: {response[:200]}...")

        # Store response in memory
        memory.add_ai_message(response)

    print(f"\nMemory contains {len(memory.get_messages())} messages")


def main():
    """Run all multi-agent demos."""
    print("Agentic AI Toolkit - Multi-Agent Systems Demo")
    print("=" * 60)

    demo_sequential_pipeline()
    demo_parallel_specialists()
    demo_agent_with_memory()

    print("\n" + "=" * 60)
    print("Demo complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
