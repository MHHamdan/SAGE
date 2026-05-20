#!/usr/bin/env python
"""
Example: Simple ReAct Agent

This example demonstrates how to create a basic ReAct agent
with custom tools using the Agentic AI Toolkit.
"""

import os
from dotenv import load_dotenv
from langchain_core.tools import tool

# Load environment variables
load_dotenv()

# Import toolkit components
from sage.core.llm_client import LLMClient
from sage.agents.react_agent import ReActAgent


# Define custom tools
@tool
def get_weather(city: str) -> str:
    """Get the current weather for a city.

    Args:
        city: Name of the city

    Returns:
        Weather information for the city
    """
    # Mock weather data - in production, call a real weather API
    weather_data = {
        "new york": "Sunny, 72°F (22°C)",
        "london": "Cloudy, 59°F (15°C)",
        "tokyo": "Rainy, 68°F (20°C)",
        "sydney": "Clear, 77°F (25°C)",
    }
    city_lower = city.lower()
    if city_lower in weather_data:
        return f"Weather in {city}: {weather_data[city_lower]}"
    return f"Weather data not available for {city}"


@tool
def search_knowledge(query: str) -> str:
    """Search a knowledge base for information.

    Args:
        query: Search query

    Returns:
        Relevant information from the knowledge base
    """
    # Mock knowledge base - in production, use vector search
    knowledge = {
        "python": "Python is a high-level programming language known for readability.",
        "ai agents": "AI agents are autonomous systems that perceive, reason, and act.",
        "langchain": "LangChain is a framework for building LLM-powered applications.",
    }

    query_lower = query.lower()
    for key, value in knowledge.items():
        if key in query_lower:
            return value
    return f"No specific information found for: {query}"


def main():
    """Main function demonstrating the simple agent."""

    # Check for API key
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("Error: OPENAI_API_KEY environment variable not set")
        print("Please set your OpenAI API key in a .env file or environment")
        return

    # Initialize LLM client
    print("Initializing LLM client...")
    llm = LLMClient(
        model="gpt-4o-mini",
        api_key=api_key,
        temperature=0.1,
    )

    # Create ReAct agent with tools
    print("Creating ReAct agent...")
    agent = ReActAgent(
        name="assistant",
        llm=llm,
        tools=[get_weather, search_knowledge],
        instructions="""You are a helpful assistant that can:
1. Get weather information for cities
2. Search a knowledge base for information

Always use the appropriate tool to answer user questions.
Provide clear and helpful responses.""",
        verbose=True,
    )

    print(f"\nAgent created: {agent}")
    print("-" * 50)

    # Test queries
    test_queries = [
        "What's the weather like in New York?",
        "Tell me about Python programming",
        "What are AI agents?",
    ]

    for query in test_queries:
        print(f"\nUser: {query}")
        print("-" * 30)

        try:
            response = agent.run(query)
            print(f"Agent: {response}")
        except Exception as e:
            print(f"Error: {e}")

        print("-" * 50)


if __name__ == "__main__":
    main()
