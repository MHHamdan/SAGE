#!/usr/bin/env python
"""
Example: Memory Systems

This example demonstrates the different memory types available
in the Agentic AI Toolkit: BufferMemory (working memory) and
VectorMemory (semantic long-term memory).
"""

import os
from dotenv import load_dotenv

load_dotenv()

from sage.memory import BufferMemory, VectorMemory


def demo_buffer_memory():
    """Demonstrate BufferMemory (working memory)."""
    print("\n" + "=" * 50)
    print("BUFFER MEMORY DEMO (Working Memory)")
    print("=" * 50)

    # Create buffer with limited capacity
    buffer = BufferMemory(max_items=5)
    print(f"\nCreated BufferMemory with max_items=5")

    # Add messages
    messages = [
        ("user", "Hello, I'm learning about AI agents"),
        ("ai", "Hello! I'd be happy to help you learn about AI agents."),
        ("user", "What is a ReAct agent?"),
        ("ai", "ReAct combines reasoning and acting in an interleaved loop."),
        ("user", "Can you give me an example?"),
        ("ai", "Sure! A ReAct agent might think, then search, observe results..."),
        ("user", "That makes sense. What about memory?"),
        ("ai", "Memory systems help agents retain context across interactions."),
    ]

    for role, content in messages:
        if role == "user":
            buffer.add_user_message(content)
        else:
            buffer.add_ai_message(content)
        print(f"Added {role} message: '{content[:40]}...'")

    # Check buffer contents
    all_messages = buffer.get_messages()
    print(f"\nBuffer now contains {len(all_messages)} messages (max was 5)")
    print("Note: Oldest messages were removed to maintain limit")

    # Display retained messages
    print("\nRetained messages:")
    for msg in all_messages:
        role = "user" if msg.type == "human" else "ai"
        print(f"  [{role}] {msg.content[:50]}...")

    # Clear and verify
    buffer.clear()
    print(f"\nAfter clear: {len(buffer.get_messages())} messages")


def demo_vector_memory():
    """Demonstrate VectorMemory (semantic long-term memory)."""
    print("\n" + "=" * 50)
    print("VECTOR MEMORY DEMO (Semantic Long-Term Memory)")
    print("=" * 50)

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("\nSkipping VectorMemory demo - OPENAI_API_KEY not set")
        print("VectorMemory requires embeddings which need an API key")
        return

    # Create vector memory
    vector_memory = VectorMemory(
        embedding_model="text-embedding-3-small",
        collection_name="demo_memory",
    )
    print(f"\nCreated VectorMemory with embedding model")

    # Add knowledge
    knowledge_items = [
        "Python is a high-level programming language known for readability.",
        "JavaScript is primarily used for web development.",
        "Machine learning is a subset of artificial intelligence.",
        "Neural networks are inspired by biological brain structures.",
        "LangChain is a framework for building LLM applications.",
        "Agents can use tools to interact with external systems.",
        "Vector databases store embeddings for semantic search.",
        "ReAct agents interleave reasoning with action execution.",
    ]

    print("\nAdding knowledge items:")
    for item in knowledge_items:
        vector_memory.add(item)
        print(f"  + {item[:50]}...")

    # Semantic search queries
    queries = [
        "What programming languages exist?",
        "Tell me about AI and machine learning",
        "How do agents work?",
    ]

    print("\nSemantic search results:")
    for query in queries:
        print(f"\nQuery: '{query}'")
        results = vector_memory.get(query, k=2)
        for i, doc in enumerate(results, 1):
            print(f"  {i}. {doc.page_content}")

    # Search with scores
    print("\nSearch with similarity scores:")
    results_with_scores = vector_memory.get_with_scores("neural networks AI", k=3)
    for doc, score in results_with_scores:
        print(f"  Score: {score:.4f} - {doc.page_content[:50]}...")


def demo_memory_with_metadata():
    """Demonstrate using metadata for filtering."""
    print("\n" + "=" * 50)
    print("METADATA FILTERING DEMO")
    print("=" * 50)

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("\nSkipping metadata demo - OPENAI_API_KEY not set")
        return

    vector_memory = VectorMemory(
        embedding_model="text-embedding-3-small",
        collection_name="metadata_demo",
    )

    # Add items with metadata
    items_with_metadata = [
        ("Python basics tutorial", {"category": "tutorial", "language": "python"}),
        ("Advanced Python patterns", {"category": "advanced", "language": "python"}),
        ("JavaScript for beginners", {"category": "tutorial", "language": "javascript"}),
        ("React component patterns", {"category": "advanced", "language": "javascript"}),
    ]

    print("\nAdding items with metadata:")
    for content, metadata in items_with_metadata:
        vector_memory.add(content, metadata=metadata)
        print(f"  + {content} | {metadata}")

    # Query with filter
    print("\nQuery: 'programming patterns' (filtered to python only)")
    results = vector_memory.get(
        "programming patterns",
        k=2,
        filter_dict={"language": "python"},
    )
    for doc in results:
        print(f"  - {doc.page_content} | {doc.metadata}")


def main():
    """Run all memory demos."""
    print("Agentic AI Toolkit - Memory Systems Demo")
    print("========================================")

    demo_buffer_memory()
    demo_vector_memory()
    demo_memory_with_metadata()

    print("\n" + "=" * 50)
    print("Demo complete!")
    print("=" * 50)


if __name__ == "__main__":
    main()
