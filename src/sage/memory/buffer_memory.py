"""Buffer memory for short-term conversation context."""

from collections import deque
from typing import Any, Dict, List, Optional

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage

from sage.memory.base_memory import BaseMemory


class BufferMemory(BaseMemory):
    """Buffer memory for maintaining recent conversation history.

    This memory type maintains a fixed-size buffer of recent messages,
    implementing a FIFO (First-In-First-Out) strategy.

    Example:
        >>> memory = BufferMemory(max_items=10)
        >>> memory.add(HumanMessage(content="Hello"))
        >>> memory.add(AIMessage(content="Hi there!"))
        >>> messages = memory.get_all()
    """

    def __init__(self, max_items: int = 10, return_messages: bool = True):
        """Initialize buffer memory.

        Args:
            max_items: Maximum number of messages to retain
            return_messages: Return as Message objects if True, else strings
        """
        super().__init__(max_items=max_items)
        self._buffer: deque = deque(maxlen=max_items)
        self.return_messages = return_messages

    def add(self, item: Any, metadata: Optional[Dict] = None) -> None:
        """Add a message to the buffer.

        Args:
            item: Message to add (string or BaseMessage)
            metadata: Optional metadata (stored with message)
        """
        if isinstance(item, str):
            item = HumanMessage(content=item)

        entry = {"message": item, "metadata": metadata or {}}
        self._buffer.append(entry)

    def add_user_message(self, content: str, metadata: Optional[Dict] = None) -> None:
        """Add a user message.

        Args:
            content: Message content
            metadata: Optional metadata
        """
        self.add(HumanMessage(content=content), metadata)

    def add_ai_message(self, content: str, metadata: Optional[Dict] = None) -> None:
        """Add an AI message.

        Args:
            content: Message content
            metadata: Optional metadata
        """
        self.add(AIMessage(content=content), metadata)

    def get(self, query: str = None, k: int = 5) -> List[Any]:
        """Get recent messages from buffer.

        Args:
            query: Ignored for buffer memory
            k: Number of recent messages to return

        Returns:
            List of recent messages
        """
        items = list(self._buffer)[-k:]
        if self.return_messages:
            return [entry["message"] for entry in items]
        return [entry["message"].content for entry in items]

    def get_all(self) -> List[Any]:
        """Get all messages in buffer.

        Returns:
            List of all buffered messages
        """
        if self.return_messages:
            return [entry["message"] for entry in self._buffer]
        return [entry["message"].content for entry in self._buffer]

    def clear(self) -> None:
        """Clear the buffer."""
        self._buffer.clear()

    def get_context_string(self, separator: str = "\n") -> str:
        """Get buffer contents as a formatted string.

        Args:
            separator: Separator between messages

        Returns:
            Formatted context string
        """
        messages = []
        for entry in self._buffer:
            msg = entry["message"]
            role = "Human" if isinstance(msg, HumanMessage) else "AI"
            messages.append(f"{role}: {msg.content}")
        return separator.join(messages)

    def __repr__(self) -> str:
        return f"BufferMemory(items={len(self._buffer)}, max={self.max_items})"
