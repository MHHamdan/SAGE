"""Base memory class for agent memory systems."""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional


class BaseMemory(ABC):
    """Abstract base class for memory implementations.

    Memory systems enable agents to maintain context across interactions,
    store and retrieve information, and learn from experience.
    """

    def __init__(self, max_items: int = 100):
        """Initialize the memory system.

        Args:
            max_items: Maximum number of items to store
        """
        self.max_items = max_items

    @abstractmethod
    def add(self, item: Any, metadata: Optional[Dict] = None) -> None:
        """Add an item to memory.

        Args:
            item: Item to store
            metadata: Optional metadata for the item
        """
        pass

    @abstractmethod
    def get(self, query: str, k: int = 5) -> List[Any]:
        """Retrieve items from memory.

        Args:
            query: Search query
            k: Number of items to retrieve

        Returns:
            List of retrieved items
        """
        pass

    @abstractmethod
    def clear(self) -> None:
        """Clear all items from memory."""
        pass

    @abstractmethod
    def get_all(self) -> List[Any]:
        """Get all items in memory.

        Returns:
            List of all stored items
        """
        pass

    def __len__(self) -> int:
        """Return number of items in memory."""
        return len(self.get_all())
