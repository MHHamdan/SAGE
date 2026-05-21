"""Vector memory for semantic search and retrieval."""

from typing import List, Dict, Any, Optional
import logging

from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document

from sage.memory.base_memory import BaseMemory

logger = logging.getLogger(__name__)


class VectorMemory(BaseMemory):
    """Vector memory using embeddings for semantic retrieval.

    This memory type stores content as vector embeddings, enabling
    semantic similarity search for relevant information retrieval.

    Example:
        >>> memory = VectorMemory(
        ...     embedding_model="text-embedding-3-small",
        ...     persist_directory="./memory_store"
        ... )
        >>> memory.add("Python is a programming language")
        >>> results = memory.get("What programming languages exist?")
    """

    def __init__(
        self,
        embedding_model: str = "text-embedding-3-small",
        api_key: Optional[str] = None,
        persist_directory: Optional[str] = None,
        collection_name: str = "agent_memory",
        max_items: int = 10000,
    ):
        """Initialize vector memory.

        Args:
            embedding_model: OpenAI embedding model name
            api_key: OpenAI API key
            persist_directory: Directory for persistent storage
            collection_name: Name for the vector collection
            max_items: Maximum items to store
        """
        super().__init__(max_items=max_items)

        self.embedding_model = embedding_model
        self.persist_directory = persist_directory
        self.collection_name = collection_name

        # Initialize embeddings
        self._embeddings = OpenAIEmbeddings(
            model=embedding_model,
            api_key=api_key,
        )

        # Initialize vector store
        self._vector_store = Chroma(
            collection_name=collection_name,
            embedding_function=self._embeddings,
            persist_directory=persist_directory,
        )

        self._documents: List[Document] = []

    def add(
        self,
        item: Any,
        metadata: Optional[Dict] = None,
    ) -> None:
        """Add content to vector memory.

        Args:
            item: Content to store (string or Document)
            metadata: Optional metadata for the content
        """
        if isinstance(item, str):
            doc = Document(page_content=item, metadata=metadata or {})
        elif isinstance(item, Document):
            doc = item
            if metadata:
                doc.metadata.update(metadata)
        else:
            doc = Document(page_content=str(item), metadata=metadata or {})

        self._vector_store.add_documents([doc])
        self._documents.append(doc)

        # Trim if exceeds max
        if len(self._documents) > self.max_items:
            self._documents = self._documents[-self.max_items :]

    def add_texts(
        self,
        texts: List[str],
        metadatas: Optional[List[Dict]] = None,
    ) -> None:
        """Add multiple texts to memory.

        Args:
            texts: List of text content
            metadatas: Optional list of metadata dicts
        """
        if metadatas is None:
            metadatas = [{}] * len(texts)

        docs = [
            Document(page_content=text, metadata=meta)
            for text, meta in zip(texts, metadatas)
        ]

        self._vector_store.add_documents(docs)
        self._documents.extend(docs)

    def get(
        self,
        query: str,
        k: int = 5,
        filter_dict: Optional[Dict] = None,
    ) -> List[Document]:
        """Retrieve relevant documents by semantic similarity.

        Args:
            query: Search query
            k: Number of documents to retrieve
            filter_dict: Optional metadata filter

        Returns:
            List of relevant documents
        """
        if filter_dict:
            results = self._vector_store.similarity_search(
                query, k=k, filter=filter_dict
            )
        else:
            results = self._vector_store.similarity_search(query, k=k)

        return results

    def get_with_scores(
        self,
        query: str,
        k: int = 5,
    ) -> List[tuple]:
        """Retrieve documents with similarity scores.

        Args:
            query: Search query
            k: Number of documents to retrieve

        Returns:
            List of (document, score) tuples
        """
        return self._vector_store.similarity_search_with_score(query, k=k)

    def get_all(self) -> List[Document]:
        """Get all stored documents.

        Returns:
            List of all documents
        """
        return self._documents.copy()

    def clear(self) -> None:
        """Clear the vector store."""
        # Recreate the collection
        self._vector_store = Chroma(
            collection_name=self.collection_name,
            embedding_function=self._embeddings,
            persist_directory=self.persist_directory,
        )
        self._documents = []

    def as_retriever(self, search_kwargs: Optional[Dict] = None):
        """Get a LangChain retriever interface.

        Args:
            search_kwargs: Search parameters (e.g., {"k": 4})

        Returns:
            Retriever interface
        """
        return self._vector_store.as_retriever(search_kwargs=search_kwargs or {"k": 4})

    def __repr__(self) -> str:
        return (
            f"VectorMemory(collection='{self.collection_name}', "
            f"docs={len(self._documents)})"
        )
