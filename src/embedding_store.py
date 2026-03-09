"""Vector store management for the RAG pipeline.

This module provides the :class:`EmbeddingStore` class, which supports both
FAISS and Chroma vector store backends and exposes a uniform interface for
creating, persisting, loading, and searching embeddings.
"""

import logging
from pathlib import Path
from typing import List, Optional

from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings

from config.settings import EmbeddingConfig, VectorStoreConfig

logger = logging.getLogger(__name__)


class EmbeddingStore:
    """Manages vector store creation, persistence, and retrieval.

    Supports FAISS and Chroma backends, selected via
    :attr:`~config.settings.VectorStoreConfig.store_type`.

    Args:
        embedding_config: Embedding model configuration.
        vector_store_config: Vector store backend configuration.

    Example::

        store = EmbeddingStore()
        store.create_store(documents)
        results = store.similarity_search("What is RAG?", k=4)
    """

    def __init__(
        self,
        embedding_config: EmbeddingConfig = None,
        vector_store_config: VectorStoreConfig = None,
    ) -> None:
        self.embedding_config = embedding_config or EmbeddingConfig()
        self.vector_store_config = vector_store_config or VectorStoreConfig()
        self._vector_store = None

        logger.info(
            "Initialising embeddings with model '%s'",
            self.embedding_config.model_name,
        )
        self._embeddings = HuggingFaceEmbeddings(
            model_name=self.embedding_config.model_name,
            model_kwargs={"device": self.embedding_config.device},
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def create_store(self, documents: List[Document]) -> None:
        """Create a new vector store from a list of documents.

        Any previously held in-memory store is replaced.

        Args:
            documents: List of :class:`~langchain.schema.Document` objects to embed.
        """
        logger.info(
            "Creating %s vector store with %d documents",
            self.vector_store_config.store_type,
            len(documents),
        )
        self._vector_store = self._build_store(documents)

    def add_documents(self, documents: List[Document]) -> None:
        """Add documents to the existing vector store, creating it if necessary.

        Args:
            documents: Documents to add.
        """
        if self._vector_store is None:
            logger.info("No existing store found; creating a new one.")
            self.create_store(documents)
        else:
            logger.info("Adding %d documents to existing store.", len(documents))
            self._vector_store.add_documents(documents)

    def save(self, path: str) -> None:
        """Persist the vector store to disk.

        For FAISS, the index is saved to *path*. For Chroma, persistence is
        handled automatically via the configured ``persist_directory``.

        Args:
            path: Directory path at which to save the FAISS index.

        Raises:
            RuntimeError: If no vector store has been created yet.
        """
        if self._vector_store is None:
            raise RuntimeError("No vector store to save. Call create_store first.")

        if self.vector_store_config.store_type == "faiss":
            Path(path).mkdir(parents=True, exist_ok=True)
            self._vector_store.save_local(path)
            logger.info("FAISS index saved to '%s'", path)
        else:
            logger.info("Chroma store auto-persists to '%s'", path)

    def load(self, path: str) -> None:
        """Load a persisted vector store from disk.

        Args:
            path: Directory path from which to load the FAISS index.

        Raises:
            FileNotFoundError: If *path* does not exist.
            ValueError: If an unsupported store type is configured.
        """
        if not Path(path).exists():
            raise FileNotFoundError(f"Vector store path not found: {path}")

        store_type = self.vector_store_config.store_type
        logger.info("Loading %s vector store from '%s'", store_type, path)

        if store_type == "faiss":
            from langchain_community.vectorstores import FAISS

            # allow_dangerous_deserialization is required by LangChain's FAISS
            # loader for pickle-backed indices. Only load indices from trusted
            # sources that you have verified yourself.
            self._vector_store = FAISS.load_local(
                path,
                self._embeddings,
                allow_dangerous_deserialization=True,
            )
        elif store_type == "chroma":
            from langchain_community.vectorstores import Chroma

            self._vector_store = Chroma(
                collection_name=self.vector_store_config.collection_name,
                embedding_function=self._embeddings,
                persist_directory=path,
            )
        else:
            raise ValueError(f"Unsupported store type: {store_type}")

    def similarity_search(self, query: str, k: int = 4) -> List[Document]:
        """Perform a similarity search against the vector store.

        Args:
            query: Natural language query string.
            k: Number of nearest documents to return.

        Returns:
            List of the *k* most similar :class:`~langchain.schema.Document` objects.

        Raises:
            RuntimeError: If no vector store has been created or loaded.
        """
        if self._vector_store is None:
            raise RuntimeError(
                "No vector store available. "
                "Call create_store() or load() first."
            )
        logger.debug("Searching for top-%d documents for query: '%s'", k, query)
        return self._vector_store.similarity_search(query, k=k)

    @property
    def retriever(self):
        """Return a LangChain retriever interface for this vector store.

        Raises:
            RuntimeError: If no vector store has been created or loaded.
        """
        if self._vector_store is None:
            raise RuntimeError(
                "No vector store available. "
                "Call create_store() or load() first."
            )
        return self._vector_store.as_retriever()

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _build_store(self, documents: List[Document]):
        """Instantiate the appropriate vector store backend.

        Args:
            documents: Documents to index.

        Returns:
            An initialised LangChain vector store instance.

        Raises:
            ValueError: If the configured store type is not recognised.
        """
        store_type = self.vector_store_config.store_type
        if store_type == "faiss":
            from langchain_community.vectorstores import FAISS

            return FAISS.from_documents(documents, self._embeddings)
        elif store_type == "chroma":
            from langchain_community.vectorstores import Chroma

            return Chroma.from_documents(
                documents,
                self._embeddings,
                collection_name=self.vector_store_config.collection_name,
                persist_directory=self.vector_store_config.persist_directory,
            )
        else:
            raise ValueError(f"Unsupported store type: {store_type}")
