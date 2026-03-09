"""RAG-Langchain-Lab source package.

This package exposes the core components of the RAG pipeline:
- RAGPipeline: High-level orchestrator (facade).
- DocumentLoader: Loads and chunks documents from disk.
- EmbeddingStore: Creates and manages vector stores.
- QueryEngine: Executes RAG queries against a vector store.
"""

from src.document_loader import DocumentLoader
from src.embedding_store import EmbeddingStore
from src.query_engine import QueryEngine
from src.rag_pipeline import RAGPipeline

__version__ = "1.0.0"

__all__ = [
    "RAGPipeline",
    "DocumentLoader",
    "EmbeddingStore",
    "QueryEngine",
]
