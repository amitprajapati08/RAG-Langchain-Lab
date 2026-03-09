"""Centralized configuration for the RAG-Langchain-Lab project.

This module defines all configuration dataclasses used throughout the
RAG pipeline. Settings are loaded from environment variables via python-dotenv.
"""

import os
from dataclasses import dataclass, field
from typing import List, Optional

from dotenv import load_dotenv

load_dotenv()


@dataclass
class EmbeddingConfig:
    """Configuration for the embedding model.

    Attributes:
        model_name: HuggingFace model identifier for sentence embeddings.
        device: Compute device ('cpu' or 'cuda').
    """

    model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    device: str = "cpu"


@dataclass
class VectorStoreConfig:
    """Configuration for the vector store backend.

    Attributes:
        store_type: Backend type, either 'faiss' or 'chroma'.
        persist_directory: Directory path for persisting the vector store.
        collection_name: Collection name (used by Chroma).
    """

    store_type: str = "faiss"
    persist_directory: str = "data/vector_store"
    collection_name: str = "rag_collection"


@dataclass
class ChunkingConfig:
    """Configuration for document chunking.

    Attributes:
        chunk_size: Maximum number of characters per chunk.
        chunk_overlap: Number of overlapping characters between consecutive chunks.
        separators: List of separator strings used by RecursiveCharacterTextSplitter.
    """

    chunk_size: int = 1000
    chunk_overlap: int = 200
    separators: List[str] = field(default_factory=lambda: ["\n\n", "\n", " ", ""])


@dataclass
class LLMConfig:
    """Configuration for the language model provider.

    Attributes:
        provider: LLM provider identifier (e.g., 'replicate').
        model_name: Model identifier on the provider platform.
        temperature: Sampling temperature controlling randomness.
        max_tokens: Maximum number of tokens to generate.
        api_key: API token loaded from the environment.
    """

    provider: str = "replicate"
    model_name: str = "meta/llama-2-70b-chat:latest"
    temperature: float = 0.3
    max_tokens: int = 512
    api_key: Optional[str] = field(
        default_factory=lambda: os.getenv("REPLICATE_API_TOKEN")
    )


@dataclass
class RetrieverConfig:
    """Configuration for the retriever component.

    Attributes:
        search_type: Type of similarity search ('similarity' or 'mmr').
        top_k: Number of documents to retrieve per query.
    """

    search_type: str = "similarity"
    top_k: int = 4


@dataclass
class RAGConfig:
    """Master configuration aggregating all sub-configs.

    Attributes:
        embedding: Embedding model configuration.
        vector_store: Vector store backend configuration.
        chunking: Document chunking configuration.
        llm: Language model configuration.
        retriever: Retriever configuration.
    """

    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    vector_store: VectorStoreConfig = field(default_factory=VectorStoreConfig)
    chunking: ChunkingConfig = field(default_factory=ChunkingConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)
    retriever: RetrieverConfig = field(default_factory=RetrieverConfig)
