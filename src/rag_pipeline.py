"""RAG pipeline orchestrator and CLI entry point.

This module exposes the :class:`RAGPipeline` facade, which ties together
document loading, embedding, vector storage, and query execution. It also
provides a command-line interface via ``argparse``.
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import Any, Dict, Optional

from config.settings import RAGConfig
from src.document_loader import DocumentLoader
from src.embedding_store import EmbeddingStore
from src.query_engine import QueryEngine

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)


class RAGPipeline:
    """High-level facade that orchestrates the full RAG workflow.

    Provides two primary workflows:

    1. **Ingest**: load documents → create embeddings → persist index → initialise query engine.
    2. **Load**: load a previously persisted index → initialise query engine.

    Args:
        config: Master RAG configuration. Defaults to :class:`~config.settings.RAGConfig`.

    Example::

        pipeline = RAGPipeline()
        pipeline.ingest("data/sample_docs")
        print(pipeline.query("What is RAG?"))
    """

    def __init__(self, config: RAGConfig = None) -> None:
        self.config = config or RAGConfig()
        self._loader = DocumentLoader(config=self.config.chunking)
        self._store = EmbeddingStore(
            embedding_config=self.config.embedding,
            vector_store_config=self.config.vector_store,
        )
        self._engine: Optional[QueryEngine] = None

    # ------------------------------------------------------------------
    # Ingestion
    # ------------------------------------------------------------------

    def ingest(self, source_path: str) -> int:
        """Load documents, build the vector store, and initialise the query engine.

        *source_path* may point to a single supported file or a directory.
        After ingestion the index is automatically persisted to
        :attr:`~config.settings.VectorStoreConfig.persist_directory`.

        Args:
            source_path: Path to a file or directory containing documents.

        Returns:
            Number of chunks ingested.
        """
        path = Path(source_path)
        if path.is_dir():
            logger.info("Ingesting directory: '%s'", source_path)
            documents = self._loader.load_directory(source_path)
        else:
            logger.info("Ingesting file: '%s'", source_path)
            documents = self._loader.load_file(source_path)

        if not documents:
            logger.warning("No documents found at '%s'.", source_path)
            return 0

        self._store.create_store(documents)
        persist_dir = self.config.vector_store.persist_directory
        self._store.save(persist_dir)
        self._engine = QueryEngine(
            retriever=self._store.retriever,
            llm_config=self.config.llm,
            retriever_config=self.config.retriever,
        )
        chunk_count = len(documents)
        logger.info("Ingestion complete. %d chunks indexed.", chunk_count)
        return chunk_count

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------

    def load_index(self, path: str) -> None:
        """Load an existing vector store from disk and initialise the query engine.

        Args:
            path: Directory path to the persisted vector store.
        """
        logger.info("Loading vector store from '%s'", path)
        self._store.load(path)
        self._engine = QueryEngine(
            retriever=self._store.retriever,
            llm_config=self.config.llm,
            retriever_config=self.config.retriever,
        )
        logger.info("Index loaded and query engine ready.")

    # ------------------------------------------------------------------
    # Querying
    # ------------------------------------------------------------------

    def query(self, question: str) -> Dict[str, Any]:
        """Answer a question using the RAG pipeline.

        Args:
            question: Natural language question.

        Returns:
            Dictionary with ``question``, ``answer``, and ``source_documents`` keys.

        Raises:
            RuntimeError: If the pipeline has not been initialised via
                :meth:`ingest` or :meth:`load_index`.
        """
        self._require_engine()
        return self._engine.query(question)

    def query_with_sources(self, question: str) -> str:
        """Answer a question and return a formatted string with source citations.

        Args:
            question: Natural language question.

        Returns:
            Formatted answer string with source file citations.

        Raises:
            RuntimeError: If the pipeline has not been initialised via
                :meth:`ingest` or :meth:`load_index`.
        """
        self._require_engine()
        return self._engine.query_with_sources(question)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _require_engine(self) -> None:
        """Raise RuntimeError if the query engine has not been initialised."""
        if self._engine is None:
            raise RuntimeError(
                "Query engine is not initialised. "
                "Call ingest() or load_index() first."
            )


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    """Build the argparse argument parser for the CLI.

    Returns:
        Configured :class:`argparse.ArgumentParser` instance.
    """
    parser = argparse.ArgumentParser(
        prog="rag_pipeline",
        description="RAG-Langchain-Lab command-line interface",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # --- ingest subcommand ---
    ingest_parser = subparsers.add_parser(
        "ingest",
        help="Load documents and build the vector index",
    )
    ingest_parser.add_argument(
        "source",
        help="Path to a file or directory containing documents to ingest",
    )
    ingest_parser.add_argument(
        "--index-dir",
        default=None,
        help="Override the directory in which to save the vector index",
    )

    # --- query subcommand ---
    query_parser = subparsers.add_parser(
        "query",
        help="Query the vector index",
    )
    query_parser.add_argument("question", help="Question to ask the RAG pipeline")
    query_parser.add_argument(
        "--index-dir",
        default=None,
        help="Directory containing the persisted vector index",
    )
    query_parser.add_argument(
        "--sources",
        action="store_true",
        help="Print source citations alongside the answer",
    )

    return parser


def main(argv=None) -> None:
    """CLI entry point for the RAG pipeline.

    Args:
        argv: Argument list (defaults to :data:`sys.argv`).
    """
    parser = _build_parser()
    args = parser.parse_args(argv)

    config = RAGConfig()
    if getattr(args, "index_dir", None):
        config.vector_store.persist_directory = args.index_dir

    pipeline = RAGPipeline(config=config)

    if args.command == "ingest":
        count = pipeline.ingest(args.source)
        print(f"Ingested {count} chunks from '{args.source}'.")

    elif args.command == "query":
        index_dir = args.index_dir or config.vector_store.persist_directory
        pipeline.load_index(index_dir)
        if args.sources:
            print(pipeline.query_with_sources(args.question))
        else:
            result = pipeline.query(args.question)
            print(result["answer"])


if __name__ == "__main__":
    main(sys.argv[1:])
