"""Document loading and chunking utilities for the RAG pipeline.

This module provides the :class:`DocumentLoader` class, which can ingest PDF,
TXT, and Markdown files and split them into overlapping text chunks suitable
for embedding and retrieval.
"""

import logging
from pathlib import Path
from typing import Dict, List, Type

from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
    UnstructuredMarkdownLoader,
)
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from config.settings import ChunkingConfig

logger = logging.getLogger(__name__)

# Mapping of lowercase file extensions to their corresponding LangChain loaders.
LOADER_MAP: Dict[str, Type] = {
    ".pdf": PyPDFLoader,
    ".txt": TextLoader,
    ".md": UnstructuredMarkdownLoader,
}


class DocumentLoader:
    """Loads documents from disk and splits them into text chunks.

    Supports PDF (``.pdf``), plain text (``.txt``), and Markdown (``.md``)
    files. Chunking behaviour is controlled by :class:`~config.settings.ChunkingConfig`.

    Args:
        config: Chunking configuration. Defaults to :class:`~config.settings.ChunkingConfig`.

    Example::

        loader = DocumentLoader()
        chunks = loader.load_file("docs/report.pdf")
        print(f"Loaded {len(chunks)} chunks")
    """

    def __init__(self, config: ChunkingConfig = None) -> None:
        self.config = config or ChunkingConfig()
        self._splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.config.chunk_size,
            chunk_overlap=self.config.chunk_overlap,
            separators=self.config.separators,
        )

    def load_file(self, file_path: str) -> List[Document]:
        """Load a single file and return a list of chunked :class:`~langchain.schema.Document` objects.

        Each returned document is enriched with the following metadata keys:

        - ``source_file``: absolute path of the source file.
        - ``chunk_index``: zero-based index of this chunk within the file.
        - ``total_chunks``: total number of chunks produced from this file.

        Args:
            file_path: Path to the file to load.

        Returns:
            List of chunked Document objects with enriched metadata.

        Raises:
            FileNotFoundError: If *file_path* does not exist.
            ValueError: If the file extension is not supported.
        """
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        extension = path.suffix.lower()
        if extension not in LOADER_MAP:
            raise ValueError(
                f"Unsupported file type '{extension}'. "
                f"Supported types: {list(LOADER_MAP.keys())}"
            )

        loader_cls = LOADER_MAP[extension]
        logger.info("Loading file '%s' with %s", file_path, loader_cls.__name__)
        loader = loader_cls(str(path))
        raw_docs = loader.load()

        chunks = self._splitter.split_documents(raw_docs)
        total = len(chunks)
        for idx, chunk in enumerate(chunks):
            chunk.metadata["source_file"] = str(path.resolve())
            chunk.metadata["chunk_index"] = idx
            chunk.metadata["total_chunks"] = total

        logger.info("Produced %d chunks from '%s'", total, file_path)
        return chunks

    def load_directory(self, dir_path: str) -> List[Document]:
        """Recursively load all supported files from a directory.

        Args:
            dir_path: Path to the directory to scan.

        Returns:
            Concatenated list of chunks from all supported files found.

        Raises:
            FileNotFoundError: If *dir_path* does not exist.
        """
        directory = Path(dir_path)
        if not directory.exists():
            raise FileNotFoundError(f"Directory not found: {dir_path}")

        all_chunks: List[Document] = []
        for ext in LOADER_MAP:
            for file_path in directory.rglob(f"*{ext}"):
                logger.info("Processing '%s'", file_path)
                try:
                    chunks = self.load_file(str(file_path))
                    all_chunks.extend(chunks)
                except (OSError, ValueError, RuntimeError, ImportError) as exc:
                    logger.warning("Skipping '%s': %s", file_path, exc)

        logger.info(
            "Loaded %d total chunks from directory '%s'", len(all_chunks), dir_path
        )
        return all_chunks
