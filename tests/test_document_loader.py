"""Tests for the DocumentLoader class.

These tests verify file loading, chunk production, metadata enrichment, and
appropriate error handling.
"""

import textwrap
from pathlib import Path
from typing import List
from unittest.mock import MagicMock, patch

import pytest

from config.settings import ChunkingConfig
from src.document_loader import DocumentLoader


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def loader() -> DocumentLoader:
    """Return a DocumentLoader with a small chunk size for testing."""
    config = ChunkingConfig(chunk_size=200, chunk_overlap=20)
    return DocumentLoader(config=config)


@pytest.fixture()
def sample_txt_file(tmp_path: Path) -> Path:
    """Write a temporary plain-text file and return its path."""
    content = textwrap.dedent(
        """\
        Retrieval Augmented Generation (RAG) is a technique that enhances
        large language models by combining them with external knowledge retrieval.

        RAG retrieves relevant documents at inference time and uses them as context
        for the language model, reducing hallucination and improving accuracy.

        The pipeline consists of document ingestion, embedding, vector storage,
        retrieval, and generation steps.
        """
    )
    file_path = tmp_path / "sample.txt"
    file_path.write_text(content)
    return file_path


@pytest.fixture()
def sample_md_file(tmp_path: Path) -> Path:
    """Write a temporary Markdown file and return its path."""
    content = "# RAG Overview\n\nRAG stands for Retrieval Augmented Generation.\n"
    file_path = tmp_path / "overview.md"
    file_path.write_text(content)
    return file_path


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestLoadFile:
    """Tests for DocumentLoader.load_file."""

    def test_load_file_returns_chunks(
        self, loader: DocumentLoader, sample_txt_file: Path
    ) -> None:
        """Loading a text file should produce at least one chunk."""
        chunks = loader.load_file(str(sample_txt_file))
        assert isinstance(chunks, list)
        assert len(chunks) >= 1

    def test_load_file_metadata(
        self, loader: DocumentLoader, sample_txt_file: Path
    ) -> None:
        """Each chunk should have source_file, chunk_index, and total_chunks metadata."""
        chunks = loader.load_file(str(sample_txt_file))
        for chunk in chunks:
            assert "source_file" in chunk.metadata
            assert "chunk_index" in chunk.metadata
            assert "total_chunks" in chunk.metadata
            assert chunk.metadata["total_chunks"] == len(chunks)

    def test_load_file_chunk_indices_are_sequential(
        self, loader: DocumentLoader, sample_txt_file: Path
    ) -> None:
        """chunk_index values should be 0-based and contiguous."""
        chunks = loader.load_file(str(sample_txt_file))
        for idx, chunk in enumerate(chunks):
            assert chunk.metadata["chunk_index"] == idx

    def test_load_file_not_found(self, loader: DocumentLoader) -> None:
        """Loading a non-existent file should raise FileNotFoundError."""
        with pytest.raises(FileNotFoundError, match="not found"):
            loader.load_file("/nonexistent/path/file.txt")

    def test_unsupported_file_type(
        self, loader: DocumentLoader, tmp_path: Path
    ) -> None:
        """Loading a file with an unsupported extension should raise ValueError."""
        bad_file = tmp_path / "document.xyz"
        bad_file.write_text("content")
        with pytest.raises(ValueError, match="Unsupported file type"):
            loader.load_file(str(bad_file))


class TestLoadDirectory:
    """Tests for DocumentLoader.load_directory."""

    def test_load_directory(
        self,
        loader: DocumentLoader,
        sample_txt_file: Path,
        sample_md_file: Path,
        tmp_path: Path,
    ) -> None:
        """Loading a directory should return chunks from all supported files."""
        # Both fixture files are already written inside tmp_path.
        chunks = loader.load_directory(str(tmp_path))
        assert len(chunks) >= 2  # at least one chunk per file

    def test_load_directory_not_found(self, loader: DocumentLoader) -> None:
        """Loading a non-existent directory should raise FileNotFoundError."""
        with pytest.raises(FileNotFoundError, match="not found"):
            loader.load_directory("/nonexistent/directory")

    def test_load_directory_empty(
        self, loader: DocumentLoader, tmp_path: Path
    ) -> None:
        """Loading an empty directory should return an empty list."""
        empty_dir = tmp_path / "empty"
        empty_dir.mkdir()
        chunks = loader.load_directory(str(empty_dir))
        assert chunks == []
