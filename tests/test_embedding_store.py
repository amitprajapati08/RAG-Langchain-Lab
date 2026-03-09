"""Tests for the EmbeddingStore class.

These tests verify vector store creation, document addition, similarity search,
and the save/load round-trip cycle using FAISS.

Heavy ML dependencies (sentence-transformers, FAISS) are mocked where necessary
to keep the test suite fast and self-contained.
"""

from pathlib import Path
from typing import List
from unittest.mock import MagicMock, patch

import pytest
from langchain_core.documents import Document

from config.settings import EmbeddingConfig, VectorStoreConfig
from src.embedding_store import EmbeddingStore


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_docs(n: int = 3) -> List[Document]:
    """Return *n* simple :class:`~langchain.schema.Document` objects."""
    return [
        Document(page_content=f"Document number {i} about RAG.", metadata={"id": i})
        for i in range(n)
    ]


def _mock_embeddings():
    """Return a MagicMock that pretends to be HuggingFaceEmbeddings."""
    mock = MagicMock()
    # embed_documents returns one 4-dim vector per document
    mock.embed_documents.side_effect = lambda texts: [[0.1 * (j + 1)] * 4 for j, _ in enumerate(texts)]
    mock.embed_query.return_value = [0.1, 0.2, 0.3, 0.4]
    return mock


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def mock_faiss_store():
    """Return a MagicMock that pretends to be a FAISS vector store."""
    store = MagicMock()
    store.similarity_search.return_value = _make_docs(2)
    store.as_retriever.return_value = MagicMock()
    return store


@pytest.fixture()
def embedding_store(mock_faiss_store) -> EmbeddingStore:
    """Return an EmbeddingStore with mocked HuggingFaceEmbeddings and FAISS."""
    emb_cfg = EmbeddingConfig(model_name="test-model", device="cpu")
    vs_cfg = VectorStoreConfig(store_type="faiss")

    with patch(
        "src.embedding_store.HuggingFaceEmbeddings",
        return_value=_mock_embeddings(),
    ), patch(
        "langchain_community.vectorstores.FAISS.from_documents",
        return_value=mock_faiss_store,
    ):
        store = EmbeddingStore(embedding_config=emb_cfg, vector_store_config=vs_cfg)
        store._vector_store = mock_faiss_store
    return store


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestCreateStore:
    """Tests for EmbeddingStore.create_store."""

    def test_create_store_sets_vector_store(self, mock_faiss_store) -> None:
        """After create_store the internal _vector_store should be set."""
        emb_cfg = EmbeddingConfig(model_name="test-model", device="cpu")
        vs_cfg = VectorStoreConfig(store_type="faiss")

        with patch(
            "src.embedding_store.HuggingFaceEmbeddings",
            return_value=_mock_embeddings(),
        ), patch(
            "langchain_community.vectorstores.FAISS.from_documents",
            return_value=mock_faiss_store,
        ):
            store = EmbeddingStore(embedding_config=emb_cfg, vector_store_config=vs_cfg)
            store.create_store(_make_docs(3))

        assert store._vector_store is mock_faiss_store

    def test_create_store_raises_without_documents(self, embedding_store: EmbeddingStore) -> None:
        """create_store should propagate any error raised by the backend."""
        with patch.object(embedding_store, "_build_store", side_effect=ValueError("no docs")):
            with pytest.raises(ValueError, match="no docs"):
                embedding_store.create_store([])


class TestSimilaritySearch:
    """Tests for EmbeddingStore.similarity_search."""

    def test_similarity_search_returns_documents(
        self, embedding_store: EmbeddingStore
    ) -> None:
        """similarity_search should delegate to the underlying store and return docs."""
        results = embedding_store.similarity_search("What is RAG?", k=2)
        assert isinstance(results, list)
        assert len(results) == 2
        embedding_store._vector_store.similarity_search.assert_called_once_with(
            "What is RAG?", k=2
        )

    def test_similarity_search_raises_without_store(self) -> None:
        """similarity_search without a store should raise RuntimeError."""
        with patch("src.embedding_store.HuggingFaceEmbeddings", return_value=MagicMock()):
            store = EmbeddingStore()
        with pytest.raises(RuntimeError, match="No vector store available"):
            store.similarity_search("question")


class TestAddDocuments:
    """Tests for EmbeddingStore.add_documents."""

    def test_add_documents_calls_add_on_existing_store(
        self, embedding_store: EmbeddingStore
    ) -> None:
        """add_documents should call add_documents on the underlying store."""
        docs = _make_docs(2)
        embedding_store.add_documents(docs)
        embedding_store._vector_store.add_documents.assert_called_once_with(docs)

    def test_add_documents_creates_store_when_none(self, mock_faiss_store) -> None:
        """add_documents should create a store if none exists."""
        with patch(
            "src.embedding_store.HuggingFaceEmbeddings",
            return_value=_mock_embeddings(),
        ), patch(
            "langchain_community.vectorstores.FAISS.from_documents",
            return_value=mock_faiss_store,
        ):
            store = EmbeddingStore(
                embedding_config=EmbeddingConfig(model_name="m", device="cpu"),
                vector_store_config=VectorStoreConfig(store_type="faiss"),
            )
            store.add_documents(_make_docs(2))

        assert store._vector_store is mock_faiss_store


class TestSaveLoad:
    """Tests for EmbeddingStore.save and EmbeddingStore.load."""

    def test_save_calls_save_local(
        self, embedding_store: EmbeddingStore, tmp_path: Path
    ) -> None:
        """save() should call save_local on the underlying FAISS store."""
        embedding_store.save(str(tmp_path))
        embedding_store._vector_store.save_local.assert_called_once_with(str(tmp_path))

    def test_save_raises_without_store(self) -> None:
        """save() without a store should raise RuntimeError."""
        with patch("src.embedding_store.HuggingFaceEmbeddings", return_value=MagicMock()):
            store = EmbeddingStore()
        with pytest.raises(RuntimeError, match="No vector store to save"):
            store.save("/tmp/index")

    def test_load_raises_when_path_missing(self) -> None:
        """load() with a non-existent path should raise FileNotFoundError."""
        with patch("src.embedding_store.HuggingFaceEmbeddings", return_value=MagicMock()):
            store = EmbeddingStore()
        with pytest.raises(FileNotFoundError):
            store.load("/nonexistent/path")

    def test_load_sets_vector_store(self, mock_faiss_store, tmp_path: Path) -> None:
        """load() should set the internal vector store from a persisted index."""
        (tmp_path / "index.faiss").touch()
        (tmp_path / "index.pkl").touch()

        with patch("src.embedding_store.HuggingFaceEmbeddings", return_value=_mock_embeddings()), \
             patch(
                 "langchain_community.vectorstores.FAISS.load_local",
                 return_value=mock_faiss_store,
             ):
            store = EmbeddingStore(
                embedding_config=EmbeddingConfig(model_name="m", device="cpu"),
                vector_store_config=VectorStoreConfig(store_type="faiss"),
            )
            store.load(str(tmp_path))

        assert store._vector_store is mock_faiss_store
