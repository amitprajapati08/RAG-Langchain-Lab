"""Tests for the QueryEngine class.

The Replicate LLM is mocked throughout to avoid API calls during testing.
"""

from typing import List
from unittest.mock import MagicMock, patch

import pytest
from langchain_core.documents import Document

from config.settings import LLMConfig, RetrieverConfig
from src.query_engine import QueryEngine


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_doc(content: str, source: str = "test_doc.txt", idx: int = 0) -> Document:
    """Return a :class:`~langchain_core.documents.Document` with standard metadata."""
    return Document(
        page_content=content,
        metadata={"source_file": source, "chunk_index": idx, "total_chunks": 1},
    )


def _mock_replicate_llm(answer: str = "Mocked answer."):
    """Return a MagicMock that behaves like a LangChain LLM returning *answer*."""
    llm = MagicMock()
    llm.invoke.return_value = answer
    return llm


def _mock_retriever(docs: List[Document] = None) -> MagicMock:
    """Return a MagicMock retriever that returns *docs* on invoke."""
    retriever = MagicMock()
    retriever.search_kwargs = {}
    retriever.invoke.return_value = docs or [_make_doc("Context text.")]
    return retriever


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def mock_retriever() -> MagicMock:
    return _mock_retriever()


@pytest.fixture()
def query_engine(mock_retriever: MagicMock) -> QueryEngine:
    """Return a QueryEngine with a mocked LLM and LCEL chain."""
    llm_cfg = LLMConfig(api_key="fake-token")
    ret_cfg = RetrieverConfig(top_k=2)

    mock_chain = MagicMock()
    mock_chain.invoke.return_value = "Mocked answer."

    with patch("src.query_engine.Replicate", return_value=_mock_replicate_llm()), \
         patch.object(QueryEngine, "_build_chain", return_value=mock_chain):
        engine = QueryEngine(
            retriever=mock_retriever,
            llm_config=llm_cfg,
            retriever_config=ret_cfg,
        )
    # Ensure chain is the mock
    engine._chain = mock_chain
    return engine


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestQueryEngineInit:
    """Tests for QueryEngine initialisation."""

    def test_init_sets_retriever_top_k(self, mock_retriever: MagicMock) -> None:
        """Initialisation should apply top_k to the retriever's search_kwargs."""
        ret_cfg = RetrieverConfig(top_k=5)
        mock_chain = MagicMock()
        with patch("src.query_engine.Replicate", return_value=_mock_replicate_llm()), \
             patch.object(QueryEngine, "_build_chain", return_value=mock_chain):
            QueryEngine(
                retriever=mock_retriever,
                llm_config=LLMConfig(api_key="token"),
                retriever_config=ret_cfg,
            )
        assert mock_retriever.search_kwargs.get("k") == 5


class TestQuery:
    """Tests for QueryEngine.query."""

    def test_query_returns_expected_keys(self, query_engine: QueryEngine) -> None:
        """query() should return a dict with question, answer, and source_documents."""
        result = query_engine.query("What is RAG?")
        assert "question" in result
        assert "answer" in result
        assert "source_documents" in result

    def test_query_preserves_question(self, query_engine: QueryEngine) -> None:
        """The returned dict should echo the original question."""
        question = "Explain RAG in simple terms."
        result = query_engine.query(question)
        assert result["question"] == question

    def test_query_returns_answer_string(self, query_engine: QueryEngine) -> None:
        """The answer field should be a non-empty string."""
        result = query_engine.query("What is RAG?")
        assert isinstance(result["answer"], str)
        assert len(result["answer"]) > 0

    def test_query_returns_source_documents(self, query_engine: QueryEngine) -> None:
        """source_documents should be a list."""
        result = query_engine.query("What is RAG?")
        assert isinstance(result["source_documents"], list)


class TestQueryWithSources:
    """Tests for QueryEngine.query_with_sources."""

    def test_query_with_sources_returns_string(self, query_engine: QueryEngine) -> None:
        """query_with_sources() should return a formatted string."""
        output = query_engine.query_with_sources("What is RAG?")
        assert isinstance(output, str)

    def test_query_with_sources_contains_answer(self, query_engine: QueryEngine) -> None:
        """The output string should contain an 'Answer:' section."""
        output = query_engine.query_with_sources("What is RAG?")
        assert "Answer:" in output

    def test_query_with_sources_contains_sources(self, query_engine: QueryEngine) -> None:
        """The output string should contain a 'Sources:' section."""
        output = query_engine.query_with_sources("What is RAG?")
        assert "Sources:" in output

