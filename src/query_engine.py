"""RAG query execution engine.

This module provides the :class:`QueryEngine` class, which wraps a LangChain
LCEL chain backed by a Replicate LLM and a configurable retriever.
"""

import logging
from typing import Any, Dict, List

from langchain_community.llms import Replicate
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough

from config.settings import LLMConfig, RetrieverConfig

logger = logging.getLogger(__name__)

_RAG_PROMPT_TEMPLATE = """You are a helpful assistant. Use ONLY the following context to answer the question. If the context doesn't contain enough information to answer the question, say "I don't have enough information to answer this question."

Context:
{context}

Question: {question}

Answer: Provide a clear, concise answer based strictly on the context above."""

RAG_PROMPT = PromptTemplate(
    template=_RAG_PROMPT_TEMPLATE,
    input_variables=["context", "question"],
)


def _format_docs(docs: List[Document]) -> str:
    """Concatenate document page content into a single context string.

    Args:
        docs: List of retrieved documents.

    Returns:
        Newline-separated string of document contents.
    """
    return "\n\n".join(doc.page_content for doc in docs)


class QueryEngine:
    """Executes RAG queries against a vector store retriever.

    Combines a Replicate LLM with a LangChain LCEL chain to answer
    natural-language questions grounded in the provided document context.

    Args:
        retriever: A LangChain retriever (e.g. from :class:`~src.embedding_store.EmbeddingStore`).
        llm_config: LLM configuration. Defaults to :class:`~config.settings.LLMConfig`.
        retriever_config: Retriever configuration. Defaults to :class:`~config.settings.RetrieverConfig`.

    Example::

        engine = QueryEngine(retriever=store.retriever)
        result = engine.query("What is RAG?")
        print(result["answer"])
    """

    def __init__(
        self,
        retriever,
        llm_config: LLMConfig = None,
        retriever_config: RetrieverConfig = None,
    ) -> None:
        self.llm_config = llm_config or LLMConfig()
        self.retriever_config = retriever_config or RetrieverConfig()
        self._retriever = retriever

        # Apply top_k to the retriever if supported
        if hasattr(self._retriever, "search_kwargs"):
            self._retriever.search_kwargs["k"] = self.retriever_config.top_k

        self._llm = self._init_llm()
        self._chain = self._build_chain()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def query(self, question: str) -> Dict[str, Any]:
        """Answer a question using the RAG chain.

        Args:
            question: Natural language question to answer.

        Returns:
            Dictionary with keys:
            - ``question``: the original question.
            - ``answer``: the LLM-generated answer.
            - ``source_documents``: list of retrieved :class:`~langchain_core.documents.Document` objects.
        """
        logger.info("Running RAG query: '%s'", question)
        source_docs = self._retriever.invoke(question)
        answer = self._chain.invoke({"question": question, "context": source_docs})
        return {
            "question": question,
            "answer": answer,
            "source_documents": source_docs,
        }

    def query_with_sources(self, question: str) -> str:
        """Answer a question and return a formatted string with citations.

        Args:
            question: Natural language question to answer.

        Returns:
            Formatted string containing the answer followed by source citations.
        """
        result = self.query(question)
        answer = result["answer"]
        sources: List[Document] = result["source_documents"]

        lines = [f"Answer: {answer}", "", "Sources:"]
        seen = set()
        for doc in sources:
            src = doc.metadata.get("source_file", doc.metadata.get("source", "unknown"))
            if src not in seen:
                seen.add(src)
                lines.append(f"  - {src}")

        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _init_llm(self):
        """Initialise the Replicate LLM.

        Returns:
            A configured LangChain ``Replicate`` LLM instance.
        """
        logger.info(
            "Initialising Replicate LLM: model='%s', temperature=%.2f",
            self.llm_config.model_name,
            self.llm_config.temperature,
        )
        return Replicate(
            model=self.llm_config.model_name,
            replicate_api_token=self.llm_config.api_key,
            model_kwargs={
                "temperature": self.llm_config.temperature,
                "max_new_tokens": self.llm_config.max_tokens,
            },
        )

    def _build_chain(self):
        """Build the LCEL RAG chain.

        Returns:
            A configured LCEL chain that accepts ``question`` and ``context`` inputs.
        """
        return (
            {
                "context": RunnablePassthrough() | (lambda inputs: _format_docs(inputs["context"])),
                "question": RunnablePassthrough() | (lambda inputs: inputs["question"]),
            }
            | RAG_PROMPT
            | self._llm
            | StrOutputParser()
        )

