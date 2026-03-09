"""Interactive Streamlit web interface for the RAG-Langchain-Lab pipeline.

Run with::

    streamlit run app/streamlit_app.py
"""

import logging
import os
import tempfile
from pathlib import Path
from typing import List

import streamlit as st

from config.settings import LLMConfig, RAGConfig, RetrieverConfig
from src.rag_pipeline import RAGPipeline

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Page configuration
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="RAG Langchain Lab",
    page_icon="🔗",
    layout="wide",
)

# ---------------------------------------------------------------------------
# Session state initialisation
# ---------------------------------------------------------------------------

if "pipeline" not in st.session_state:
    st.session_state.pipeline: RAGPipeline = None  # type: ignore[assignment]
if "chat_history" not in st.session_state:
    st.session_state.chat_history: List[dict] = []
if "pipeline_ready" not in st.session_state:
    st.session_state.pipeline_ready: bool = False


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------


def _build_pipeline(top_k: int, temperature: float) -> RAGPipeline:
    """Construct a :class:`~src.rag_pipeline.RAGPipeline` from UI settings.

    Args:
        top_k: Number of documents to retrieve per query.
        temperature: Sampling temperature for the LLM.

    Returns:
        A configured :class:`~src.rag_pipeline.RAGPipeline` instance.
    """
    config = RAGConfig(
        llm=LLMConfig(temperature=temperature),
        retriever=RetrieverConfig(top_k=top_k),
    )
    return RAGPipeline(config=config)


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

with st.sidebar:
    st.title("🔗 RAG Langchain Lab")
    st.markdown("---")

    # --- Settings ---
    st.subheader("⚙️ Settings")
    top_k = st.slider("Top-K documents", min_value=1, max_value=10, value=4, step=1)
    temperature = st.slider(
        "LLM Temperature", min_value=0.0, max_value=1.0, value=0.3, step=0.05
    )

    st.markdown("---")

    # --- Document Upload ---
    st.subheader("📄 Document Upload")
    uploaded_files = st.file_uploader(
        "Upload PDF, TXT, or Markdown files",
        type=["pdf", "txt", "md"],
        accept_multiple_files=True,
    )

    if st.button("⚙️ Process Documents", disabled=not uploaded_files):
        if uploaded_files:
            with st.spinner("Processing documents…"):
                with tempfile.TemporaryDirectory() as tmp_dir:
                    for uploaded_file in uploaded_files:
                        dest = Path(tmp_dir) / uploaded_file.name
                        dest.write_bytes(uploaded_file.read())

                    pipeline = _build_pipeline(top_k, temperature)
                    try:
                        chunk_count = pipeline.ingest(tmp_dir)
                        st.session_state.pipeline = pipeline
                        st.session_state.pipeline_ready = True
                        st.success(f"✅ Ingested {chunk_count} chunks from {len(uploaded_files)} file(s).")
                    except Exception as exc:
                        st.error(f"❌ Ingestion failed: {exc}")
                        logger.exception("Ingestion error")

    st.markdown("---")

    # --- Load Existing Index ---
    st.subheader("💾 Load Existing Index")
    index_dir = st.text_input(
        "Index directory", value="data/vector_store", key="index_dir_input"
    )
    if st.button("📂 Load Existing Index"):
        with st.spinner("Loading index…"):
            pipeline = _build_pipeline(top_k, temperature)
            try:
                pipeline.load_index(index_dir)
                st.session_state.pipeline = pipeline
                st.session_state.pipeline_ready = True
                st.success(f"✅ Index loaded from '{index_dir}'.")
            except Exception as exc:
                st.error(f"❌ Failed to load index: {exc}")
                logger.exception("Load index error")

# ---------------------------------------------------------------------------
# Main area
# ---------------------------------------------------------------------------

st.title("💬 RAG Chat Interface")

if not st.session_state.pipeline_ready:
    st.info(
        "👈 Upload documents and click **Process Documents**, "
        "or load an existing index from the sidebar to get started."
    )

# --- Render chat history ---
for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message["role"] == "assistant" and message.get("sources"):
            with st.expander("📚 Source Documents"):
                for src_doc in message["sources"]:
                    src = src_doc.metadata.get(
                        "source_file", src_doc.metadata.get("source", "unknown")
                    )
                    chunk_idx = src_doc.metadata.get("chunk_index", "?")
                    st.markdown(f"**Source:** `{src}` (chunk {chunk_idx})")
                    st.markdown(f"> {src_doc.page_content[:300]}…")
                    st.markdown("---")

# --- Chat input ---
if prompt := st.chat_input(
    "Ask a question about your documents…",
    disabled=not st.session_state.pipeline_ready,
):
    # Display user message
    st.session_state.chat_history.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Generate and display assistant response
    with st.chat_message("assistant"):
        with st.spinner("Thinking…"):
            try:
                result = st.session_state.pipeline.query(prompt)
                answer = result["answer"]
                sources = result.get("source_documents", [])

                st.markdown(answer)

                if sources:
                    with st.expander("📚 Source Documents"):
                        for src_doc in sources:
                            src = src_doc.metadata.get(
                                "source_file",
                                src_doc.metadata.get("source", "unknown"),
                            )
                            chunk_idx = src_doc.metadata.get("chunk_index", "?")
                            st.markdown(f"**Source:** `{src}` (chunk {chunk_idx})")
                            st.markdown(f"> {src_doc.page_content[:300]}…")
                            st.markdown("---")

                st.session_state.chat_history.append(
                    {
                        "role": "assistant",
                        "content": answer,
                        "sources": sources,
                    }
                )
            except Exception as exc:
                error_msg = f"❌ Error: {exc}"
                st.error(error_msg)
                st.session_state.chat_history.append(
                    {"role": "assistant", "content": error_msg, "sources": []}
                )
                logger.exception("Query error")
