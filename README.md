# 🔗 RAG-Langchain-Lab

![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python)
![LangChain](https://img.shields.io/badge/LangChain-0.2%2B-green?logo=chainlink)
![License: MIT](https://img.shields.io/badge/License-MIT-yellow)

> **Hands-on lab on Retrieval Augmented Generation (RAG) using LangChain and the Replicate API.**

## Overview

RAG-Langchain-Lab is a production-quality RAG pipeline that demonstrates professional ML
engineering patterns. It ingests documents (PDF, TXT, Markdown), creates semantic embeddings,
stores them in a vector database, and answers natural-language questions grounded in the
retrieved context—minimising hallucination and enabling real-time knowledge updates.

### 5-Step RAG Pipeline

```
┌──────────────────────────────────────────────────────────────────┐
│                        RAG Pipeline                              │
│                                                                  │
│  ┌──────────┐   ┌──────────┐   ┌──────────┐   ┌──────────────┐ │
│  │ Document │   │  Chunk   │   │  Embed   │   │    Vector    │ │
│  │  Loader  │──▶│ Splitter │──▶│  Model   │──▶│    Store     │ │
│  │(PDF/TXT/ │   │(Recursive│   │(MiniLM-  │   │(FAISS/Chroma)│ │
│  │  MD)     │   │  Split)  │   │  L6-v2)  │   │              │ │
│  └──────────┘   └──────────┘   └──────────┘   └──────┬───────┘ │
│                                                        │         │
│  ┌─────────────────────────────────────────────────────┼───────┐ │
│  │                  Query Time                         ▼       │ │
│  │  User Query ──▶ Embed Query ──▶ ANN Search ──▶ Top-K Docs  │ │
│  │                                                      │       │ │
│  │                  LLM (Replicate / Llama-2) ◀─────────┘       │ │
│  │                          │                                   │ │
│  │                    Grounded Answer                           │ │
│  └─────────────────────────────────────────────────────────────┘ │
└──────────────────────────────────────────────────────────────────┘
```

## Project Structure

```
RAG-Langchain-Lab/
├── src/
│   ├── __init__.py          # Package exports
│   ├── rag_pipeline.py      # Facade / CLI orchestrator
│   ├── document_loader.py   # PDF, TXT, Markdown ingestion & chunking
│   ├── embedding_store.py   # FAISS / Chroma vector store management
│   └── query_engine.py      # RetrievalQA chain with custom prompt
├── app/
│   └── streamlit_app.py     # Interactive chat UI
├── data/
│   └── sample_docs/
│       └── sample.txt       # Sample RAG knowledge base document
├── tests/
│   ├── __init__.py
│   ├── test_document_loader.py
│   ├── test_embedding_store.py
│   └── test_query_engine.py
├── config/
│   └── settings.py          # Centralised dataclass configuration
├── .env.example
├── .gitignore
├── requirements.txt
└── README.md
```

## Installation

### Prerequisites

- Python 3.10+
- A [Replicate](https://replicate.com/) API token

### Clone & Set Up Virtual Environment

```bash
git clone https://github.com/amitprajapati08/RAG-Langchain-Lab.git
cd RAG-Langchain-Lab

python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

pip install -r requirements.txt
```

### Configure Environment

```bash
cp .env.example .env
# Edit .env and set your REPLICATE_API_TOKEN
```

## Usage

### Python API

```python
from src.rag_pipeline import RAGPipeline

pipeline = RAGPipeline()

# Ingest documents (returns number of chunks created)
chunk_count = pipeline.ingest("data/sample_docs")
print(f"Indexed {chunk_count} chunks")

# Ask a question
result = pipeline.query("What are the benefits of RAG?")
print(result["answer"])

# Include source citations
print(pipeline.query_with_sources("What is RAG?"))
```

### Command-Line Interface

> **Note:** Run all commands from the project root directory so that the `src` and `config` packages are on the Python path.

```bash
# Ingest a directory of documents
python -m src.rag_pipeline ingest data/sample_docs

# Ingest a single file with a custom index location
python -m src.rag_pipeline ingest report.pdf --index-dir /tmp/my_index

# Query the index
python -m src.rag_pipeline query "What is Retrieval Augmented Generation?"

# Query with source citations
python -m src.rag_pipeline query "What are common vector databases?" --sources
```

### Streamlit Web UI

```bash
streamlit run app/streamlit_app.py
```

Open [http://localhost:8501](http://localhost:8501) in your browser. Upload PDF/TXT/MD
files via the sidebar, click **Process Documents**, then ask questions in the chat box.

## Testing

```bash
# Run all tests
pytest

# Run with coverage report
pytest --cov=src --cov-report=term-missing

# Run a specific test file
pytest tests/test_document_loader.py -v
```

## Configuration

All parameters are centralised in `config/settings.py` and can be overridden programmatically
or via environment variables.

| Parameter | Default | Description |
|-----------|---------|-------------|
| `EmbeddingConfig.model_name` | `sentence-transformers/all-MiniLM-L6-v2` | HuggingFace embedding model |
| `EmbeddingConfig.device` | `cpu` | Compute device (`cpu` or `cuda`) |
| `VectorStoreConfig.store_type` | `faiss` | Backend: `faiss` or `chroma` |
| `VectorStoreConfig.persist_directory` | `data/vector_store` | Index save location |
| `ChunkingConfig.chunk_size` | `1000` | Max characters per chunk |
| `ChunkingConfig.chunk_overlap` | `200` | Overlap between chunks |
| `LLMConfig.model_name` | `meta/llama-2-70b-chat:latest` | Replicate model identifier |
| `LLMConfig.temperature` | `0.3` | LLM sampling temperature |
| `LLMConfig.max_tokens` | `512` | Max generated tokens |
| `RetrieverConfig.top_k` | `4` | Documents retrieved per query |

## Future Improvements

- [ ] Hybrid search (BM25 + dense retrieval)
- [ ] Cross-encoder re-ranking of retrieved chunks
- [ ] RAGAS evaluation framework integration
- [ ] Async ingestion and querying
- [ ] Multi-modal support (images, tables)
- [ ] Docker / docker-compose deployment
- [ ] CI/CD pipeline (GitHub Actions)
- [ ] Guardrails / output filtering
- [ ] Multi-tenant vector store namespaces
- [ ] Streaming LLM responses in Streamlit

## License

This project is licensed under the [MIT License](LICENSE).

## Contributing

Contributions are welcome! Please open an issue or submit a pull request with your proposed
changes. Ensure all tests pass (`pytest`) and code is formatted (`black .`, `ruff check .`)
before submitting.
