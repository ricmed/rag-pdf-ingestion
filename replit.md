# rag-pdf-ingestion

## Project Overview

A production-ready Python pipeline that ingests PDF files into a hybrid SQLite + FAISS vector store for Retrieval-Augmented Generation (RAG). Uses [OpenDataLoader PDF](https://github.com/opendataloader-project/opendataloader-pdf) as the parsing engine.

GitHub: https://github.com/ricmed/rag-pdf-ingestion

## Architecture

Single-language Python project with a clean module layout under `rag_pipeline/`:

- `ingest.py` — main ingestion CLI
- `query.py` — RAG query CLI
- `config.py` — shared configuration dataclasses
- `converter.py` — OpenDataLoader PDF wrapper (local + hybrid AI mode)
- `chunker.py` — four chunking strategies (element / section / merged / table)
- `embedder.py` — pluggable embedding backends (Sentence-Transformers, OpenAI, Cohere)
- `vector_store.py` — FAISS index management
- `db.py` — SQLite schema creation and CRUD helpers
- `llm_provider.py` — switchable LLM backends (OpenAI, Anthropic, Ollama)
- `edit_metadata.py` — interactive metadata override before ingesting
- `smoke_test.py` — unit + integration tests

## Key Dependencies

- Python 3.10+
- Java 11+ (required by OpenDataLoader PDF)
- `opendataloader-pdf` — PDF parsing engine
- `sentence-transformers` — default open embedding model
- `faiss-cpu` — approximate nearest-neighbour vector search
- `langchain` — optional LangChain document loader integration
- Optional: `openai`, `cohere`, `anthropic` for proprietary model support

## Data Storage

- `rag_pipeline/data/chunks.db` — SQLite database (documents + chunks tables)
- `rag_pipeline/data/faiss.index` — FAISS flat-L2 index

Both are excluded from git via `.gitignore`.

## Replit Setup

- Python 3.12 module configured
- No web frontend — CLI-only tool
- Workflow: runs the ingest CLI with `--export-options` for reference

## Environment Variables

| Variable | Purpose |
|----------|---------|
| `OPENAI_API_KEY` | OpenAI embeddings + LLM |
| `COHERE_API_KEY` | Cohere embeddings |
| `ANTHROPIC_API_KEY` | Anthropic LLM |
| `OLLAMA_BASE_URL` | Ollama local LLM server URL |
| `GITHUB_PERSONAL_ACCESS_TOKEN` | GitHub push authentication |
